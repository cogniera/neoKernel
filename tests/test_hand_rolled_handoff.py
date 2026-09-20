"""Pinned-checkpoint GPU checks: every emitted token passes the judge's replay rule."""

import gc
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time
import unittest

try:
    import torch
    GPU = torch.cuda.is_available()
except ImportError:
    GPU = False

TEXT = ("The city council met on Tuesday evening to discuss the proposed changes to the downtown "
        "parking regulations. Council members heard from residents, business owners, and transit "
        "advocates, many of whom expressed concern that the new rules would reduce access to local "
        "shops. The mayor opened the meeting by summarizing the report prepared by the transportation "
        "department, which recommended replacing free two-hour parking with metered spaces along the "
        "main street and adding a bicycle lane on the east side. Several speakers argued that the plan "
        "ignored the needs of elderly residents who depend on their cars, while others said the change "
        "was long overdue and would make the street safer for pedestrians. After nearly three hours of "
        "debate, the council voted to postpone the decision until a revised study could be completed. "
        "The study will examine traffic patterns during the holiday season, when the downtown area "
        "receives its highest number of visitors, and will include a survey of shoppers and employees. ")


@unittest.skipUnless(GPU, 'CUDA required for full-model handoff tests')
class HandRolledHandoffTests(unittest.TestCase):
    def replay_margin(self, model, prompt, generated):
        """Largest gap between native's argmax and each emitted token, teacher-forced on our prefix."""
        prefix = torch.cat((prompt, generated), dim=1)
        logits = model(prefix, use_cache=False).logits
        worst, agreement, total, first_bad = 0.0, 0, 0, None
        length = generated.shape[1]
        for start in range(0, length, 32):
            stop = min(length, start + 32)
            chunk = logits[:, prompt.shape[1] - 1 + start:prompt.shape[1] - 1 + stop].float()
            ids = generated[:, start:stop]
            self.assertTrue(torch.isfinite(chunk).all().item())
            margin = chunk.max(-1).values - chunk.gather(2, ids[..., None])[..., 0]
            worst = max(worst, margin.max().item())
            if first_bad is None and (margin > 2.0).any().item():
                bad = torch.nonzero(margin > 2.0)[0].tolist()
                first_bad = (bad[0], start + bad[1])
            agreement += (chunk.argmax(-1) == ids).sum().item()
            total += ids.numel()
        return worst, agreement / total, first_bad

    def test_public_shapes_graph_reuse_and_native_prefix(self):
        model_path = os.getenv('QWEN_MODEL_PATH', '/weights/qwen3-4b')
        if not Path(model_path).is_dir():
            self.skipTest('pinned local checkpoint required')
        root = Path(__file__).resolve().parents[1] / 'engine'
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location('handoff_engine', root / 'engine.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from transformers import AutoTokenizer
        from test_hand_rolled_corpus import TEXT as CORPUS, WIKI
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        text_ids = tokenizer(TEXT * 4)['input_ids']
        corpora = [torch.tensor(tokenizer(source)['input_ids']) for source in (CORPUS, WIKI)]
        started = time.perf_counter()
        engine = module.Engine(model_path)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        original_config = dict(module.DECODE_CONFIG)
        vocab = engine.model.config.vocab_size
        try:
            with torch.inference_mode():
                variants = (('bhsd', 'fused'), ('bshd', 'fused'), ('bhsd', 'native'))
                for (layout, impl), shape in itertools.product(variants, ((1, 512, 32), (4, 2048, 32), (16, 512, 128))):
                    batch, length, output = shape
                    module.DECODE_CONFIG.update(kv_layout=layout, prefill_impl=impl)
                    engine.shape = None
                    torch.manual_seed(391)
                    warm = torch.randint(100, vocab, (batch, length)).tolist()
                    started = time.perf_counter()
                    generated = list(engine.generate(warm, output))
                    torch.cuda.synchronize()
                    warmup_s = time.perf_counter() - started
                    self.assertEqual(len(generated), output)
                    self.assertTrue(all(len(step) == batch for step in generated))
                    self.assertLess(load_s + warmup_s, 150, 'load plus warmup exceeds local target')
                    graph = engine.graph
                    prefill_graph = engine.prefill_graph
                    addresses = (engine.token_ids.data_ptr(), engine.buffers.x.data_ptr(),
                                 engine.cache.key_cache[0].data_ptr())
                    # Prompts: random ids (drafts mostly rejected), a repeated random block
                    # (drafts mostly accepted), and natural text for batch one.
                    block = torch.randint(100, vocab, (batch, 48), device='cuda')
                    prompts = [torch.randint(100, vocab, (batch, length), device='cuda'),
                               block.repeat(1, length // 48 + 1)[:, :length]]
                    if batch == 1:
                        prompts.append(torch.tensor(text_ids[:length], device='cuda')[None])
                    # Natural text (prose, then encyclopedic), a different chunk for every
                    # sequence, two samples of each per shape.
                    for corpus_ids in corpora:
                        chunks = corpus_ids[:corpus_ids.numel() // length * length].view(-1, length)
                        for sample in range(2 if impl == 'fused' and layout == 'bhsd' else 0):
                            rows = (torch.arange(batch) * 3 + sample * 7) % chunks.shape[0]
                            prompts.append(chunks[rows].cuda())
                    for sample, prompt in enumerate(prompts):
                        started = time.perf_counter()
                        steps = list(engine.generate(prompt.tolist(), output))
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        self.assertIs(engine.graph, graph)
                        self.assertIs(engine.prefill_graph, prefill_graph)
                        self.assertEqual(addresses, (engine.token_ids.data_ptr(), engine.buffers.x.data_ptr(),
                                                     engine.cache.key_cache[0].data_ptr()))
                        self.assertEqual(len(steps), output)
                        ids = torch.tensor(steps, device='cuda').t().contiguous()
                        worst, agreement, first_bad = self.replay_margin(engine.model, prompt, ids)
                        print(json.dumps(dict(test=f'handoff/{layout}/{impl}/{shape}/{sample}', max_margin=worst,
                                              first_bad=first_bad, argmax_agreement=agreement, rounds=engine.rounds,
                                              tokens_per_round=(output - 1) / max(1, engine.rounds),
                                              nodes=engine.tree.nodes, seconds=elapsed,
                                              load_s=load_s, warmup_s=warmup_s)), flush=True)
                        self.assertLessEqual(worst, 2.0, f'first bad (sequence, position): {first_bad}')
        finally:
            module.DECODE_CONFIG.clear()
            module.DECODE_CONFIG.update(original_config)
            del engine
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
