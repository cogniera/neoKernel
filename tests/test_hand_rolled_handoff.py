"""Pinned-checkpoint GPU checks for prefill handoff, graph capture, and reset."""

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


@unittest.skipUnless(GPU, 'CUDA required for full-model handoff tests')
class HandRolledHandoffTests(unittest.TestCase):
    def test_public_shapes_graph_reuse_and_native_prefix(self):
        model_path = os.getenv('QWEN_MODEL_PATH', '/weights/qwen3-4b')
        if not Path(model_path).is_dir():
            self.skipTest('pinned local checkpoint required')
        root = Path(__file__).resolve().parents[1] / 'engine'
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location('handoff_engine', root / 'engine.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        started = time.perf_counter()
        engine = module.Engine(model_path)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        original_config = dict(module.DECODE_CONFIG)
        try:
            with torch.inference_mode():
                for impl, layout, shape in itertools.product(
                        ('triton', 'sdpa_grouped'), ('bhsd', 'bshd'),
                        ((1, 512, 32), (4, 2048, 32), (16, 512, 128))):
                    batch, length, output = shape
                    module.DECODE_CONFIG.update(attention_impl=impl, kv_layout=layout)
                    engine.shape = None
                    torch.manual_seed(391)
                    warm = torch.randint(100, engine.model.config.vocab_size, (batch, length)).tolist()
                    started = time.perf_counter()
                    generated = list(engine.generate(warm, output))
                    torch.cuda.synchronize()
                    warmup_s = time.perf_counter() - started
                    self.assertEqual(len(generated), output)
                    self.assertTrue(all(len(step) == batch for step in generated))
                    self.assertLess(load_s + warmup_s, 120, 'load plus warmup exceeds local target')
                    graph = engine.graph
                    addresses = (engine.token_ids.data_ptr(), engine.buffers.x.data_ptr(),
                                 engine.cache.key_cache[0].data_ptr())
                    for sample in range(2):
                        prompt = torch.randint(100, engine.model.config.vocab_size, (batch, length), device='cuda')
                        generator = engine.generate(prompt.tolist(), output)
                        first = next(generator)
                        second = next(generator)
                        actual_logits = engine.logits.clone()
                        generator.close()
                        self.assertIs(engine.graph, graph)
                        self.assertEqual(addresses, (engine.token_ids.data_ptr(), engine.buffers.x.data_ptr(),
                                                     engine.cache.key_cache[0].data_ptr()))
                        first_ids = torch.tensor(first, device='cuda').view(batch, 1)
                        second_ids = torch.tensor(second, device='cuda').view(batch, 1)
                        first_logits = engine.model(prompt, use_cache=False, logits_to_keep=1).logits[:, -1]
                        prefix = torch.cat((prompt, first_ids), dim=1)
                        ref = engine.model(prefix, use_cache=False, logits_to_keep=1).logits[:, -1]
                        for label, logits, ids in (('prefill', first_logits, first_ids), ('decode', ref, second_ids)):
                            margin = logits.float().max(-1).values - logits.float().gather(1, ids).squeeze(1)
                            self.assertTrue(torch.isfinite(logits).all().item())
                            self.assertLessEqual(margin.max().item(), 2.0)
                            print(json.dumps(dict(test=f'handoff/{impl}/{layout}/{shape}/{sample}/{label}',
                                                  max_margin=margin.max().item(),
                                                  argmax_agreement=(logits.argmax(-1) == ids[:, 0]).float().mean().item(),
                                                  max_abs_diff=(actual_logits.float()-ref.float()).abs().max().item()
                                                  if label == 'decode' else None,
                                                  load_s=load_s, warmup_s=warmup_s)), flush=True)
        finally:
            module.DECODE_CONFIG.clear()
            module.DECODE_CONFIG.update(original_config)
            del engine
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
