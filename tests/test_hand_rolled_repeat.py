"""Repeat-call checks: every call's tokens must pass the judge's replay rule.

The judge runs one warmup generation and then several samples against one live
Engine, so a fault that needs a particular cache or draft-table state shows up
on a later sample and not on the first. Interleaving prompts and re-running
each one exercises that, and replaying every call teacher-forced says whether a
disagreement between calls is a harmless tie or a token the judge would reject.
"""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

try:
    import torch
    GPU = torch.cuda.is_available()
except ImportError:
    GPU = False

MARGIN = 2.0


@unittest.skipUnless(GPU, 'CUDA required for repeat-call tests')
class HandRolledRepeatTests(unittest.TestCase):
    def load_engine(self):
        model_path = os.getenv('QWEN_MODEL_PATH', '/weights/qwen3-4b')
        if not Path(model_path).is_dir():
            self.skipTest('pinned local checkpoint required')
        root = Path(__file__).resolve().parents[1] / 'engine'
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location('repeat_engine', root / 'engine.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, module.Engine(model_path)

    def prompts(self, batch, length, vocab):
        """Random ids reject most drafts; a repeated block accepts most of them."""
        torch.manual_seed(1571)
        block = torch.randint(100, vocab, (batch, 48))
        repeated = block.repeat(1, length // 48 + 1)[:, :length]
        mixed = torch.randint(100, vocab, (batch, length))
        mixed[:, length // 2:] = repeated[:, :length - length // 2]
        return {'random': torch.randint(100, vocab, (batch, length)).tolist(),
                'repeated': repeated.tolist(), 'mixed': mixed.tolist()}

    def worst_margin(self, model, prompt, generated, rows=8):
        """Largest gap between native's argmax and our token, teacher-forced on our own prefix."""
        worst, where = 0.0, None
        for start in range(0, prompt.shape[0], rows):
            prefix = torch.cat((prompt[start:start + rows], generated[start:start + rows]), dim=1)
            logits = model(prefix, use_cache=False).logits
            for first in range(0, generated.shape[1], 32):
                last = min(generated.shape[1], first + 32)
                chunk = logits[:, prompt.shape[1] - 1 + first:prompt.shape[1] - 1 + last].float()
                ids = generated[start:start + rows, first:last]
                self.assertTrue(torch.isfinite(chunk).all().item())
                margin = chunk.max(-1).values - chunk.gather(2, ids[..., None])[..., 0]
                peak = margin.max().item()
                if peak > worst:
                    bad = torch.nonzero(margin == margin.max())[0].tolist()
                    worst, where = peak, (start + bad[0], first + bad[1])
            del logits
        return worst, where

    def run_shape(self, engine, batch, length, output, rounds=3):
        """Warm up, then interleave prompts; replay every call and report disagreements."""
        vocab = engine.model.config.vocab_size
        prompts = self.prompts(batch, length, vocab)
        label = f'batch {batch} prompt {length} output {output}'
        failures, first = [], {}
        with torch.inference_mode():
            engine.shape = None
            list(engine.generate(prompts['random'], output))
            for call, name in enumerate([n for _ in range(rounds) for n in prompts]):
                generated = list(engine.generate(prompts[name], output))
                self.assertEqual(len(generated), output)
                self.assertTrue(all(len(step) == batch for step in generated))
                tokens = torch.tensor(generated, device='cuda').t().contiguous()
                worst, where = self.worst_margin(engine.model, torch.tensor(prompts[name], device='cuda'), tokens)
                same = first.setdefault(name, tokens)
                agree = bool(torch.equal(tokens, same))
                print(f'{label} call {call} {name}: worst margin {worst:.4f} at {where}; '
                      f'agrees with first {name} call: {agree}; rounds {engine.rounds}', flush=True)
                if worst > MARGIN:
                    failures.append(f'{label} call {call} ({name}) margin {worst:.4f} at sequence/token {where}')
        if failures:
            self.fail('; '.join(failures))

    def test_repeat_calls_pass_replay_on_hidden_like_shapes(self):
        module, engine = self.load_engine()
        # 32/256/64 is the shape that failed one sample of a six-workload bench.
        for batch, length, output in ((32, 256, 64), (16, 512, 128)):
            with self.subTest(batch=batch, prompt=length, output=output):
                self.run_shape(engine, batch, length, output, rounds=2)

    def test_repeat_calls_pass_replay_with_single_token_decode(self):
        """The same check with one-node trees: isolates the draft path from the rest."""
        module, engine = self.load_engine()
        original = dict(module.DECODE_CONFIG)
        try:
            module.DECODE_CONFIG.update(draft_nodes={1: 1})
            self.run_shape(engine, 16, 512, 128, rounds=2)
        finally:
            module.DECODE_CONFIG.clear()
            module.DECODE_CONFIG.update(original)


if __name__ == '__main__':
    unittest.main()
