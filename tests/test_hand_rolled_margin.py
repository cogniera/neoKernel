"""Numeric headroom: how close the engine runs to the judge's 2.0 tie budget.

The judge draws a fresh uniform-random prompt for every sample, so a workload
fails when some draw lands on a position where the engine's logits deviate far
enough from a teacher-forced replay to change the argmax. What matters is not
whether one run passes but how much margin is left, and which fused operation
spends it. Each config variant is measured over the same prompts so the numbers
are comparable, and the fusions are CONFIG flags, so ablating them is free.
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

BUDGET = 2.0
SHAPES = ((16, 512, 128), (32, 256, 64))
VARIANTS = (('all fused', {}),
            ('no fuse_norm_residual', {'fuse_norm_residual': False}),
            ('no fuse_qk_norm_rope', {'fuse_qk_norm_rope': False}),
            ('no fuse_silu_mul', {'fuse_silu_mul': False}),
            ('bshd cache', {'kv_layout': 'bshd'}))


@unittest.skipUnless(GPU, 'CUDA required for margin tests')
class HandRolledMarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.getenv('QWEN_MODEL_PATH', '/weights/qwen3-4b')
        if not Path(model_path).is_dir():
            raise unittest.SkipTest('pinned local checkpoint required')
        root = Path(__file__).resolve().parents[1] / 'engine'
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location('margin_engine', root / 'engine.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.engine = cls.module.Engine(model_path)
        cls.baseline = dict(cls.module.DECODE_CONFIG)

    def prompts(self, batch, length, draws):
        """Uniform random ids, the same draw the judge uses (make_prompt)."""
        vocab = self.engine.model.config.vocab_size
        generator = torch.Generator(device='cpu').manual_seed(20260922)
        return [torch.randint(100, vocab, (batch, length), generator=generator).tolist()
                for _ in range(draws)]

    def worst_margin(self, prompt, generated, rows=4):
        """Largest gap between native's argmax and our token, teacher-forced on our prefix."""
        prompt = torch.tensor(prompt, device='cuda')
        worst, where = 0.0, None
        for start in range(0, prompt.shape[0], rows):
            prefix = torch.cat((prompt[start:start + rows], generated[start:start + rows]), dim=1)
            logits = self.engine.model(prefix, use_cache=False).logits
            for first in range(0, generated.shape[1], 32):
                last = min(generated.shape[1], first + 32)
                chunk = logits[:, prompt.shape[1] - 1 + first:prompt.shape[1] - 1 + last].float()
                ids = generated[start:start + rows, first:last]
                self.assertTrue(torch.isfinite(chunk).all().item())
                margin = chunk.max(-1).values - chunk.gather(2, ids[..., None])[..., 0]
                if margin.max().item() > worst:
                    bad = torch.nonzero(margin == margin.max())[0].tolist()
                    worst, where = margin.max().item(), (start + bad[0], first + bad[1])
            del logits
        return worst, where

    def measure(self, batch, length, output, draws=3):
        worst, where = 0.0, None
        with torch.inference_mode():
            for prompt in self.prompts(batch, length, draws):
                self.engine.shape = None
                generated = torch.tensor(list(self.engine.generate(prompt, output)),
                                         device='cuda').t().contiguous()
                peak, at = self.worst_margin(prompt, generated)
                if peak > worst:
                    worst, where = peak, at
        return worst, where

    def test_margin_headroom_by_variant(self):
        """Report the worst margin each fusion leaves; fail only if one exceeds the budget."""
        over = []
        for batch, length, output in SHAPES:
            for label, overrides in VARIANTS:
                self.module.DECODE_CONFIG.clear()
                self.module.DECODE_CONFIG.update(self.baseline)
                self.module.DECODE_CONFIG.update(overrides)
                worst, where = self.measure(batch, length, output)
                print(f'batch {batch} prompt {length} output {output} | {label:<24} | '
                      f'worst margin {worst:.4f} at {where}', flush=True)
                if worst > BUDGET:
                    over.append(f'{batch}/{length}/{output} {label}: {worst:.4f} at {where}')
        self.module.DECODE_CONFIG.clear()
        self.module.DECODE_CONFIG.update(self.baseline)
        if over:
            self.fail('; '.join(over))


if __name__ == '__main__':
    unittest.main()
