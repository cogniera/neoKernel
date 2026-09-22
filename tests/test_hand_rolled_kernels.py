"""GPU-only numerical tests, before engine integration; unittest skips on CPU.

Run with: python -m unittest discover -s tests -p test_hand_rolled_kernels.py -v
All numeric reports include max absolute error and last-dimension argmax agreement.
The test tolerances diagnose individual BF16 operations; sequence correctness is
still decided exclusively by the unchanged neokernel judge.
"""

import itertools
import json
from pathlib import Path
import sys
import unittest

try:
    import torch
    GPU = torch.cuda.is_available()
except ImportError:
    GPU = False


class GroupedSDPALayoutTests(unittest.TestCase):
    def test_copy_preserves_head_order_for_noncontiguous_sdpa_output(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch required')
        # SDPA may return B,H,Q,D backed by B,Q,H,D. Flattening H,Q with
        # view is invalid; copy into a matching grouped destination instead.
        expected = torch.arange(2*8*4*128).reshape(2, 8, 4, 128)
        output = expected.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(output.is_contiguous())
        storage = torch.empty((2, 32, 128), dtype=expected.dtype)
        storage.view(2, 8, 4, 128).copy_(output)
        torch.testing.assert_close(storage, expected.reshape(2, 32, 128))


@unittest.skipUnless(GPU, 'CUDA required for Triton decode tests')
class HandRolledKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'engine'))
        import triton
        import transformers
        if triton.__version__ != '3.1.0' or transformers.__version__ != '4.51.3':
            raise RuntimeError('GPU tests require pinned Triton 3.1.0 / Transformers 4.51.3')
        torch.backends.cuda.matmul.allow_tf32 = False
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        cls.config = Qwen3Config(hidden_size=2560, intermediate_size=9728,
                                num_attention_heads=32, num_key_value_heads=8,
                                head_dim=128, num_hidden_layers=1, rope_theta=5e6,
                                rms_norm_eps=1e-6)
        cls.config._attn_implementation = 'sdpa'
        rotary = Qwen3RotaryEmbedding(cls.config, device='cuda')
        cls.cos, cls.sin = rotary(torch.empty(1, device='cuda', dtype=torch.bfloat16),
                                 torch.arange(2080, device='cuda')[None])
        cls.cos, cls.sin = cls.cos[0].contiguous(), cls.sin[0].contiguous()

    def setUp(self):
        torch.manual_seed(714)

    def report(self, label, actual, expected, atol, rtol=0.01):
        a, e = actual.float(), expected.float()
        result = {'test': label, 'max_abs_diff': (a-e).abs().max().item(),
                  'argmax_agreement': (a.argmax(-1) == e.argmax(-1)).float().mean().item()}
        print(json.dumps(result), flush=True)
        torch.testing.assert_close(a, e, atol=atol, rtol=rtol)

    def rand(self, *shape):
        return torch.randn(shape, device='cuda', dtype=torch.bfloat16)

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_norm_and_residual(self):
        from kernels.elementwise import norm_out
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        native = Qwen3RMSNorm(2560, eps=1e-6).cuda().bfloat16()
        native.weight.copy_(self.rand(2560))
        for batch, residual in itertools.product((1, 4, 16), (False, True)):
            x, r, out = self.rand(batch, 2560), self.rand(batch, 2560), self.rand(batch, 2560)
            expected_sum = x + r
            norm_out(x, native.weight, out, residual=r if residual else None,
                     summed=r if residual else None)
            if residual:
                torch.testing.assert_close(r, expected_sum, atol=0, rtol=0)
            self.report(f'norm/{batch}/{residual}', out, native(expected_sum if residual else x), .03125)

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_qk_rope_cache(self):
        from kernels.elementwise import qk_rope_cache_out
        from kernels.decode import cache_storage
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb
        qn, kn = [Qwen3RMSNorm(128, eps=1e-6).cuda().bfloat16() for _ in range(2)]
        qn.weight.copy_(self.rand(128)); kn.weight.copy_(self.rand(128))
        for batch, layout, fused in itertools.product((1, 4, 16), ('bhsd', 'bshd'), (False, True)):
            p = self.rand(batch, 6144)
            q = self.rand(batch, 32, 128)
            k, v = [cache_storage(batch, 2080, 'cuda', layout) for _ in range(2)]
            for pos in (0, 512, 2048, 2079):
                position = torch.tensor([pos], device='cuda')
                qk_rope_cache_out(p, qn.weight, kn.weight, self.cos, self.sin, position,
                                 q, k, v, torch.empty_like(p), fused)
                qref = qn(p[:, :4096].reshape(batch, 32, 1, 128))
                kref = kn(p[:, 4096:5120].reshape(batch, 8, 1, 128))
                qr, kr = apply_rotary_pos_emb(qref, kref, self.cos[None, pos:pos+1], self.sin[None, pos:pos+1])
                label = f'qk/{batch}/{layout}/{fused}/{pos}'
                self.report(label+'/q', q, qr[:, :, 0], .0625)
                self.report(label+'/k', k[:, :, pos], kr[:, :, 0], .0625)
                torch.testing.assert_close(v[:, :, pos], p[:, 5120:].reshape(batch, 8, 128), atol=0, rtol=0)
            self.assertEqual(k[:, :, 1:512].count_nonzero().item(), 0)

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_silu(self):
        from kernels.elementwise import silu_mul_out
        from transformers.activations import ACT2FN
        for batch, fused in itertools.product((1, 4, 16), (False, True)):
            p = self.rand(batch, 19456)
            out = self.rand(batch, 9728)
            silu_mul_out(p, out, torch.empty_like(out), fused)
            ref = ACT2FN['silu'](p[:, :9728]) * p[:, 9728:]
            self.report(f'silu/{batch}/{fused}', out, ref, .03125)

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_attention(self):
        from kernels.attention import attention_out
        from kernels.decode import cache_storage, DecodeBuffers
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv
        import torch.nn.functional as F
        for batch, capacity, layout in itertools.product((1, 4, 16), (640, 2080), ('bhsd', 'bshd')):
            q = self.rand(batch, 32, 128)
            k, v = [cache_storage(batch, capacity, 'cuda', layout) for _ in range(2)]
            k.copy_(self.rand(*k.shape)); v.copy_(self.rand(*v.shape))
            buf = DecodeBuffers(batch, capacity, 'cuda')
            for pos in (0, 127, 512, capacity-1):
                position = torch.tensor([pos], device='cuda')
                attention_out(q, k, v, position, buf.attn, buf.partial, buf.lse)
                ref = F.scaled_dot_product_attention(q[:, :, None], repeat_kv(k[:, :, :pos+1], 4),
                                                      repeat_kv(v[:, :, :pos+1], 4))[:, :, 0]
                self.report(f'attn/{batch}/{capacity}/{layout}/{pos}', buf.attn, ref, .015625)
            # Poison inactive slots: masked capacity must have no influence.
            k[:, :, 513:].fill_(float('nan')); v[:, :, 513:].fill_(float('nan'))
            position.fill_(512)
            attention_out(q, k, v, position, buf.attn, buf.partial, buf.lse)
            self.assertTrue(buf.attn.isfinite().all().item())

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_layer_all_switch_combinations(self):
        from kernels.decode import LayerWeights, DecodeBuffers, cache_storage
        from transformers import StaticCache
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        native = Qwen3DecoderLayer(self.config, 0).cuda().bfloat16().eval()
        # DecoderLayer alone does not run PreTrainedModel's Qwen initialization.
        for name, p in native.named_parameters():
            if 'norm' not in name:
                p.normal_(0, .02)
        w = LayerWeights(native)
        for batch in (1, 4, 16):
            capacity, pos = 640, 512
            x = self.rand(batch, 1, 2560)
            prefix_k, prefix_v = [self.rand(batch, 8, pos, 128) for _ in range(2)]
            position = torch.tensor([pos], device='cuda')
            mask = torch.arange(capacity, device='cuda')[None, None, None] <= position
            cache = StaticCache(self.config, max_batch_size=batch, max_cache_len=capacity,
                                device='cuda', dtype=torch.bfloat16)
            cache.key_cache[0][:, :, :pos].copy_(prefix_k)
            cache.value_cache[0][:, :, :pos].copy_(prefix_v)
            ref = native(x, attention_mask=mask, position_ids=position[None], past_key_value=cache,
                         cache_position=position, use_cache=True,
                         position_embeddings=(self.cos[None, pos:pos+1], self.sin[None, pos:pos+1]))[0]
            for nr, qk, sm, attn, layout in itertools.product((False, True), (False, True), (False, True),
                                                            ('triton',), ('bhsd', 'bshd')):
                config = dict(fuse_norm_residual=nr, fuse_qk_norm_rope=qk, fuse_silu_mul=sm,
                              attention_impl=attn, kv_layout=layout)
                buf = DecodeBuffers(batch, capacity, 'cuda', config)
                k, v = [cache_storage(batch, capacity, 'cuda', layout) for _ in range(2)]
                k[:, :, :pos].copy_(prefix_k); v[:, :, :pos].copy_(prefix_v)
                buf.x.copy_(x[:, 0])
                buf.layer(w, k, v, position, self.cos, self.sin)
                self.report(f'layer/{batch}/{config}', buf.x, ref[:, 0], .125, .03)


if __name__ == '__main__':
    unittest.main()
