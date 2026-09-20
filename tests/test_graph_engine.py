import importlib.util
from pathlib import Path
import unittest
import sys


class GraphEngineTests(unittest.TestCase):
    def test_packed_weights_preserve_prefill_and_share_storage(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        root = Path(__file__).resolve().parents[1] / 'engine'
        sys.path.insert(0, str(root))
        from kernels.weights import LayerWeights
        torch.manual_seed(23)
        config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).eval()
        ids = torch.randint(0, 32, (2, 5))
        with torch.inference_mode():
            before = model(ids, use_cache=False).logits
            weights = [LayerWeights(layer) for layer in model.model.layers]
            after = model(ids, use_cache=False).logits
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        for layer, w in zip(model.model.layers, weights):
            for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                self.assertEqual(projection.weight.untyped_storage().data_ptr(), w.qkv_weight.untyped_storage().data_ptr())
                self.assertTrue(projection.weight.is_contiguous())
            for projection in (layer.mlp.gate_proj, layer.mlp.up_proj):
                self.assertEqual(projection.weight.untyped_storage().data_ptr(), w.gate_up_weight.untyped_storage().data_ptr())
                self.assertTrue(projection.weight.is_contiguous())

    def test_causal_prefill_boolean_decode_and_reset(self):
        for layout in ('bhsd', 'bshd'):
            with self.subTest(layout=layout):
                self._check_prefill_and_reset(layout)

    def _check_prefill_and_reset(self, layout):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('grouped-query flash prefill needs CUDA')
        from transformers import Qwen3Config, Qwen3ForCausalLM
        path = Path(__file__).resolve().parents[1]/'engine/engine.py'
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location('tested_graph_engine', path)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        torch.manual_seed(17)
        config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).eval().cuda().bfloat16()
        shape = (2, 2, 2, 9, 16) if layout == 'bhsd' else (2, 2, 9, 2, 16)
        storage = [torch.zeros(shape, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        if layout == 'bshd':
            storage = [s.transpose(2, 3) for s in storage]
        cache = engine.PrefixCache(*storage)
        with torch.inference_mode():
            for _ in range(2):
                # A fresh prompt must overwrite every active cache slot; the
                # prefill writes only the prompt's slots and returns native K/V.
                for tensor in cache.key_cache + cache.value_cache:
                    tensor.fill_(123.0)
                sequence = torch.randint(0, 32, (2, 5), device='cuda')
                positions = torch.arange(5, device='cuda')
                cache.prefill_length = 5
                actual = engine.qwen_forward(model, sequence, cache, positions, positions[None], None)
                expected = model(sequence, use_cache=False, logits_to_keep=1).logits
                torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)
                for tensor in cache.key_cache + cache.value_cache:
                    self.assertTrue((tensor[:, :, 5:] == 123.0).all().item())
                    self.assertFalse((tensor[:, :, :5] == 123.0).all().item())
