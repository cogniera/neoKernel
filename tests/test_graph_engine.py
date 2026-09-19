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
        from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache
        path = Path(__file__).resolve().parents[1]/'engine/engine.py'
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location('tested_graph_engine', path)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        torch.manual_seed(17)
        config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).eval()
        cache = engine.PrefixStaticCache(config, max_batch_size=2, max_cache_len=9, device='cpu', dtype=torch.float32,
                                         layout=layout)
        mask = torch.zeros((2, 1, 1, 9), dtype=torch.bool)
        with torch.inference_mode():
            for _ in range(2):
                # A fresh prompt must overwrite every active cache slot and
                # attention must never consume the unused capacity.
                for tensor in cache.key_cache + cache.value_cache:
                    tensor.fill_(123.0)
                mask.zero_()
                mask[:, :, :, :5].fill_(True)
                sequence = torch.randint(0, 32, (2, 5))
                current = sequence
                for start in [0, 5, 6, 7]:
                    positions = torch.arange(start, start+current.shape[1])
                    if start:
                        mask.index_fill_(3, positions, True)
                    cache.prefill_length = 0 if start else 5
                    actual = engine.qwen_forward(model, current, cache, positions, positions[None], mask if start else None)
                    expected = model(sequence, use_cache=False, logits_to_keep=1).logits
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                    current = actual[:, -1].argmax(-1, keepdim=True)
                    sequence = torch.cat((sequence, current), dim=1)
