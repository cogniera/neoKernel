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
