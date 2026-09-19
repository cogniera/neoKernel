import importlib.util
from pathlib import Path
import unittest


class GraphEngineTests(unittest.TestCase):
    def test_causal_prefill_boolean_decode_and_reset(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache
        path = Path(__file__).resolve().parents[1]/'engine/engine.py'
        spec = importlib.util.spec_from_file_location('tested_graph_engine', path)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        torch.manual_seed(17)
        config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).eval()
        cache = engine.PrefixStaticCache(config, max_batch_size=2, max_cache_len=9, device='cpu', dtype=torch.float32)
        mask = torch.zeros((2, 1, 1, 9), dtype=torch.bool)
        with torch.inference_mode():
            for _ in range(2):
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
