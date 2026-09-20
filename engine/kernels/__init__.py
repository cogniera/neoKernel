"""Decode choices; numeric values are compatible with neokernel.sweep."""

CONFIG = {'fuse_norm_residual': True, 'fuse_qk_norm_rope': True, 'fuse_silu_mul': True, 'attention_impl': 'triton', 'kv_layout': 'bhsd', 'skinny_gemm': False,
          'draft_nodes': {1: 64, 2: 32, 3: 16, 5: 8, 17: 4, 33: 2, 65: 1}, 'draft_ranks': 8, 'pace': {1: 1.5, 2: 1.3, 3: 1.15},
          'prefill_impl': 'fused', 'warm_window': 1024, 'ngram': 3}

TUNABLES = {
    "norm.BLOCK": 4096,
    "norm.num_warps": 4,
    "norm.num_stages": 1,
    "qk.BLOCK": 128,
    "qk.num_warps": 4,
    "qk.num_stages": 1,
    "silu.BLOCK": 1024,
    "silu.num_warps": 4,
    "silu.num_stages": 1,
    "attention.BLOCK": 128,
    "attention.num_warps": 4,
    "attention.num_stages": 1,
    "merge.num_warps": 4,
    "merge.num_stages": 1,
    "gemm.BLOCK": 128,
    "gemm.num_warps": 4,
    "gemm.num_stages": 4,
}
