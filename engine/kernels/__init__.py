"""Decode choices; numeric values are compatible with neokernel.sweep."""

CONFIG = {'fuse_norm_residual': True, 'fuse_qk_norm_rope': True, 'fuse_silu_mul': True, 'attention_impl': 'triton', 'kv_layout': 'bhsd'}

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
}


def _select_attention(batch, prompt_length, output_length):
    if batch == 1:
        return 'sdpa_grouped', 'bshd'
    return 'triton', 'bhsd'


def configure_for_shape(batch, prompt_length, output_length):
    impl, layout = _select_attention(batch, prompt_length, output_length)
    CONFIG['attention_impl'] = impl
    CONFIG['kv_layout'] = layout
