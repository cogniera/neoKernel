"""Loaded weight views and reusable scratch for a hand-rolled Qwen3 layer.

Every buffer holds batch * tree.nodes rows: row b*T+j is draft-tree node j of
sequence b. With a one-node tree this is ordinary single-token decode.
"""

import torch
from kernels import CONFIG, TUNABLES
from kernels.elementwise import norm_out, qk_rope_cache_out, silu_mul_out
from kernels.attention import attention_out
from kernels.gemm import skinny_mm
from kernels.weights import LayerWeights


class DecodeBuffers:
    def __init__(self, batch, capacity, device, config=None, tree=None, scratch=True):
        from kernels.tree import DraftTree
        self.config = dict(CONFIG if config is None else config)
        if self.config['attention_impl'] != 'triton':
            raise ValueError('invalid attention_impl')
        if self.config['kv_layout'] not in ('bhsd', 'bshd'):
            raise ValueError('invalid kv_layout')
        self.tree = DraftTree(1, 1, device) if tree is None else tree
        rows = batch * self.tree.nodes
        self.rows = rows
        def alloc(*shape):
            return torch.empty(shape, device=device, dtype=torch.bfloat16)
        self.mm = skinny_mm if self.config.get('skinny_gemm', False) else (lambda a, b, out: torch.mm(a, b, out=out))
        self.x = alloc(rows, 2560)
        self.norm = alloc(rows, 2560)
        self.branch = alloc(rows, 2560)
        self.qkv = alloc(rows, 6144)
        self.qk_scratch = alloc(rows, 6144)
        self.q = alloc(rows, 32, 128)
        self.attn = alloc(rows, 32, 128)
        self.attn_flat = self.attn.view(rows, 4096)
        self.gate_up = alloc(rows, 19456)
        self.product = alloc(rows, 9728)
        self.activation = alloc(rows, 9728)
        splits = (capacity + TUNABLES['attention.BLOCK'] - 1) // TUNABLES['attention.BLOCK']
        self.partial = self.lse = None
        if scratch:
            self.partial = torch.empty((rows, 32, splits, 128), device=device, dtype=torch.float32)
            self.lse = torch.empty((rows, 32, splits), device=device, dtype=torch.float32)

    def layer(self, w, k, v, base, cos, sin, carry_in=False, defer_out=False):
        """One decoder layer on fixed buffers.

        base is the int64 [B] tensor of root cache slots. With
        fuse_norm_residual, carry_in folds the previous layer's pending
        down-projection output (still in branch) into this layer's input norm,
        and defer_out leaves this layer's down output in branch for the next
        norm instead of adding it into x. Defaults reproduce the unfused chain.
        """
        if carry_in and self.config['fuse_norm_residual']:
            norm_out(self.branch, w.input_norm, self.norm, w.eps, self.x, self.x)
        else:
            norm_out(self.x, w.input_norm, self.norm, w.eps)
        self.mm(self.norm, w.qkv, self.qkv)
        qk_rope_cache_out(self.qkv, w.q_norm, w.k_norm, cos, sin, base, self.tree,
                          self.q, k, v, self.qk_scratch, self.config['fuse_qk_norm_rope'])
        self.attend(k, v, base)
        self.mm(self.attn_flat, w.o, self.branch)
        if self.config['fuse_norm_residual']:
            norm_out(self.branch, w.post_norm, self.norm, w.eps, self.x, self.x)
        else:
            torch.add(self.x, self.branch, out=self.x)
            norm_out(self.x, w.post_norm, self.norm, w.eps)
        self.mm(self.norm, w.gate_up, self.gate_up)
        silu_mul_out(self.gate_up, self.product, self.activation, self.config['fuse_silu_mul'])
        self.mm(self.product, w.down, self.branch)
        if not (defer_out and self.config['fuse_norm_residual']):
            torch.add(self.x, self.branch, out=self.x)

    def attend(self, k, v, base):
        attention_out(self.q, k, v, base, self.tree, self.attn, self.partial, self.lse)


class PrefillTree:
    """Prompt rows as one chain: token p of every sequence sits at position and slot p."""

    def __init__(self, length, device):
        self.nodes = length
        self.max_depth = length - 1
        self.depth = torch.arange(length, device=device, dtype=torch.int32)


class PrefillBuffers(DecodeBuffers):
    """The decode layer over batch * prompt rows with causal FlashAttention reading the cache.

    K and V are consumed straight from the cache slots the fused RoPE kernel
    wrote, with the eight KV heads shared by their four query heads inside the
    kernel, so no 32-head copy of the prompt's keys and values is ever made.
    """

    def __init__(self, batch, prompt_length, device, config=None):
        super().__init__(batch, prompt_length, device, config, PrefillTree(prompt_length, device), scratch=False)
        self.batch, self.length = batch, prompt_length

    def attend(self, k, v, base):
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        query = self.q.view(self.batch, self.length, 32, 128).transpose(1, 2)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(query, k[:, :, :self.length], v[:, :, :self.length],
                                                 dropout_p=0.0, is_causal=self.length > 1,
                                                 scale=128 ** -0.5, enable_gqa=True)
        self.attn.copy_(out.transpose(1, 2).reshape(self.rows, 32, 128))


def cache_storage(batch, capacity, device, layout, layers=1):
    """Return [layers,B,8,capacity,128] logical BHSD storage in the selected physical layout."""
    if layout == 'bhsd':
        return torch.zeros((layers, batch, 8, capacity, 128), dtype=torch.bfloat16, device=device)
    if layout == 'bshd':
        return torch.zeros((layers, batch, capacity, 8, 128), dtype=torch.bfloat16, device=device).transpose(2, 3)
    raise ValueError('invalid kv_layout')
