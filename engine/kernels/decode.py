"""Loaded weight views and reusable scratch for a hand-rolled Qwen3 layer."""

import torch
import torch.nn.functional as F
from kernels import CONFIG, TUNABLES
from kernels.elementwise import norm_out, qk_rope_cache_out, silu_mul_out
from kernels.attention import attention_out
from kernels.weights import LayerWeights


class DecodeBuffers:
    def __init__(self, batch, capacity, device, config=None):
        self.config = dict(CONFIG if config is None else config)
        if self.config['attention_impl'] not in ('triton', 'sdpa_grouped'):
            raise ValueError('invalid attention_impl')
        if self.config['kv_layout'] not in ('bhsd', 'bshd'):
            raise ValueError('invalid kv_layout')
        def alloc(*shape):
            return torch.empty(shape, device=device, dtype=torch.bfloat16)
        self.x = alloc(batch, 2560)
        self.norm = alloc(batch, 2560)
        self.branch = alloc(batch, 2560)
        self.qkv = alloc(batch, 6144)
        self.qk_scratch = alloc(batch, 6144)
        self.q = alloc(batch, 32, 128)
        self.attn = alloc(batch, 32, 128)
        self.attn_flat = self.attn.view(batch, 4096)
        self.attn_grouped = self.attn.view(batch, 8, 4, 128)
        self.q_grouped = self.q.view(batch, 8, 4, 128)
        self.gate_up = alloc(batch, 19456)
        self.product = alloc(batch, 9728)
        self.activation = alloc(batch, 9728)
        splits = (capacity + TUNABLES['attention.BLOCK'] - 1) // TUNABLES['attention.BLOCK']
        self.partial = torch.empty((batch, 32, splits, 128), device=device, dtype=torch.float32)
        self.lse = torch.empty((batch, 32, splits), device=device, dtype=torch.float32)

    def layer(self, w, k, v, position, cos, sin, mask, carry_in=False, defer_out=False):
        """One decoder layer on fixed buffers.

        With fuse_norm_residual, carry_in folds the previous layer's pending
        down-projection output (still in branch) into this layer's input norm,
        and defer_out leaves this layer's down output in branch for the next
        norm instead of adding it into x. Defaults reproduce the unfused chain.
        """
        if carry_in and self.config['fuse_norm_residual']:
            norm_out(self.branch, w.input_norm, self.norm, w.eps, self.x, self.x)
        else:
            norm_out(self.x, w.input_norm, self.norm, w.eps)
        torch.mm(self.norm, w.qkv, out=self.qkv)
        qk_rope_cache_out(self.qkv, w.q_norm, w.k_norm, cos, sin, position,
                          self.q, k, v, self.qk_scratch, self.config['fuse_qk_norm_rope'])
        if self.config['attention_impl'] == 'triton':
            attention_out(self.q, k, v, position, self.attn, self.partial, self.lse)
        else:
            # Four query rows per KV head; all four use the same visible prefix.
            # SDPA's output/workspace is owned by the CUDA graph's private pool.
            grouped = F.scaled_dot_product_attention(self.q_grouped, k, v,
                                                      attn_mask=mask, is_causal=False,
                                                      scale=128 ** -0.5)
            self.attn_grouped.copy_(grouped)
        torch.mm(self.attn_flat, w.o, out=self.branch)
        if self.config['fuse_norm_residual']:
            norm_out(self.branch, w.post_norm, self.norm, w.eps, self.x, self.x)
        else:
            torch.add(self.x, self.branch, out=self.x)
            norm_out(self.x, w.post_norm, self.norm, w.eps)
        torch.mm(self.norm, w.gate_up, out=self.gate_up)
        silu_mul_out(self.gate_up, self.product, self.activation, self.config['fuse_silu_mul'])
        torch.mm(self.product, w.down, out=self.branch)
        if not (defer_out and self.config['fuse_norm_residual']):
            torch.add(self.x, self.branch, out=self.x)


def cache_storage(batch, capacity, device, layout):
    """Return logical BHSD cache backed by the selected physical layout."""
    if layout == 'bhsd':
        return torch.zeros((batch, 8, capacity, 128), dtype=torch.bfloat16, device=device)
    if layout == 'bshd':
        return torch.zeros((batch, capacity, 8, 128), dtype=torch.bfloat16, device=device).transpose(1, 2)
    raise ValueError('invalid kv_layout')
