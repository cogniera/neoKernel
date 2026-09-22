"""Reusable scratch for a hand-rolled Qwen3 layer over draft-tree rows.

Every buffer holds batch * tree.nodes rows: row b*T+j is draft-tree node j of
sequence b. With a one-node tree this is ordinary single-token decode.
"""

import torch
from kernels import CONFIG, TUNABLES
from kernels.elementwise import silu_mul_out
from kernels.rmsnorm import norm_out
from kernels.tree import DraftTree
from kernels.tree_ops import qk_rope_cache_out
from kernels.tree_attention import attention_out


class TreeDecodeBuffers:
    def __init__(self, batch, capacity, device, config=None, tree=None):
        self.config = dict(CONFIG if config is None else config)
        self.tree = DraftTree(1, 1, device) if tree is None else tree
        self.rows = rows = batch * self.tree.nodes
        def alloc(*shape):
            return torch.empty(shape, device=device, dtype=torch.bfloat16)
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
        torch.mm(self.norm, w.qkv, out=self.qkv)
        qk_rope_cache_out(self.qkv, w.q_norm, w.k_norm, cos, sin, base, self.tree,
                          self.q, k, v, self.qk_scratch, self.config['fuse_qk_norm_rope'])
        attention_out(self.q, k, v, base, self.tree, self.attn, self.partial, self.lse)
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


def cache_storage(batch, capacity, device, layout, layers=1):
    """Return [layers,B,8,capacity,128] logical BHSD storage in the selected physical layout."""
    if layout == 'bhsd':
        return torch.zeros((layers, batch, 8, capacity, 128), dtype=torch.bfloat16, device=device)
    if layout == 'bshd':
        return torch.zeros((layers, batch, capacity, 8, 128), dtype=torch.bfloat16, device=device).transpose(2, 3)
    raise ValueError('invalid kv_layout')
