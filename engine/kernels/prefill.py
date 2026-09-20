"""Packed prefill operations preserving native BF16 rounding boundaries.

Inputs and outputs are BF16 CUDA tensors. Q is returned in logical BHSD form;
K/V are written into the caller-owned static cache, only within the prompt.
Allocation during capture is owned by the prefill CUDA graph's private pool.
"""

import torch
import triton
import triton.language as tl
from kernels.rmsnorm import _norm


def norm(x, module):
    """Normalize the last dimension, rounding before the learned gain multiply."""
    width = x.shape[-1]
    flat = x.contiguous().view(-1, width)
    out = torch.empty_like(flat)
    _norm[(flat.shape[0],)](
        flat, flat, module.weight, out, out, width, module.variance_epsilon,
        False, triton.next_power_of_2(width), num_warps=4,
        num_stages=1, enable_fp_fusion=False,
    )
    return out.view(x.shape)


def add_norm(branch, residual, module):
    """Return (normalized, rounded residual sum) in one pass over each row."""
    width = branch.shape[-1]
    flat = branch.contiguous().view(-1, width)
    skip = residual.contiguous().view(-1, width)
    out = torch.empty_like(flat)
    summed = torch.empty_like(flat)
    _norm[(flat.shape[0],)](
        flat, skip, module.weight, out, summed, width, module.variance_epsilon,
        True, triton.next_power_of_2(width), num_warps=4,
        num_stages=1, enable_fp_fusion=False,
    )
    return out.view(branch.shape), summed.view(branch.shape)


@triton.jit
def _qkv(P, QW, KW, COS, SIN, Q, K, V, LENGTH: tl.constexpr,
         KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
         VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr,
         QEPS: tl.constexpr, KEPS: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    batch = row // LENGTH
    pos = row % LENGTH
    h = group * 8 + tl.arange(0, 8)
    d = tl.arange(0, 128)
    other = (d + 64) % 128
    x = tl.load(P + row * 6144 + h[:, None] * 128 + d[None, :]).to(tl.float32)
    xr = tl.load(P + row * 6144 + h[:, None] * 128 + other[None, :]).to(tl.float32)
    if group < 4:
        gain = tl.load(QW + d).to(tl.float32)
        gain_r = tl.load(QW + other).to(tl.float32)
        eps = QEPS
    else:
        gain = tl.load(KW + d).to(tl.float32)
        gain_r = tl.load(KW + other).to(tl.float32)
        eps = KEPS
    inv = tl.rsqrt(tl.sum(x * x, 1) / 128 + eps)
    # Reference RMSNorm rounds the normalized value, then the weighted value.
    x = ((x * inv[:, None]).to(tl.bfloat16).to(tl.float32) * gain[None, :]).to(tl.bfloat16)
    xr = ((xr * inv[:, None]).to(tl.bfloat16).to(tl.float32) * gain_r[None, :]).to(tl.bfloat16)
    co = tl.load(COS + pos * 128 + d).to(tl.float32)
    si = tl.load(SIN + pos * 128 + d).to(tl.float32)
    rotated = tl.where(d[None, :] < 64, -xr.to(tl.float32), xr.to(tl.float32))
    # Each native RoPE product rounds before the addition.
    a = (x.to(tl.float32) * co[None, :]).to(tl.bfloat16)
    z = (rotated * si[None, :]).to(tl.bfloat16)
    y = a.to(tl.float32) + z.to(tl.float32)
    if group < 4:
        tl.store(Q + row * 4096 + h[:, None] * 128 + d[None, :], y)
    else:
        kh = h - 32
        tl.store(K + batch * KB + kh[:, None] * KH + pos * KS + d[None, :], y)
        value = tl.load(P + row * 6144 + 5120 + kh[:, None] * 128 + d[None, :])
        tl.store(V + batch * VB + kh[:, None] * VH + pos * VS + d[None, :], value)


def qkv_cache(packed, attn, cos, sin, key_cache, value_cache):
    """Qwen3 packed [B,S,6144] -> Q[B,32,S,128], prompt K/V cache views."""
    batch, length = packed.shape[:2]
    q = torch.empty((batch, length, 32, 128), device=packed.device, dtype=packed.dtype)
    _qkv[(batch * length, 5)](
        packed, attn.q_norm.weight, attn.k_norm.weight, cos, sin,
        q, key_cache, value_cache, length,
        *key_cache.stride()[:3], *value_cache.stride()[:3],
        attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return q.transpose(1, 2), key_cache[:, :, :length, :], value_cache[:, :, :length, :]


@triton.jit
def _activation(P, O, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    g = tl.load(P + row * (2 * WIDTH) + col, i < N, 0).to(tl.float32)
    u = tl.load(P + row * (2 * WIDTH) + WIDTH + col, i < N, 0).to(tl.float32)
    activated = (g / (1.0 + tl.exp(-g))).to(P.dtype.element_ty)
    tl.store(O + i, activated.to(tl.float32) * u, i < N)


def packed_silu_mul(packed):
    """Read concatenated gate/up projections without making separate copies."""
    width = packed.shape[-1] // 2
    out = torch.empty((*packed.shape[:-1], width), device=packed.device, dtype=packed.dtype)
    count = out.numel()
    _activation[(triton.cdiv(count, 1024),)](
        packed, out, count, width, 1024, num_warps=4, num_stages=1,
        enable_fp_fusion=False,
    )
    return out
