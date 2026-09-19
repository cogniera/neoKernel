"""Allocation-free BF16 decode kernels. Outputs are caller-owned CUDA tensors.

Hidden inputs are contiguous [B,H]; packed projections are [B,6144] and
[B,19456]. Cache views have logical [B,8,L,128] shape and arbitrary strides.
Position is a one-element CUDA int64 tensor. No wrapper synchronizes the host.
"""

import torch
import triton
import triton.language as tl
from kernels import TUNABLES


@triton.jit
def _norm(X, R, W, Y, SUM, H: tl.constexpr, EPS: tl.constexpr,
          ADD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + c, c < H, 0).to(tl.float32)
    if ADD:
        r = tl.load(R + row * H + c, c < H, 0).to(tl.float32)
        x = (x + r).to(X.dtype.element_ty).to(tl.float32)
        tl.store(SUM + row * H + c, x, c < H)
    var = tl.sum(x * x, 0) / H
    n = (x * tl.rsqrt(var + EPS)).to(X.dtype.element_ty)
    w = tl.load(W + c, c < H, 0).to(tl.float32)
    tl.store(Y + row * H + c, n.to(tl.float32) * w, c < H)


def norm_out(x, weight, out, eps=1e-6, residual=None, summed=None):
    """Adapt the starter RMSNorm cast rule; optionally write rounded x+residual.

    summed may alias residual: each program reads and writes only its own row.
    out must not alias x or summed.
    """
    block = TUNABLES["norm.BLOCK"]
    if block < x.shape[-1] or block & (block - 1):
        raise ValueError("norm.BLOCK must be a power of two covering hidden width")
    _norm[(x.shape[0],)](x, residual if residual is not None else x, weight,
                         out, summed if summed is not None else out,
                         x.shape[-1], eps, residual is not None, block,
                         num_warps=TUNABLES["norm.num_warps"],
                         num_stages=TUNABLES["norm.num_stages"], enable_fp_fusion=False)


@triton.jit
def _head_norm(P, QW, KW, O, BLOCK: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK)
    x = tl.load(P + b * 6144 + h * 128 + d, d < 128, 0).to(tl.float32)
    gain = tl.load(tl.where(h < 32, QW + d, KW + d), d < 128, 0).to(tl.float32)
    x = (x * tl.rsqrt(tl.sum(x*x, 0) / 128 + 1.e-6)).to(P.dtype.element_ty)
    tl.store(O + b * 6144 + h * 128 + d, x.to(tl.float32) * gain, d < 128)


@triton.jit
def _qk(P, N, QW, KW, COS, SIN, POS, Q, K, V,
        KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
        VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr,
        NORMALIZE: tl.constexpr, BLOCK: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK)
    other = (d + 64) % 128
    pos = tl.load(POS)
    if NORMALIZE:
        x = tl.load(P + b * 6144 + h * 128 + d, d < 128, 0).to(tl.float32)
        xr = tl.load(P + b * 6144 + h * 128 + other, d < 128, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x*x, 0) / 128 + 1.e-6)
        w = tl.load(tl.where(h < 32, QW + d, KW + d), d < 128, 0).to(tl.float32)
        wr = tl.load(tl.where(h < 32, QW + other, KW + other), d < 128, 0).to(tl.float32)
        x = ((x * inv).to(P.dtype.element_ty).to(tl.float32) * w).to(P.dtype.element_ty)
        xr = ((xr * inv).to(P.dtype.element_ty).to(tl.float32) * wr).to(P.dtype.element_ty)
    else:
        x = tl.load(N + b * 6144 + h * 128 + d, d < 128, 0)
        xr = tl.load(N + b * 6144 + h * 128 + other, d < 128, 0)
    co = tl.load(COS + pos * 128 + d, d < 128, 0).to(tl.float32)
    si = tl.load(SIN + pos * 128 + d, d < 128, 0).to(tl.float32)
    rotated = tl.where(d < 64, -xr.to(tl.float32), xr.to(tl.float32))
    # Native apply_rotary_pos_emb rounds EACH product before the addition.
    a = (x.to(tl.float32) * co).to(P.dtype.element_ty)
    z = (rotated * si).to(P.dtype.element_ty)
    y = a.to(tl.float32) + z.to(tl.float32)
    if h < 32:
        tl.store(Q + b * 4096 + h * 128 + d, y, d < 128)
    else:
        kh = h - 32
        tl.store(K + b * KB + kh * KH + pos * KS + d, y, d < 128)
        v = tl.load(P + b * 6144 + 5120 + kh * 128 + d, d < 128, 0)
        tl.store(V + b * VB + kh * VH + pos * VS + d, v, d < 128)


def qk_rope_cache_out(packed, qw, kw, cos, sin, position, q, k, v, scratch, fused=True):
    """Normalize Q/K, apply absolute-position native RoPE tables, update K/V slot.

    cos/sin are contiguous [capacity,128] BF16 native rotary outputs, computed
    once during preparation with the loaded module (theta=5e6). scratch is
    [B,6144] and is used only by the unfused per-head norm fallback.
    """
    block = TUNABLES["qk.BLOCK"]
    if block < 128 or block & (block - 1):
        raise ValueError("qk.BLOCK must be a power of two >= 128")
    launch = dict(num_warps=TUNABLES["qk.num_warps"],
                  num_stages=TUNABLES["qk.num_stages"], enable_fp_fusion=False)
    if not fused:
        _head_norm[(packed.shape[0], 40)](packed, qw, kw, scratch, block, **launch)
    _qk[(packed.shape[0], 40)](packed, scratch, qw, kw, cos, sin, position, q, k, v,
                              *k.stride()[:3], *v.stride()[:3], fused, block, **launch)


@triton.jit
def _silu(P, O, I: tl.constexpr, BLOCK: tl.constexpr):
    b, tile = tl.program_id(0), tl.program_id(1)
    d = tile * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(P + b * 2 * I + d, d < I, 0).to(tl.float32)
    u = tl.load(P + b * 2 * I + I + d, d < I, 0).to(tl.float32)
    act = (g / (1. + tl.exp(-g))).to(P.dtype.element_ty)
    tl.store(O + b * I + d, act.to(tl.float32) * u, d < I)


def silu_mul_out(packed, out, activation, fused=True):
    """BF16 SiLU then BF16 product; activation is caller-owned fallback storage."""
    width = out.shape[-1]
    if fused:
        _silu[(out.shape[0], triton.cdiv(width, TUNABLES["silu.BLOCK"]))](
            packed, out, width, TUNABLES["silu.BLOCK"],
            num_warps=TUNABLES["silu.num_warps"],
            num_stages=TUNABLES["silu.num_stages"], enable_fp_fusion=False)
    else:
        torch.ops.aten.silu.out(packed[:, :width], out=activation)
        torch.mul(activation, packed[:, width:], out=out)
