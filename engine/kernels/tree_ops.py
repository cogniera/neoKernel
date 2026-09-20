"""Q/K norm, native RoPE and cache writes for draft-tree rows.

Rows are B*T tree nodes: row b*T+j is node j of sequence b, at position
BASE[b]+DEPTH[j]; its K/V land in cache slot BASE[b]+j. Cache views have
logical [B,8,L,128] shape and arbitrary strides. base is an int64 [B] CUDA
tensor. No wrapper synchronizes the host.
"""

import triton
import triton.language as tl
from kernels import TUNABLES
from kernels.elementwise import _head_norm


@triton.jit
def _qk(P, N, QW, KW, COS, SIN, BASE, DEPTH, Q, K, V,
        KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
        VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr,
        NORMALIZE: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr):
    row, h = tl.program_id(0), tl.program_id(1)
    b = row // T
    j = row % T
    d = tl.arange(0, BLOCK)
    other = (d + 64) % 128
    base = tl.load(BASE + b)
    pos = base + tl.load(DEPTH + j)
    slot = base + j
    if NORMALIZE:
        x = tl.load(P + row * 6144 + h * 128 + d, d < 128, 0).to(tl.float32)
        xr = tl.load(P + row * 6144 + h * 128 + other, d < 128, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x*x, 0) / 128 + 1.e-6)
        w = tl.load(tl.where(h < 32, QW + d, KW + d), d < 128, 0).to(tl.float32)
        wr = tl.load(tl.where(h < 32, QW + other, KW + other), d < 128, 0).to(tl.float32)
        x = ((x * inv).to(P.dtype.element_ty).to(tl.float32) * w).to(P.dtype.element_ty)
        xr = ((xr * inv).to(P.dtype.element_ty).to(tl.float32) * wr).to(P.dtype.element_ty)
    else:
        x = tl.load(N + row * 6144 + h * 128 + d, d < 128, 0)
        xr = tl.load(N + row * 6144 + h * 128 + other, d < 128, 0)
    co = tl.load(COS + pos * 128 + d, d < 128, 0).to(tl.float32)
    si = tl.load(SIN + pos * 128 + d, d < 128, 0).to(tl.float32)
    rotated = tl.where(d < 64, -xr.to(tl.float32), xr.to(tl.float32))
    # Native apply_rotary_pos_emb rounds EACH product before the addition.
    a = (x.to(tl.float32) * co).to(P.dtype.element_ty)
    z = (rotated * si).to(P.dtype.element_ty)
    y = a.to(tl.float32) + z.to(tl.float32)
    if h < 32:
        tl.store(Q + row * 4096 + h * 128 + d, y, d < 128)
    else:
        kh = h - 32
        tl.store(K + b * KB + kh * KH + slot * KS + d, y, d < 128)
        v = tl.load(P + row * 6144 + 5120 + kh * 128 + d, d < 128, 0)
        tl.store(V + b * VB + kh * VH + slot * VS + d, v, d < 128)


def qk_rope_cache_out(packed, qw, kw, cos, sin, base, tree, q, k, v, scratch, fused=True):
    """Normalize Q/K, apply native RoPE at position base[b]+depth[j], write K/V to slot base[b]+j.

    packed is [B*T,6144], row b*T+j being tree node j of sequence b. cos/sin
    are contiguous [capacity,128] BF16 native rotary outputs computed once
    during preparation with the loaded module (theta=5e6). scratch is
    [B*T,6144] and is used only by the unfused per-head norm fallback.
    """
    block = TUNABLES["qk.BLOCK"]
    if block < 128 or block & (block - 1):
        raise ValueError("qk.BLOCK must be a power of two >= 128")
    if packed.shape[0] != base.shape[0] * tree.nodes:
        raise ValueError("qk_rope_cache_out expects batch * tree.nodes rows")
    launch = dict(num_warps=TUNABLES["qk.num_warps"],
                  num_stages=TUNABLES["qk.num_stages"], enable_fp_fusion=False)
    if not fused:
        _head_norm[(packed.shape[0], 40)](packed, qw, kw, scratch, block, **launch)
    _qk[(packed.shape[0], 40)](packed, scratch, qw, kw, cos, sin, base, tree.depth, q, k, v,
                              *k.stride()[:3], *v.stride()[:3], fused, tree.nodes, block, **launch)
