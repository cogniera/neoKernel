"""Allocation-free BF16 decode kernels. Outputs are caller-owned CUDA tensors.

Packed projections are [B,6144] and [B,19456]. No wrapper synchronizes the host.
"""

import torch
import triton
import triton.language as tl
from kernels import TUNABLES


@triton.jit
def _head_norm(P, QW, KW, O, BLOCK: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK)
    x = tl.load(P + b * 6144 + h * 128 + d, d < 128, 0).to(tl.float32)
    gain = tl.load(tl.where(h < 32, QW + d, KW + d), d < 128, 0).to(tl.float32)
    x = (x * tl.rsqrt(tl.sum(x*x, 0) / 128 + 1.e-6)).to(P.dtype.element_ty)
    tl.store(O + b * 6144 + h * 128 + d, x.to(tl.float32) * gain, d < 128)


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
