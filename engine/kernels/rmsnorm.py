"""Qwen3 RMSNorm with the native BF16 cast before gain multiplication.

The optional residual add is rounded to BF16 before normalization. This is
the allocation-free decode extension of the starter RMSNorm cast rule.
"""

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
