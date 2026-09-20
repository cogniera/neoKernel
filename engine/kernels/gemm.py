"""Skinny BF16 GEMM for decode: out[M,N] = x[M,K] @ W[N,K]^T with M <= 64.

cuBLAS at M <= 16 launches 40 to 96 blocks for these widths and leaves a third
of an H100 idle. Each program here streams BLOCK_N weight rows over all of K,
accumulates in fp32 on tensor cores, and rounds once to BF16, which is the
reference's rounding point. The tile width is chosen so at least MIN_PROGRAMS
programs run; W is the contiguous [N, K] module weight and x is contiguous.
"""

import triton
import triton.language as tl
from kernels import TUNABLES

MIN_PROGRAMS = 160


@triton.jit
def _skinny(X, W, OUT, M, N: tl.constexpr, K: tl.constexpr, XS: tl.constexpr, OS: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    m = tl.arange(0, BLOCK_M)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + kk
        x = tl.load(X + m[:, None] * XS + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0.)
        w = tl.load(W + n[:, None] * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), 0.)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(OUT + m[:, None] * OS + n[None, :], acc.to(tl.bfloat16), (m[:, None] < M) & (n[None, :] < N))


def block_n(n):
    for width in (128, 64, 32, 16):
        if triton.cdiv(n, width) >= MIN_PROGRAMS:
            return width
    return 16


def skinny_mm(x, w_t, out):
    """out = x @ w_t, where w_t is the transposed view of a contiguous [N, K] weight."""
    w = w_t.t()
    rows, k = x.shape
    n = w.shape[0]
    block_k = TUNABLES["gemm.BLOCK"]
    if block_k < 16 or block_k & (block_k - 1):
        raise ValueError("gemm.BLOCK must be a power of two of at least 16")
    if rows > 64 or w.shape[1] != k or not (x.is_contiguous() and w.is_contiguous() and out.is_contiguous()):
        raise ValueError("skinny_mm expects contiguous operands and at most 64 rows")
    if out.shape != (rows, n):
        raise ValueError("skinny_mm output shape mismatch")
    width = block_n(n)
    _skinny[(triton.cdiv(n, width),)](
        x, w, out, rows, n, k, x.stride(0), out.stride(0),
        max(16, triton.next_power_of_2(rows)), width, block_k,
        num_warps=TUNABLES["gemm.num_warps"], num_stages=TUNABLES["gemm.num_stages"],
        enable_fp_fusion=False)
