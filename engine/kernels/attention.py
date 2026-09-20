"""Exact grouped decode attention, split over cache blocks, then stable merge.

Q/out: contiguous BF16 [B,32,128]. K/V: logical [B,8,capacity,128], with
either BHSD or BSHD physical strides. Only indices <= device position are read.
One program serves one KV head's four query heads over one cache block, so K
and V are read once per group. The four heads occupy rows 0..3 of a 16-row
tile: QK^T and PV run on tensor cores with fp32 accumulation, and P is rounded
to BF16 before PV as the reference flash path does. Partial output and
log-normalizers are caller-owned FP32 buffers, indexed per query head.
"""

import triton
import triton.language as tl
from kernels import TUNABLES


@triton.jit
def _partial(Q, K, V, POS, PART, LSE, CAP: tl.constexpr,
             KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
             VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr,
             SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    b, g, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = split * BLOCK
    pos = tl.load(POS)
    r = tl.arange(0, 16)
    d = tl.arange(0, 128)
    rows = r < 4
    offset = (b * 32 + g * 4 + r) * SPLITS + split
    if start > pos:
        # Whole block beyond the visible prefix: contributes nothing to the merge.
        tl.store(PART + offset[:, None] * 128 + d[None, :], tl.zeros((16, 128), tl.float32), rows[:, None])
        tl.store(LSE + offset, tl.full((16,), -float("inf"), tl.float32), rows)
    else:
        seq = start + tl.arange(0, BLOCK)
        valid = (seq <= pos) & (seq < CAP)
        q = tl.load(Q + b * 4096 + (g * 4 + r)[:, None] * 128 + d[None, :], rows[:, None], 0.)
        k = tl.load(K + b * KB + g * KH + seq[:, None] * KS + d[None, :], valid[:, None], 0.)
        scores = tl.dot(q, tl.trans(k)) * 0.08838834764831845
        scores = tl.where(valid[None, :], scores, -float("inf"))
        mx = tl.max(scores, 1)
        safe_mx = tl.where(mx == -float("inf"), 0., mx)
        p = tl.exp(scores - safe_mx[:, None])
        denom = tl.sum(p, 1)
        v = tl.load(V + b * VB + g * VH + seq[:, None] * VS + d[None, :], valid[:, None], 0.)
        acc = tl.dot(p.to(tl.bfloat16), v) / tl.maximum(denom, 1.e-20)[:, None]
        tl.store(PART + offset[:, None] * 128 + d[None, :], acc, rows[:, None])
        tl.store(LSE + offset, tl.where(denom > 0., safe_mx + tl.log(denom), -float("inf")), rows)


@triton.jit
def _merge(PART, LSE, OUT, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    bh = tl.program_id(0)
    s = tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    logs = tl.load(LSE + bh*SPLITS + s, s < SPLITS, -float("inf"))
    p = tl.exp(logs - tl.max(logs, 0))
    p = p / tl.sum(p, 0)
    x = tl.load(PART + (bh*SPLITS + s[:, None])*128 + d[None, :], s[:, None] < SPLITS, 0)
    tl.store(OUT + bh*128 + d, tl.sum(x * p[:, None], 0))


def attention_out(q, k, v, position, out, partial, lse):
    block = TUNABLES["attention.BLOCK"]
    if block < 16 or block & (block - 1):
        raise ValueError("attention.BLOCK must be a power of two of at least 16")
    splits = triton.cdiv(k.shape[2], block)
    _partial[(q.shape[0], 8, splits)](
        q, k, v, position, partial, lse, k.shape[2],
        *k.stride()[:3], *v.stride()[:3], splits, block,
        num_warps=TUNABLES["attention.num_warps"],
        num_stages=TUNABLES["attention.num_stages"], enable_fp_fusion=False)
    _merge[(q.shape[0]*32,)](partial, lse, out, splits, triton.next_power_of_2(splits),
                            num_warps=TUNABLES["merge.num_warps"],
                            num_stages=TUNABLES["merge.num_stages"], enable_fp_fusion=False)
