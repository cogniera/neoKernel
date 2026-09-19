"""Exact grouped decode attention, split over cache blocks, then stable merge.

Q/out: contiguous BF16 [B,32,128]. K/V: logical [B,8,capacity,128], with
either BHSD or BSHD physical strides. Only indices <= device position are read.
Partial output and log-normalizers are caller-owned FP32 buffers. No repeat_kv.
"""

import triton
import triton.language as tl
from kernels import TUNABLES


@triton.jit
def _partial(Q, K, V, POS, PART, LSE, CAP: tl.constexpr,
             KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
             VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr,
             SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    b, h, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq = split * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    pos = tl.load(POS)
    valid = (seq <= pos) & (seq < CAP)
    q = tl.load(Q + b * 4096 + h * 128 + d).to(tl.float32)
    k = tl.load(K + b*KB + (h//4)*KH + seq[:, None]*KS + d[None, :],
                valid[:, None], 0).to(tl.float32)
    scores = tl.sum(k * q[None, :], 1) * 0.08838834764831845
    scores = tl.where(valid, scores, -float("inf"))
    mx = tl.max(scores, 0)
    safe_mx = tl.where(mx == -float("inf"), 0., mx)
    p = tl.exp(scores - safe_mx)
    denom = tl.sum(p, 0)
    v = tl.load(V + b*VB + (h//4)*VH + seq[:, None]*VS + d[None, :],
                valid[:, None], 0).to(tl.float32)
    acc = tl.sum(p[:, None] * v, 0) / tl.maximum(denom, 1.e-20)
    offset = (b*32+h)*SPLITS + split
    tl.store(PART + offset*128 + d, acc)
    tl.store(LSE + offset, tl.where(denom > 0., safe_mx + tl.log(denom), -float("inf")))


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
    splits = triton.cdiv(k.shape[2], block)
    _partial[(q.shape[0], 32, splits)](
        q, k, v, position, partial, lse, k.shape[2],
        *k.stride()[:3], *v.stride()[:3], splits, block,
        num_warps=TUNABLES["attention.num_warps"],
        num_stages=TUNABLES["attention.num_stages"], enable_fp_fusion=False)
    _merge[(q.shape[0]*32,)](partial, lse, out, splits, triton.next_power_of_2(splits),
                            num_warps=TUNABLES["merge.num_warps"],
                            num_stages=TUNABLES["merge.num_stages"], enable_fp_fusion=False)
