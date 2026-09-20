"""Exact grouped attention over a static cache for T query tokens per sequence.

Q/out: contiguous BF16 [B*T,32,128]; row b*T+j is token j of sequence b and
sits at position BASE[b]+j. K/V: logical [B,8,capacity,128] with BHSD or BSHD
strides; the slots for the T new tokens are already written. Token j reads
cache indices <= BASE[b]+j, which is the causal rule for a chain of drafts.
One program serves one KV head's four query heads for all T tokens over one
cache block, so K and V are read once per (sequence, group, block). The 4*T
rows fill a 16- or 32-row tile: QK^T and PV run on tensor cores with fp32
accumulation and P is rounded to BF16 before PV, as the reference flash path
does. Partial outputs and log-normalizers are caller-owned FP32 buffers
indexed per (row, query head), merged by a second stable kernel.
"""

import triton
import triton.language as tl
from kernels import TUNABLES


@triton.jit
def _partial(Q, K, V, BASE, PART, LSE, CAP: tl.constexpr,
             KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
             VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr,
             SPLITS: tl.constexpr, BLOCK: tl.constexpr, T: tl.constexpr, ROWS: tl.constexpr,
             POS_STRIDE: tl.constexpr):
    b, g, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = split * BLOCK
    base = tl.load(BASE + b * POS_STRIDE)
    r = tl.arange(0, ROWS)
    j = r // 4
    h = r % 4
    d = tl.arange(0, 128)
    rows = j < T
    qpos = base + j
    index = (b * T + j) * 32 + g * 4 + h
    offset = index * SPLITS + split
    if start > base + (T - 1):
        # Whole block beyond every visible prefix: contributes nothing to the merge.
        tl.store(PART + offset[:, None] * 128 + d[None, :], tl.zeros((ROWS, 128), tl.float32), rows[:, None])
        tl.store(LSE + offset, tl.full((ROWS,), -float("inf"), tl.float32), rows)
    else:
        seq = start + tl.arange(0, BLOCK)
        kvalid = (seq < CAP) & (seq <= base + T - 1)
        q = tl.load(Q + index[:, None] * 128 + d[None, :], rows[:, None], 0.)
        k = tl.load(K + b * KB + g * KH + seq[:, None] * KS + d[None, :], kvalid[:, None], 0.)
        scores = tl.dot(q, tl.trans(k)) * 0.08838834764831845
        allowed = kvalid[None, :] & (seq[None, :] <= qpos[:, None])
        scores = tl.where(allowed, scores, -float("inf"))
        mx = tl.max(scores, 1)
        safe_mx = tl.where(mx == -float("inf"), 0., mx)
        p = tl.exp(scores - safe_mx[:, None])
        denom = tl.sum(p, 1)
        v = tl.load(V + b * VB + g * VH + seq[:, None] * VS + d[None, :], kvalid[:, None], 0.)
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


def attention_out(q, k, v, base, out, partial, lse, tokens=1):
    """out[b*T+j] attends over cache indices <= base[b]+j; base is an int64 [B] device tensor."""
    block = TUNABLES["attention.BLOCK"]
    if block < 16 or block & (block - 1):
        raise ValueError("attention.BLOCK must be a power of two of at least 16")
    if tokens < 1 or tokens > 8 or q.shape[0] != k.shape[0] * tokens:
        raise ValueError("attention_out serves 1 to 8 tokens per sequence")
    splits = triton.cdiv(k.shape[2], block)
    _partial[(k.shape[0], 8, splits)](
        q, k, v, base, partial, lse, k.shape[2],
        *k.stride()[:3], *v.stride()[:3], splits, block, tokens, 16 if tokens <= 4 else 32,
        0 if base.numel() == 1 else 1,
        num_warps=TUNABLES["attention.num_warps"],
        num_stages=TUNABLES["attention.num_stages"], enable_fp_fusion=False)
    _merge[(q.shape[0]*32,)](partial, lse, out, splits, triton.next_power_of_2(splits),
                            num_warps=TUNABLES["merge.num_warps"],
                            num_stages=TUNABLES["merge.num_stages"], enable_fp_fusion=False)
