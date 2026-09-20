"""Split-K BF16 decode with FP32 partials and fused, native-rounded consumers.

Weights retain contiguous [N,K] storage. Each producer partitions K into
independent tensor-core tiles; consumers sum FP32 partials before rounding
once to BF16, at the same boundary as the native linear operation.
"""

import torch
import triton
import triton.language as tl
from kernels.decode import DecodeBuffers
from kernels.rmsnorm import norm_out


@triton.jit
def _project(X, W, P, B: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             S: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    nt, mt, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mt * BM + tl.arange(0, BM)
    n = nt * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for tile in range(tl.cdiv(K, S * BK)):
        k = (tile * S + split) * BK + kk
        a = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < B) & (k[None, :] < K), 0.)
        w = tl.load(W + n[:, None] * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), 0.)
        acc = tl.dot(a, tl.trans(w), acc)
    tl.store(P + split * B * N + m[:, None] * N + n[None, :], acc,
             (m[:, None] < B) & (n[None, :] < N))


class Projection:
    def __init__(self, batch, n, k, splits, device):
        self.batch, self.n, self.k, self.splits = batch, n, k, splits
        self.partial = torch.empty((splits, batch, n), device=device, dtype=torch.float32)

    def run(self, x, weight):
        _project[(triton.cdiv(self.n, 64), triton.cdiv(self.batch, 16), self.splits)](
            x, weight, self.partial, self.batch, self.n, self.k, self.splits,
            16, 64, 128, num_warps=4, num_stages=3,
        )


@triton.jit
def _reduce_norm(P, R, W, O, SUM, B: tl.constexpr, H: tl.constexpr,
                 S: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    s = tl.arange(0, S)
    h = tl.arange(0, BLOCK)
    partial = tl.load(P + s[:, None] * B * H + b * H + h[None, :], h[None, :] < H, 0.)
    projected = tl.sum(partial, 0).to(tl.bfloat16).to(tl.float32)
    residual = tl.load(R + b * H + h, h < H, 0.).to(tl.float32)
    value = (projected + residual).to(tl.bfloat16).to(tl.float32)
    tl.store(SUM + b * H + h, value, h < H)
    inv = tl.rsqrt(tl.sum(value * value, 0) / H + EPS)
    normalized = (value * inv).to(tl.bfloat16).to(tl.float32)
    gain = tl.load(W + h, h < H, 0.).to(tl.float32)
    tl.store(O + b * H + h, normalized * gain, h < H)


def reduce_norm(plan, residual, weight, out, summed, eps):
    _reduce_norm[(plan.batch,)](
        plan.partial, residual, weight, out, summed, plan.batch, plan.n,
        plan.splits, eps, triton.next_power_of_2(plan.n),
        num_warps=8, num_stages=1, enable_fp_fusion=False,
    )


@triton.jit
def _reduce_qkv(P, QW, KW, COS, SIN, POS, Q, K, V,
                B: tl.constexpr, S: tl.constexpr,
                KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
                VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    s = tl.arange(0, S)
    d = tl.arange(0, 128)
    other = (d + 64) % 128
    pos = tl.load(POS)
    offsets = s[:, None] * B * 6144 + b * 6144 + h * 128
    x = tl.sum(tl.load(P + offsets + d[None, :]), 0).to(tl.bfloat16).to(tl.float32)
    xr = tl.sum(tl.load(P + offsets + other[None, :]), 0).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / 128 + 1.e-6)
    gain = tl.load(tl.where(h < 32, QW + d, KW + d)).to(tl.float32)
    gain_r = tl.load(tl.where(h < 32, QW + other, KW + other)).to(tl.float32)
    x = ((x * inv).to(tl.bfloat16).to(tl.float32) * gain).to(tl.bfloat16)
    xr = ((xr * inv).to(tl.bfloat16).to(tl.float32) * gain_r).to(tl.bfloat16)
    co = tl.load(COS + pos * 128 + d).to(tl.float32)
    si = tl.load(SIN + pos * 128 + d).to(tl.float32)
    rotated = tl.where(d < 64, -xr.to(tl.float32), xr.to(tl.float32))
    a = (x.to(tl.float32) * co).to(tl.bfloat16)
    z = (rotated * si).to(tl.bfloat16)
    value = a.to(tl.float32) + z.to(tl.float32)
    if h < 32:
        tl.store(Q + b * 4096 + h * 128 + d, value)
    else:
        kh = h - 32
        tl.store(K + b * KB + kh * KH + pos * KS + d, value)
        vp = tl.load(P + s[:, None] * B * 6144 + b * 6144 + 5120 + kh * 128 + d[None, :])
        tl.store(V + b * VB + kh * VH + pos * VS + d, tl.sum(vp, 0).to(tl.bfloat16))


@triton.jit
def _reduce_silu(P, O, B: tl.constexpr, S: tl.constexpr,
                 I: tl.constexpr, BLOCK: tl.constexpr):
    b, tile = tl.program_id(0), tl.program_id(1)
    s = tl.arange(0, S)
    d = tile * BLOCK + tl.arange(0, BLOCK)
    offsets = s[:, None] * B * (2 * I) + b * (2 * I) + d[None, :]
    gate = tl.sum(tl.load(P + offsets, d[None, :] < I, 0.), 0).to(tl.bfloat16).to(tl.float32)
    up = tl.sum(tl.load(P + offsets + I, d[None, :] < I, 0.), 0).to(tl.bfloat16).to(tl.float32)
    act = (gate / (1. + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(O + b * I + d, act * up, d < I)


class SplitDecodeBuffers(DecodeBuffers):
    """Drop-in decode state for the fused residual chain used by Engine."""
    def __init__(self, batch, capacity, device, config=None):
        super().__init__(batch, capacity, device, config)
        self.qkv_plan = Projection(batch, 6144, 2560, 4, device)
        self.o_plan = Projection(batch, 2560, 4096, 8, device)
        self.gate_plan = Projection(batch, 19456, 2560, 2, device)
        self.down_plan = Projection(batch, 2560, 9728, 8, device)

    def layer(self, w, k, v, position, cos, sin, mask, carry_in=False, defer_out=True, normalized_input=False):
        if not defer_out:
            raise ValueError('split decode requires the fused residual chain')
        if carry_in:
            reduce_norm(self.down_plan, self.x, w.input_norm, self.norm, self.x, w.eps)
        elif not normalized_input:
            norm_out(self.x, w.input_norm, self.norm, w.eps)
        self.qkv_plan.run(self.norm, w.qkv_weight)
        _reduce_qkv[(self.x.shape[0], 40)](
            self.qkv_plan.partial, w.q_norm, w.k_norm, cos, sin, position,
            self.q, k, v, self.x.shape[0], self.qkv_plan.splits,
            *k.stride()[:3], *v.stride()[:3], num_warps=4, num_stages=1,
            enable_fp_fusion=False,
        )
        self.attend(k, v, position, mask)
        self.o_plan.run(self.attn_flat, w.o.t())
        reduce_norm(self.o_plan, self.x, w.post_norm, self.norm, self.x, w.eps)
        self.gate_plan.run(self.norm, w.gate_up_weight)
        _reduce_silu[(self.x.shape[0], triton.cdiv(9728, 512))](
            self.gate_plan.partial, self.product, self.x.shape[0], self.gate_plan.splits,
            9728, 512, num_warps=4, num_stages=1, enable_fp_fusion=False,
        )
        self.down_plan.run(self.product, w.down.t())

    def finish_norm(self, weight, eps):
        reduce_norm(self.down_plan, self.x, weight, self.norm, self.x, eps)
