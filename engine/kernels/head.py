"""Exact vocabulary selection without materializing the full logits matrix.

Every vocabulary row is evaluated in BF16 with FP32 accumulation. Logits
round to BF16 before selection; equal scores choose the lowest token ID.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _tiles(X, W, VALUES, IDS, B: tl.constexpr, V: tl.constexpr,
           H: tl.constexpr, TILES: tl.constexpr, BM: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr):
    nt, mt = tl.program_id(0), tl.program_id(1)
    m = mt * BM + tl.arange(0, BM)
    n = nt * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for tile in range(tl.cdiv(H, BK)):
        k = tile * BK + kk
        x = tl.load(X + m[:, None] * H + k[None, :], (m[:, None] < B) & (k[None, :] < H), 0.)
        w = tl.load(W + n[:, None] * H + k[None, :], (n[:, None] < V) & (k[None, :] < H), 0.)
        acc = tl.dot(x, tl.trans(w), acc)
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    rounded = tl.where(n[None, :] < V, rounded, -float('inf'))
    best = tl.max(rounded, 1)
    ids = tl.min(tl.where((rounded == best[:, None]) & (n[None, :] < V), n[None, :], 2147483647), 1)
    tl.store(VALUES + m * TILES + nt, best, m < B)
    tl.store(IDS + m * TILES + nt, ids, m < B)


@triton.jit
def _select(VALUES, IDS, OUT, TILES: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    t = tl.arange(0, BLOCK)
    scores = tl.load(VALUES + b * TILES + t, t < TILES, -float('inf'))
    ids = tl.load(IDS + b * TILES + t, t < TILES, 2147483647)
    best = tl.max(scores, 0)
    index = tl.min(tl.where(scores == best, ids, 2147483647), 0)
    tl.store(OUT + b, index.to(tl.int64))


class Head:
    def __init__(self, batch, vocab, hidden, device):
        self.batch, self.vocab, self.hidden = batch, vocab, hidden
        self.tiles = triton.cdiv(vocab, 128)
        self.values = torch.empty((batch, self.tiles), device=device, dtype=torch.float32)
        self.ids = torch.empty((batch, self.tiles), device=device, dtype=torch.int32)

    def out(self, x, weight, tokens):
        _tiles[(self.tiles, triton.cdiv(self.batch, 16))](
            x, weight, self.values, self.ids, self.batch, self.vocab, self.hidden,
            self.tiles, 16, 128, 128, num_warps=4, num_stages=3,
        )
        _select[(self.batch,)](self.values, self.ids, tokens, self.tiles,
                              triton.next_power_of_2(self.tiles), num_warps=4)


@triton.jit
def _commit(NEXT, TOKENS, HISTORY, POS, B: tl.constexpr, PROMPT: tl.constexpr,
            BLOCK: tl.constexpr):
    b = tl.arange(0, BLOCK)
    pos = tl.load(POS)
    value = tl.load(NEXT + b, b < B, 0)
    tl.store(TOKENS + b, value, b < B)
    tl.store(HISTORY + (pos - PROMPT + 1) * B + b, value, b < B)
    tl.store(POS, pos + 1)


def commit_step(next_tokens, tokens, history, position, prompt_length):
    batch = tokens.numel()
    _commit[(1,)](next_tokens, tokens, history, position, batch, prompt_length,
                  triton.next_power_of_2(batch), num_warps=4)
