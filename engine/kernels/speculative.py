"""Prefix-only draft lookup and exact greedy acceptance on device.

History is int64 [B,capacity]. Position is [B], pointing to the last emitted
token, which is the next input to the target. Drafts never authorize output:
the target predicts every emitted token, and only a matching prefix survives.
"""

import triton
import triton.language as tl


@triton.jit
def _draft(H, POS, INPUT, CAP: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    pos = tl.load(POS + b)
    current = tl.load(H + b * CAP + pos)
    tl.store(INPUT + b * T, current)
    if T > 1:
        i = tl.arange(0, BLOCK)
        valid = (i < CAP) & (i + T - 1 <= pos)
        a0 = tl.load(H + b * CAP + i, valid, -1)
        match = valid & (a0 == current)
        length = tl.where(match, 1, 0)
        for back in tl.static_range(1, 4):
            previous = tl.load(H + b * CAP + pos - back, pos >= back, -2)
            old = tl.load(H + b * CAP + i - back, valid & (i >= back), -1)
            match = match & (i >= back) & (pos >= back) & (old == previous)
            length = tl.where(match, back + 1, length)
        longest = tl.max(length, 0)
        source = tl.max(tl.where((length == longest) & (length > 0), i, -1), 0)
        for j in tl.static_range(1, T):
            token = tl.load(H + b * CAP + source + j, source >= 0, 0)
            tl.store(INPUT + b * T + j, tl.where(source >= 0, token, current))


def draft_out(history, position, inputs):
    tokens = inputs.shape[1]
    _draft[(history.shape[0],)](history, position, inputs, history.shape[1], tokens,
                               triton.next_power_of_2(history.shape[1]), num_warps=4)


@triton.jit
def _accept(INPUT, PRED, H, POS, PACKET, CAP: tl.constexpr, END: tl.constexpr,
            T: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    pos = tl.load(POS + b)
    j = tl.arange(0, BLOCK)
    predicted = tl.load(PRED + b * T + j, j < T, 0)
    proposed = tl.load(INPUT + b * T + j + 1, j < T - 1, 0)
    mismatch = (j < T - 1) & (predicted != proposed)
    first = tl.min(tl.where(mismatch, j, T - 1), 0)
    count = tl.minimum(first + 1, tl.maximum(END - pos, 0))
    tl.store(PACKET + b * STRIDE, count)
    tl.store(PACKET + b * STRIDE + j + 1, predicted, j < T)
    tl.store(H + b * CAP + pos + j + 1, predicted, j < count)
    tl.store(POS + b, pos + count)


def accept_out(inputs, predictions, history, position, packet, end):
    tokens = inputs.shape[1]
    _accept[(history.shape[0],)](inputs, predictions, history, position, packet,
                                history.shape[1], end, tokens, packet.shape[1],
                                triton.next_power_of_2(tokens), num_warps=4)
