"""Draft-tree expansion, exact greedy acceptance, and cache compaction on device.

The draft source is the target's own recent predictions: TABLE[b, t] holds the
top-k next tokens the target computed the last time token t was an input in
sequence b (seeded from that sequence's prompt bigrams). Drafts never authorize
output: every emitted token is the argmax the target computed on the accepted
prefix, and a draft node only extends the path when its token equals that argmax.

The table is per sequence, not shared. Rows are keyed by token id, so one table
would be written by every sequence at once and the winner of a collision would
be whichever program happened to land last; the drafts, the accepted run length
and therefore the step schedule would all vary between two calls on the same
prompt. One slice per sequence makes every write deterministic without
serializing anything, and keeps one sequence's continuations out of another's
drafts. TS is the stride between slices.

context is int64 [B,C]; count[b] is the number of committed tokens, whose last
one (the root) has no cache entry yet. base[b] = count[b]-1 is the root's slot.
"""

import triton
import triton.language as tl


@triton.jit
def _find(row, n, NGRAM: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    """Most recent start i with row[i:i+NGRAM] == row[n-NGRAM:n] and a token after it; -1 if none."""
    best = (n * 0 - 1).to(tl.int32)
    for start in range(0, C, BLOCK):
        i = start + tl.arange(0, BLOCK)
        ok = (i + NGRAM) < n
        match = ok
        for t in tl.static_range(NGRAM):
            tok = tl.load(row + i + t, ok, -1)
            ref = tl.load(row + n - NGRAM + t)
            match = match & (tok == ref)
        best = tl.maximum(best, tl.max(tl.where(match, i, -1), 0))
    return best


@triton.jit
def _draft(CTX, COUNT, TABLE, PARENT, RANK, DEPTH, SPINE, TOKENS, BASE,
           C: tl.constexpr, T: tl.constexpr, TP: tl.constexpr, K: tl.constexpr,
           TS: tl.constexpr, MAXDEPTH: tl.constexpr, NGRAM: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    n = tl.load(COUNT + b)
    table = TABLE + b.to(tl.int64) * TS
    row = CTX + b * C
    root = tl.load(row + n - 1).to(tl.int32)
    idx = tl.arange(0, TP)
    valid = idx < T
    parent = tl.load(PARENT + idx, valid, 0)
    rank = tl.load(RANK + idx, valid, 0)
    depth = tl.load(DEPTH + idx, valid, -1)
    spine = tl.load(SPINE + idx, valid, -1)
    # A repeated n-gram in the committed context overrides the rank-0 chain
    # with what followed it last time; the table fills every other node.
    found = _find(row, n, NGRAM, C, BLOCK)
    toks = tl.zeros((TP,), tl.int32) + root
    for level in tl.static_range(1, MAXDEPTH + 1):
        # Every node at this level extends its parent's token, resolved last level.
        ptok = tl.sum(tl.where(idx[None, :] == parent[:, None], toks[None, :], 0), 1)
        at = depth == level
        cand = tl.load(table + ptok.to(tl.int64) * K + rank, at, 0)
        source = found + NGRAM - 1 + level
        use = (found >= 0) & (source < n)
        lookup = tl.load(row + tl.where(use, source, 0)).to(tl.int32)
        cand = tl.where((spine == level) & use, lookup, cand)
        toks = tl.where(at, cand, toks)
    tl.store(TOKENS + b * T + idx, toks.to(tl.int64), valid)
    tl.store(BASE + b, n - 1)


def draft_out(context, count, table, tree, tokens, base, ngram=3):
    batch, capacity = context.shape
    _draft[(batch,)](context, count, table, tree.parent, tree.rank, tree.depth, tree.spine, tokens, base,
                     capacity, tree.nodes, triton.next_power_of_2(tree.nodes), table.shape[2],
                     table.stride(0), tree.max_depth, ngram, 1024, num_warps=4)


@triton.jit
def _seed(PROMPT, TOPK, TABLE, PS: tl.constexpr, W: tl.constexpr, LIMIT: tl.constexpr,
          N: tl.constexpr, K: tl.constexpr, TS: tl.constexpr):
    """Bigrams for the positions no warm pass covers, then the target's own top-k."""
    b = tl.program_id(0)
    kk = tl.arange(0, K)
    table = TABLE + b.to(tl.int64) * TS
    row = PROMPT + b * PS
    # Ascending, one position at a time: a token repeated in the prompt must end
    # up with what followed its LAST occurrence, whichever program runs it.
    for i in range(0, LIMIT):
        tok = tl.load(row + i).to(tl.int64)
        tl.store(table + tok * K, tl.load(row + i + 1).to(tl.int32))
    for i in range(0, W):
        tok = tl.load(row + N - W + i).to(tl.int64)
        tl.store(table + tok * K + kk, tl.load(TOPK + (b * W + i) * K + kk).to(tl.int32))


def seed_out(prompt, topk, table, window):
    """table[b, prompt[b, i]] = the target's top-k at position i, later positions winning."""
    batch, length = prompt.shape
    if window and topk.shape != (batch * window, table.shape[2]):
        raise ValueError("seed_out expects [B*W, K] predictions for the last W prompt positions")
    if prompt.stride(1) != 1 or table.shape[0] != batch:
        raise ValueError("seed_out expects a contiguous [B, N] prompt and one table slice per sequence")
    _seed[(batch,)](prompt, topk, table, prompt.stride(0), window,
                    max(0, min(length - 1, length - window)), length, table.shape[2],
                    table.stride(0), num_warps=4)


@triton.jit
def _accept(TOKENS, TOPK, CHILD, CTX, COUNT, TABLE, STEP, PACKET, PATH, ACCEPTED,
            C: tl.constexpr, T: tl.constexpr, K: tl.constexpr, MC: tl.constexpr, MCP: tl.constexpr,
            MAXDEPTH: tl.constexpr, DP: tl.constexpr, WIDTH: tl.constexpr, PW: tl.constexpr,
            PROMPT: tl.constexpr, OUTPUT: tl.constexpr, STEPS: tl.constexpr, TS: tl.constexpr):
    b = tl.program_id(0)
    n = tl.load(COUNT + b)
    table = TABLE + b.to(tl.int64) * TS
    c = tl.arange(0, MCP)
    di = tl.arange(0, DP)
    kk = tl.arange(0, K)
    cur = tl.load(CHILD) * 0
    pred = tl.load(TOPK + (b * T) * K)
    emitted = tl.where(di == 0, pred, 0)
    path = tl.where(di == 0, 0, -1)
    accepted = cur
    alive = cur == 0
    for i in tl.static_range(1, MAXDEPTH + 1):
        kids = tl.load(CHILD + cur * MC + c, c < MC, -1)
        ktok = tl.load(TOKENS + b * T + kids, kids >= 0, -1)
        hit = (kids >= 0) & (ktok == pred)
        found = alive & (tl.max(tl.where(hit, kids, -1), 0) >= 0)
        nxt = tl.max(tl.where(hit, kids, -1), 0)
        cur = tl.where(found, nxt, cur)
        pred = tl.load(TOPK + (b * T + cur) * K)
        emitted = tl.where(found & (di == i), pred, emitted)
        path = tl.where(found & (di == i), cur, path)
        accepted = accepted + found.to(tl.int32)
        alive = found
    raw = accepted
    generated = (n - PROMPT).to(tl.int32)
    if STEPS > 0:
        step = tl.load(STEP).to(tl.int32)
        allowed = ((OUTPUT - 1) * step + STEPS - 1) // STEPS - generated
        accepted = tl.minimum(accepted, tl.maximum(allowed, 0))
    accepted = tl.minimum(accepted, tl.maximum(OUTPUT - 1 - generated, 0))
    tl.store(CTX + b * C + n + di, emitted, di <= accepted)
    tl.store(COUNT + b, tl.minimum(n + accepted + 1, PROMPT + OUTPUT))
    tl.store(PACKET + b * WIDTH + di, emitted, di <= MAXDEPTH)
    tl.store(PACKET + b * WIDTH + MAXDEPTH + 1, accepted.to(tl.int64))
    tl.store(PACKET + b * WIDTH + MAXDEPTH + 2, raw.to(tl.int64))
    tl.store(PATH + b * PW + di, path, di <= MAXDEPTH)
    tl.store(ACCEPTED + b, accepted)
    # Refresh the table with every node's prediction; the accepted path is
    # written last so verified contexts win over rejected branches.
    for j in tl.static_range(T):
        tok = tl.load(TOKENS + b * T + j)
        tl.store(table + tok * K + kk, tl.load(TOPK + (b * T + j) * K + kk).to(tl.int32))
    for i in tl.static_range(MAXDEPTH + 1):
        node = tl.sum(tl.where(di == i, path, 0), 0)
        on = ((i <= accepted) & (node >= 0)) | (kk < 0)
        safe = tl.where(node >= 0, node, 0)
        tok = tl.load(TOKENS + b * T + safe)
        tl.store(table + tok * K + kk, tl.load(TOPK + (b * T + safe) * K + kk).to(tl.int32), on)


def accept_out(tokens, topk, tree, context, count, table, step, packet, path, accepted,
               prompt_length, output_length, schedule_steps):
    batch, capacity = context.shape
    width = packet.shape[1]
    if width != tree.max_depth + 3 or path.shape[1] < tree.max_depth + 1:
        raise ValueError("packet needs max_depth + 3 columns and path max_depth + 1")
    _accept[(batch,)](tokens, topk, tree.child, context, count, table, step, packet, path, accepted,
                      capacity, tree.nodes, table.shape[2], tree.max_children,
                      triton.next_power_of_2(tree.max_children), tree.max_depth,
                      triton.next_power_of_2(tree.max_depth + 1), width, path.shape[1],
                      prompt_length, output_length, schedule_steps, table.stride(0), num_warps=4)


@triton.jit
def _compact(KC, VC, BASE, PATH, ACCEPTED,
             LS: tl.constexpr, KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
             DP: tl.constexpr, MAXDEPTH: tl.constexpr):
    b, layer = tl.program_id(0), tl.program_id(1)
    base = tl.load(BASE + b)
    accepted = tl.load(ACCEPTED + b)
    hh = tl.arange(0, 8)
    d = tl.arange(0, 128)
    tile = hh[:, None] * KH + d[None, :]
    origin = layer * LS + b * KB
    # In-order moves are safe: a path node's slot index is at least its depth
    # and grows along the path, so no destination is a later source.
    for i in tl.static_range(1, MAXDEPTH + 1):
        node = tl.load(PATH + b * DP + i)
        on = (i <= accepted) & (node >= 0) & (node != i)
        k = tl.load(KC + origin + (base + node) * KS + tile, on, 0.)
        v = tl.load(VC + origin + (base + node) * KS + tile, on, 0.)
        tl.store(KC + origin + (base + i) * KS + tile, k, on)
        tl.store(VC + origin + (base + i) * KS + tile, v, on)


def compact_out(k_cache, v_cache, base, path, accepted, tree):
    """Move accepted tree nodes' K/V rows to slots base+1..base+depth in every layer."""
    if tree.max_depth == 0:
        return
    if k_cache.stride() != v_cache.stride() or k_cache.stride(4) != 1:
        raise ValueError("compact_out expects matching [L,B,8,capacity,128] cache views")
    layers, batch = k_cache.shape[0], k_cache.shape[1]
    _compact[(batch, layers)](k_cache, v_cache, base, path, accepted,
                              *k_cache.stride()[:4], path.shape[1], tree.max_depth, num_warps=4)
