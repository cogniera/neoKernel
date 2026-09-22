"""Row allocation, draft-tree expansion, exact greedy acceptance, and cache compaction.

The draft source is the target's own recent predictions: TABLE[t] holds the
top-k next tokens the target computed the last time token t was an input
(seeded from the prompt's bigrams and warmed over the prompt). Drafts never
authorize output: every emitted token is the argmax the target computed on
the accepted prefix, and a draft node only extends the path when its token
equals that argmax.

Each step has a fixed budget of R rows. The allocator gives every unfinished
sequence a prefix of one template tree, sized by how many tokens it still
needs, so sequences that fall behind get the most nodes and finished ones
get none. Row r belongs to sequence SEQ[r] as node NODE[r] (-1 when unused);
sequence b owns rows STARTS[b] .. STARTS[b]+NODES[b]-1.

context is int64 [B,C]; count[b] is the number of committed tokens, whose last
one (the root) has no cache entry yet. base[b] = count[b]-1 is the root's slot.
"""

import triton
import triton.language as tl


@triton.jit
def _allocate(COUNT, STEP, NODES, STARTS, SEQ, NODE, FLAT,
              B: tl.constexpr, BP: tl.constexpr, R: tl.constexpr, RB: tl.constexpr,
              TMAX: tl.constexpr, PROMPT: tl.constexpr, OUTPUT: tl.constexpr, STEPS: tl.constexpr):
    b = tl.arange(0, BP)
    valid = b < B
    n = tl.load(COUNT + b, valid, PROMPT + OUTPUT)
    generated = (n - PROMPT).to(tl.int32)
    need = tl.maximum(OUTPUT - generated, 0)
    if STEPS > 0:
        # Pacing: a sequence on schedule may add at most room+1 tokens this step.
        step = tl.load(STEP).to(tl.int32)
        room = ((OUTPUT - 1) * step + STEPS - 1) // STEPS - generated
        need = tl.minimum(need, tl.maximum(room, 0) + 1)
    need = tl.where(valid, need, 0)
    active = (need > 0).to(tl.int32)
    spare = R - tl.sum(active, 0)
    total = tl.maximum(tl.sum(need, 0), 1)
    share = active + (spare * need) // total
    nodes = tl.minimum(tl.minimum(share, need), TMAX)
    starts = tl.cumsum(nodes, 0) - nodes
    tl.store(NODES + b, nodes, valid)
    tl.store(STARTS + b, starts, valid)
    for start in range(0, R, RB):
        r = start + tl.arange(0, RB)
        tl.store(SEQ + r, tl.full((RB,), -1, tl.int32), r < R)
        tl.store(NODE + r, tl.full((RB,), -1, tl.int32), r < R)
        tl.store(FLAT + r, tl.zeros((RB,), tl.int64), r < R)


def allocate_out(count, step, rows, prompt_length, output_length, schedule_steps):
    """Size every sequence's tree for this step and clear the row map."""
    batch = count.shape[0]
    _allocate[(1,)](count, step, rows.nodes, rows.starts, rows.seq, rows.node, rows.tokens,
                    batch, triton.next_power_of_2(batch), rows.count, min(rows.count, 256),
                    rows.tree.nodes, prompt_length, output_length, schedule_steps, num_warps=4)


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
def _draft(CTX, COUNT, TABLE, PARENT, RANK, DEPTH, SPINE, NODES, STARTS, SEQ, NODE, FLAT, BASE,
           C: tl.constexpr, T: tl.constexpr, TP: tl.constexpr, K: tl.constexpr,
           MAXDEPTH: tl.constexpr, NGRAM: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    n = tl.load(COUNT + b)
    size = tl.load(NODES + b)
    first = tl.load(STARTS + b)
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
        cand = tl.load(TABLE + ptok.to(tl.int64) * K + rank, at, 0)
        source = found + NGRAM - 1 + level
        use = (found >= 0) & (source < n)
        lookup = tl.load(row + tl.where(use, source, 0)).to(tl.int32)
        cand = tl.where((spine == level) & use, lookup, cand)
        toks = tl.where(at, cand, toks)
    live = idx < size
    tl.store(FLAT + first + idx, toks.to(tl.int64), live)
    tl.store(SEQ + first + idx, tl.zeros((TP,), tl.int32) + b, live)
    tl.store(NODE + first + idx, idx, live)
    tl.store(BASE + b, n - 1)


def draft_out(context, count, table, rows, base, ngram=3):
    """Fill every sequence's allocated rows with its root token and draft nodes."""
    batch, capacity = context.shape
    tree = rows.tree
    _draft[(batch,)](context, count, table, tree.parent, tree.rank, tree.depth, tree.spine,
                     rows.nodes, rows.starts, rows.seq, rows.node, rows.tokens, base,
                     capacity, tree.nodes, triton.next_power_of_2(tree.nodes), table.shape[1],
                     tree.max_depth, ngram, 1024, num_warps=4)


@triton.jit
def _warm(TOK, TOPK, TABLE, W: tl.constexpr, TS: tl.constexpr, K: tl.constexpr):
    b = tl.program_id(0)
    kk = tl.arange(0, K)
    for i in range(0, W):
        tok = tl.load(TOK + b * TS + i)
        tl.store(TABLE + tok * K + kk, tl.load(TOPK + (b * W + i) * K + kk).to(tl.int32))


def warm_out(tokens, topk, table):
    """table[tokens[b, i]] = topk[b*W + i] in position order, so later positions win."""
    batch, window = tokens.shape
    if topk.shape != (batch * window, table.shape[1]) or tokens.stride(1) != 1:
        raise ValueError("warm_out expects [B*W, K] predictions for a [B, W] token view")
    _warm[(batch,)](tokens, topk, table, window, tokens.stride(0), table.shape[1], num_warps=4)


@triton.jit
def _accept(FLAT, TOPK, CHILD, NODES, STARTS, CTX, COUNT, TABLE, STEP, PACKET, PATH, ACCEPTED,
            C: tl.constexpr, T: tl.constexpr, K: tl.constexpr, MC: tl.constexpr, MCP: tl.constexpr,
            MAXDEPTH: tl.constexpr, DP: tl.constexpr, WIDTH: tl.constexpr, PW: tl.constexpr,
            PROMPT: tl.constexpr, OUTPUT: tl.constexpr, STEPS: tl.constexpr):
    b = tl.program_id(0)
    size = tl.load(NODES + b)
    active = size > 0
    first = tl.where(active, tl.load(STARTS + b), 0)
    n = tl.load(COUNT + b)
    c = tl.arange(0, MCP)
    di = tl.arange(0, DP)
    kk = tl.arange(0, K)
    cur = size * 0
    pred = tl.load(TOPK + (first + cur) * K)
    emitted = tl.where(di == 0, pred, 0)
    path = tl.where(di == 0, 0, -1)
    accepted = cur
    alive = active
    for i in tl.static_range(1, MAXDEPTH + 1):
        kids = tl.load(CHILD + cur * MC + c, c < MC, -1)
        kids = tl.where(kids < size, kids, -1)
        ktok = tl.load(FLAT + first + kids, kids >= 0, -1)
        hit = (kids >= 0) & (ktok == pred)
        nxt = tl.max(tl.where(hit, kids, -1), 0)
        found = alive & (nxt >= 0)
        cur = tl.where(found, nxt, cur)
        pred = tl.load(TOPK + (first + cur) * K)
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
    accepted = tl.where(active, accepted, 0)
    tl.store(CTX + b * C + n + di, emitted, active & (di <= accepted))
    tl.store(COUNT + b, tl.where(active, tl.minimum(n + accepted + 1, PROMPT + OUTPUT), n))
    tl.store(PACKET + b * WIDTH + di, emitted, di <= MAXDEPTH)
    tl.store(PACKET + b * WIDTH + MAXDEPTH + 1, tl.where(active, accepted + 1, 0).to(tl.int64))
    tl.store(PACKET + b * WIDTH + MAXDEPTH + 2, raw.to(tl.int64))
    tl.store(PATH + b * PW + di, path, di <= MAXDEPTH)
    tl.store(ACCEPTED + b, accepted)
    # Refresh the table with every node's prediction; the accepted path is
    # written last so verified contexts win over rejected branches.
    for j in tl.static_range(T):
        on = (j < size) | (kk < 0)
        at = tl.where(j < size, first + j, 0)
        tok = tl.load(FLAT + at)
        tl.store(TABLE + tok * K + kk, tl.load(TOPK + at * K + kk).to(tl.int32), on)
    for i in tl.static_range(MAXDEPTH + 1):
        node = tl.sum(tl.where(di == i, path, 0), 0)
        on = (active & (i <= accepted) & (node >= 0)) | (kk < 0)
        safe = tl.where(node >= 0, node, 0)
        tok = tl.load(FLAT + first + safe)
        tl.store(TABLE + tok * K + kk, tl.load(TOPK + (first + safe) * K + kk).to(tl.int32), on)


def accept_out(rows, topk, context, count, table, step, packet, path, accepted,
               prompt_length, output_length, schedule_steps):
    batch, capacity = context.shape
    tree = rows.tree
    width = packet.shape[1]
    if width != tree.max_depth + 3 or path.shape[1] < tree.max_depth + 1:
        raise ValueError("packet needs max_depth + 3 columns and path max_depth + 1")
    _accept[(batch,)](rows.tokens, topk, tree.child, rows.nodes, rows.starts, context, count, table,
                      step, packet, path, accepted,
                      capacity, tree.nodes, table.shape[1], tree.max_children,
                      triton.next_power_of_2(tree.max_children), tree.max_depth,
                      triton.next_power_of_2(tree.max_depth + 1), width, path.shape[1],
                      prompt_length, output_length, schedule_steps, num_warps=4)


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


class RowMap:
    """Per-step ownership of the R decode rows, written on device by the allocator and draft."""

    def __init__(self, batch, count, tree, device):
        import torch
        self.count = count
        self.tree = tree
        self.nodes = torch.zeros((batch,), device=device, dtype=torch.int32)
        self.starts = torch.zeros((batch,), device=device, dtype=torch.int32)
        self.seq = torch.full((count,), -1, device=device, dtype=torch.int32)
        self.node = torch.full((count,), -1, device=device, dtype=torch.int32)
        self.tokens = torch.zeros((count,), device=device, dtype=torch.int64)
