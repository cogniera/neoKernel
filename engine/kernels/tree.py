"""Static draft tree: which top-k continuation each node extends.

Node 0 is the verified token whose next logits are still unknown. Node j > 0
holds candidate rank RANK[j] of the top-k table row of its parent's token.
Nodes are numbered so a parent precedes its children and a node's index is at
least its depth, so an accepted root-to-leaf path can be compacted into cache
slots base..base+depth in increasing order without clobbering unread rows.
The shape is chosen once by best-first expansion under a fixed guess of the
per-rank hit probabilities; the tables that describe it are plain tensors.
"""

import torch

RANK_PROBABILITY = (0.55, 0.14, 0.07, 0.045, 0.03, 0.022, 0.017, 0.013)


class DraftTree:
    def __init__(self, nodes, branching, device):
        if nodes < 1 or branching < 1 or branching > len(RANK_PROBABILITY):
            raise ValueError("draft tree needs at least one node and 1..8 ranks")
        parent, rank, depth = [-1], [0], [0]
        # Candidates are (probability, parent, rank); the earliest of equals wins.
        candidates = [(RANK_PROBABILITY[0], 0, 0)]
        while len(parent) < nodes and candidates:
            best = max(range(len(candidates)), key=lambda i: (candidates[i][0], -i))
            prob, p, r = candidates.pop(best)
            j = len(parent)
            parent.append(p)
            rank.append(r)
            depth.append(depth[p] + 1)
            candidates.append((prob * RANK_PROBABILITY[0], j, 0))
            if r + 1 < branching:
                candidates.append((prob * RANK_PROBABILITY[r + 1] / RANK_PROBABILITY[r], p, r + 1))
        self.nodes = len(parent)
        self.branching = branching
        self.max_depth = max(depth)
        children = [[] for _ in range(self.nodes)]
        for j in range(1, self.nodes):
            children[parent[j]].append(j)
        self.max_children = max(1, max(len(c) for c in children))
        # The spine is the rank-0 chain from the root; a prompt-lookup match overrides it.
        spine_depth = [-1] * self.nodes
        node = 0
        while node >= 0:
            spine_depth[node] = depth[node]
            node = next((c for c in children[node] if rank[c] == 0), -1)
        ancestor = [[0] * self.nodes for _ in range(self.nodes)]
        for j in range(self.nodes):
            a = j
            while a >= 0:
                ancestor[j][a] = 1
                a = parent[a]
        as_tensor = lambda values, dtype=torch.int32: torch.tensor(values, dtype=dtype, device=device)
        self.parent = as_tensor(parent)
        self.rank = as_tensor(rank)
        self.depth = as_tensor(depth)
        self.child = as_tensor([c + [-1] * (self.max_children - len(c)) for c in children]).contiguous()
        self.ancestor = as_tensor(ancestor).contiguous()
        self.spine = as_tensor(spine_depth)
        self.depth_list = depth
        self.parent_list = parent
        self.rank_list = rank
