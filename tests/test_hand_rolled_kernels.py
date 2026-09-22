"""GPU-only numerical tests, before engine integration; unittest skips on CPU.

Run with: python -m unittest discover -s tests -p test_hand_rolled_kernels.py -v
All numeric reports include max absolute error and last-dimension argmax agreement.
The test tolerances diagnose individual BF16 operations; sequence correctness is
still decided exclusively by the unchanged neokernel judge.
"""

import itertools
import json
from pathlib import Path
import sys
import unittest

try:
    import torch
    TORCH, GPU = True, torch.cuda.is_available()
except ImportError:
    TORCH = GPU = False


@unittest.skipUnless(TORCH, 'PyTorch required')
class GroupedSDPALayoutTests(unittest.TestCase):
    def test_copy_preserves_head_order_for_noncontiguous_sdpa_output(self):
        # SDPA may return B,H,Q,D backed by B,Q,H,D. Flattening H,Q with
        # view is invalid; copy into a matching grouped destination instead.
        expected = torch.arange(2*8*4*128).reshape(2, 8, 4, 128)
        output = expected.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(output.is_contiguous())
        storage = torch.empty((2, 32, 128), dtype=expected.dtype)
        storage.view(2, 8, 4, 128).copy_(output)
        torch.testing.assert_close(storage, expected.reshape(2, 32, 128))


@unittest.skipUnless(TORCH, 'PyTorch required')
class DraftTreeTests(unittest.TestCase):
    def test_tree_orders_parents_first_and_depth_bounds_index(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'engine'))
        from kernels.tree import DraftTree
        for nodes in (1, 4, 8, 16, 32, 64):
            tree = DraftTree(nodes, 8, 'cpu')
            self.assertEqual(tree.nodes, nodes)
            for j in range(1, nodes):
                self.assertLess(tree.parent_list[j], j)
                self.assertGreaterEqual(j, tree.depth_list[j])
                self.assertEqual(tree.depth_list[j], tree.depth_list[tree.parent_list[j]] + 1)
            self.assertEqual(tree.depth_list[0], 0)
            ancestor = tree.ancestor.tolist()
            for j in range(nodes):
                chain, a = set(), j
                while a >= 0:
                    chain.add(a)
                    a = tree.parent_list[a]
                self.assertEqual({i for i in range(nodes) if ancestor[j][i]}, chain)
            for j in range(nodes):
                kids = [c for c in range(nodes) if tree.parent_list[c] == j]
                self.assertEqual([c for c in tree.child[j].tolist() if c >= 0], kids)
                self.assertEqual(sorted(tree.rank_list[c] for c in kids), list(range(len(kids))))


@unittest.skipUnless(GPU, 'CUDA required for Triton decode tests')
class HandRolledKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'engine'))
        import triton
        import transformers
        if triton.__version__ != '3.1.0' or transformers.__version__ != '4.51.3':
            raise RuntimeError('GPU tests require pinned Triton 3.1.0 / Transformers 4.51.3')
        torch.backends.cuda.matmul.allow_tf32 = False
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        cls.config = Qwen3Config(hidden_size=2560, intermediate_size=9728,
                                num_attention_heads=32, num_key_value_heads=8,
                                head_dim=128, num_hidden_layers=1, rope_theta=5e6,
                                rms_norm_eps=1e-6)
        cls.config._attn_implementation = 'sdpa'
        rotary = Qwen3RotaryEmbedding(cls.config, device='cuda')
        cls.cos, cls.sin = rotary(torch.empty(1, device='cuda', dtype=torch.bfloat16),
                                 torch.arange(2080, device='cuda')[None])
        cls.cos, cls.sin = cls.cos[0].contiguous(), cls.sin[0].contiguous()

    def setUp(self):
        torch.manual_seed(714)

    def report(self, label, actual, expected, atol, rtol=0.01):
        a, e = actual.float(), expected.float()
        result = {'test': label, 'max_abs_diff': (a-e).abs().max().item(),
                  'argmax_agreement': (a.argmax(-1) == e.argmax(-1)).float().mean().item()}
        print(json.dumps(result), flush=True)
        torch.testing.assert_close(a, e, atol=atol, rtol=rtol)

    def rand(self, *shape):
        return torch.randn(shape, device='cuda', dtype=torch.bfloat16)

    def trees(self):
        from kernels.tree import DraftTree
        return [DraftTree(nodes, 8, 'cuda') for nodes in (1, 4, 8, 16, 64)]

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_norm_and_residual(self):
        from kernels.rmsnorm import norm_out
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        native = Qwen3RMSNorm(2560, eps=1e-6).cuda().bfloat16()
        native.weight.copy_(self.rand(2560))
        for batch, residual in itertools.product((1, 4, 16, 128), (False, True)):
            x, r, out = self.rand(batch, 2560), self.rand(batch, 2560), self.rand(batch, 2560)
            expected_sum = x + r
            norm_out(x, native.weight, out, residual=r if residual else None,
                     summed=r if residual else None)
            if residual:
                torch.testing.assert_close(r, expected_sum, atol=0, rtol=0)
            self.report(f'norm/{batch}/{residual}', out, native(expected_sum if residual else x), .03125)

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_qk_rope_cache(self):
        from kernels.tree_ops import qk_rope_cache_out
        from kernels.tree_decode import cache_storage
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb
        qn, kn = [Qwen3RMSNorm(128, eps=1e-6).cuda().bfloat16() for _ in range(2)]
        qn.weight.copy_(self.rand(128)); kn.weight.copy_(self.rand(128))
        for batch, layout, fused, tree in itertools.product((1, 4), ('bhsd', 'bshd'), (False, True), self.trees()):
            rows = batch * tree.nodes
            p = self.rand(rows, 6144)
            q = self.rand(rows, 32, 128)
            k, v = [cache_storage(batch, 2080, 'cuda', layout)[0] for _ in range(2)]
            for pos in (0, 512, 2080 - tree.nodes):
                base = torch.full((batch,), pos, device='cuda', dtype=torch.int64) - torch.arange(batch, device='cuda')
                base.clamp_(min=0)
                qk_rope_cache_out(p, qn.weight, kn.weight, self.cos, self.sin, base, tree,
                                 q, k, v, torch.empty_like(p), fused)
                qref = qn(p[:, :4096].reshape(rows, 32, 1, 128))
                kref = kn(p[:, 4096:5120].reshape(rows, 8, 1, 128))
                positions = (base[:, None] + tree.depth[None, :].to(torch.int64)).reshape(-1)
                qr, kr = apply_rotary_pos_emb(qref, kref, self.cos[positions][:, None], self.sin[positions][:, None])
                label = f'qk/{batch}/{layout}/{fused}/{tree.nodes}/{pos}'
                self.report(label+'/q', q, qr[:, :, 0], .0625)
                slots = base[:, None] + torch.arange(tree.nodes, device='cuda')[None, :]
                written_k = torch.stack([k[b, :, slots[b]].transpose(0, 1) for b in range(batch)]).reshape(rows, 8, 128)
                written_v = torch.stack([v[b, :, slots[b]].transpose(0, 1) for b in range(batch)]).reshape(rows, 8, 128)
                self.report(label+'/k', written_k, kr[:, :, 0], .0625)
                torch.testing.assert_close(written_v, p[:, 5120:].reshape(rows, 8, 128), atol=0, rtol=0)
            self.assertEqual(k[:, :, tree.nodes:512 - batch].count_nonzero().item(), 0)

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_silu(self):
        from kernels.elementwise import silu_mul_out
        from transformers.activations import ACT2FN
        for batch, fused in itertools.product((1, 4, 16, 128), (False, True)):
            p = self.rand(batch, 19456)
            out = self.rand(batch, 9728)
            silu_mul_out(p, out, torch.empty_like(out), fused)
            ref = ACT2FN['silu'](p[:, :9728]) * p[:, 9728:]
            self.report(f'silu/{batch}/{fused}', out, ref, .03125)

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_attention(self):
        from kernels.tree_attention import attention_out
        from kernels.tree_decode import cache_storage, TreeDecodeBuffers as DecodeBuffers
        import torch.nn.functional as F
        for batch, capacity, layout, tree in itertools.product((1, 4), (640, 2080), ('bhsd', 'bshd'), self.trees()):
            rows = batch * tree.nodes
            q = self.rand(rows, 32, 128)
            k, v = [cache_storage(batch, capacity, 'cuda', layout)[0] for _ in range(2)]
            k.copy_(self.rand(*k.shape)); v.copy_(self.rand(*v.shape))
            buf = DecodeBuffers(batch, capacity, 'cuda', tree=tree)
            ancestor = tree.ancestor.bool().cpu()
            for pos in (0, 127, 512, capacity - tree.nodes):
                base = (torch.full((batch,), pos, dtype=torch.int64) - torch.arange(batch)).clamp_(min=0)
                attention_out(q, k, v, base.cuda(), tree, buf.attn, buf.partial, buf.lse)
                ref = torch.empty_like(buf.attn)
                for b in range(batch):
                    for j in range(tree.nodes):
                        keep = torch.cat((torch.arange(base[b]), base[b] + torch.nonzero(ancestor[j]).flatten()))
                        keys = k[b][:, keep.cuda()].repeat_interleave(4, dim=0)
                        values = v[b][:, keep.cuda()].repeat_interleave(4, dim=0)
                        ref[b * tree.nodes + j] = F.scaled_dot_product_attention(
                            q[b * tree.nodes + j][:, None], keys, values)[:, 0]
                self.report(f'attn/{batch}/{capacity}/{layout}/{tree.nodes}/{pos}', buf.attn, ref, .015625)
            # Poison inactive slots: masked capacity must have no influence.
            k[:, :, 512 + tree.nodes:].fill_(float('nan')); v[:, :, 512 + tree.nodes:].fill_(float('nan'))
            base = torch.full((batch,), 512, device='cuda', dtype=torch.int64)
            attention_out(q, k, v, base, tree, buf.attn, buf.partial, buf.lse)
            self.assertTrue(buf.attn.isfinite().all().item())

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_warm(self):
        from kernels.speculative import warm_out
        vocab, batch, window = 500, 3, 40
        table = torch.randint(0, vocab, (vocab, 8), device='cuda', dtype=torch.int32)
        before = table.clone()
        # Distinct tokens across sequences: programs racing on one row would be arbitrary.
        prompt = torch.randperm(vocab, device='cuda')[:batch * 64].view(batch, 64).contiguous()
        prompt[1, 64 - window + 3] = prompt[1, 64 - 5]
        topk = torch.randint(0, vocab, (batch * window, 8), device='cuda')
        warm_out(prompt[:, 64 - window:], topk, table)
        expected = before.clone()
        for b in range(batch):
            for i in range(window):
                expected[prompt[b, 64 - window + i]] = topk[b * window + i].to(torch.int32)
        torch.testing.assert_close(table, expected, atol=0, rtol=0)

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_draft_accept_compact(self):
        from kernels.speculative import draft_out, accept_out, compact_out
        from kernels.tree_decode import cache_storage
        vocab, batch, prompt, output, layers = 300, 3, 40, 24, 2
        for tree in self.trees():
            nodes, depth = tree.nodes, tree.max_depth
            capacity = prompt + output + nodes
            spine = tree.spine.tolist()
            for trial in range(3):
                table = torch.randint(0, vocab, (vocab, 8), device='cuda', dtype=torch.int32)
                context = torch.randint(0, vocab, (batch, prompt + output + depth + 1), device='cuda')
                count = torch.randint(prompt + 1, prompt + output + 1, (batch,), device='cuda')
                if trial == 0:
                    count.fill_(prompt + 1)
                if trial == 1:
                    # Plant a repeat of the last three tokens early in sequence 1 and
                    # a nearer one in sequence 2 (only a token or two follow it).
                    n = count[1].item()
                    context[1, 5:8] = context[1, n - 3:n].clone()
                    n = count[2].item()
                    context[2, n - 5:n - 2] = context[2, n - 3:n].clone()
                tokens = torch.empty((batch, nodes), device='cuda', dtype=torch.int64)
                base = torch.empty((batch,), device='cuda', dtype=torch.int64)
                draft_out(context, count, table, tree, tokens, base)
                expect = torch.empty_like(tokens)
                for b in range(batch):
                    n = count[b].item()
                    row = context[b].tolist()
                    found = max([i for i in range(n - 3) if row[i:i + 3] == row[n - 3:n]], default=-1)
                    expect[b, 0] = context[b, n - 1]
                    for j in range(1, nodes):
                        d = tree.depth_list[j]
                        if spine[j] == d and found >= 0 and found + 2 + d < n:
                            expect[b, j] = row[found + 2 + d]
                        else:
                            expect[b, j] = table[expect[b, tree.parent_list[j]], tree.rank_list[j]]
                torch.testing.assert_close(tokens, expect, atol=0, rtol=0)
                torch.testing.assert_close(base, count - 1, atol=0, rtol=0)
                # Predictions: force partial agreement so paths of every length occur.
                topk = torch.randint(0, vocab, (batch * nodes, 8), device='cuda')
                for b in range(batch):
                    for j in range(nodes):
                        if torch.rand(()) < 0.7:
                            kids = [c for c in range(nodes) if tree.parent_list[c] == j]
                            if kids:
                                topk[b * nodes + j, 0] = tokens[b, kids[int(torch.randint(len(kids), ()))]]
                step = torch.tensor([1 + trial], device='cuda')
                schedule = 0 if trial < 2 else max(1, output // 2)
                packet = torch.zeros((batch, depth + 3), device='cuda', dtype=torch.int64)
                path = torch.zeros((batch, depth + 1), device='cuda', dtype=torch.int32)
                accepted = torch.zeros((batch,), device='cuda', dtype=torch.int32)
                before_context, before_count = context.clone(), count.clone()
                accept_out(tokens, topk, tree, context, count, table, step, packet, path, accepted,
                           prompt, output, schedule)
                all_tokens = tokens.tolist()
                for b in range(batch):
                    n = before_count[b].item()
                    cur, emitted, walk = 0, [topk[b * nodes, 0].item()], [0]
                    while True:
                        kids = [c for c in range(nodes)
                                if tree.parent_list[c] == cur and all_tokens[b][c] == emitted[-1]]
                        if not kids:
                            break
                        cur = max(kids)
                        walk.append(cur)
                        emitted.append(topk[b * nodes + cur, 0].item())
                    raw = len(walk) - 1
                    allowed = raw
                    if schedule:
                        quota = ((output - 1) * step.item() + schedule - 1) // schedule - (n - prompt)
                        allowed = min(allowed, max(0, quota))
                    allowed = min(allowed, max(0, output - 1 - (n - prompt)))
                    self.assertEqual(packet[b, depth + 2].item(), raw)
                    self.assertEqual(packet[b, depth + 1].item(), allowed)
                    self.assertEqual(accepted[b].item(), allowed)
                    self.assertEqual(packet[b, :allowed + 1].tolist(), emitted[:allowed + 1])
                    self.assertEqual(path[b, :allowed + 1].tolist(), walk[:allowed + 1])
                    self.assertEqual(count[b].item(), min(n + allowed + 1, prompt + output))
                    self.assertEqual(context[b, n:n + allowed + 1].tolist(), emitted[:allowed + 1])
                    torch.testing.assert_close(context[b, :n], before_context[b, :n], atol=0, rtol=0)
                    # Within a sequence the accepted path's rows win the table; another
                    # sequence's program may still overwrite a shared token afterwards.
                    elsewhere = [t for bb, row in enumerate(all_tokens) if bb != b for t in row]
                    on_path = [all_tokens[b][i] for i in walk[:allowed + 1]]
                    for i in walk[:allowed + 1]:
                        if all_tokens[b][i] not in elsewhere and on_path.count(all_tokens[b][i]) == 1:
                            torch.testing.assert_close(table[all_tokens[b][i]], topk[b * nodes + i].to(torch.int32),
                                                       atol=0, rtol=0)
                    seen = [t for row in all_tokens for t in row]
                    for j in range(nodes):
                        tok = all_tokens[b][j]
                        if seen.count(tok) == 1:
                            torch.testing.assert_close(table[tok], topk[b * nodes + j].to(torch.int32), atol=0, rtol=0)
                # Compaction moves accepted nodes' rows into path order for every layer.
                for layout in ('bhsd', 'bshd'):
                    k = cache_storage(batch, capacity, 'cuda', layout, layers)
                    v = cache_storage(batch, capacity, 'cuda', layout, layers)
                    k.copy_(self.rand(*k.shape)); v.copy_(self.rand(*v.shape))
                    ek, ev = k.clone(), v.clone()
                    for b in range(batch):
                        for i in range(1, accepted[b].item() + 1):
                            ek[:, b, :, base[b] + i] = k[:, b, :, base[b] + path[b, i]]
                            ev[:, b, :, base[b] + i] = v[:, b, :, base[b] + path[b, i]]
                    compact_out(k, v, base, path, accepted, tree)
                    torch.testing.assert_close(k, ek, atol=0, rtol=0)
                    torch.testing.assert_close(v, ev, atol=0, rtol=0)

    @torch.inference_mode() if GPU else (lambda f: f)
    def test_layer_all_switch_combinations(self):
        from kernels.weights import LayerWeights
        from kernels.tree_decode import TreeDecodeBuffers as DecodeBuffers, cache_storage
        from kernels.tree import DraftTree
        from transformers import StaticCache
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        native = Qwen3DecoderLayer(self.config, 0).cuda().bfloat16().eval()
        # DecoderLayer alone does not run PreTrainedModel's Qwen initialization.
        for name, p in native.named_parameters():
            if 'norm' not in name:
                p.normal_(0, .02)
        w = LayerWeights(native)
        tree = DraftTree(1, 8, 'cuda')
        for batch in (1, 4, 16):
            capacity, pos = 640, 512
            x = self.rand(batch, 1, 2560)
            prefix_k, prefix_v = [self.rand(batch, 8, pos, 128) for _ in range(2)]
            position = torch.tensor([pos], device='cuda')
            base = torch.full((batch,), pos, device='cuda', dtype=torch.int64)
            mask = torch.arange(capacity, device='cuda')[None, None, None] <= position
            cache = StaticCache(self.config, max_batch_size=batch, max_cache_len=capacity,
                                device='cuda', dtype=torch.bfloat16)
            cache.key_cache[0][:, :, :pos].copy_(prefix_k)
            cache.value_cache[0][:, :, :pos].copy_(prefix_v)
            ref = native(x, attention_mask=mask, position_ids=position[None], past_key_value=cache,
                         cache_position=position, use_cache=True,
                         position_embeddings=(self.cos[None, pos:pos+1], self.sin[None, pos:pos+1]))[0]
            for nr, qk, sm, layout in itertools.product((False, True), (False, True), (False, True),
                                                        ('bhsd', 'bshd')):
                config = dict(fuse_norm_residual=nr, fuse_qk_norm_rope=qk, fuse_silu_mul=sm, kv_layout=layout)
                buf = DecodeBuffers(batch, capacity, 'cuda', config, tree)
                k, v = [cache_storage(batch, capacity, 'cuda', layout)[0] for _ in range(2)]
                k[:, :, :pos].copy_(prefix_k); v[:, :, :pos].copy_(prefix_v)
                buf.x.copy_(x[:, 0])
                buf.layer(w, k, v, base, self.cos, self.sin)
                self.report(f'layer/{batch}/{config}', buf.x, ref[:, 0], .125, .03)


if __name__ == '__main__':
    unittest.main()
