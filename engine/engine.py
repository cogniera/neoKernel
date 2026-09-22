"""Fused causal prefill, then decode that verifies a draft tree per sequence.

Every decode step runs the target once over batch * T rows: the last committed
token of each sequence (the root) plus T-1 draft nodes proposed from the
target's own recent top-k predictions. The accepted root-to-leaf path is the
greedy continuation the target computed itself, so output is exact by
construction; drafts only decide how far one step advances. A one-node tree
is ordinary single-token decode.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM
from kernels import CONFIG
from kernels.prefill import norm, add_norm, qkv_cache, packed_silu_mul
from kernels.rmsnorm import norm_out
from kernels.speculative import draft_out, accept_out, compact_out, warm_out
from kernels.tree import DraftTree
from kernels.tree_decode import TreeDecodeBuffers, cache_storage
from kernels.weights import LayerWeights


DECODE_CONFIG = dict(CONFIG)


class PrefixCache:
    """Per-layer [B,8,capacity,128] views of one [L,B,8,capacity,128] allocation per K and V."""

    def __init__(self, key_storage, value_storage):
        self.key_storage, self.value_storage = key_storage, value_storage
        self.key_cache = [key_storage[i] for i in range(key_storage.shape[0])]
        self.value_cache = [value_storage[i] for i in range(value_storage.shape[0])]


@torch.inference_mode()
def qwen_forward(model, input_ids, cache, packed_layers, keep_rows=1):
    """Prefill; returns the final-normalized hidden rows the caller asked to keep (last ones)."""
    base = model.model
    batch, length = input_ids.shape
    x = base.embed_tokens(input_ids)
    positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
    cos, sin = base.rotary_emb(x, positions)
    normalized = norm(x, base.layers[0].input_layernorm)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for layer_idx, layer in enumerate(base.layers):
            weights = packed_layers[layer_idx]
            packed = F.linear(normalized, weights.qkv_weight)
            q, k, v = qkv_cache(
                packed, layer.self_attn, cos, sin,
                cache.key_cache[layer_idx], cache.value_cache[layer_idx],
            )
            last = layer_idx == len(base.layers) - 1
            trim = last and keep_rows == 1
            if trim:
                # All prompt K/V are already cached. Earlier query outputs in
                # the final layer have no consumers; only the last predicts a token.
                q = q[:, :, -1:, :]
                x = x[:, -1:, :]
            attended = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=not trim and length > 1,
                scale=layer.self_attn.scaling, enable_gqa=True,
            )
            attended = attended.transpose(1, 2).contiguous().view(batch, 1 if trim else length, -1)
            branch = layer.self_attn.o_proj(attended)
            normalized, residual = add_norm(branch, x, layer.post_attention_layernorm)
            gate_up = F.linear(normalized, weights.gate_up_weight)
            branch = layer.mlp.down_proj(packed_silu_mul(gate_up))
            if last:
                normalized, _ = add_norm(branch, residual, base.norm)
            else:
                normalized, x = add_norm(branch, residual, base.layers[layer_idx + 1].input_layernorm)
    return normalized[:, -keep_rows:, :]


def by_batch(table, batch, default):
    """Value for the largest configured batch that does not exceed this one."""
    keys = [key for key in table if key <= batch]
    return table[max(keys)] if keys else default


def draft_nodes(batch):
    return by_batch(DECODE_CONFIG["draft_nodes"], batch, 1)


def schedule_steps(batch, output_length):
    """Planned decode steps; 0 leaves acceptance unpaced. Pacing bounds sample spread."""
    target = by_batch(DECODE_CONFIG["pace"], batch, 0.0)
    if target <= 1.0 or output_length < 2:
        return 0
    return max(1, round((output_length - 1) / target))


class Engine:
    def __init__(self, model_path):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True,
        ).eval().to("cuda:0")
        with torch.inference_mode():
            self.layers = [LayerWeights(layer) for layer in self.model.model.layers]
        self.embedding = self.model.model.embed_tokens.weight
        self.lm_head = self.model.lm_head.weight.t()
        self.final_norm = self.model.model.norm.weight
        self.final_eps = self.model.model.norm.variance_epsilon
        vocab = self.model.config.vocab_size
        self.ranks = DECODE_CONFIG["draft_ranks"]
        # Fresh draft table for every prompt: rank r of token t starts as t itself.
        self.identity = torch.arange(vocab, device="cuda:0", dtype=torch.int32)[:, None].expand(vocab, self.ranks).contiguous()
        self.table = torch.empty_like(self.identity)
        self.shape = None
        self.graph = None
        self.prefill_graph = None

    def prepare(self, batch, prompt_length, output_length):
        self.graph = None
        self.prefill_graph = None
        self.shape = (batch, prompt_length, output_length)
        vocab = self.model.config.vocab_size
        self.tree = DraftTree(draft_nodes(batch) if output_length > 1 else 1, self.ranks, "cuda:0")
        nodes, depth = self.tree.nodes, self.tree.max_depth
        self.steps = schedule_steps(batch, output_length)
        capacity = prompt_length + output_length + nodes
        layers = len(self.layers)
        self.key_storage = cache_storage(batch, capacity, "cuda:0", DECODE_CONFIG["kv_layout"], layers)
        self.value_storage = cache_storage(batch, capacity, "cuda:0", DECODE_CONFIG["kv_layout"], layers)
        self.cache = PrefixCache(self.key_storage, self.value_storage)
        self.prompt_ids = torch.empty((batch, prompt_length), device="cuda:0", dtype=torch.int64)
        self.host_prompt = torch.empty((batch, prompt_length), dtype=torch.int64, pin_memory=True)
        self.token_ids = torch.zeros((batch, 1), device="cuda:0", dtype=torch.int64)
        # Table warming: the target's top-k at the last W prompt positions of every sequence.
        self.warm_window = min(prompt_length, DECODE_CONFIG["warm_window"]) if nodes > 1 else 0
        self.first_logits = torch.empty((batch, vocab), device="cuda:0", dtype=torch.bfloat16)
        if self.warm_window:
            total = batch * self.warm_window
            self.warm_chunk = min(total, 2048)
            self.warm_logits = torch.empty((self.warm_chunk, vocab), device="cuda:0", dtype=torch.bfloat16)
            self.warm_values = torch.empty((self.warm_chunk, self.ranks), device="cuda:0", dtype=torch.bfloat16)
            self.warm_indices = torch.empty((total, self.ranks), device="cuda:0", dtype=torch.int64)
        self.context = torch.zeros((batch, prompt_length + output_length + depth + 1),
                                   device="cuda:0", dtype=torch.int64)
        self.count = torch.zeros((batch,), device="cuda:0", dtype=torch.int64)
        self.base = torch.zeros((batch,), device="cuda:0", dtype=torch.int64)
        self.step = torch.zeros((1,), device="cuda:0", dtype=torch.int64)
        self.tokens = torch.zeros((batch, nodes), device="cuda:0", dtype=torch.int64)
        self.packet = torch.zeros((batch, depth + 3), device="cuda:0", dtype=torch.int64)
        self.path = torch.zeros((batch, depth + 1), device="cuda:0", dtype=torch.int32)
        self.accepted = torch.zeros((batch,), device="cuda:0", dtype=torch.int32)
        # Two pinned landing buffers and two streams: step t+1 is queued on the
        # GPU before the host waits for step t, so the yield never idles the GPU.
        self.host_first = torch.empty((batch, 1), dtype=torch.int64, pin_memory=True)
        self.host_packets = [torch.empty(self.packet.shape, dtype=torch.int64, pin_memory=True) for _ in range(2)]
        self.copy_streams = [torch.cuda.Stream() for _ in range(3)]
        self.buffers = TreeDecodeBuffers(batch, capacity, "cuda:0", DECODE_CONFIG, self.tree)
        rows = self.buffers.rows
        self.logits = torch.empty((rows, vocab), device="cuda:0", dtype=torch.bfloat16)
        self.top_values = torch.empty((rows, self.ranks), device="cuda:0", dtype=torch.bfloat16)
        self.top_indices = torch.empty((rows, self.ranks), device="cuda:0", dtype=torch.int64)
        positions = torch.arange(capacity, device="cuda:0").unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(self.buffers.x, positions)
        self.cos = cos[0].contiguous()
        self.sin = sin[0].contiguous()

    def tree_decode(self):
        """One verification step: draft, target forward over all nodes, accept, compact."""
        prompt_length, output_length = self.shape[1:]
        draft_out(self.context, self.count, self.table, self.tree, self.tokens, self.base,
                  DECODE_CONFIG["ngram"])
        torch.index_select(self.embedding, 0, self.tokens.view(-1), out=self.buffers.x)
        fuse = DECODE_CONFIG["fuse_norm_residual"]
        for i, weights in enumerate(self.layers):
            self.buffers.layer(weights, self.cache.key_cache[i], self.cache.value_cache[i],
                               self.base, self.cos, self.sin, carry_in=fuse and i > 0, defer_out=fuse)
        if fuse:
            # The last down projection is still pending in branch; the final
            # norm consumes it as its residual and stores the rounded sum in x.
            norm_out(self.buffers.branch, self.final_norm, self.buffers.norm,
                     self.final_eps, self.buffers.x, self.buffers.x)
        else:
            norm_out(self.buffers.x, self.final_norm, self.buffers.norm, self.final_eps)
        torch.mm(self.buffers.norm, self.lm_head, out=self.logits)
        # Rank 0 is the greedy token; on an exact BF16 tie any tied token is greedy.
        torch.topk(self.logits, self.ranks, dim=-1, out=(self.top_values, self.top_indices))
        accept_out(self.tokens, self.top_indices, self.tree, self.context, self.count,
                   self.table, self.step, self.packet, self.path, self.accepted,
                   prompt_length, output_length, self.steps)
        compact_out(self.key_storage, self.value_storage, self.base, self.path,
                    self.accepted, self.tree)
        self.step.add_(1)

    def capture_tree(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.tree_decode()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.tree_decode()
        torch.cuda.current_stream().wait_stream(stream)

    def reset_tree(self):
        """Commit prompt and first token; seed the draft table with prompt bigrams, then warm it."""
        prompt_length = self.shape[1]
        self.context[:, :prompt_length].copy_(self.prompt_ids)
        self.context[:, prompt_length:prompt_length + 1].copy_(self.token_ids)
        self.count.fill_(prompt_length + 1)
        self.step.fill_(1)
        self.table.copy_(self.identity)
        if prompt_length > 1:
            self.table[:, 0].scatter_(0, self.prompt_ids[:, :-1].reshape(-1),
                                      self.prompt_ids[:, 1:].reshape(-1).to(torch.int32))
        if self.warm_window:
            warm_out(self.prompt_ids[:, prompt_length - self.warm_window:], self.warm_indices, self.table)

    def fetch(self, slot, source, target):
        """Queue an async copy of a device tensor; the next step waits for it on the device."""
        stream = self.copy_streams[slot]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            target.copy_(source, non_blocking=True)
        torch.cuda.current_stream().wait_stream(stream)

    def launch(self, slot):
        self.graph.replay()
        self.fetch(slot, self.packet, self.host_packets[slot])

    def generate_tree(self, max_new_tokens):
        if self.graph is None:
            # Warmup runs real steps and advances the state: rebuild it from the prefill.
            self.reset_tree()
            self.capture_tree()
        self.reset_tree()
        self.fetch(2, self.token_ids, self.host_first)
        self.launch(0)
        self.copy_streams[2].synchronize()
        queues = [[t] for t in self.host_first[:, 0].tolist()]
        yield [queue[0] for queue in queues]
        depth = self.tree.max_depth
        inflight, steps = 0, 1
        emitted = 1
        while emitted < max_new_tokens:
            if all(len(queue) > emitted for queue in queues):
                yield [queue[emitted] for queue in queues]
                emitted += 1
                continue
            if inflight is None:
                inflight = steps % 2
                self.launch(inflight)
                steps += 1
            shortest = min(len(queue) for queue in queues)
            # Queue the next step before waiting unless this one can finish every sequence.
            queue_next = shortest + depth + 1 < max_new_tokens
            if queue_next:
                self.launch(1 - inflight)
                steps += 1
            self.copy_streams[inflight].synchronize()
            for queue, row in zip(queues, self.host_packets[inflight].tolist()):
                take = min(row[depth + 1] + 1, max_new_tokens - len(queue))
                queue.extend(row[:take])
            inflight = 1 - inflight if queue_next else None
        self.rounds = steps

    def prefill(self):
        batch = self.shape[0]
        rows = max(1, self.warm_window)
        normalized = qwen_forward(self.model, self.prompt_ids, self.cache, self.layers, rows)
        torch.mm(normalized[:, -1, :], self.lm_head, out=self.first_logits)
        torch.argmax(self.first_logits, dim=-1, keepdim=True, out=self.token_ids)
        if self.warm_window:
            flat = normalized.reshape(batch * self.warm_window, -1)
            total, chunk = flat.shape[0], self.warm_chunk
            for start in range(0, total, chunk):
                size = min(chunk, total - start)
                logits = self.warm_logits[:size]
                torch.mm(flat[start:start + size], self.lm_head, out=logits)
                torch.topk(logits, self.ranks, dim=-1,
                           out=(self.warm_values[:size], self.warm_indices[start:start + size]))

    def capture_prefill(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.prefill()
        torch.cuda.current_stream().wait_stream(stream)
        self.prefill_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.prefill_graph, stream=stream):
            self.prefill()
        torch.cuda.current_stream().wait_stream(stream)

    def generate(self, input_ids, max_new_tokens):
        batch, prompt_length = len(input_ids), len(input_ids[0])
        with torch.inference_mode():
            if self.shape != (batch, prompt_length, max_new_tokens):
                self.prepare(batch, prompt_length, max_new_tokens)
            self.host_prompt.copy_(torch.tensor(input_ids, dtype=torch.int64))
            self.prompt_ids.copy_(self.host_prompt, non_blocking=True)
            if self.prefill_graph is None:
                self.capture_prefill()
            self.prefill_graph.replay()
            yield from self.generate_tree(max_new_tokens)
