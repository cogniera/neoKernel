"""Causal native prefill and a hand-rolled Qwen3 decode graph that verifies a draft tree.

Each decode step runs the target once over batch * T rows: the last committed
token of every sequence (the root) plus T-1 draft nodes proposed from the
target's own recent top-k predictions. The accepted root-to-leaf path is the
greedy continuation the target computed itself, so output is exact by
construction; drafts only decide how many positions one step can advance.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers import AutoModelForCausalLM
from kernels import CONFIG, TUNABLES


# Configuration is fixed for the lifetime of an imported engine.
DECODE_CONFIG = dict(CONFIG)


class PrefixCache:
    """Per-layer [B,8,capacity,128] views of one [L,B,8,capacity,128] allocation per K and V."""

    prefill_length = 0

    def __init__(self, key_storage, value_storage):
        self.key_storage, self.value_storage = key_storage, value_storage
        self.key_cache = [key_storage[i] for i in range(key_storage.shape[0])]
        self.value_cache = [value_storage[i] for i in range(value_storage.shape[0])]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # generate() prefills the entire prompt starting at position zero.
        # Populate the fixed decode cache without indexed writes, and let
        # prefill attention consume the original native K/V tensors.
        self.key_cache[layer_idx][:, :, :self.prefill_length, :].copy_(key_states)
        self.value_cache[layer_idx][:, :, :self.prefill_length, :].copy_(value_states)
        return key_states, value_states


@torch.inference_mode()
def qwen_forward(model, input_ids, cache, cache_position, position_ids, attention_mask):
    base = model.model
    x = base.embed_tokens(input_ids)
    cos, sin = base.rotary_emb(x, position_ids)
    batch, length = input_ids.shape
    # Keep the native projection and BF16 rounding boundaries. Only attention
    # dispatch changes: FlashAttention consumes eight KV heads directly instead
    # of materializing repeat_kv's 32-head copies at every layer.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for layer_idx, layer in enumerate(base.layers):
            residual = x
            normalized = layer.input_layernorm(x)
            attn = layer.self_attn
            head_shape = (batch, length, -1, attn.head_dim)
            q = attn.q_norm(attn.q_proj(normalized).view(head_shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(normalized).view(head_shape)).transpose(1, 2)
            v = attn.v_proj(normalized).view(head_shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k, v = cache.update(k, v, layer_idx, {
                "sin": sin, "cos": cos, "cache_position": cache_position,
            })
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0,
                is_causal=attention_mask is None and length > 1,
                scale=attn.scaling, enable_gqa=True,
            )
            attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
            x = residual + attn.o_proj(attended)
            residual = x
            x = residual + layer.mlp(layer.post_attention_layernorm(x))
    # RMSNorm is independent across tokens; only the final logits are used.
    x = base.norm(x[:, -1:, :])
    return model.lm_head(x)


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
        # Lazy device imports allow the native-prefill CPU regression to run on
        # hosts without Triton. These execute during loading, never in a step.
        from kernels.decode import DecodeBuffers, PrefillBuffers, cache_storage
        from kernels.weights import LayerWeights
        from kernels.rmsnorm import norm_out
        from kernels.tree import DraftTree
        from kernels.speculative import draft_out, accept_out, compact_out, warm_out
        self.warm_out = warm_out
        self.buffer_type = DecodeBuffers
        self.prefill_type = PrefillBuffers
        self.cache_storage = cache_storage
        self.tree_type = DraftTree
        self.norm_out = norm_out
        self.draft_out, self.accept_out, self.compact_out = draft_out, accept_out, compact_out
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True,
        ).eval().to("cuda:0")
        self.model.requires_grad_(False)
        with torch.inference_mode():
            self.layers = [LayerWeights(layer) for layer in self.model.model.layers]
        self.embedding = self.model.model.embed_tokens.weight
        self.lm_head = self.model.lm_head.weight.t()
        self.final_norm = self.model.model.norm.weight
        self.final_eps = self.model.model.norm.variance_epsilon
        vocab = self.model.config.vocab_size
        self.ranks = DECODE_CONFIG["draft_ranks"]
        # Fresh table for every prompt: rank r of token t starts as t itself.
        self.identity = torch.arange(vocab, device="cuda:0", dtype=torch.int32)[:, None].expand(vocab, self.ranks).contiguous()
        self.table = torch.empty_like(self.identity)
        self.shape = None
        self.graph = None
        self.prefill_graph = None

    def prepare(self, batch, prompt_length, output_length):
        self.graph = None
        self.prefill_graph = None
        self.shape = (batch, prompt_length, output_length)
        self.tree = self.tree_type(draft_nodes(batch) if output_length > 1 else 1, self.ranks, "cuda:0")
        nodes, depth = self.tree.nodes, self.tree.max_depth
        self.steps = schedule_steps(batch, output_length)
        capacity = prompt_length + output_length + nodes
        layers = len(self.layers)
        self.key_storage = self.cache_storage(batch, capacity, "cuda:0", DECODE_CONFIG["kv_layout"], layers)
        self.value_storage = self.cache_storage(batch, capacity, "cuda:0", DECODE_CONFIG["kv_layout"], layers)
        self.cache = PrefixCache(self.key_storage, self.value_storage)
        self.prompt_ids = torch.empty((batch, prompt_length), device="cuda:0", dtype=torch.int64)
        self.prefill_positions = torch.arange(prompt_length, device="cuda:0")
        self.token_ids = torch.zeros((batch, 1), device="cuda:0", dtype=torch.int64)
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
        self.buffers = self.buffer_type(batch, capacity, "cuda:0", DECODE_CONFIG, self.tree)
        rows = self.buffers.rows
        self.logits = torch.empty((rows, self.model.config.vocab_size), device="cuda:0", dtype=torch.bfloat16)
        self.top_values = torch.empty((rows, self.ranks), device="cuda:0", dtype=torch.bfloat16)
        self.top_indices = torch.empty((rows, self.ranks), device="cuda:0", dtype=torch.int64)
        positions = torch.arange(capacity, device="cuda:0").unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(self.buffers.x, positions)
        self.cos = cos[0].contiguous()
        self.sin = sin[0].contiguous()
        self.prefill_buffers = None
        self.warm_window = 0
        if DECODE_CONFIG["prefill_impl"] == "fused":
            self.prefill_buffers = self.prefill_type(batch, prompt_length, "cuda:0", DECODE_CONFIG)
            self.prefill_base = torch.zeros((batch,), device="cuda:0", dtype=torch.int64)
            self.last_rows = torch.arange(batch, device="cuda:0") * prompt_length + prompt_length - 1
            self.last_x = torch.empty((batch, 2560), device="cuda:0", dtype=torch.bfloat16)
            self.last_branch = torch.empty((batch, 2560), device="cuda:0", dtype=torch.bfloat16)
            self.last_norm = torch.empty((batch, 2560), device="cuda:0", dtype=torch.bfloat16)
            self.first_logits = torch.empty((batch, self.model.config.vocab_size),
                                            device="cuda:0", dtype=torch.bfloat16)
            # Table warming: the target's top-k at the last W prompt positions of every sequence.
            window = min(prompt_length, DECODE_CONFIG["warm_window"]) if output_length > 1 else 0
            self.warm_window = window
            if window:
                total = batch * window
                self.warm_chunk = min(total, 2048)
                self.warm_rows = (torch.arange(batch, device="cuda:0")[:, None] * prompt_length
                                  + prompt_length - window + torch.arange(window, device="cuda:0")[None, :]).reshape(-1)
                self.warm_x = torch.empty((self.warm_chunk, 2560), device="cuda:0", dtype=torch.bfloat16)
                self.warm_branch = torch.empty_like(self.warm_x)
                self.warm_norm = torch.empty_like(self.warm_x)
                self.warm_logits = torch.empty((self.warm_chunk, self.model.config.vocab_size),
                                               device="cuda:0", dtype=torch.bfloat16)
                self.warm_values = torch.empty((self.warm_chunk, self.ranks), device="cuda:0", dtype=torch.bfloat16)
                self.warm_indices = torch.empty((total, self.ranks), device="cuda:0", dtype=torch.int64)

    def decode(self):
        """One verification step: draft, target forward over all nodes, accept, compact."""
        batch, prompt_length, output_length = self.shape
        self.draft_out(self.context, self.count, self.table, self.tree, self.tokens, self.base,
                       DECODE_CONFIG["ngram"])
        torch.index_select(self.embedding, 0, self.tokens.view(-1), out=self.buffers.x)
        fuse = DECODE_CONFIG["fuse_norm_residual"]
        for i, weights in enumerate(self.layers):
            self.buffers.layer(weights, self.cache.key_cache[i], self.cache.value_cache[i],
                               self.base, self.cos, self.sin, carry_in=fuse and i > 0, defer_out=fuse)
        if fuse:
            # The last down projection is still pending in branch; the final
            # norm consumes it as its residual and stores the rounded sum in x.
            self.norm_out(self.buffers.branch, self.final_norm, self.buffers.norm,
                          self.final_eps, self.buffers.x, self.buffers.x)
        else:
            self.norm_out(self.buffers.x, self.final_norm, self.buffers.norm, self.final_eps)
        torch.mm(self.buffers.norm, self.lm_head, out=self.logits)
        # Rank 0 is the greedy token; on an exact BF16 tie any tied token is greedy.
        torch.topk(self.logits, self.ranks, dim=-1, out=(self.top_values, self.top_indices))
        self.accept_out(self.tokens, self.top_indices, self.tree, self.context, self.count,
                        self.table, self.step, self.packet, self.path, self.accepted,
                        prompt_length, output_length, self.steps)
        self.compact_out(self.key_storage, self.value_storage, self.base, self.path,
                         self.accepted, self.tree)
        self.step.add_(1)

    def capture_decode(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.decode()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.decode()
        torch.cuda.current_stream().wait_stream(stream)

    def fused_prefill(self):
        """The decode layer over every prompt row; final norm and logits for the last row only."""
        pb = self.prefill_buffers
        torch.index_select(self.embedding, 0, self.prompt_ids.view(-1), out=pb.x)
        fuse = DECODE_CONFIG["fuse_norm_residual"]
        for i, weights in enumerate(self.layers):
            pb.layer(weights, self.cache.key_cache[i], self.cache.value_cache[i],
                     self.prefill_base, self.cos, self.sin, carry_in=fuse and i > 0, defer_out=fuse)
        torch.index_select(pb.x, 0, self.last_rows, out=self.last_x)
        if fuse:
            torch.index_select(pb.branch, 0, self.last_rows, out=self.last_branch)
            self.norm_out(self.last_branch, self.final_norm, self.last_norm, self.final_eps,
                          self.last_x, self.last_x)
        else:
            self.norm_out(self.last_x, self.final_norm, self.last_norm, self.final_eps)
        torch.mm(self.last_norm, self.lm_head, out=self.first_logits)
        if self.warm_window:
            total, chunk = self.warm_rows.shape[0], self.warm_chunk
            for start in range(0, total, chunk):
                size = min(chunk, total - start)
                rows = self.warm_rows[start:start + size]
                x, norm = self.warm_x[:size], self.warm_norm[:size]
                torch.index_select(pb.x, 0, rows, out=x)
                if fuse:
                    branch = self.warm_branch[:size]
                    torch.index_select(pb.branch, 0, rows, out=branch)
                    self.norm_out(branch, self.final_norm, norm, self.final_eps, x, x)
                else:
                    self.norm_out(x, self.final_norm, norm, self.final_eps)
                logits = self.warm_logits[:size]
                torch.mm(norm, self.lm_head, out=logits)
                torch.topk(logits, self.ranks, dim=-1,
                           out=(self.warm_values[:size], self.warm_indices[start:start + size]))
        return self.first_logits

    def prefill(self):
        batch, prompt_length, output_length = self.shape
        if self.prefill_buffers is not None:
            logits = self.fused_prefill()
        else:
            logits = qwen_forward(self.model, self.prompt_ids, self.cache, self.prefill_positions,
                                  self.prefill_positions.unsqueeze(0), None)[:, -1, :]
        self.token_ids.copy_(logits.argmax(dim=-1, keepdim=True))
        # Commit prompt and first token; seed the draft table with prompt bigrams.
        self.context[:, :prompt_length].copy_(self.prompt_ids)
        self.context[:, prompt_length:prompt_length + 1].copy_(self.token_ids)
        self.count.fill_(prompt_length + 1)
        self.step.fill_(1)
        self.table.copy_(self.identity)
        if prompt_length > 1:
            self.table[:, 0].scatter_(0, self.prompt_ids[:, :-1].reshape(-1),
                                      self.prompt_ids[:, 1:].reshape(-1).to(torch.int32))
        if self.prefill_buffers is not None and self.warm_window:
            self.warm_out(self.prompt_ids[:, prompt_length - self.warm_window:], self.warm_indices, self.table)

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

    def generate(self, input_ids, max_new_tokens):
        if max_new_tokens <= 0:
            return
        batch, prompt_length = len(input_ids), len(input_ids[0])
        with torch.inference_mode():
            if self.shape != (batch, prompt_length, max_new_tokens):
                self.prepare(batch, prompt_length, max_new_tokens)
            self.prompt_ids.copy_(torch.tensor(input_ids, dtype=torch.int64, device="cuda:0"))
            self.cache.prefill_length = prompt_length
            if self.prefill_graph is None:
                self.capture_prefill()
            self.prefill_graph.replay()
            if max_new_tokens > 1 and self.graph is None:
                # Warmup runs real steps and advances the state: prefill again.
                self.capture_decode()
                self.prefill_graph.replay()
            self.cache.prefill_length = 0
            self.fetch(2, self.token_ids, self.host_first)
            if max_new_tokens == 1:
                self.copy_streams[2].synchronize()
                yield self.host_first[:, 0].tolist()
                return
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
