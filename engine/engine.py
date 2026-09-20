"""Causal native prefill and a hand-rolled, fixed-buffer Qwen3 decode graph."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, StaticCache
from kernels import CONFIG, TUNABLES


# Configuration is fixed for the lifetime of an imported engine.
DECODE_CONFIG = dict(CONFIG)


class PrefixStaticCache(StaticCache):
    prefill_length = 0

    def __init__(self, *args, layout="bhsd", **kwargs):
        super().__init__(*args, **kwargs)
        if layout == "bshd":
            for i in range(len(self.key_cache)):
                self.key_cache[i] = self.key_cache[i].transpose(1, 2).contiguous().transpose(1, 2)
                self.value_cache[i] = self.value_cache[i].transpose(1, 2).contiguous().transpose(1, 2)
        elif layout != "bhsd":
            raise ValueError("invalid kv_layout")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.prefill_length:
            # generate() prefills the entire prompt starting at position zero.
            # Populate the fixed decode cache without indexed writes, and let
            # prefill attention consume the original native K/V tensors.
            self.key_cache[layer_idx][:, :, :self.prefill_length, :].copy_(key_states)
            self.value_cache[layer_idx][:, :, :self.prefill_length, :].copy_(value_states)
            return key_states, value_states
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


@torch.inference_mode()
def qwen_forward(model, input_ids, cache, packed_layers):
    from kernels.prefill import norm, add_norm, qkv_cache, packed_silu_mul

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
            if last:
                # All prompt K/V are already cached. Earlier query outputs in
                # the final layer have no consumers; only the last predicts a token.
                q = q[:, :, -1:, :]
                x = x[:, -1:, :]
            attended = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=not last and length > 1,
                scale=layer.self_attn.scaling, enable_gqa=True,
            )
            attended = attended.transpose(1, 2).contiguous().view(batch, 1 if last else length, -1)
            branch = layer.self_attn.o_proj(attended)
            normalized, residual = add_norm(branch, x, layer.post_attention_layernorm)
            gate_up = F.linear(normalized, weights.gate_up_weight)
            branch = layer.mlp.down_proj(packed_silu_mul(gate_up))
            if last:
                normalized, _ = add_norm(branch, residual, base.norm)
            else:
                normalized, x = add_norm(branch, residual, base.layers[layer_idx + 1].input_layernorm)
    return model.lm_head(normalized)


class Engine:
    def __init__(self, model_path):
        from kernels.decode import DecodeBuffers
        from kernels.split_decode import SplitDecodeBuffers
        from kernels.weights import LayerWeights
        from kernels.head import Head, commit_step
        from kernels.rmsnorm import embedding_norm_out
        self.buffer_type = DecodeBuffers
        self.split_buffer_type = SplitDecodeBuffers
        self.head_type = Head
        self.commit_step = commit_step
        self.embedding_norm_out = embedding_norm_out
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
        self.shape = None
        self.graph = None

    def prepare(self, batch, prompt_length, output_length):
        self.graph = None
        self.prefill_graph = None
        self.shape = (batch, prompt_length, output_length)
        self.prompt_ids = torch.empty((batch, prompt_length), device="cuda:0", dtype=torch.int64)
        self.host_prompt = torch.empty((batch, prompt_length), dtype=torch.int64, pin_memory=True)
        capacity = prompt_length + output_length
        self.cache = PrefixStaticCache(
            self.model.config, max_batch_size=batch, max_cache_len=capacity,
            device="cuda:0", dtype=torch.bfloat16, layout=DECODE_CONFIG["kv_layout"],
        )
        self.decode_mask = torch.zeros((batch, 1, 1, capacity), device="cuda:0", dtype=torch.bool)
        self.token_ids = torch.zeros((batch, 1), device="cuda:0", dtype=torch.int64)
        self.flat_token_ids = self.token_ids.view(batch)
        self.next_tokens = torch.empty_like(self.token_ids)
        self.cache_position = torch.zeros((1,), device="cuda:0", dtype=torch.int64)
        # Each generated token has a stable landing row. Copies never race with
        # a later step's token_ids update, so compute need not wait for copies.
        self.output_tokens = torch.empty((output_length, batch), device="cuda:0", dtype=torch.int64)
        self.host_tokens = torch.empty((output_length, batch), dtype=torch.int64, pin_memory=True)
        self.copy_stream = torch.cuda.Stream()
        self.ready_events = [torch.cuda.Event() for _ in range(output_length)]
        self.done_events = [torch.cuda.Event() for _ in range(output_length)]
        self.buffer_options = {"cublas": self.buffer_type(batch, capacity, "cuda:0", DECODE_CONFIG)}
        if batch <= 32 and DECODE_CONFIG["fuse_norm_residual"]:
            self.buffer_options["splitk"] = self.split_buffer_type(batch, capacity, "cuda:0", DECODE_CONFIG)
        self.buffers = self.buffer_options["cublas"]
        self.fused_head = False
        self.head = self.head_type(batch, self.model.config.vocab_size, self.embedding.shape[1], "cuda:0")
        self.logits = torch.empty((batch, self.model.config.vocab_size), device="cuda:0", dtype=torch.bfloat16)
        positions = torch.arange(capacity, device="cuda:0").unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(self.buffers.x, positions)
        self.cos = cos[0].contiguous()
        self.sin = sin[0].contiguous()

    def decode(self):
        if DECODE_CONFIG["attention_impl"] == "sdpa_grouped":
            self.decode_mask.index_fill_(3, self.cache_position, True)
        self.embedding_norm_out(self.embedding, self.flat_token_ids, self.layers[0].input_norm,
                                self.buffers.x, self.buffers.norm, self.layers[0].eps)
        fuse = DECODE_CONFIG["fuse_norm_residual"]
        for i, weights in enumerate(self.layers):
            self.buffers.layer(weights, self.cache.key_cache[i], self.cache.value_cache[i],
                               self.cache_position, self.cos, self.sin, self.decode_mask,
                               carry_in=fuse and i > 0, defer_out=fuse, normalized_input=i == 0)
        self.buffers.finish_norm(self.final_norm, self.final_eps)
        if self.fused_head:
            self.head.out(self.buffers.norm, self.embedding, self.next_tokens)
        else:
            torch.mm(self.buffers.norm, self.lm_head, out=self.logits)
            torch.argmax(self.logits, dim=-1, keepdim=True, out=self.next_tokens)
        return self.next_tokens

    def reset_decode(self):
        # Only positions below prompt_length are valid after a reset. Prefill
        # overwrites those slots for every generate(), including after warmup.
        prompt_length = self.shape[1]
        self.cache_position.fill_(prompt_length)
        self.token_ids.copy_(self.output_tokens[0].view(-1, 1))
        if DECODE_CONFIG["attention_impl"] == "sdpa_grouped":
            self.decode_mask.zero_()
            self.decode_mask[:, :, :, :prompt_length].fill_(True)

    def capture_decode(self):
        """Choose an entire decode graph during untimed, shape-specific warmup.

        Both projection paths evaluate the full model. Attention always reads
        the full visible cache, and both heads evaluate the entire vocabulary.
        A variant must beat the current best by 3% to replace it. Selection is
        fixed for later samples; timing data and warmup tokens are not reused.
        """
        batch, prompt_length, output_length = self.shape
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        attention_modes = ["split128"]
        if DECODE_CONFIG["attention_impl"] == "triton":
            attention_modes.append("online" if batch >= 8 else "split256")
        heads = [False, True] if batch <= 32 else [False]
        timed_steps = min(3, output_length - 1)
        best = None
        with torch.cuda.stream(stream):
            for projection, buffers in self.buffer_options.items():
                self.buffers = buffers
                for attention in attention_modes:
                    buffers.attention_mode = attention
                    for fused_head in heads:
                        self.fused_head = fused_head
                        self.reset_decode()
                        for _ in range(2):
                            self.decode()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            self.decode()
                            self.commit_step(self.next_tokens, self.token_ids, self.output_tokens,
                                             self.cache_position, prompt_length)
                        elapsed = []
                        for _ in range(3):
                            self.reset_decode()
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record(stream)
                            for _ in range(timed_steps):
                                graph.replay()
                            end.record(stream)
                            end.synchronize()
                            elapsed.append(start.elapsed_time(end) / timed_steps)
                        median = sorted(elapsed)[1]
                        if best is None or median < best[0] * 0.97:
                            best = (median, graph, projection, attention, fused_head)
        torch.cuda.current_stream().wait_stream(stream)
        _, self.graph, projection, attention, self.fused_head = best
        self.buffers = self.buffer_options[projection]
        self.buffers.attention_mode = attention
        self.reset_decode()
        print("decode plan: " + projection + "/" + attention +
              ("/fused-head" if self.fused_head else "/cublas-head"), flush=True)

    def prefill(self):
        logits = qwen_forward(self.model, self.prompt_ids, self.cache, self.layers)
        self.token_ids.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.output_tokens[0].copy_(self.flat_token_ids)

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

    def fetch(self, step):
        self.ready_events[step].record(torch.cuda.current_stream())
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.ready_events[step])
            self.host_tokens[step].copy_(self.output_tokens[step], non_blocking=True)
            self.done_events[step].record(self.copy_stream)

    def emit(self, step):
        # Wait only for this token's copy, not later queued copies or compute.
        self.done_events[step].synchronize()
        return self.host_tokens[step].tolist()

    def generate(self, input_ids, max_new_tokens):
        if max_new_tokens <= 0:
            return
        batch, prompt_length = len(input_ids), len(input_ids[0])
        with torch.inference_mode():
            if self.shape != (batch, prompt_length, max_new_tokens):
                self.prepare(batch, prompt_length, max_new_tokens)
            self.host_prompt.copy_(torch.tensor(input_ids, dtype=torch.int64))
            self.prompt_ids.copy_(self.host_prompt, non_blocking=True)
            if self.prefill_graph is None:
                self.capture_prefill()
            self.prefill_graph.replay()
            self.reset_decode()
            if max_new_tokens > 1 and self.graph is None:
                self.capture_decode()
            self.fetch(0)
            queued = 0
            for step in range(max_new_tokens):
                # A bounded lookahead keeps the GPU busy while Python yields.
                # Every step has distinct device/host storage, including tails.
                while queued < min(max_new_tokens - 1, step + 2):
                    self.graph.replay()
                    queued += 1
                    self.fetch(queued)
                yield self.emit(step)
