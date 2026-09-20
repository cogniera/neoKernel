"""Causal native prefill and a hand-rolled, fixed-buffer Qwen3 decode graph."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
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
def qwen_forward(model, input_ids, cache, cache_position, position_ids, attention_mask):
    from kernels.prefill import norm, silu_mul

    base = model.model
    x = base.embed_tokens(input_ids)
    cos, sin = base.rotary_emb(x, position_ids)
    batch, length = input_ids.shape
    # Keep the native projections and BF16 rounding boundaries.
    # FlashAttention consumes eight KV heads directly instead
    # of materializing repeat_kv's 32-head copies at every layer.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for layer_idx, layer in enumerate(base.layers):
            residual = x
            normalized = norm(x, layer.input_layernorm)
            attn = layer.self_attn
            head_shape = (batch, length, -1, attn.head_dim)
            q = norm(attn.q_proj(normalized).view(head_shape), attn.q_norm).transpose(1, 2)
            k = norm(attn.k_proj(normalized).view(head_shape), attn.k_norm).transpose(1, 2)
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
            normalized = norm(x, layer.post_attention_layernorm)
            mlp = layer.mlp
            product = silu_mul(mlp.gate_proj(normalized), mlp.up_proj(normalized))
            x = residual + mlp.down_proj(product)
    # RMSNorm is independent across tokens; only the final logits are used.
    x = norm(x[:, -1:, :], base.norm)
    return model.lm_head(x)


class Engine:
    def __init__(self, model_path):
        # Lazy device imports allow the native-prefill CPU regression to run on
        # hosts without Triton. These execute during loading, never in a step.
        from kernels.decode import DecodeBuffers
        from kernels.weights import LayerWeights
        from kernels.rmsnorm import norm_out
        self.buffer_type = DecodeBuffers
        self.norm_out = norm_out
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
        self.prompt_ids = torch.empty((batch, prompt_length), device="cuda:0", dtype=torch.int64)
        self.shape = (batch, prompt_length, output_length)
        capacity = prompt_length + output_length
        self.cache = PrefixStaticCache(
            self.model.config, max_batch_size=batch, max_cache_len=capacity,
            device="cuda:0", dtype=torch.bfloat16, layout=DECODE_CONFIG["kv_layout"],
        )
        self.prefill_positions = torch.arange(prompt_length, device="cuda:0")
        self.decode_mask = torch.zeros((batch, 1, 1, capacity), device="cuda:0", dtype=torch.bool)
        self.token_ids = torch.zeros((batch, 1), device="cuda:0", dtype=torch.int64)
        self.cache_position = torch.zeros((1,), device="cuda:0", dtype=torch.int64)
        self.position_ids = self.cache_position.view(1, 1)
        self.flat_token_ids = self.token_ids.view(batch)
        self.next_tokens = torch.empty_like(self.token_ids)
        # Two pinned landing buffers and two streams: step t+1 is queued on the
        # GPU before the host waits for token t, so the yield never idles the GPU.
        self.host_tokens = [torch.empty((batch, 1), dtype=torch.int64, pin_memory=True) for _ in range(2)]
        self.copy_streams = [torch.cuda.Stream() for _ in range(2)]
        self.buffers = self.buffer_type(batch, capacity, "cuda:0", DECODE_CONFIG)
        self.logits = torch.empty((batch, self.model.config.vocab_size),
                                  device="cuda:0", dtype=torch.bfloat16)
        positions = torch.arange(capacity, device="cuda:0").unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(self.buffers.x, positions)
        self.cos = cos[0].contiguous()
        self.sin = sin[0].contiguous()

    def decode(self):
        if DECODE_CONFIG["attention_impl"] == "sdpa_grouped":
            self.decode_mask.index_fill_(3, self.cache_position, True)
        torch.index_select(self.embedding, 0, self.flat_token_ids, out=self.buffers.x)
        fuse = DECODE_CONFIG["fuse_norm_residual"]
        for i, weights in enumerate(self.layers):
            self.buffers.layer(weights, self.cache.key_cache[i], self.cache.value_cache[i],
                               self.cache_position, self.cos, self.sin, self.decode_mask,
                               carry_in=fuse and i > 0, defer_out=fuse)
        if fuse:
            # The last down projection is still pending in branch; the final
            # norm consumes it as its residual and stores the rounded sum in x.
            self.norm_out(self.buffers.branch, self.final_norm, self.buffers.norm,
                          self.final_eps, self.buffers.x, self.buffers.x)
        else:
            self.norm_out(self.buffers.x, self.final_norm, self.buffers.norm, self.final_eps)
        torch.mm(self.buffers.norm, self.lm_head, out=self.logits)
        # torch.argmax chooses the lowest vocabulary index on exact ties.
        torch.argmax(self.logits, dim=-1, keepdim=True, out=self.next_tokens)
        return self.next_tokens

    def capture_decode(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.decode()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.next_tokens = self.decode()
            # One replay is a complete step: it consumes token_ids at the
            # current position and leaves the next token and position in place.
            self.token_ids.copy_(self.next_tokens)
            self.cache_position.add_(1)
        torch.cuda.current_stream().wait_stream(stream)

    def prefill(self):
        logits = qwen_forward(self.model, self.prompt_ids, self.cache, self.prefill_positions,
                              self.prefill_positions.unsqueeze(0), None)
        self.token_ids.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))

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

    def fetch(self, slot):
        """Queue an async copy of the current token_ids; the next step waits for it on the device."""
        stream = self.copy_streams[slot]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.host_tokens[slot].copy_(self.token_ids, non_blocking=True)
        torch.cuda.current_stream().wait_stream(stream)

    def emit(self, slot):
        self.copy_streams[slot].synchronize()
        return self.host_tokens[slot][:, 0].tolist()

    def generate(self, input_ids, max_new_tokens):
        if max_new_tokens <= 0:
            return
        batch, prompt_length = len(input_ids), len(input_ids[0])
        with torch.inference_mode():
            if self.shape != (batch, prompt_length, max_new_tokens):
                self.prepare(batch, prompt_length, max_new_tokens)
            self.decode_mask.zero_()
            self.decode_mask[:, :, :, :prompt_length].fill_(True)
            self.cache_position.fill_(prompt_length)
            self.prompt_ids.copy_(torch.tensor(input_ids, dtype=torch.int64, device="cuda:0"))
            self.cache.prefill_length = prompt_length
            if self.prefill_graph is None:
                self.capture_prefill()
            self.prefill_graph.replay()
            self.cache.prefill_length = 0
            if max_new_tokens > 1 and self.graph is None:
                self.capture_decode()
            self.fetch(0)
            for step in range(1, max_new_tokens):
                self.graph.replay()
                self.fetch(step % 2)
                yield self.emit((step - 1) % 2)
            yield self.emit((max_new_tokens - 1) % 2)
