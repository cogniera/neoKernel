"""Causal native prefill and a hand-rolled, fixed-buffer Qwen3 decode graph."""

import torch
from transformers import AutoModelForCausalLM, StaticCache
from kernels import CONFIG, TUNABLES, configure_for_shape


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
            self.key_cache[layer_idx][:, :, :self.prefill_length, :].copy_(key_states)
            self.value_cache[layer_idx][:, :, :self.prefill_length, :].copy_(value_states)
            return key_states, value_states
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


@torch.inference_mode()
def qwen_forward(model, input_ids, cache, cache_position, position_ids, attention_mask):
    base = model.model
    x = base.embed_tokens(input_ids)
    position_embeddings = base.rotary_emb(x, position_ids)
    for layer in base.layers:
        x = layer(
            x, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=cache, use_cache=True, cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
    x = base.norm(x[:, -1:, :])
    return model.lm_head(x)


class Engine:
    def __init__(self, model_path):
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
        configure_for_shape(batch, prompt_length, output_length)
        DECODE_CONFIG['attention_impl'] = CONFIG['attention_impl']
        DECODE_CONFIG['kv_layout'] = CONFIG['kv_layout']
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
        for i, weights in enumerate(self.layers):
            self.buffers.layer(weights, self.cache.key_cache[i], self.cache.value_cache[i],
                               self.cache_position, self.cos, self.sin, self.decode_mask)
        self.norm_out(self.buffers.x, self.final_norm, self.buffers.norm, self.final_eps)
        torch.mm(self.buffers.norm, self.lm_head, out=self.logits)
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
            yield self.token_ids[:, 0].tolist()
            for _ in range(max_new_tokens - 1):
                self.graph.replay()
                self.token_ids.copy_(self.next_tokens)
                self.cache_position.add_(1)
                yield self.token_ids[:, 0].tolist()
