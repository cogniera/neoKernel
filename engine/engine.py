"""Greedy Qwen3 using its loaded layers, fixed KV storage, and graphed decode."""

import torch
from transformers import AutoModelForCausalLM, StaticCache


class PrefixStaticCache(StaticCache):
    prefill_length = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if self.prefill_length:
            return keys[:, :, :self.prefill_length, :], values[:, :, :self.prefill_length, :]
        return keys, values


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
    x = base.norm(x)
    return model.lm_head(x[:, -1:, :])


class Engine:
    def __init__(self, model_path):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True,
        ).eval().to("cuda:0")
        self.shape = None
        self.graph = None

    def prepare(self, batch, prompt_length, output_length):
        self.graph = None
        self.shape = (batch, prompt_length, output_length)
        capacity = prompt_length + output_length
        self.cache = PrefixStaticCache(
            self.model.config, max_batch_size=batch, max_cache_len=capacity,
            device="cuda:0", dtype=torch.bfloat16,
        )
        self.prefill_positions = torch.arange(prompt_length, device="cuda:0")
        self.decode_mask = torch.zeros((batch, 1, 1, capacity), device="cuda:0", dtype=torch.bool)
        self.token_ids = torch.zeros((batch, 1), device="cuda:0", dtype=torch.int64)
        self.cache_position = torch.zeros((1,), device="cuda:0", dtype=torch.int64)
        self.position_ids = self.cache_position.view(1, 1)

    def decode(self):
        self.decode_mask.index_fill_(3, self.cache_position, True)
        logits = qwen_forward(self.model, self.token_ids, self.cache,
                              self.cache_position, self.position_ids, self.decode_mask)
        return logits[:, -1, :].argmax(dim=-1, keepdim=True)

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

    def generate(self, input_ids, max_new_tokens):
        batch, prompt_length = len(input_ids), len(input_ids[0])
        with torch.inference_mode():
            if self.shape != (batch, prompt_length, max_new_tokens):
                self.prepare(batch, prompt_length, max_new_tokens)
            self.decode_mask.zero_()
            self.decode_mask[:, :, :, :prompt_length].fill_(True)
            self.cache_position.fill_(prompt_length)
            prompt = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
            self.cache.prefill_length = prompt_length
            logits = qwen_forward(self.model, prompt, self.cache, self.prefill_positions,
                                  self.prefill_positions.unsqueeze(0), None)
            self.cache.prefill_length = 0
            self.token_ids.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
            if max_new_tokens > 1 and self.graph is None:
                self.capture_decode()
            yield self.token_ids[:, 0].tolist()
            for _ in range(max_new_tokens - 1):
                self.graph.replay()
                self.token_ids.copy_(self.next_tokens)
                self.cache_position.add_(1)
                yield self.token_ids[:, 0].tolist()
