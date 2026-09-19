"""Pack from loaded modules and share storage with the unchanged prefill path."""

import torch


class LayerWeights:
    @torch.no_grad()
    def __init__(self, layer):
        a, m = layer.self_attn, layer.mlp
        self.qkv_weight = torch.cat((a.q_proj.weight, a.k_proj.weight, a.v_proj.weight))
        self.gate_up_weight = torch.cat((m.gate_proj.weight, m.up_proj.weight))
        self.qkv = self.qkv_weight.t()
        self.gate_up = self.gate_up_weight.t()
        # Each native module retains identical contiguous values and dimensions.
        # Shared slices avoid holding another 4.7 GB of projection weights.
        q_end = a.q_proj.out_features
        k_end = q_end + a.k_proj.out_features
        a.q_proj.weight = torch.nn.Parameter(self.qkv_weight[:q_end], requires_grad=False)
        a.k_proj.weight = torch.nn.Parameter(self.qkv_weight[q_end:k_end], requires_grad=False)
        a.v_proj.weight = torch.nn.Parameter(self.qkv_weight[k_end:], requires_grad=False)
        width = m.gate_proj.out_features
        m.gate_proj.weight = torch.nn.Parameter(self.gate_up_weight[:width], requires_grad=False)
        m.up_proj.weight = torch.nn.Parameter(self.gate_up_weight[width:], requires_grad=False)
        self.o = a.o_proj.weight.t()
        self.down = m.down_proj.weight.t()
        self.input_norm = layer.input_layernorm.weight
        self.post_norm = layer.post_attention_layernorm.weight
        self.q_norm, self.k_norm = a.q_norm.weight, a.k_norm.weight
        self.eps = layer.input_layernorm.variance_epsilon
