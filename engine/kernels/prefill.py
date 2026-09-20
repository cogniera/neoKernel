"""Prefill-only elementwise fusion with native BF16 rounding boundaries.

Inputs are CUDA BF16 tensors. Outputs are new contiguous tensors owned by the
caller (or the capturing CUDA graph). No cache or model weights are mutated.
"""

import torch
import triton
import triton.language as tl
from kernels.rmsnorm import _norm


def norm(x, module):
    """Normalize the last dimension, rounding before the learned gain multiply."""
    width = x.shape[-1]
    # The final-token slice can have a batch stride larger than width.
    flat = x.contiguous().view(-1, width)
    out = torch.empty_like(flat)
    _norm[(flat.shape[0],)](
        flat, flat, module.weight, out, out, width, module.variance_epsilon,
        False, triton.next_power_of_2(width), num_warps=4,
        num_stages=1, enable_fp_fusion=False,
    )
    return out.view(x.shape)


@triton.jit
def _activation(G, U, O, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    u = tl.load(U + i, i < N, 0).to(tl.float32)
    # Native SiLU materializes BF16 before multiplying by up_proj's BF16 output.
    activated = (g / (1.0 + tl.exp(-g))).to(G.dtype.element_ty)
    tl.store(O + i, activated.to(tl.float32) * u, i < N)


def silu_mul(gate, up):
    """Fuse SiLU and product for contiguous native projection outputs."""
    out = torch.empty_like(gate)
    count = gate.numel()
    _activation[(triton.cdiv(count, 1024),)](
        gate, up, out, count, 1024, num_warps=4, num_stages=1,
        enable_fp_fusion=False,
    )
    return out
