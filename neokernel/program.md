# neoKernel operator program

Raise the geometric mean of output tokens per second across the selected workloads without failing any gate. Propose one structural hypothesis at a time. Numeric Triton parameters belong to the LLM-free sweep. Return the exact proposal JSON schema supplied in the system context. Keep reasoning to at most three sentences and the hypothesis to one sentence.

The archive root exports Engine with __init__(self, model_path) and generate(self, input_ids, max_new_tokens). Yield exactly max_new_tokens lists of batch Python ints in input order. EOS is an ordinary token. Load only the pinned local Qwen/Qwen3-4B-Instruct-2507 revision cdbee75f17c01a7cc42f958dc650907174af0554 in BF16. Python 3.11, CUDA 12.4, PyTorch 2.5.1, Triton 3.1.0, Transformers 4.51.3, safetensors 0.5.3, and tokenizers 0.21.1 are fixed. No network or installation in the engine. Source only, below 2 MiB and 200 files. Every token is checked against native teacher-forced logits on your emitted prefix, with at most 2.0 logits below the argmax. Reset prompt-dependent state each call.

Only engine/engine.py and Python files under engine/kernels/ may change. Never change the judge, guard, this program, workloads, keep threshold, timing, or correctness code. The harness alone owns exp/<id> branches, snapshots, keep decisions, commits, merges and tags. A keep requires passing every selected gate and improving the matching baseline by more than 1 percent. Never issue Git operations from engine code. The harness never pushes.

## Kept state and ordered playbook

The kept engine is the hand-rolled decode lineage labeled #18, with Triton decode attention and the later native-prefill CUDA graph fix. Official Dryft run 1bf05baa scored 670.7 tok/s on commit 012646d. Local log #18 itself failed h-c TTFT; the passing all-workload local freeze is #25. Do not describe the earlier failure as a kept benchmark. Native-prefill graph capture is already present; the final remaining item below concerns further measured improvements, not adding a missing first graph.

Kept items: **static_kv_cache**, **bypass_wrapper**, **cuda_graph_decode**, **concat_qkv**, **concat_gate_up**, **fused_rmsnorm**, **fused_qk_norm_rope_kv_write**, **fused_silu_mul**, **decode_attention_kernel**. Norm and activation fusions describe decode; they do not claim that fused prefill norms passed. Preserve those implementations and propose the first remaining item not kept and not exhausted by five reverts. Explain a deviation only when the profile supports a larger opportunity. Numeric launch parameters belong to the LLM-free sweep.

1. **attention_impl_kv_layout**: Choose `attention_impl` (`triton` or `sdpa_grouped`) and `kv_layout` (`bhsd` or `bshd`) per runtime shape through `CONFIG`. Select deterministically during shape setup before warmup/capture, using actual batch, prompt and cache capacity, with a valid fallback for unseen shapes. This touches `engine/kernels/__init__.py` for CONFIG, `engine/engine.py` for dispatch/cache setup and `engine/kernels/decode.py` or `engine/kernels/attention.py` only where needed to pass the selected path and strides. Preserve full visible attention, causal prefill, fresh cache state, stable graph addresses and dtype boundaries. Do not branch on workload names, seeds or prompt content.

2. **lm_head_argmax**: Fuse the tied vocabulary projection and global argmax so decode avoids materializing full logits and a separate selection pass. Preserve the reference BF16 projection output rounding before comparison and choose the lowest token index on exact ties. This touches a new Python/Triton module under `engine/kernels/`, plus `engine/kernels/decode.py` and the relevant call site in `engine/engine.py`. Reuse the loaded tied embedding weight; compare against the native projection before complete-generation replay.

3. **prefill_packed_weights**: Route prefill through concatenated QKV and gate/up projections and fused norms while retaining the exact causal flash attention path. Current native prefill modules share contiguous views of the packed storage but still issue separate projections. This touches `engine/engine.py`, `engine/kernels/weights.py` and the normalization/projection adapters under `engine/kernels/`; update `engine/kernels/decode.py` only for shared interfaces. Preserve head norms, absolute RoPE, residual additions, every BF16 cast boundary and the cache handoff into decode. The earlier fused-prefill-norm attempt failed fresh public-2 replay at a 2.125-logit deficit (#23); investigate that numerical failure rather than repeating the rejected formulation. Keep the working native causal path as the fallback.

4. **prefill_cuda_graph**: Improve or extend the existing native-prefill CUDA graph after the preceding changes, if measured TTFT and total throughput justify it. This primarily touches graph construction, persistent buffers, shape setup and reset in `engine/engine.py`, and only needed allocation interfaces in `engine/kernels/decode.py`. Capture each fixed shape during warmup, keep the causal flash path, avoid sample-dependent compilation, preserve fresh prompts and exact output counts, and remain inside load/warmup and memory limits. Never claim the native-prefill graph is absent or remove its already demonstrated TTFT benefit without a passing replacement.

## Numerics and integration

### Hand-rolled decode controls

The hand-rolled decode candidate exposes `CONFIG` and scalar `TUNABLES` in
`engine/kernels/__init__.py`. Values are fixed before engine construction and
graph capture; changing them requires a new process and a new harness run.
Do not claim a configuration is frozen until its complete freeze passes.

| CONFIG key | True / selected path | Fallback / alternative |
| --- | --- | --- |
| `fuse_norm_residual` | Rounded attention residual plus post-attention RMSNorm in one kernel | Separate BF16 add followed by RMSNorm |
| `fuse_qk_norm_rope` | Q/K head norm, RoPE and K/V write in one kernel | Separate head norm then RoPE and K/V write |
| `fuse_silu_mul` | SiLU and product with intermediate BF16 rounding | Native ATen SiLU then multiply |
| `attention_impl` | `triton`: two-pass split attention with a device position | `sdpa_grouped`: four query rows per KV head and a boolean prefix mask |
| `kv_layout` | `bhsd`: contiguous head/sequence cache | `bshd`: sequence/head storage exposed as a logical BHSD view |

`TUNABLES` supplies `norm`, `qk`, `silu`, and `attention` BLOCK sizes, num_warps,
and num_stages; `merge` supplies num_warps and num_stages. Norm BLOCK must be a
power of two at least 2560, and Q/K BLOCK a power of two at least 128. Attention
and SiLU BLOCK sizes must be powers of two. Invalid launch sizes are rejected,
not silently substituted. Scalar values describe the current candidate;
the numeric sweep can replace each with search choices before staging trials.

Packed QKV and gate/up weights come from loaded modules. Native prefill modules
retain identical contiguous weight views into the packed storage. Current kept prefill stays
on causal Transformers and writes the same cache read by custom decode. Tests
must cover prefill handoff, graph replay, consecutive calls with fresh prompts,
and both attention implementations before any keep decision.

Reduce in fp32, cast the normalized value to bf16 before multiplying by the weight. As the starter explains: "Reorder arithmetic freely; do not reformulate it." Never move a cast across an operation. Every new kernel must be compared with the Transformers 4.51.3 module it replaces on random inputs before integration. Then run a complete-generation comparison, including two calls with different prompts. A per-kernel check does not replace sequence replay.

TTFT and TPOT each above 1.05 times local native (official Dryft uses 1.10) fail even when throughput improves. Memory above 90 percent fails. Spread above 25 percent fails. An occasional compile or slow path can violate spread. Load plus one warmup and each sample must each fit 300 seconds. Allocate, compile, and capture during warmup when shapes are known.

No quantization, approximate or sparse attention, cache eviction, unverified draft models, timing access, CUDA events, cached outputs across calls, evaluator mutation, or edits outside scope. Never repeat a patch that failed guard. After five reverted attempts at an item, move on. One hypothesis per proposal and one line of hypothesis per experiment.

The dynamic context contains the current engine and kernels, current diff, at most 15 verbatim log lines, counts summarizing older attempts, a coverage list of unattempted items, and the current profile. Read the kernel table alongside gap_ms, the bandwidth and prefill floors, and per-kernel speed-of-light ratios where bytes are derivable. Unknown byte counts are null, not zero. Prioritize the operation or host gap with the largest distance to its floor. Profiler overhead and overlapping streams can distort summed kernel time; use the interval-union metric too.

## Whole-file proposal format

Return a files object mapping scoped paths to their complete new contents. Return complete files, never fragments or unified diffs. Do not add comments describing what changed. A file not listed remains unchanged. An empty string deletes a file only under engine/kernels/; engine/engine.py may never be deleted or emptied. The context contains the complete current content of every file in the write scope. The harness writes replacements and asks Git to generate the logged diff.
