# neoKernel operator program

Raise the geometric mean of output tokens per second across the selected workloads without failing any gate. Propose one structural hypothesis at a time. Numeric Triton parameters belong to the LLM-free sweep. Return the exact proposal JSON schema supplied in the system context. Keep reasoning to at most three sentences and the hypothesis to one sentence.

The archive root exports Engine with __init__(self, model_path) and generate(self, input_ids, max_new_tokens). Yield exactly max_new_tokens lists of batch Python ints in input order. EOS is an ordinary token. Load only the pinned local Qwen/Qwen3-4B-Instruct-2507 revision cdbee75f17c01a7cc42f958dc650907174af0554 in BF16. Python 3.11, CUDA 12.4, PyTorch 2.5.1, Triton 3.1.0, Transformers 4.51.3, safetensors 0.5.3, and tokenizers 0.21.1 are fixed. No network or installation in the engine. Source only, below 2 MiB and 200 files. Every token is checked against native teacher-forced logits on your emitted prefix, with at most 2.0 logits below the argmax. Reset prompt-dependent state each call.

Only engine/engine.py and Python files under engine/kernels/ may change. Never change the judge, guard, this program, workloads, keep threshold, timing, or correctness code. The harness uses snapshots and changes the working tree only. It never creates commits, merges, or pushes.

## Ordered playbook

Propose the first item not yet kept unless the profile points to a larger opportunity. Explain any deviation. This is the expected order of payoff for these shapes; prerequisites make static cache and wrapper work useful before graph capture.

1. **static_kv_cache**: Preallocate correctly strided per-layer KV storage. Reset logical length each call. Attention must see initialized slots only. See the guide's "Replace the cache and capture decode", "Tensor shapes", and "Execution graph".
2. **bypass_wrapper**: Use the guide's qwen_forward approach to remove generic top-level model dispatch while preserving Qwen layer semantics. See "Bypass the top-level wrapper". Chunked prefill, verification, and static caches require explicit causal masks.
3. **cuda_graph_decode**: Capture a fixed-shape decode step with stable input, output, position, and cache addresses. Keep host list conversion and yield outside capture. See "Replace the cache and capture decode". This is the largest expected single win.
4. **concat_qkv**: Relayout Q, K, and V weights during construction and issue a shared projection. Preserve head reshape, normalization, and dtype boundaries. See "Modules and weights" and "Map model operations to kernels".
5. **concat_gate_up**: Relayout gate/up weights together; preserve separate SwiGLU values and BF16 cast boundaries. See "Execution graph", "Modules and weights", and "Replace full blocks last".
6. **prefill_cuda_graph**: Capture fixed workload prefill during warmup if it improves TTFT without prompt-dependent compilation or excessive memory. Keep a causal attention mask. See "Are megakernels possible?" and "Replace the cache and capture decode".
7. **fused_rmsnorm**: Reuse the starter's exact RMSNorm kernel and adapter discipline. See "Replace one leaf module first" and engine/kernels/rmsnorm.py. Compare random inputs against the native module before wiring it in.
8. **fused_qk_norm_rope_kv_write**: Fuse per-head Q/K normalization, absolute-position RoPE, and writes into initialized cache slots, preserving Q/K head grouping. See "Tensor shapes", "Map model operations to kernels", and "Replace full blocks last".
9. **fused_silu_mul**: Fuse the activation and product while keeping the reference cast points. See "Execution graph" and "Map model operations to kernels".
10. **lm_head_argmax**: Fuse vocabulary projection and argmax while preserving lowest-index selection on exact ties. See "Map model operations to kernels". Use the tied embedding weight once.
11. **decode_attention_kernel**: Implement grouped-query decode attention on valid cache positions, with scale 1/sqrt(128) and KV head h//4. See "Tensor shapes" and "Define your own kernel interface". Preserve full exact attention.
12. **speculative_prompt_lookup**: Attempt only after every preceding item is kept. Verify proposed tokens exactly using the native-equivalent target model. Multi-token verification requires a correct causal mask. Declare SPECULATIVE = True at engine.py module scope so the judge explains its relaxed weight floor. The prefill floor remains unchanged.

Official native decode accounts for approximately 96, 76, and 94 percent of public generation time. The 17 to 22 ms decode step is dominated by overhead compared with a roughly 2.4 ms weight-bandwidth floor. With unchanged prefill, the planning targets after decode graphs are roughly 330, 460, and 3,700 tokens/s. These are estimates, not measurements or promised results. After decode approaches the floor, prefill becomes the bottleneck on the two 8,192-token shapes: about 200 ms of an approximately 280 ms total. That is why weight concatenation and prefill graphs precede elementwise fusions, each usually worth well under a millisecond per step.

## Numerics and integration

Reduce in fp32, cast the normalized value to bf16 before multiplying by the weight. As the starter explains: "Reorder arithmetic freely; do not reformulate it." Never move a cast across an operation. Every new kernel must be compared with the Transformers 4.51.3 module it replaces on random inputs before integration. Then run a complete-generation comparison, including two calls with different prompts. A per-kernel check does not replace sequence replay.

TTFT and TPOT each above 1.05 times local native (official Dryft uses 1.10) fail even when throughput improves. Memory above 90 percent fails. Spread above 25 percent fails. An occasional compile or slow path can violate spread. Load plus one warmup and each sample must each fit 300 seconds. Allocate, compile, and capture during warmup when shapes are known.

No quantization, approximate or sparse attention, cache eviction, unverified draft models, timing access, CUDA events, cached outputs across calls, evaluator mutation, or edits outside scope. Never repeat a patch that failed guard. After five reverted attempts at an item, move on. One hypothesis per proposal and one line of hypothesis per experiment.

The dynamic context contains the current engine and kernels, current diff, at most 15 verbatim log lines, counts summarizing older attempts, a coverage list of unattempted items, and the current profile. Read the kernel table alongside gap_ms, the bandwidth and prefill floors, and per-kernel speed-of-light ratios where bytes are derivable. Unknown byte counts are null, not zero. Prioritize the operation or host gap with the largest distance to its floor. Profiler overhead and overlapping streams can distort summed kernel time; use the interval-union metric too.
