# neoKernel operator program

Raise the geometric mean of output tokens per second across the selected workloads without failing any gate. Propose one structural hypothesis at a time. Numeric Triton parameters belong to the LLM-free sweep. Return the exact proposal JSON schema supplied in the system context. Keep reasoning to at most three sentences and the hypothesis to one sentence.

The archive root exports Engine with __init__(self, model_path) and generate(self, input_ids, max_new_tokens). Yield exactly max_new_tokens lists of batch Python ints in input order. EOS is an ordinary token. Load only the pinned local Qwen/Qwen3-4B-Instruct-2507 revision cdbee75f17c01a7cc42f958dc650907174af0554 in BF16. Python 3.11, CUDA 12.4, PyTorch 2.5.1, Triton 3.1.0, Transformers 4.51.3, safetensors 0.5.3, and tokenizers 0.21.1 are fixed. No network or installation in the engine. Source only, below 2 MiB and 200 files. Every token is checked against native teacher-forced logits on your emitted prefix, with at most 2.0 logits below the argmax. Reset prompt-dependent state each call.

Only engine/engine.py and Python files under engine/kernels/ may change. Never change the judge, guard, this program, workloads, keep rule, timing, or correctness code. The harness alone owns exp/<id> branches, snapshots, keep decisions, commits, merges and tags. Never issue Git operations from engine code. The harness never pushes.

## Keep rule and current distance

A keep requires passing every selected gate, a geometric mean more than 1 percent above the matching baseline, **and no selected workload more than 3 percent below its baseline throughput**. A global choice that wins the geomean by helping one shape and hurting another is not kept: experiment #37 switched attention to grouped SDPA with BSHD storage for every shape, measured +9 percent locally, lost 4 percent on Dryft, and was reverted.

Where the engine stands, so you know the distance:

- Kept engine #18 on Dryft: official run 1bf05baa, **670.7 tok/s** hidden-workload geomean on commit 012646d. Local six-workload freeze #25: 722.6 tok/s.
- Public TPOT on Dryft: **4.75 / 6.06 / 6.16 ms** for B1 / B4 / B16 against a **2.40 ms** weight-read floor (8.04 GB at 3.35 TB/s). Decode is at roughly half the bandwidth bound; the rest is launch gap and non-GEMM kernels.
- Public TTFT on Dryft: **19.7 / 201.6 / 190.8 ms**. Prefill on 8,192 tokens runs at about 330 achieved TFLOPS against a 600 TFLOPS assumption, so prefill has about 1.8x of headroom on the long shapes.
- The leader is at **1,385 tok/s**, above the no-speculation ceiling for these shapes. Items 1 to 6 below are worth roughly 1.5x together; item 7 is the multiple that reaches the leader's band.
- H100 decode profile of the kept engine (profiler overhead included): batch 1 step 6.65 ms wall, 4.28 ms of kernels in 439 launches, 2.37 ms gap; batch 16 step 8.44 ms, 5.81 ms kernels, 2.63 ms gap. GEMMs are 3.2 ms of the batch-1 kernel time; `_partial` decode attention is 0.34 ms at batch 1 and 1.59 ms at batch 16; the LM-head GEMM is 0.27 ms; `_norm` is 0.19 ms over 73 launches.

## Kept state

Kept items: **static_kv_cache**, **bypass_wrapper**, **cuda_graph_decode**, **concat_qkv**, **concat_gate_up**, **fused_rmsnorm**, **fused_qk_norm_rope_kv_write**, **fused_silu_mul**, **decode_attention_kernel**, plus the native-prefill CUDA graph. Read `engine/kernels/decode.py` before proposing: the decode layer already runs packed QKV and gate/up GEMMs, a fused Q/K norm + RoPE + cache write, a fused SiLU product, the post-attention residual fused into `norm_out`, and a split-K Triton attention (`_partial` over cache blocks, `_merge` by log-sum-exp). Do not re-propose those. Preserve their implementations and interfaces; the unit tests import `norm_out`, `qk_rope_cache_out`, `silu_mul_out`, `attention_out`, `cache_storage`, `DecodeBuffers`, `LayerWeights`, and the engine attributes `graph`, `prefill_graph`, `token_ids`, `buffers`, `cache`, `logits`, `model`, `shape`, `DECODE_CONFIG`.

Propose the first item below that is neither kept nor exhausted by five reverts. Deviate only when the profile in context shows a larger gap elsewhere, and say so in the reasoning.

## Ordered playbook

### 1. residual_fuse_norm

**Mechanism.** The kept fusion covers only the attention residual: `norm_out(branch, post_norm, norm, eps, residual=x, summed=x)` sums in fp32, rounds the sum to bf16, stores it back into the residual buffer, and normalizes the rounded bf16 sum with the existing cast placement. The reference adds in bf16, so rounding before both the store and the normalization is the matching point. What remains unfused is the down-projection residual: `torch.add(x, branch, out=x)` at the end of each layer followed by the next layer's `norm_out(x, input_norm, ...)` and, after the last layer, the final norm. Fuse that pair the same way: carry `branch` (down_proj output) into the next input norm and the final norm as the residual argument. One elementwise launch and one activation round-trip per layer disappear; 36 launches per step. Do not change the rounding point or the weight multiply order.

**Files.** `engine/kernels/rmsnorm.py` or `engine/kernels/elementwise.py` if the kernel needs a new entry point, and the layer/step sequence in `engine/kernels/decode.py` plus the final-norm call in `engine/engine.py`.

**Expected.** 0.3 to 0.5 ms per step at batch 1 including the launch gap it removes; the kernel time itself is small.

**Test.** `test_norm_and_residual` already diffs `norm_out` with a residual against the module pair (bf16 add then `Qwen3RMSNorm`) on random bf16 inputs: max abs diff 0 on the stored residual, within bf16 ulp (0.03125 absolute) on the normalized output. Keep that contract for the new call site and the handoff test must still see identical first and second tokens on all public shapes.

**Gate most likely to bite.** Correctness, if the fp32 sum is normalized without the bf16 rounding first; that is the reformulation the numerics section forbids.

### 2. attention_splitk

**Mechanism.** The kept kernel already splits the cache into `attention.BLOCK`-sized chunks per program with an LSE merge, but its grid is `(batch, 32 query heads, splits)`: every K/V block is loaded four times, once per grouped query head, and the split count comes from cache capacity rather than the valid length, so at position 512 of a 2,080 capacity three quarters of the programs do masked work. Rewrite `_partial` as grouped split-K: one program per `(batch, KV head, chunk)` that loads K and V once and computes all four grouped query rows; keep bf16 in, fp32 accumulation, scale 1/sqrt(128), the `pos`-masked validity, the fp32 partial and LSE buffers, and the existing `_merge`. The grid is fixed at graph capture, so chunk count cannot follow `pos` at run time; make the chunk size a TUNABLE so the sweep and item 6 can size it per shape, and let programs whose whole chunk is beyond `pos` exit after writing `-inf` LSE.

**Files.** `engine/kernels/attention.py` only, plus buffer shapes in `engine/kernels/decode.py` if the partial layout changes.

**Expected.** Most of the attention share at batch 1 and 4: `_partial` is 0.34 ms at batch 1 and 1.59 ms at batch 16 in the profile above, and the four-fold K/V re-read is the bandwidth it wastes.

**Test.** `test_attention` compares against SDPA on random inputs at capacities 640 and 2,080, batches 1, 4 and 16, positions 0, 127, 512 and capacity-1, with poisoned inactive slots; add 544 if you change the chunking. Argmax agreement and max abs diff under 4e-3 (the kept kernel is at 0.0039). Also compare against the existing kernel before replacing it.

**Gate most likely to bite.** TPOT ratio at batch 16 if the grouped program spills registers or serializes the four query rows; profile both before proposing.

### 3. lm_head_argmax_tiled

**Mechanism.** The step ends with `torch.mm(norm, lm_head, out=logits)` over the tied embedding `[151936, 2560]` and a separate `torch.argmax`. Replace both with one Triton kernel that tiles the vocabulary in `BLOCK_N` columns, accumulates each tile in fp32 with `tl.dot` over the 2,560 reduction in `BLOCK_K` steps, rounds the tile to bf16 exactly as the cuBLAS output is rounded, and keeps a running `(max, index)` per row in the epilogue across tiles, choosing the lowest index on exact ties; then a tiny reduction over the per-tile winners. The full logits tensor is never materialized. Expose `lm_head.BLOCK_N`, `lm_head.BLOCK_K` and `lm_head.num_warps` in TUNABLES. Five earlier attempts (#26, #28, #30 to #32) died in Triton compilation at `d = tl.arange(0, D)` with `D = 2560`: `tl.arange` extents must be powers of two, so the reduction has to be a loop over power-of-two `BLOCK_K` tiles, never one 2,560-wide range. Keep `engine.logits` allocated and written with the winning value per row if you remove the full projection, because the handoff test reads it; or keep the full projection on the prefill path only.

**Files.** `engine/kernels/lm_head.py` (new), the decode step in `engine/kernels/decode.py`, the call site in `engine/engine.py`.

**Expected.** 0.2 to 0.4 ms per step at batch 1, more at batch 16.

**Test.** Argmax equals `torch.argmax` on random bf16 hidden states, including constructed exact ties (duplicate a column of the weight and check the lower index wins), at batches 1, 4 and 16; the winning logit value within bf16 tolerance of the cuBLAS projection. Compare against the native projection before any complete-generation replay.

**Gate most likely to bite.** Correctness on near ties if the tile rounding differs from cuBLAS's single rounding of the fp32 accumulator.

### 4. prefill_packed_weights

**Mechanism.** Prefill still runs the Transformers modules with separate q/k/v and gate/up projections over views of the packed storage. Route prefill through the concatenated QKV weight and the concatenated gate/up weight and the fused RMSNorm kernels, keeping the Transformers causal SDPA flash path for attention, the per-head Q/K norms, absolute RoPE, every bf16 cast boundary, and the cache handoff into decode. Two failure modes are on record: #33 and #34 reshaped the packed projection output per head with the wrong dimension order (`size of tensor a (128) must match ... (4096)`), and #38 failed the handoff test inside `qwen_forward` at `layer.self_attn.q_norm(q)` twice through repair, because the packed output was split before the head norm with shapes the module does not accept. Split `[B, S, 6144]` into `q [B, S, 32, 128]`, `k [B, S, 8, 128]`, `v [B, S, 8, 128]` before the head norms, apply the norms per head, then transpose to `[B, H, S, D]` for RoPE and SDPA. The earlier fused-prefill-norm attempt (#23) failed a fresh public-2 replay at a 2.125-logit deficit; the prefill norm must keep the exact `rmsnorm.py` cast placement over the token axis, and the native causal path stays as the fallback.

**Files.** The prefill path in `engine/engine.py` (`qwen_forward` and graph capture), `engine/kernels/weights.py`; `engine/kernels/decode.py` only for shared interfaces.

**Expected.** 10 to 20 percent on the 2,048-token and batch-16 shapes, where prefill is the larger share of total time.

**Test.** First decode token after prefill matches native on all public shapes (the handoff test), and per-layer hidden-state max abs diff against the native module stack on one 512-token prompt under 2e-2.

**Gate most likely to bite.** Correctness on the 2,048-token shape, where a reordered prefill norm accumulates the largest deviation; then TTFT ratio if the packed GEMMs are not captured in the prefill graph.

### 5. gemm_epilogue_residual

**Mechanism.** Fuse the residual add into the o_proj and down_proj GEMMs' epilogues: a Triton matmul computing `C = A @ B + R` with fp32 accumulation and bf16 output. The reference is cuBLAS bf16 output followed by a bf16 add, that is two roundings; a single rounding of `acc + R` in fp32 differs from it by up to one bf16 ulp of the sum. Implement both epilogues (round the accumulator to bf16, add R in fp32, round again; versus one rounding at the end), measure which matches native within the margin on random inputs and on the handoff test, and keep that one. This item overlaps item 1: with the residual folded into the norm kernel, the epilogue fusion saves the fp32 read of `branch` rather than a launch, so justify it from the profile rather than from launch count. A Triton GEMM at M = 1 to 16 must at least match cuBLAS's tile choice for these shapes; benchmark against `torch.mm` before proposing.

**Files.** `engine/kernels/matmul.py` (new), the o_proj and down_proj call sites in `engine/kernels/decode.py`.

**Expected.** 0.2 to 0.3 ms per step.

**Test.** Diff against `torch.mm` plus add on random bf16 inputs for both rounding variants; report which one matches native, and the max abs diff of the other.

**Gate most likely to bite.** TPOT ratio: a Triton GEMM slower than cuBLAS at small M loses more than the fused add saves.

### 6. per_shape_config

**Mechanism.** The judge already knows `(B, S, N)` per workload and the engine learns it at warmup. Make CONFIG and TUNABLES a table keyed by `(B, S bucket, N bucket)` with a default row; select the row deterministically during shape setup before capture, from actual batch, prompt length and cache capacity, with the default row as the fallback for unseen shapes. Never branch on workload names, seeds or prompt content. This recovers what #37 lost: grouped SDPA with BSHD storage was faster at some shapes and slower at others, and a global switch cannot express that. The engine side is yours: the table in `engine/kernels/__init__.py`, the lookup in the warmup path of `engine/engine.py`, and the launch-parameter reads in the kernels. The harness side is the operator's: `neokernel/sweep.py` evaluates kept kernel variants per shape and writes the winning row per shape, and `neokernel/cli.py` freeze bundles the table into `engine/`.

**Files.** `engine/kernels/__init__.py`, the warmup and shape setup in `engine/engine.py`, kernels only where they read TUNABLES.

**Expected.** Recovers the shape-specific gains, about 9 percent locally for the attention layout at the shapes where it won, without the regression on the others.

**Test.** The engine selects the row at warmup (assert the chosen row for each public shape), the default row is used for an unseen shape, all public shapes pass check, and the freeze output contains the table.

**Gate most likely to bite.** Sample spread, if row selection ever depends on anything but the fixed shape; and the 3 percent per-workload rule, which is exactly what this item exists to satisfy.

### 7. speculative_prompt_lookup

**Mechanism.** Exact speculative decoding with n-gram drafts. Draft: at each step, search the prompt plus generated tokens for the last n tokens (n from 3 down to 1); on a match propose the k tokens that followed it, k up to 8. Verify: one forward over the k draft tokens through the static cache with a causal mask that admits key j for query i only when j <= L + i, compare the argmax at each draft position with the draft, accept the longest matching prefix plus the first non-matching argmax token, and reset the cache's logical length to the accepted position (a length pointer, not eviction). Fallback: no match, one normal step; after two consecutive misses stay on normal steps for a while so verification cost does not dominate on prompts without repetition. The verification forward must be graph-captured for a fixed k (pad drafts to k and mask the padding) or run eager with fixed shapes; measure both. The attention kernel must accept T > 1 queries per sequence. Declare `SPECULATIVE = True` at module level so the judge relaxes the physics floor to N x step_floor / 8. TTFT is unaffected because prefill is unchanged. This is legal and passes the correctness rule by construction because verification reproduces the same argmax.

**Files.** The generate loop in `engine/engine.py`, a multi-token step in `engine/kernels/decode.py`, `engine/kernels/attention.py` for T > 1 queries.

**Expected.** 1.3x to 2x on decode depending on acceptance; the only item that reaches the leader's band.

**Local measurement caveat.** The local judge draws prompts uniformly at random from the vocabulary (`neokernel.judge.make_prompt`), so n-gram matches inside the prompt almost never occur locally and acceptance will read near zero; Dryft derives prompts from a fixed text corpus, where repetition is common. Report acceptance rate per sample in the engine's stderr. The operator owns a corpus-derived prompt option for the local judge; until it exists, a local bench of this item measures only its fallback cost, which must stay within the TPOT gate.

**Test.** Tokens identical to greedy on 20 prompts (include prompts with deliberate repetition, since random ones exercise only the fallback); acceptance rate logged per prompt; the multi-token attention path compared against SDPA with the causal prefix mask at T = 1, 4 and 8.

**Gates most likely to bite.** 25 percent sample spread when acceptance varies by prompt (bench on at least 10 prompts), and TPOT ratio if verification is slower than a plain step on prompts with no matches.

### 8. tolerance_budget (measurement, not a kernel)

Operator-owned; never a proposal item. For each fused op, `neokernel/tolerance.py` swaps it to native, runs three prompts, and records the distribution of final-logit shift it introduces (median, p99, max), printed in the judge report next to the op's time saving and the engine's remaining headroom under the 2.0 margin. When that report is in your context, respect it: keep p99 headroom above 0.5 logits, and do not stack a new fusion on an op whose budget is already spent. The recorded reference is native replayed against itself: up to 0.75 logits at a handful of positions in every twelve thousand.

## Repair stage

After a proposal passes the guard, the harness runs `tests/test_hand_rolled_kernels.py` and `tests/test_hand_rolled_handoff.py` on an L4 against your files before any judge call: every kernel against the Transformers 4.51.3 module it replaces on random inputs, and the whole engine for prefill handoff, graph and buffer reuse, exact output counts and fresh-prompt reset on the pinned checkpoint. If they fail, you receive your own proposal as an assistant turn and a bounded excerpt of the test output, and you return corrected complete files in the same schema with the same item and hypothesis; omitted files stay as last applied. Corrected files are guarded and retested. You get two repair turns; a repair that fails the guard or is not valid JSON consumes a turn without being applied. After two failed repairs the experiment is reverted with the note `failed after 2 repairs`, and the log line records `repairs` and `first_error`. Only a candidate that passes the unit tests reaches the public correctness check and the judge. The tests are the harness's own files; a proposal cannot ship or change tests. Fix the cause of a failure; do not remove or stub what the tests exercise.

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

Reduce in fp32, cast the normalized value to bf16 before multiplying by the weight. As the starter explains: "Reorder arithmetic freely; do not reformulate it." Never move a cast across an operation; `engine/kernels/rmsnorm.py` matches the reference's cast placement exactly and explains why the obvious version does not. Every new kernel must be compared with the Transformers 4.51.3 module it replaces on random inputs before integration. Then run a complete-generation comparison, including two calls with different prompts. A per-kernel check does not replace sequence replay.

TTFT and TPOT each above 1.05 times local native (official Dryft uses 1.10) fail even when throughput improves. Memory above 90 percent fails. Spread above 25 percent fails. An occasional compile or slow path can violate spread. Load plus one warmup and each sample must each fit 300 seconds. Allocate, compile, and capture during warmup when shapes are known.

Forbidden: quantization, approximate or sparse attention, cache eviction, draft models without exact verification, any edit outside engine/, any use of time or CUDA events, output caching across calls, evaluator mutation, and timing access. Never repeat a patch that failed guard. After five reverted attempts at an item, move on. One hypothesis per proposal and one line of hypothesis per experiment.

The dynamic context contains the current engine and kernels, current diff, at most 15 verbatim log lines, counts summarizing older attempts, a coverage list of unattempted items, and the current profile. Read the kernel table alongside gap_ms, the bandwidth and prefill floors, and per-kernel speed-of-light ratios where bytes are derivable. Unknown byte counts are null, not zero. Prioritize the operation or host gap with the largest distance to its floor. Profiler overhead and overlapping streams can distort summed kernel time; use the interval-union metric too.

## Whole-file proposal format

Return a files object mapping scoped paths to their complete new contents. Return complete files, never fragments or unified diffs. Do not add comments describing what changed. A file not listed remains unchanged. An empty string deletes a file only under engine/kernels/; engine/engine.py may never be deleted or emptied. The context contains the complete current content of every file in the write scope. The harness writes replacements and asks Git to generate the logged diff.
