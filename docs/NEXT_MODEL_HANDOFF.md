# Next-model handoff: 932.9 tok/s to the next substantial gain

## User objective and working agreement

Improve the Qwen3 engine toward >1400 tok/s and ultimately the previously reported
1431.9 leader. The latest user request is for a pathway for the next model, not
another engine candidate in this turn. The preceding working agreement is to make
substantial combined changes, commit and push engine candidates, and let the user
supply the website measurement. Do not run local tests, benchmarks, Modal, the old
LLM harness, or website submissions unless the user changes that instruction.
Warmup-time selection inside the submitted engine is already part of the design.
Do not ask for permissions already granted. Do not claim an unmeasured speedup.

## Start from the correct checkout

- Measured candidate: `688780e`, user-reported **932.9 tok/s**.
- Working checkout: `C:\Users\trend\.codex\worktrees\prefill-fusion\neoKernel`.
- Its branch is `codex/prefill-fusion`; its HEAD was `688780e` when this handoff was written.
- Last successful push was that commit to `origin/main` at `https://github.com/cogniera/neoKernel.git`.
- The original checkout is `C:\Users\trend\Documents\Programming\Projects\neoKernel`.
  Its local main was still `8ca6b9b` and it contains uncommitted engine, tree/speculative,
  harness, and test edits. Preserve those edits. Do not reset it, stage everything,
  or assume it represents the submitted engine. Continue in the isolated checkout.
- This handoff and the latest result record are local documentation changes, not
  a new pushed engine. A copy of this handoff is in both checkouts.
- Read AGENTS.md, QWEN_ENGINE_CONTRACT.md and OPTIMIZATION_GUIDE.md before changing operations.
- PowerShell; Python is `py -3.11`. Before every commit, run the required credential
  scan of the staged tree and abort if the existing forbidden credential markers
  occur. Never print credential values or save them to repository files.
- Stage explicit paths. A subsequent authorized engine candidate can be published
  from this checkout with `git push origin HEAD:main`; never force-push.

## What is actually measured

| Commit | Reported official TPS | Change |
| --- | ---: | --- |
| 3ae5b52 | 779.2 | Pipelined single-token decode with grouped split-K attention |
| 0694a14 | 766.7 | Four-token prompt-lookup speculation; regressed |
| 8ca6b9b | 774.9 | Native grouped-query FlashAttention prefill |
| 4fcf3c1 | 837.1 | Fused prefill RMSNorm and SiLU/product |
| ffcf084 | 859.0 | Packed prefill, fused Q/K/rotary/cache and residual norms |
| 688780e | 932.9 | Combined decode alternatives, warmup selection, independent copies |

The last three scores were supplied without full run pages, run IDs, or public
per-workload timings. We do NOT know the latest TTFT/TPOT breakdown or which decode
variants won for which shapes. Do not attribute the entire improvement to split-K,
the fused head, or any other single component. The latest gain is about 8.6% over
859. Reaching 1400 from 932.9 requires about 1.50x throughput, or 33.4% less total
time for an equivalent workload mix. Public workloads do not determine the score.

## Current implementation map

- `engine/engine.py`: prefill, CUDA graphs, whole-step warmup selection, streaming.
- `engine/kernels/prefill.py`: packed QKV norm/RoPE/cache writes, residual/norm,
  packed SwiGLU activation. Native BF16 projections remain via F.linear.
- `engine/kernels/weights.py`: packed QKV and gate/up weights; native modules share
  their storage. Do not accidentally allocate another full model while repacking.
- `engine/kernels/decode.py`: cuBLAS projections and fused elementwise decode chain.
- `engine/kernels/split_decode.py`: four split-K projection producers with FP32
  partials, followed by fused reductions into Q/K/RoPE/cache, SiLU/product, and
  residual/RMSNorm. Current splits: QKV=4, attention-output=8, gate/up=2, down=8.
  Current matmul tiles: M16, N64, K128, four warps, three stages.
- `engine/kernels/attention.py`: split/merge attention (128 or 256 keys/block), plus
  single-launch online full-prefix attention for an alternative at larger batches.
- `engine/kernels/head.py`: full-vocabulary tiled BF16 projection plus argmax,
  and a one-program token/position/history commit kernel.
- `engine/kernels/rmsnorm.py`: native cast boundaries; fused first embedding/norm.

Prefill already has causal GQA FlashAttention, graph replay, packed projections,
fused norms/activation/rotary/cache/residuals, and final-layer last-query-only
attention/output/MLP. Every layer still populates all prompt K/V needed by decode.

Decode currently chooses up to eight full-step graphs at first-call warmup:
cuBLAS versus split-K for ALL four projections together; two attention modes;
and cuBLAS versus tiled vocabulary head. Batches above 32 retain cuBLAS projections
and head. Each graph is timed over three repetitions of up to three decode steps
near the start of generation. A new candidate needs a 3% improvement to replace
current best; selection is fixed for later samples. This is speed selection, not
a correctness check. Selected plan names are printed, but official hidden-run
stdout is withheld by the platform.

Streaming already gives each output a unique device row and pinned host row.
Ready/done events permit copies without making compute wait; lookahead is two
steps. Preserve these dependencies and exact output counts when changing graphs.

## Recommended next combined candidate: mixed projection plans and better tiling

This is the most concrete limitation visible in the current code. The engine
must choose all-cuBLAS or all-split-K, although QKV, attention-output, gate/up, and
down have different matrix dimensions. A winning whole-step split-K graph can
still contain a losing projection. This is a hypothesis, not a measured diagnosis.

1. Make each projection and its consumer independently dispatchable. Its output
   representation must be explicit: materialized BF16 versus FP32 split partials.
   QKV feeds Q/K norm/RoPE/cache; O feeds residual/post-norm; gate/up feeds SiLU;
   down feeds the next input norm or final norm. Do not feed a partial accumulator
   into a consumer that expects a rounded BF16 projection.
2. Add a SMALL shape-dependent tiling set: choose split count and N/K tile sizes
   independently by projection; consider a dedicated BF16-weight FP32-reduction
   GEMV path for batch 1 and tensor-core GEMM for larger batches. Existing skinny
   GEMM was slower: do not simply turn its old CONFIG switch back on.
3. If a weight layout enables coalesced/vectorized reads, create it during loading
   or warmup from the exact BF16 values. Account for extra model-sized copies and
   preserve native prefill access and tied embeddings. No quantization.
4. Select producer-plus-consumer pairs, then capture and compare complete mixed
   graphs. Keep a bounded search, such as greedy coordinate selection across the
   four projections with a final complete-graph comparison, instead of a Cartesian
   explosion. Warmup plus loading must remain below 300 seconds. Ensure trial
   buffers and logical cache positions are reset, and no timed sample retunes.
5. Keep the current complete graph as an option. Short near-start timings are
   noisy and underrepresent long output/context behavior. Where improving this,
   use properly initialized cache states and complete valid warmup decoding; do
   not jump position into uninitialized cache just to time a later position.

Touch: `split_decode.py`, `decode.py`, `weights.py`, `engine.py`; possibly a new
focused matmul module. Preserve the existing prefill and streaming gains.

## Additional fusion opportunities to include when they fit the plan

- Consider fusion of the attention split merge into its output projection. First
  normalize/recombine the split outputs correctly and round the attention result
  to BF16 BEFORE the O projection. Applying the projection separately to unrounded
  partials is not equivalent to the native BF16 operation. Repeatedly recomputing
  the attention merge per output tile may cost more than its saved launch.
- Improve the vocabulary path by shape. It currently uses one fixed tile. Native
  rounded logits and lowest-index ties are mandatory. Selection must still cover
  every vocabulary row; heuristic vocabulary pruning is not equivalent.
- Preserve separate causal prefill and decode policies. Prefill is already much
  more optimized; do not spend the entire iteration polishing host code or a
  one-layer shortcut while ignoring the per-token weight stream.
- A monolithic whole-model megakernel is not a safe default. Triton 3.1 has no
  automatic safe grid-wide barrier. Spin-waiting on unscheduled GPU blocks can
  deadlock. Keep inter-stage launches unless a concrete safe synchronization
  mechanism and occupancy bound are established.

## Architectural route if weight streaming remains the ceiling

A larger jump may require reusing a full-model weight read across multiple
accepted tokens, rather than slightly accelerating another single-token kernel.
Exact verified speculation can do this, but the previous prompt-lookup proposal
already regressed (766.7 versus 779.2). Do not reintroduce it unchanged or assume
that a wider tree fixes its acceptance rate.

A new design needs a substantially better, cheap draft source and full-target
verification. Possibilities to reason about include a same-checkpoint reduced-
layer draft or a bounded candidate tree, with causal ancestry masks, but these are
unproven options and may fail the cost model. No external checkpoint downloads or
extra unverified model outputs are allowed. Preserve the existing uncommitted tree
work as reference; it is not a tested baseline and must not be silently merged.

Use the break-even calculation before building a large speculative engine:
if K draft positions cost D each, a verification block costs V, accepted/emitted
target-valid tokens average A, and ordinary decode costs T per token, speedup is
approximately A*T/(K*D+V+handoff overhead), before prefill. A true 1.5x total gain
needs more than 1.5x decode gain when prefill is unchanged. Draft cost, rejected
branches, head cost, and unequal per-sequence acceptance all matter. Switching
to a slower verifier and calling it speculation does not create speed.

If implemented, include per-shape warmup acceptance/cost selection and a fast
single-token fallback, exact longest-prefix acceptance, correct rejected-cache
handling, and full-target replacements. Every sequence must yield exactly N tokens;
keep TTFT independent of expensive speculative blocks. Never reuse prompt content
or generated-token results between samples.

## Non-negotiable numerics and contract

Qwen3-4B BF16, H2560, I9728, 36 layers, Q32/KV8, D128, vocab151936, tied embeddings.
Runtime is PyTorch 2.5.1 / Triton 3.1 / Transformers 4.51.3 / CUDA12.4 on one H100.
RMSNorm: normalize FP32 -> BF16 -> learned gain -> BF16. Residual sums round to
BF16 before normalization. Q/K normalize per head before RoPE; each RoPE product
rounds to BF16 before addition. SiLU rounds to BF16 before multiplying by up.
Linear outputs and final logits round to BF16 before their consumers/selection.
Full visible attention, correct scale and absolute positions, no cross-prompt cache.
Exactly N lists of B IDs, EOS ordinary, no network/downloads in the engine.
The 2.0 logit margin is not permission to quantize, sparsify, or omit computation
whose output is actually required. Never manipulate the timing or judging harness.

## Deliverable for the next model

Build a coherent larger candidate from `688780e`, prioritizing mixed projection
plans, consumer fusion and shape-specific tiling. Use architectural speculation
only with a credible acceptance/cost argument and exact verification. Preserve
all known wins and the source commit as the fallback. Follow the user's no-local-
verification agreement, stage only intended files, scan credentials, commit and
push the engine candidate. Report the commit and what changed without claiming
1400 was reached. The user will provide the next actual website result.
