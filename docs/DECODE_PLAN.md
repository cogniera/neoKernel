# Combined decode candidate

The user reported **859 tok/s** for `ffcf084`. That remains the measured baseline.
The combined candidate has not been tested or benchmarked locally, following the
user's instruction to push changes and use the website for measurement.

## Retained prefill

Packed QKV and gate/up projections, fused RMSNorm and SiLU/product, fused
Q/K normalization and native-rounded rotary/cache writes, grouped-query causal
FlashAttention, fused residual normalization, final-layer last-query evaluation,
and the prefill CUDA graph are retained.

## Decode alternatives

The cuBLAS projection path remains available. A new split-K implementation
partitions each projection's reduction into independent tensor-core programs.
Partials accumulate in FP32. Their consumers perform the final sum and native
BF16 rounding before Q/K normalization, rotary/cache writes, SiLU/product, or
residual normalization. The last down projection feeds the final norm directly.
All weights remain BF16; no quantization or approximate attention is introduced.

Attention can use the existing split/merge implementation, a larger split block
for small batches, or a single-launch online softmax for larger batches. All paths
attend to every initialized position through the current device-side position.

Vocabulary selection can use cuBLAS plus argmax, or a tiled full-vocabulary BF16
matrix product that retains only each tile's maximum and index. Logits round to
BF16 before comparison; ties select the smallest vocabulary index. The tied
embedding weight is unchanged.

The first decode embedding gather and RMSNorm share a kernel in both paths.

## Shape-specific warmup selection

For batches up to 32, warmup considers combinations of two projection paths,
two attention modes, and two vocabulary paths. Larger batches retain cuBLAS
projections/head and consider the attention variants. Each candidate is a full
decode CUDA graph, timed with CUDA events over three short repetitions. A
candidate must be at least 3% faster than the current best median to replace it.
The chosen graph is fixed for subsequent samples of that shape. There is no
runtime speed claim until the external benchmark measures it.

Every timed repetition restores the original first token and prompt position.
Only cache slots at or after the prompt are overwritten during selection; the
logical position excludes stale slots. Selection is followed by another reset.
Every subsequent generate call overwrites the complete prompt cache in prefill.
Warmup output tokens are never used as a later sample's prompt or continuation.

## Streaming

Each output step owns a distinct device history row and pinned host row. A
separate copy stream waits for the event following that step. Compute does not
wait for the copy because later steps cannot overwrite its source row. The host
waits on the requested token's copy event, rather than the entire copy stream.
At most two decode steps are queued ahead of the token currently being emitted.
The final token's event completes before it is yielded. Exactly the requested
number of steps is computed and yielded, including short output lengths and tails.

## Measurement still required

The website must establish correctness, load/warmup time, latency, sample spread,
memory, and the official score. Warmup selection measures speed only; it is not a
correctness test. Numerical reduction order differs in the split-K and online
attention alternatives, while BF16 cast locations and model formulas are retained.
The goal remains above 1400 tok/s; this document does not claim it was reached.
