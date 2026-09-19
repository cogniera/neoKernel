# CUDA graph diagnosis

**Performance targets remain unmet.** The revised engine passed one complete six-workload benchmark and was kept as Codex log #11. A subsequent attended baseline failed the public-0 TTFT gate, so no GLM proposals ran.

## Graph replay evidence

| Version | step_wall_ms | sum_kernel_ms | gap_ms | CUDA kernels | cudaGraphLaunch | Host syncs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rejected engine | 15.982 | 8.715 | 7.266 | 2150 | 1 | 1 |
| Persistent boolean-mask revision | 15.625 | 8.684 | 6.942 | 2292 | 1 | 1 |

Both traces contain cudaGraphLaunch, so the graph was already replaying. Thousands of CUDA kernels are nodes in that graph, not evidence that capture was skipped. The gap includes profiling overhead. The existing GPU kernel work itself took about 8.7 ms on public-0; correcting launch or mask plumbing alone did not achieve 5 ms.

Exactly one cudaStreamSynchronize appears inside the measured decode region, from .tolist(). The profiler adds no explicit synchronization inside that region. The steady-state source calls graph.replay(), token_ids.copy_(), cache_position.add_(), and .tolist(), without .item(), .cpu(), or Python-side tensor construction.

## Changes

The engine was derived from results/codex_graph_rejected/engine.py. Loaded Transformers layers are retained. Prefill uses attention_mask=None and initialized-prefix cache views so SDPA can use its causal flash path. K/V writes still target the fixed-capacity StaticCache tensors through cache_position. No prefill 4D mask is constructed.

Decode uses a persistent boolean [B,1,1,S+N] mask. A captured index_fill_ marks the current cache position valid before attention. Other slots stay masked. Generation resets validity and positions; stale cache contents need not be zeroed because prefill overwrites the prompt prefix and decode overwrites each position before exposing it. The graph and input addresses stay fixed across samples.

The profiled boolean-mask revision preceded the prefix-only prefill change; that final change affects prefill and reset, not the captured decode tensor operations.

## Validation and passing benchmark

Guard passed. L4 public correctness passed with near-tie counts 0 / 0 / 2. The earlier boolean-mask check also passed with 0 / 0 / 7. Peak candidate memory stayed below 43.3% of the L4. CPU tests cover causal prefill, fixed-capacity boolean decode masking and stale-cache reuse across prompts.

| Workload | tok/s | TTFT ms | Native TTFT ms | TTFT ratio | TPOT ms | Native TPOT ms | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| public-0 | 98.12 | 25.507 | 24.772 | 1.0296 | 9.691 | 22.273 | pass |
| public-1 | 169.69 | 202.599 | 201.660 | 1.0047 | 17.798 | 22.836 | pass |
| public-2 | 780.26 | 191.741 | 190.176 | 1.0082 | 19.158 | 22.901 | pass |
| h-a | 157.27 | 55.327 | 55.365 | 0.9993 | 12.044 | 22.322 | pass |
| h-b | 396.10 | 194.575 | 194.181 | 1.0020 | 17.431 | 24.029 | pass |
| h-c | 1420.38 | 190.683 | 188.529 | 1.0114 | 19.817 | 23.171 | pass |

The judge geomean was 323.657 tok/s, +28.775% against the previous kept native baseline. Log #11 marks static_kv_cache, bypass_wrapper and cuda_graph_decode kept. This is not a claim that the requested stricter targets passed: public-0 TPOT is 9.691 ms, public-2 TPOT is 19.158 ms, and all public TTFT ratios remain slightly above 1.0.

An earlier run on the same code had pipe delays up to 107.7 ms and 13–24% public spread, with native public-0 throughput dropping to 24.45 tok/s. Its results are retained in prefix_graph_bench.json; the quieter repeat is prefix_graph_clean_bench.json.

## Attended continuation

Log #12: baseline failed gates. public-0 TTFT/native=1.177056. Other shapes passed. The three-step auto command exited before proposing or applying another engine change. No gate was relaxed and no failed run was declared eligible.

## Spending

This step: $1.3473 of $3.00. Cumulative Modal estimate across the conversation: $3.0875. Estimates conservatively include allocation waits and CPU/RAM; they are not billing records.

No commit or push was performed. The kept engine remains in the working tree; its measured snapshot is in results/codex_graph_kept/.
