# Prefill latency follow-up

The website reported a latency failure for B1 on commit `31857c8`. The
screenshots do not include native timings or name the failing latency metric.
They cannot establish whether TTFT or TPOT exceeded the website gate.

The five-sample local baseline (experiment 21) passed correctness and latency:
TTFT 21.882 ms versus native 21.475 ms (1.018944x), TPOT 4.674 ms versus
native 17.663 ms (0.264618x). This points to TTFT as the tighter local constraint;
it does not prove which metric failed remotely.

The first candidate replaced the two hidden-state RMSNorm modules in each prefill
layer with the existing Triton norm kernel. Each prompt row uses FP32 reduction,
rounds the normalized value to BF16, then multiplies by the learned gain.
The causal Transformers attention path and the decode graph remain intact.
Prompt rows are flattened for the kernel and restored to their original shape.

L4 independent replay passed all three public workloads (near ties: 0 / 0 / 4).
The CPU suite passed 55 tests with six GPU-only skips. Five H100 B1 samples
(experiment 22) passed with TTFT 28.804 ms, TTFT/native 1.019787, TPOT 4.911 ms,
TPOT/native 0.195314, and 2.86% spread. Different host timings prevent claiming
a speedup from this comparison alone. A subsequent freeze stopped at L4:
public-2 token [sample 0, sequence 14, position 40] had a 2.125 logit deficit,
exceeding 2.0. Experiment 23 records this rejection. The fused prefill norms
were removed; the H100 freeze stage was not launched for that candidate.

The replacement candidate captures native prefill operations in a separate
CUDA graph during warmup. Every generation copies fresh prompt IDs into a
persistent input buffer and replays prefill to overwrite the active cache
prefix and select the first token. Native RMSNorm arithmetic is retained.
Prefill and decode graphs are recreated when batch, prompt length or output
length changes. The graph handoff test also checks prefill graph identity
across fresh prompts.

The accounting helper now reads the already-recorded incremental task budget:
$3.700718 historical spending plus the authorized $4 task allowance. Reservation
checks remain enforced and have regression coverage. Judge, guard, correctness,
latency and stability gates are unchanged.

Website public and official runs will be performed by the user. No release tag
is requested. The completed freeze result follows below.

## Accepted native-prefill graph freeze

All six workloads passed five H100 samples after the L4 public correctness check.

| Workload | TTFT ms | TTFT/native | TPOT ms | TPOT/native | Spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| public-0 | 18.588 | 0.8063 | 4.703 | 0.2430 | 0.53% |
| public-1 | 200.494 | 0.9906 | 6.014 | 0.2876 | 0.22% |
| public-2 | 189.453 | 0.9894 | 6.204 | 0.2781 | 0.29% |
| h-a | 52.550 | 0.9525 | 5.120 | 0.2620 | 6.75% |
| h-b | 193.010 | 0.9900 | 6.054 | 0.2825 | 0.24% |
| h-c | 188.015 | 0.9878 | 6.347 | 0.2762 | 0.22% |

Experiment 25. Freeze: `neokernel/results/freeze_prefill_graph/FREEZE.json`.
Submission archive: `neokernel/results/freeze_prefill_graph/submission.tar.gz`.
Archive SHA256: `d37970354a465173b0f4a4f451ec995e29a4a89f5c3e1a964d3d5657bf4a40cb`.
Task estimated spend: $1.733876 / $4.00.
The freeze records the pre-edit Git HEAD; its copied engine and archive hash
identify the tested candidate independently of the subsequent commit.
Website validation is pending and will be performed by the user. No additional
GPU runs are needed for this handoff.
