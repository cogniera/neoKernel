# Hand-rolled decode

Status: the original hand-rolled candidate failed its final h-c TTFT freeze
(log #20). The later native-prefill graph candidate passed all six local
workloads at five samples each (log #25), then scored 670.7 tok/s in official
Dryft run `1bf05baa` on commit `012646d`. The earlier failures remain failures.
Triton decode attention is selected. The chronology below separates those
stages; the final subsection records the successful prefill follow-up.

Sources: [log export](log.md),
[`freeze_prefill_graph/FREEZE.json`](../neokernel/results/freeze_prefill_graph/FREEZE.json),
and [official run records](../neokernel/results/dryft_runs.json).

## Implementation

Decode reads loaded weight tensors directly. QKV and gate/up projections are
packed during construction; the original Transformers prefill modules retain
identical contiguous views into those allocations. Both residual boundaries,
per-head Q/K RMSNorm, BF16 intermediate casts, absolute RoPE, head grouping,
full visible attention, tied embedding/head, and lowest-index argmax are retained.

Prefill still follows the kept causal Transformers path. It writes into the
same cache used by the hand-rolled step. The cache exposes logical BHSD shape
with BHSD or BSHD backing storage. The Triton attention path reads a device
position; grouped SDPA uses a persistent prefix mask and four query rows per
KV head. No K/V head replication is needed.

The decode graph uses persistent activations and outputs. Triton specializations
are exercised before capture during warmup. SDPA's output/workspace belongs to
the CUDA graph's private memory pool. The token copy, position advance, and one
host list conversion remain outside the graph. Each generate resets positions
and cache visibility, and emits exactly the requested number of steps.

`engine/kernels/__init__.py` contains the five CONFIG switches and 14 scalar
TUNABLES. `neokernel/program.md` describes their alternatives and constraints.
The accepted snapshot is under `neokernel/results/freeze_prefill_graph/engine/`.

## Numerical evidence

The second attended L4 unit run passed all six tests, including 32 configuration
combinations at each of batches 1, 4, and 16. It used Triton 3.1.0 and
Transformers 4.51.3. Every comparison prints maximum absolute difference and
argmax agreement in the saved report.

| Component | Comparisons | Maximum absolute difference | Minimum per-case argmax agreement |
| --- | ---: | ---: | ---: |
| Hidden RMSNorm and rounded residual | 6 | 0 | 100% |
| Q/K head norm and RoPE | 96 | 0 | 100% |
| SiLU product | 6 | 0 | 100% |
| Decode attention | 48 | 0.00390625 | 93.75% |
| Complete decoder layer | 96 | 0.03125 | 75% |

These argmax statistics concern activation coordinates, not vocabulary tokens.
They do not replace the unchanged judge's teacher-forced replay. The first
attempt found a grouped-SDPA noncontiguous-view bug; copying into a matching
preallocated grouped destination fixed it. The failed report is preserved.

The integrated source then passed all seven GPU test methods. Across 48
full-model handoff checks (three public shapes, two attention paths, two cache
layouts, two fresh prompts, prefill and first decode), every emitted token was
at native's maximum logit, including exact ties. Maximum complete-logit
difference was 0.1875. Maximum observed load plus warmup was 20.191 seconds,
below the requested local 120-second target. Tests also check graph/address
reuse and exact output counts during warmup.

The offline suite passed 54 tests with six GPU tests skipped. It checks that
packed shared weights preserve prefill logits exactly and that both cache
layouts preserve causal prefill and reset across prompts. Source guard and
whitespace checks pass. The starter-only adapter test now uses a preserved
starter fixture. No judge, guard, or gate was changed.

## Artifacts and remaining measurements

Reports are under `neokernel/results/`:

- `hand_rolled_unit_tests_attempt1.json`: failed first validation.
- `hand_rolled_unit_tests_attempt2.json`: passing pre-integration validation.
- `hand_rolled_integration_tests.json`: passing integrated-source GPU tests.
- `hand_rolled_budget.json`: task's historical-spend offset and $4 limit.
- `hand_rolled_before/engine/`: preserved kept engine for paired profiling.

Log entries use proposer `codex`, item `hand_rolled_decode`. Entries #13 and
#14 record staged validation and integration; neither declares a kept benchmark.
Entry #15 records passing integrated L4 tests and public correctness. Public
near-tie counts were 1 / 0 / 6, with no incorrect position. Total estimated task
spending after these stages is $0.1638 of $4.00. The public check report is
`hand_rolled_public_check.json`. L4 throughput is not an H100 performance result.
The sections below record the H100 comparison, full bench, profiles, and failed
final freeze, including the calibration snapshots used for estimates.

## H100 attention comparison

Three samples per public workload, with freshly paired native measurements.
Both implementations passed every gate. Triton had the higher geometric mean
of public throughput: 483.48 versus 438.24 tok/s. This is a public local comparison,
not a hidden-workload or official leaderboard score. Log entries #16/#17 contain
the Triton/SDPA results; neither marks the engine frozen or kept.

| Workload | Triton tok/s | Triton TPOT ms | Triton TTFT/native | Triton spread | Dryft-equivalent tok/s estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| public-0 | 174.16 | 4.987 | 1.00 | 1.19% | 264.65 |
| public-1 | 318.76 | 6.243 | 1.02 | 0.90% | 405.54 |
| public-2 | 2035.73 | 6.368 | 1.03 | 0.63% | 2490.19 |

Direct comparison from [Triton](../neokernel/results/hand_rolled_triton.json) and [grouped SDPA](../neokernel/results/hand_rolled_sdpa_grouped.json):

| Workload | Triton tok/s | Grouped SDPA tok/s | Triton TPOT ms | Grouped SDPA TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| public-0 | 174.16 | 151.17 | 4.987 | 5.904 |
| public-1 | 318.76 | 273.05 | 6.243 | 8.433 |
| public-2 | 2035.73 | 2039.07 | 6.368 | 6.351 |

Batch-1 TPOT is only 0.013 ms below the requested 5 ms target in this run; the
full benchmark and freeze must establish whether it holds across samples.
Dryft-equivalent values use the accepted host calibration and are estimates,
never inputs to gates. Total estimated task spending after comparison is
$0.4069 of $4.00. Exact per-run calibration snapshots are stored alongside
`hand_rolled_triton.json` and `hand_rolled_sdpa_grouped.json`.

## Full six-workload benchmark: attempt 1

Log #18 records a failed benchmark, not a kept result. All six workloads passed
teacher-forced correctness. h-c median TTFT was 201.442768 ms against native's
191.495673 ms: ratio 1.051944, above the unchanged local limit of 1.05. The
allowed median was 201.070457 ms, so the miss was 0.372311 ms. Its three candidate
TTFT samples were 201.443, 203.121, and 200.304 ms. This overlap with the threshold
allows timing variation as an explanation but does not prove it; the failure
remains a failure. No prefill code or gate was changed to hide the result.

| Workload | tok/s | TTFT ms | TTFT/native | TPOT ms | Spread | Correctness | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| public-0 | 190.02 | 22.694 | 1.03832 | 4.700 | 2.13% | pass | pass |
| public-1 | 326.50 | 206.533 | 1.01261 | 5.984 | 3.65% | pass | pass |
| public-2 | 2088.94 | 197.192 | 1.02154 | 6.167 | 0.80% | pass | pass |
| h-a | 351.65 | 55.964 | 0.98532 | 4.891 | 1.24% | pass | pass |
| h-b | 873.69 | 201.387 | 1.02123 | 6.101 | 0.45% | pass | pass |
| h-c | 3391.07 | 201.443 | 1.05194 | 6.389 | 0.37% | pass | latency_limit |

Public Dryft-equivalent throughput estimates were 199.20 / 325.95 / 2089.36 tok/s
using that run's paired host factors. Maximum H100 load plus warmup was 9.075 s.
The full failed report and calibration are preserved as
`hand_rolled_full_bench_attempt1.json`; the original timestamped run and log also
remain. Total estimated task spending is $0.6154 of $4.00. Further GPU work needs
attended approval because the approved sequence was conditional on no failures.

## Five-sample h-c rerun

The operator approved a five-sample h-c rerun without changing the engine, then
profiles and a five-sample all-workload freeze. The rerun passed every local gate:
TTFT/native 1.046791, TTFT 196.610 ms, TPOT 6.233 ms, throughput 3476.52 tok/s,
and spread 0.353%. Correctness passed with 45 near ties. Log #19 and
`hand_rolled_bench_h-c.json` preserve the result. The previous failure remains
recorded; no gate was relaxed. Task spending after this rerun is $0.6947.

## Before/after H100 profiles

The before snapshot is the #11 graph baseline from the start of this task;
the after snapshot is the #18 hand-rolled decode stage. Each trace
captures a warmed decode step. Profiling overhead is included in wall time and
gap, so these are not substitutes for streamed TPOT measurements.

| Workload | Version | Kernels | Kernel ms | Gap ms | Graph launches | Host synchronizations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| public-0 | before | 2292 | 8.070 | 6.311 | 1 | 1 |
| public-0 | hand-rolled | 439 | 4.279 | 2.369 | 1 | 1 |
| public-2 | before | 2400 | 17.457 | 6.004 | 1 | 1 |
| public-2 | hand-rolled | 475 | 5.810 | 2.625 | 1 | 1 |

Sources: [B1 before](../neokernel/results/hand_rolled_profile_before_public-0.json),
[B1 after](../neokernel/results/hand_rolled_profile_after_public-0.json),
[B16 before](../neokernel/results/hand_rolled_profile_before_public-2.json),
[B16 after](../neokernel/results/hand_rolled_profile_after_public-2.json).
These are separate captures from the earlier 8.684 ms #11 diagnostic in
GRAPH_DIAGNOSIS.md; neither capture replaces the other.

Both measured shapes are below the requested 600-kernel target. The single host
synchronization is `cudaStreamSynchronize` from list conversion. Full traces are
in `hand_rolled_profile_{before,after}_{public-0,public-2}.json`. Estimated task
spending after all four profiles is $0.8107. The unchanged CLI's five-sample
freeze with `--workloads all` was run next; its result is below.

## Final five-sample all-workload freeze

Command: `py -3.11 -m neokernel freeze --out neokernel/results/freeze_hand_rolled --workloads all`.
Public L4 correctness passed, followed by five H100 samples on all six workloads.
The freeze failed h-c TTFT; every other workload and all correctness, memory,
load/warmup, timeout, physics, and sample-spread checks passed.

| Workload | tok/s | TTFT ms | TTFT/native | TPOT ms | Spread | Near ties | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| public-0 | 173.54 | 30.895 | 1.023280 | 4.952 | 3.68% | 0 | pass |
| public-1 | 316.42 | 207.674 | 1.011487 | 6.350 | 0.68% | 1 | pass |
| public-2 | 1999.92 | 199.448 | 1.023157 | 6.482 | 1.61% | 21 | pass |
| h-a | 331.15 | 57.396 | 1.005905 | 5.226 | 3.88% | 0 | pass |
| h-b | 848.37 | 200.888 | 1.016128 | 6.396 | 1.41% | 8 | pass |
| h-c | 3310.16 | 202.323 | 1.051394 | 6.602 | 7.39% | 40 | latency_limit |

h-c native median TTFT was 192.433439 ms. The 1.05 local bound was 202.055111 ms;
candidate median was 202.323444 ms, a miss of **0.268333 ms**. Its candidate TTFT
samples were 245.948, 201.654, 202.323, 201.918, and 202.924 ms. The ratio would
be below the contract's 1.10 Dryft gate, but the requested local freeze did not
pass, so no commit was made. The local gate was not changed.

Maximum candidate memory was 12.114% of H100 capacity. Maximum load plus warmup
was 11.491 s. Freeze public-0 and public-2 median TPOT meet the requested
5 ms / 7 ms targets; the profiles meet the 600-kernel target. h-c TTFT lacks
reliable headroom under the local gate.

| Public workload | Dryft-equivalent tok/s estimate | TTFT ms estimate | TPOT ms estimate |
| --- | ---: | ---: | ---: |
| public-0 | 249.52 | 22.522 | 3.421 |
| public-1 | 445.53 | 205.119 | 4.151 |
| public-2 | 2464.77 | 197.091 | 5.197 |

These are baseline-ratio estimates using the freeze's paired calibration,
not official results or gate inputs. No eligible aggregate score is claimed.

The full failed freeze, L4 check, and calibration are preserved in
`neokernel/results/freeze_hand_rolled_failed.json`. H100 raw run:
`neokernel/results/runs/20260919T215914_852095_20e929bc2907.json`.
Archive SHA256:
`83bca8f8effa2adfaa9f20497975857e20c5780f58f2761f845b07f55ad2c3a5`.

Final experiment log summary (complete row in `neokernel/results/log.jsonl`):

```json
{"id":20,"proposer":"codex","item":"hand_rolled_decode","guard":"pass","kept":false,"geomean_tps":null,"note":"Final five-sample all-workload freeze failed h-c TTFT; all correctness checks passed; no commit or push."}
```

Total estimated task spending, excluding the preserved $3.7007 historical
offset, is **$1.166325 of $4.00**. GPU work stopped after the failed final freeze.
The subsequent authorization to commit and push supersedes the original
freeze-before-commit restriction and no-push instruction. It does not change
the measurements or the historical log #20 above.

Triton API references checked against the pinned release:
[standard operations](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/standard.py),
[JIT launch interface](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/runtime/jit.py).

## Resolution of the h-c TTFT episode

The sequence was failure at 1.051944x native (#18), an isolated five-sample
pass at 1.046791x (#19), then another all-workload freeze failure at
1.051394x (#20). Passing the isolated rerun did not establish repeatable
headroom. These are local ratios; none should be substituted for a website
native measurement. Source: [log.md](log.md) and the saved reports above.

A later attempt to fuse prefill norms passed one check but failed a fresh L4
public-2 replay at a 2.125 logit deficit (#23). It was removed. The retained
change captures native prefill operations in a CUDA graph and preserves native
normalization arithmetic. The full #25 freeze passed with h-c TTFT 188.015 ms,
ratio 0.9878; all six TTFT ratios were at most 0.9906. B1 TTFT was 18.588 ms,
ratio 0.8063. Source: [freeze record](../neokernel/results/freeze_prefill_graph/FREEZE.json)
and [prefill follow-up](PREFILL_LATENCY_FIX.md).

The original kernel and handoff unit reports describe the hand-rolled decode
integration before this prefill change. They are not a claim that every GPU
unit test was rerun after prefill capture; the final candidate's end-to-end
correctness evidence is the L4 check and five-sample H100 freeze.
Numerical source reports: [unit attempt 2](../neokernel/results/hand_rolled_unit_tests_attempt2.json)
and [integrated handoff](../neokernel/results/hand_rolled_integration_tests.json).

Official run `1bf05baa` subsequently passed with public TTFT 19.68 / 201.59 /
190.75 ms, TPOT 4.75 / 6.06 / 6.16 ms and throughput 191.6 / 328.6 / 2104.4
for B1 / B4 / B16. Its hidden-workload score was 670.7 tok/s. The run page
identifies commit `012646d`, not the original failed #18 candidate. Source:
[official record](../neokernel/results/dryft_runs.json). Cumulative spending
by tier is generated in [RESULTS.md](RESULTS.md); the $1.1663 figure earlier
in this document is the historical task subtotal at the failed freeze.
