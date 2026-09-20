# neoKernel

neoKernel is a local harness and Qwen3 4B inference engine for the Dryft benchmark, with native, graph-based and hand-rolled stages recorded in [the official results](neokernel/results/dryft_runs.json).
The saved experiments compare complete generation, check emitted tokens against an independent native model, and record accepted and rejected changes in [the experiment log](docs/log.md).
The latest accepted local candidate uses hand-rolled decode and native prefill CUDA graphs, with its five-sample freeze saved in [FREEZE.json](neokernel/results/freeze_prefill_graph/FREEZE.json).

## Recorded numbers

| Engine stage | Official run | Hidden-workload score, tok/s | Rank at capture | Public B1 / B4 / B16, tok/s |
| --- | --- | ---: | ---: | --- |
| Native starter | 4877cddd | 217.3 | 12 | 57.0 / 150.0 / 671.2 |
| #11 static cache and decode graph | 69b39b11 | 288.5 | 42 | 99.7 / 170.1 / 779.5 |
| #18 hand-rolled lineage, with native prefill graph | 1bf05baa | 670.7 | 38 | 191.6 / 328.6 / 2104.4 |

Source: [hand-kept official records and run-page links](neokernel/results/dryft_runs.json).
Ranks are historical snapshots. The last row is commit `012646d`, Dryft submission #5 and local freeze #25; it is not the failed local benchmark at log #18.

| H100 decode profile | B1 kernels / kernel ms / gap ms | B16 kernels / kernel ms / gap ms |
| --- | --- | --- |
| #11 baseline | 2292 / 8.070 / 6.311 | 2400 / 17.457 / 6.004 |
| #18 hand-rolled decode | 439 / 4.279 / 2.369 | 475 / 5.810 / 2.625 |

Sources: `neokernel/results/hand_rolled_profile_{before,after}_{public-0,public-2}.json`, linked individually in [RESULTS.md](docs/RESULTS.md).
These are instrumented local decode profiles, not website latency measurements.
The subsequent prefill graph freeze passed all six local workloads at five samples each; its worst TTFT/native ratio was 0.9906. Source: [freeze record](neokernel/results/freeze_prefill_graph/FREEZE.json).

## Commands

Run from the repository root with Python 3.11. GPU commands require the dependencies in [neokernel/requirements.txt](neokernel/requirements.txt), an authenticated Modal account and the pinned checkpoint already present in the configured Modal volume. The auto command also requires `BASETEN_API_KEY` in the environment. Do not put credentials in source or result files.

```powershell
# Offline source validation and documentation
py -3.11 -m neokernel guard
py -3.11 -m neokernel report

# Public correctness on L4
py -3.11 -m neokernel check --workloads public

# Paired native timing on H100
py -3.11 -m neokernel bench --workloads all --samples 5 --refresh-native

# One warm decode step on L4; do not compare directly with H100 profiles
py -3.11 -m neokernel profile --workload public-0

# One attended structural proposal through GLM
py -3.11 -m neokernel auto --attended --steps 1 --workloads all --max-gpu-minutes 5

# Optional numeric search without a model API
py -3.11 -m neokernel sweep --steps 1 --workloads public --max-gpu-minutes 5

# Public L4 correctness, then five H100 samples per selected workload
py -3.11 -m neokernel freeze --out neokernel/results/freeze_candidate --workloads all
```

GPU commands spend money and enforce the ledger's reservation checks. `--max-gpu-minutes` is an additional bound for auto and sweep, not a dollar estimate. Freeze requires a new output directory and does not submit to Dryft. Command definitions are in [cli.py](neokernel/cli.py); measured invocations and spending are recorded in [log.md](docs/log.md) and [RESULTS.md](docs/RESULTS.md).

`report` reads saved files only. Edit `neokernel/results/dryft_runs.json` by hand when a new official result is available, then regenerate both Markdown tables. Bulk raw results are ignored by Git; a fresh clone retains the published tables and the versioned official record but needs the local raw artifacts to reproduce every local table.

## Repository layout

| Path | Purpose |
| --- | --- |
| `engine/` | Submitted Engine and imported Python/Triton kernels. |
| `neokernel/` | Local judge, guard, Modal commands, proposal loop, numeric sweep and report generator. |
| `neokernel/results/` | Local logs, raw runs, profiles, spend ledger and frozen artifacts; official records are in `dryft_runs.json`. |
| `agent/` | Dryft API client and source archive packager. |
| `tests/` | CPU regressions and GPU numerical checks. |
| `experiments/` | Saved experiment source outside the submitted engine. |
| `docs/` | Results, design, calibration, numerical evidence and the HTML overview. |

The [freeze record](neokernel/results/freeze_prefill_graph/FREEZE.json) identifies the measured engine archive. Harness and agent files are excluded from that archive.

## Documentation

- [Results and spending](docs/RESULTS.md)
- [Experiment log](docs/log.md)
- [Design and proposal format](docs/DESIGN.md)
- [Hand-rolled decode measurements](docs/HAND_ROLLED_DECODE.md)
- [Prefill latency follow-up](docs/PREFILL_LATENCY_FIX.md)
- [Calibration and host differences](docs/CALIBRATION.md)
- [Graph diagnosis](docs/GRAPH_DIAGNOSIS.md)
- [Judge review](docs/JUDGE_REVIEW.md)
- [HTML overview](docs/index.html), a separately maintained historical narrative; use RESULTS.md for the generated measurement tables.

## Sources

- [Official Dryft runs, including page URLs and measurement provenance](neokernel/results/dryft_runs.json).
- [Local results, profiles, floors and spend source links](docs/RESULTS.md) and [append-only log export](docs/log.md).
- [Engine contract](QWEN_ENGINE_CONTRACT.md) and [Qwen implementation guide](OPTIMIZATION_GUIDE.md).
- [InferenceBench](https://arxiv.org/abs/2607.20468), motivation for separating structural exploration and numeric search; this project does not reproduce its paper results.
- [Design references](docs/DESIGN.md#design-decisions-and-their-sources), including AutoKernel, autoresearch, KernelGuard, Speed-of-Light Guidance, Hazy Research and KernelAgent.
