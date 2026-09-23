# neoKernel

An automated research loop for Qwen3 inference. A model proposes a source change. The harness checks it, measures it and accepts or restores the candidate. Every attempt leaves a record.

**Version 1** marks the project as it stood on September 22, 2026. The documentation covers the research loop and selected local measurements from direct engine development and automated harness proposals.

[Read the interactive documentation](docs/index.html) · [Download run records](docs/log.md)

## Open the documentation

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open [the local preview](http://127.0.0.1:8765/). The site has a rotatable 3D model of the loop, a step-through experiment diagram, a token replay example and a score progression with one baseline. It is static HTML, CSS and JavaScript with no build step. It also opens directly from `docs/index.html`.

## The loop

1. Measure the current engine as a baseline.
2. Propose a bounded change using the source, profile and experiment history.
3. Validate replacement files and run the source guard.
4. Run unit tests with up to two model repair attempts, then public correctness checks.
5. Benchmark the candidate against the native reference.
6. Save the result and accept or restore the candidate.

Accepting a candidate requires every gate to pass, more than 1% aggregate throughput improvement and no selected workload more than 3% below baseline. The candidate cannot replace the judge or its tests.

## What the record shows

The documented results are selected local six-workload measurements. Direct development established the engine foundation, and the automated harness subsequently explored proposals on top of it. The highest locally measured candidate reached 796.0 tok/s, an 8.92% improvement over the engine used as its immediate baseline. Individual trials were not monotonic. See [the results from the runs](docs/index.html#evidence) or download the [run records](docs/log.md).

## Run the harness

Use Python 3.11 from the repository root. Start with the offline commands:

```powershell
py -3.11 -m neokernel guard
py -3.11 -m neokernel report
```

The automated loop also needs the dependencies in [neokernel/requirements.txt](neokernel/requirements.txt), a configured Modal account, the pinned checkpoint in its volume and `BASETEN_API_KEY` in the environment. The working tree must be clean on `main`.

```powershell
# One proposal attempt. Uses paid GPU and model inference.
py -3.11 -m neokernel auto --steps 1 --workloads all --max-gpu-minutes 60
```

Budget reservations can stop dispatch before the requested attempt completes. Use `--resume` to recover an interrupted loop. Accepted changes are committed and merged locally. The loop never pushes. See [setup and recovery](neokernel/README.md) for details.

## Repository

| Path | Purpose |
| --- | --- |
| `engine/` | Inference engine and the kernels it imports. |
| `neokernel/` | Proposal loop, guard, judge, GPU orchestration and recovery. |
| `agent/` | Source packager and external evaluation client. |
| `tests/` | CPU regressions and GPU numerical checks. |
| `docs/` | Version 1 site, evidence snapshot and technical notes. |
| `experiments/` | Saved experimental source outside the engine archive. |

Earlier [design notes](docs/DESIGN.md), [measurement tables](docs/RESULTS.md) and [experiment log](docs/log.md) remain as historical records. They have broader scope than the main article and can describe earlier implementations.

## Future work

Version 2 will focus on multiple agent swarms working in parallel, API provider independence and GPU provider independence. The longer term goal is an installable Python package for reproducible inference research. These are planned features. Check the [Future work section](docs/index.html#future) for more details and technical sources.
