# neoKernel

An automated research loop for Qwen3 inference. A model proposes a source change. The harness checks it, measures it and keeps or restores the candidate. Every attempt leaves a record.

**Version 1** marks the project as it stood on September 22, 2026. The main documentation covers only the automated harness loop. Manually authored engine optimizations are outside that narrative.

[Read the interactive documentation](docs/index.html) · [Version 1 notes](docs/v1.html) · [Experiment evidence](docs/v1-evidence.json)

## Open the documentation

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open [the local preview](http://127.0.0.1:8765/). The site has a rotatable 3D model of the loop, a step-through experiment diagram, a token replay example and a browser for the recorded automated trials. It is static HTML, CSS and JavaScript with no build step. It also opens directly from `docs/index.html`.

## The loop

1. Measure the current engine as a baseline.
2. Propose a bounded change using the source, profile and experiment history.
3. Validate replacement files and run the source guard.
4. Run unit tests with up to two model repair attempts, then public correctness checks.
5. Benchmark the candidate against the native reference.
6. Save the result and keep or restore the candidate.

A local keep needs every gate to pass, more than 1% aggregate throughput improvement and no selected workload more than 3% below baseline. The candidate cannot replace the judge or its tests.

## What the record shows

The Version 1 snapshot contains 27 automated entries: three baseline checks and 24 proposal attempts. These include failed and interrupted attempts. Baselines are measurements of the current engine, not improvements produced by the loop.

Trial 37 was kept locally and later reverted after external evaluation. Trial 39 failed local replay and was later marked kept after external evaluation. Trial 44 regressed and was reverted. The [evidence notes](docs/v1.html#evidence) preserve those distinctions and the limits of the recorded sources.

The external evaluator is Dryft. Its hidden workload score is separate from the local harness's public and proxy workload measurements. See the [engine contract](QWEN_ENGINE_CONTRACT.md) for evaluation rules.

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

Budget reservations can stop dispatch before the requested attempt completes. Use `--resume` to recover an interrupted loop. Kept changes are committed and merged locally. The loop never pushes. See [setup and recovery](neokernel/README.md) for details.

## Repository

| Path | Purpose |
| --- | --- |
| `engine/` | Inference engine and the kernels it imports. |
| `neokernel/` | Proposal loop, guard, judge, GPU orchestration and recovery. |
| `agent/` | Source packager and external evaluation client. |
| `tests/` | CPU regressions and GPU numerical checks. |
| `docs/` | Version 1 site, evidence snapshot and technical notes. |
| `experiments/` | Saved experimental source outside the engine archive. |

The site uses a checked-in evidence export so it works without the ignored raw results directory. [build_v1_evidence.py](docs/build_v1_evidence.py) regenerates that fixed snapshot when the local source log is available.

Earlier [design notes](docs/DESIGN.md), [measurement tables](docs/RESULTS.md) and [experiment log](docs/log.md) remain as historical records. They have broader scope than the Version 1 article and can describe earlier implementations. For current harness behavior, use the source linked from the Version 1 notes.

## Future work

Version 2 will focus on multiple agent swarms working in parallel, API provider independence and GPU provider independence. The longer term goal is an installable Python package for reproducible inference research. These are planned features. Check the [Future work section](docs/index.html#future) for more details and technical sources.
