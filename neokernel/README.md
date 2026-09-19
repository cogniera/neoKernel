# neoKernel v1

An opt-in research harness around the untouched Dryft starter. Nothing under neokernel/ is submitted. No command commits, merges, or pushes. Auto and sweep keep winning source in the working tree and restore unsuccessful experiments from snapshots.

## Offline use

From the repository root:

```powershell
python -m neokernel --help
python -m neokernel guard
python -m neokernel log
python -m unittest discover -s tests -v
```

The guard and arithmetic tests use the standard library. Tensor replay tests run on CPU when torch and transformers==4.51.3 are installed; otherwise unittest reports them as skipped. No test needs credentials or contacts an API. Python 3.11 is the deployment target.

## Remote setup and commands

The following commands contact external services and are for later authorized use. Install neokernel/requirements.txt, authenticate Modal, and explicitly download the pinned weights using download-weights. The CLI never installs packages or downloads weights automatically. Configure BASETEN_API_KEY in the environment before auto. Do not put credentials inside engine/.

```powershell
python -m neokernel download-weights
python -m neokernel check --workloads public
python -m neokernel bench --workloads all --samples 3
python -m neokernel bench --workloads public --refresh-native
python -m neokernel profile --workload public-0
python -m neokernel gauge --workload public-0
python -m neokernel race --a HEAD --b current --workload public-0
python -m neokernel sweep --steps 3 --wire-rmsnorm --max-gpu-minutes 180
python -m neokernel auto --steps 1 --attended --max-gpu-minutes 300
python -m neokernel log --last 15 --kept
python -m neokernel freeze --out ./frozen-v1 --workloads all
```

Selections are public, hidden, all, or comma-separated workload names. Hidden means guessed shapes, not the private leaderboard workloads. check enforces output correctness, deadlines, and memory; bench additionally enforces latency, spread, and physics estimates. Native and candidate run sequentially in alternating order for each workload. Native medians are cached only within that container; --refresh-native repeats them. Latency gates use those local native medians. Official numbers appear only in calibration comparisons and host correction factors. Bench records child CPU time and parent-received pipe latency; --transport binary selects fixed-size token frames instead of flushed JSON lines.

Every remote invocation has a 3,600-second timeout. Search budgets conservatively reserve that entire possible allocation before dispatch and debit reported GPU seconds on completion. Failed or interrupted remote calls charge the full reservation because their actual usage is unknown. This intentionally refuses dispatch with fewer than 60 GPU minutes remaining. It is a dispatch budget, not a provider billing cap: idle time, image startup, and asynchronous cancellation can differ from reported time. No other work is scheduled automatically.

race uses the same secret prompt seed for both candidates. It streams them sequentially on one H100 to avoid timing contamination from GPU contention, then displays both streams and their profile bars. Display is sequence zero for batched workloads. It is a demo, not a correctness certificate or a keep decision.

The RMSNorm sweep hook is opt-in and initially staged in a temporary copy. It does not edit the starter merely by importing or installing this harness. Existing kernels can expose a literal TUNABLES dict with numeric lists in kernels/__init__.py; sweep replaces the assignment with scalar values without discarding other module content. A successful point becomes the live configuration. Preserve original ranges separately if repeated searches need them.

The agent returns a files mapping containing complete replacement contents. Omitted paths are unchanged; an empty string deletes only a kernel file. The main engine cannot be deleted. Paths are validated before writing, and Git generates the diff from before/after snapshots, including new and deleted files, for each log entry. No index or commit changes are made. On Ctrl-C, the in-flight live experiment is restored. Source snapshots are also persisted under results/snapshots/ for inspection after a hard process termination. Hard termination cannot execute Python cleanup; restore manually from that snapshot if needed.

freeze requires a new output directory, public correctness checks, and a five-sample benchmark for the selected workloads. It writes engine/ and FREEZE.json only after passing. The output is for review or a later submission; freeze does not submit it.

See [calibration status](../docs/CALIBRATION.md) and [design and limitations](../docs/DESIGN.md). External acceptance, calibration, and one live agent iteration remain pending.

The attended Modal dollar ceiling is $3 cumulative until an agent optimization passes the judge and is kept, then $6 cumulative. Baseline measurements do not raise the ceiling. Each GPU call reserves its maximum allowed cost before dispatch.
