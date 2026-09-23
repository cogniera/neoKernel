# neoKernel local research loop

Run orchestration on the laptop with Python 3.11. GPU work in auto and sweep is restricted to Modal judge calls returning JSON. The engine archive contains only engine source. The harness never pushes.

## Overnight commands

```powershell
py -3.11 -m unittest discover -s tests -v
py -3.11 -m neokernel sweep
py -3.11 -m neokernel auto --steps 30
# After a killed process or sleep interruption:
py -3.11 -m neokernel auto --steps 30 --resume
```

Sweep defaults to public-0 and public-2, with ten numeric coordinate trials over scalar TUNABLES. List-valued TUNABLES still support Cartesian search. Auto defaults to all six local workloads. Local h-a/h-b/h-c are proxy shapes, not disclosed official hidden shapes. Neither command pauses for approval; --attended is retained only for command-line compatibility.

Both commands require a clean main branch before remote dispatch. Every trial owns exp/<id>. Only the judge's passing result with geomean improvement strictly greater than 1% against the matching workload baseline, and no workload more than 3% below its baseline, permits a keep. A keep commits scoped engine files after a credential scan, fast-forward merges main, tags kept-<id>, writes results/best.json and copies results to results_backup/<id>/. Results live under neokernel/; neither results nor backups are committed.

Between guard and the public correctness check, auto runs the kernel and engine unit tests (tests/test_hand_rolled_kernels.py, tests/test_hand_rolled_handoff.py and tests/test_hand_rolled_corpus.py) on L4 against the candidate. If they fail, the same model receives its own files and the test output and returns corrected complete files, which are applied, guarded and retested; after two failed repair turns the experiment is reverted with the note `failed after 2 repairs`. Only a candidate that passes the unit tests reaches check and the judge. Each log line records `repairs` and `first_error`; raw test output is under results/unit_tests/<id>-<turn>.json and repair proposals under results/proposals/<id>-repair<turn>.json. Each unit-test call reserves an L4 call like check; a passing run of the kept engine took about 206 GPU seconds.

Reverted candidates are preserved under results/snapshots/<id>/engine/, with original files under before/ and a diff in log.jsonl. The durable transaction journal makes log finalization idempotent. --resume snapshots and discards an interrupted, unmerged experiment; if the keep merge already happened, it completes the tag/log/best/backup instead of undoing that accepted commit. Unknown changes outside engine are preserved and block recovery. A process lock prevents simultaneous loops.

Candidate correctness, latency, memory, timing-spread and other gate failures are ordinary reverts. A harness exception writes results/CRASH.txt and leaves its branch, source and journal in place. A clean budget stop snapshots and reverts an unjudged candidate. No gate is relaxed.

The first command creates results/night_budget.json with an $8 incremental ceiling shared by sweep, auto, Modal and Baseten. Reservations are persisted before dispatch. Unresolved reservations remain charged after termination. Each L4/H100 call reserves 900 seconds plus startup and 60 seconds idle; containers scale down after 60 idle seconds. Baseten uses conservative input bounds and a 16,384-token output cap, then settles against reported usage at GLM-5.2's published $1.40/M input and $4.40/M output rates. These are conservative local estimates, not a provider-side invoice cap. Dispatch stops early if its reservation cannot fit. The ledger is not reset on resume.

Baseten 429/5xx responses use exponential backoff; persistent API failures pause five minutes and retry. Structured-output incompatibility falls back to JSON mode. Set BASETEN_API_KEY only in the environment, never in source. The credential scan checks the complete index before every experiment commit.

An hourly local worker appends time, auto steps, kept ids, best geomean and spend to results/NIGHT.md. results/MORNING.md contains the final log count, keeps, best record, spend by tier, crash text, saved power settings and operator-only freeze/push commands. GPU tests are skipped locally; CPU tests run without network calls.

```powershell
py -3.11 -m neokernel guard
py -3.11 -m neokernel report
py -3.11 -m neokernel log --last 15
py -3.11 -m neokernel freeze --out neokernel/results/review_freeze --workloads all
```

Freeze performs public L4 correctness followed by five H100 samples per selected workload. Its output directory must not already exist. It never submits or pushes.
