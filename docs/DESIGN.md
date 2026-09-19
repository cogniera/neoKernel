# neoKernel v1 design

neoKernel is a research harness for whole-generation optimization of the pinned Qwen3 4B model. The submission remains the source under engine/. The harness packages that source using agent/package.py and never ships its judge, guard, logs, credentials, or agent to Dryft. The initial starter is unchanged. A frozen copy in neokernel/native_engine.py supplies reference timings after candidate changes.

## Components and trust boundary

```text
Local operator
  CLI -> source guard -> starter packager -> explicit Modal invocation
    |                                        |
    +-> results, snapshots, logs              v
    +-> numeric sweep or Baseten proposer   trusted parent judge
        scoped patch and keep/revert          | clocks, fresh prompts, JSON pipe
                                              v
                                        fresh candidate interpreter
                                        temporary engine extraction
                                              | token lists, memory report
                                              v
                                        kill process group, confirm exit
                                              |
                                              v
                                        native teacher-forced replay
                                              |
                                              v
                                        gates and structured report
```

The parent owns prompts, seeds, timestamps, replay, and decisions. The child starts with an isolated interpreter, a sanitized environment, offline model flags, no harness entry on sys.path, and a separate process group. Its stdout is redirected to stderr; a retained stream carries JSON records. JSON avoids deserializing executable pickle payloads. Source and archive lint run before model loading. Extraction rejects traversal, duplicate names, links, oversized archives, and non-source files. The parent bounds wire messages and enforces deadlines. Each workload constructs a fresh candidate, loads once, warms up once on a distinct prompt, then runs the requested samples. The child is killed and reaped before replay begins.

This is process isolation, not the full Dryft security sandbox. Parent and child currently use the same operating-system identity. No network namespace or seccomp policy is installed. The child reports its own allocator peak, and static Python lint cannot prove integrity against arbitrary obfuscated code or hostile native libraries. A production adversarial runner needs a separate UID, read-only mounts, no network, and OS-level process protections. Do not treat local gates as an attestation. The explicit download_weights setup function is separate from engine execution, and candidate environments omit provider credentials.

The AST guard follows direct aliases, restricts imports and reflective operations, blocks timers, event construction, unsafe OS calls, write-mode opens, environment writes, and assignments into protected libraries. There is one deliberate exception to the brief's blanket protected-module rule: literal False assignments to the two TF32 flags used by the unchanged starter. Requiring both starter acceptance and a ban on those exact assignments would be contradictory. Normal model.eval() is allowed; the Python eval builtin is forbidden. The guard is conservative, so a legitimate unusual kernel may need a reviewed lint enhancement.

## Measurement and correctness

The parent reader timestamps each JSON token message. TTFT is first arrival minus request start. TPOT is last minus first arrival divided by N-1. Total time ends at the final token, and a separate completion message verifies no extra yields, reports memory, and ensures generation terminates before its deadline. N=1 has zero TPOT. Bool, float, tensor, tuple, out-of-vocabulary IDs, excess yields, and early termination fail. Wire parsing, scheduling, and prompt serialization contribute small protocol overhead. Native reference timing uses the same path.

Replay concatenates each sequence's prompt and emitted tokens, forwards with use_cache=False, and examines positions S-1 through S+N-2. Argmax or a gap of at most 2.0 logits passes. Non-finite logits fail. The first bad position includes sample, sequence, and token indices; near_tie_count counts passing non-argmax positions. The native model is loaded independently once per Modal container, stays resident, and is never shared with the child. Tied parameters are counted once through model.parameters(). Keeping the reference resident consumes physical GPU capacity outside the child's allocator peak and must be measured during calibration.

The median total gives batch*N/median_total tokens/s. Spread is (maximum-minimum)/median. Latency compares medians with the native baseline. The run has no geomean when any selected workload fails. Guessed hidden shapes are explicitly separate from Dryft's unknown scoring set. A correctness-only run bypasses latency, spread, and physics gates; it still validates memory and execution deadlines. The unchanged native starter and candidate run sequentially, native first on even workload indices and candidate first on odd indices. Replay occurs after both processes exit. Native medians are cached per container, workload shape, sample count, and transport; client files and official measurements cannot supply gates.

Judge containers request 8 CPU cores and 32 GiB RAM. Parent and child log CPU count, cgroup cpu.max, and PyTorch thread count, then use eight PyTorch threads. The child records process CPU time around each generator step and a monotonic timestamp before sending each yield. The parent timestamps receipt independently; only parent timestamps determine measured latency. Child-to-parent transport latency is diagnostic. Both flushed JSON lines and unbuffered binary token frames are supported. CPU decode summaries exclude the prefill step.

## Physics estimates and profiles

For measured parameter payload W bytes and unique parameter count P:

```text
step_floor_s = W / 3.35e12
decode_floor_s = (N - 1) * step_floor_s
prefill_floor_s = 2 * P * batch * S / 6e14
floor_tps = batch * N / (prefill_floor_s + decode_floor_s)
measured_to_floor_ratio = measured_tps / floor_tps
```

Actual weight_bytes is deliberately pending. The native loader prints and the run JSON stores the observed value; no model was downloaded or loaded during code-only implementation. Populate it and residual protocol differences in CALIBRATION.md after authorized evaluation. Do not present the planning estimate near 2.4 ms as a measurement.

A sample below the estimated decode floor, or TTFT below the prefill floor, is flagged physics_violation and cannot be kept. A literal module-level SPECULATIVE = True relaxes the decode bound to N*step_floor_s/8, visibly recorded in each result, while leaving prefill unchanged. These are heuristic plausibility limits. Exact speculation can legitimately cross a per-token weight bound, and other hardware effects can make a theoretical bound imperfect.

Profiling measures one decode step after complete warmup and fresh prefill. It records CUDA kernel events, aggregates by name, and reports wall time, summed kernel time, and gap_ms = wall minus sum. A separate interval-union calculation reports uncovered time when kernels overlap. Profiling overhead is present, so profile timing is not used for benchmark gates. Byte estimates are null when kernel input/output ownership cannot be determined reliably; a generic kernel name alone does not identify a tensor shape. Gauge shows stacked kernel segments, grey gap, and the weight-floor marker. Overlapping kernels can yield a negative arithmetic gap, which is reported rather than silently rewritten.

## Search and persistence

Sweep enumerates or deterministically samples numeric choices from a literal TUNABLES dictionary without querying an LLM. Candidates are staged in a disposable tree; one judge_many invocation reuses the warm container and native model. Every candidate gets a one-workload correctness check followed by a two-sample benchmark. A passing improvement replaces the live tree only after search finishes and only if the operator has not changed that tree concurrently. The optional starter RMSNorm hook stages the guide's adapter; importing this package never wires it into the engine.

The agent uses the Baseten OpenAI-compatible endpoint only on an explicit auto command. It validates the chosen model slug, requests structured JSON where supported, falls back only on explicit schema-support errors, and strictly parses fields with one repair attempt. Transient 429/5xx errors receive bounded exponential retries with jitter. The static operator program and schema lead the prompt unchanged; the current engine, profile, diff, last 15 log lines, and summarized older coverage follow.

The loop measures the current baseline, scopes patches to engine.py and kernels/*.py, checks them in a temporary directory, applies source changes, runs guard, public correctness, and the selected three-sample benchmark. A keep requires every gate and greater than 1 percent geomean improvement. Five reverts at an item prohibit another attempt. Exact prompt-lookup speculation requires every earlier playbook item kept. A finally block restores rejected or interrupted experiments, and snapshots remain on disk for hard-crash recovery.

The loop itself never commits, merges, or pushes. The operator authorized one harness commit on main followed by one experiment on a codex/ branch; candidate edits remain uncommitted and are kept or restored by the judge. git apply checks plain text patches without --3way and without touching the index. Version records include the existing SHA and an archive digest because SHA alone does not identify uncommitted candidates. All run JSON, native records, profiles, snapshots, and append-only experiment logs live under ignored results/.

Freeze evaluates the current archive, requires public correctness and five-sample benchmark success, checks that source stayed unchanged, and copies it into a new output directory with FREEZE.json. It never publishes or submits the artifact.

## Design decisions and their sources

The following sources and research claims were specified in the supplied build brief. Links were not fetched or independently verified because the user requested code only and no API calls. The implementation uses the stated ideas; it does not claim reproduction of those papers' results.

[Dryft contract](../QWEN_ENGINE_CONTRACT.md), [optimization guide](../OPTIMIZATION_GUIDE.md), and [challenge docs](https://htn.dryft.ai/docs) supply generation shape, local checkpoint loading, fresh workload processes, warmup, latency and memory gates, median scoring, and replay on the emitted prefix. The guide and starter RMSNorm supply the BF16 cast-placement rule. The private corpus prompt derivation and full OS sandbox cannot be replicated from the public contract alone.

[AutoKernel, arXiv 2603.21331](https://arxiv.org/abs/2603.21331) and [its repository](https://github.com/RightNow-AI/autokernel) motivate iterative keep/revert experiments, the 1 percent threshold, a five-revert move-on rule, and correctness-aware logs. The brief describes its scope as isolated kernels, with cross-kernel fusion future work; neoKernel instead evaluates the entire generation step. The no-commit instruction changes Git keep/revert into file snapshots.

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) motivates a small editable scope, frozen evaluation, and program.md as the operator's policy lever. The brief identifies its train.py versus prepare.py separation; neoKernel uses engine.py and kernels/ versus the harness.

[InferenceBench, arXiv 2607.20468](https://arxiv.org/abs/2607.20468) is cited by the brief for premature convergence, median exploration of one configuration, a numeric search outperforming agents, and evaluator integrity violations under optimization pressure. The corresponding design choices are LLM-free numeric sweep, explicit playbook coverage, and independent integrity checks. Those comparative claims remain unverified here.

[KernelGuard, Lacuna / Tiptree Systems](https://lacuna.tiptreesystems.com/work/we-let-agents-compete-and-they-tried-to-cheat-kernelguard-defending-gpu/wrk_8addb4ee547c113ab8c7edb2c64b9408) motivates defenses against cached output replay, timer monkeypatching, and evaluator mutation. neoKernel takes conservative source lint, parent-owned timing and replay, process separation, and physics plausibility bounds. These do not amount to a security proof.

[Speed-of-Light Guidance, arXiv 2603.29010](https://arxiv.org/abs/2603.29010) motivates providing roofline estimates and the measured distance from them to an optimization agent. neoKernel exposes per-step bandwidth and prefill compute estimates in reports and profiles, and per-kernel HBM fractions when byte estimates are available. Prioritization follows the largest measured gap rather than a generic fusion preference.

[Hazy Research, Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) motivates measuring time between small-model decode operations as a first-class quantity. neoKernel exposes gap_ms alongside individual kernel duration, with an interval-union correction available for overlap.

[AutoMegaKernel, arXiv 2606.09682](https://arxiv.org/abs/2606.09682) motivates sequence-level verification against the Hugging Face model for whole-model implementations. neoKernel uses Dryft's teacher-forced margin rule rather than claiming strict token identity when a valid near tie occurs.

[Meta KernelAgent](https://github.com/meta-pytorch/KernelAgent) motivates per-kernel comparisons against a PyTorch reference before composition and an end-to-end check after integration. program.md explicitly requires this discipline for newly generated kernels. The agent must supply those comparisons; a source lint pass cannot establish their numeric equivalence.

## Acceptance status

Local tests cover guard rules, archive attacks, gate and floor arithmetic, output validation, patch boundaries, snapshot restoration, tunable staging, logs, budgets, profile aggregation, paired native caching, and pipe frames. CPU tensor tests use a tiny two-layer Qwen3 with hidden width 64. All 49 tests pass on Python 3.11 with CPU torch 2.5.1 and transformers 4.51.3, without skips. Modal version verification and L4 public correctness have passed. CALIBRATION.md records the latest H100 results; later profile and agent validation remain pending until calibration passes.
