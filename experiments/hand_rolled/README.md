# Hand-rolled decode development record

Final status: decode and measurement work completed; the final five-sample
all-workload freeze failed h-c TTFT at 1.051394 versus the local 1.05 gate.
All correctness checks passed. Profiles show 439/475 kernels at batches 1/16;
freeze TPOT is 4.952/6.482 ms. The operator subsequently authorized committing
and pushing this candidate despite the local freeze failure. Total task spend is
$1.1663 of $4. Full results are in `docs/HAND_ROLLED_DECODE.md`; the sections
below preserve the development sequence.

The user requires kernel comparisons before integration. The copies here record
the pre-integration implementation validated on L4. Active source now lives in
`engine/`; unit tests and the validation runner target that active source.

First validation (requires attended GPU approval):

```powershell
py -3.11 -m neokernel.hand_rolled_validation
```

This uses the existing pinned Modal image, L4 options, runtime check, and spend
ledger. It starts no benchmark, downloads no model, and alters no judge or gate.
The test report is saved to `neokernel/results/hand_rolled_unit_tests.json`.
Maximum reservation for this first call is approximately $0.383 under the existing
ledger's conservative 900-second timeout plus startup and idle allowance.
This task has its own $4 incremental limit; historical spend is not charged twice.

Tests cover random BF16 norms, rounded residuals, normalized Q/K and absolute RoPE,
direct K/V writes in both layouts, SiLU with the intermediate BF16 rounding,
grouped attention against native repeated-KV SDPA, poisoned inactive slots,
and a complete layer for batches 1/4/16 across all 32 switch combinations.
Every comparison prints maximum absolute error and argmax agreement.

Pending after successful kernel validation: integrate loaded weights and static
buffers into Engine; test causal prefill handoff and graph replay/reset on the
three public shapes; compare both attention implementations on H100; run guard,
L4 check, full H100 bench, before/after profiles, codex/hand_rolled_decode log,
freeze, staged credential-pattern scan, and conditional main commit. Never push.

The implementation uses Triton 3.1.0 APIs; source references:
https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/standard.py
https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/runtime/jit.py

No complete-engine performance or correctness result is claimed before its
integration validation and harness benchmark.

## First attended validation

L4 attempt 1 completed in 15.613 seconds of reported GPU work; conservative
allocation estimate was 32.50 seconds ($0.0129). Four GPU test methods passed.
RMSNorm (including residual), Q/K norm plus RoPE, and SiLU were exact on all
tested inputs. Attention maximum absolute error was 0.00390625; per-case argmax
agreement ranged from 93.75% to 100% (these are attention coordinates, not tokens).
Two initial one-layer comparisons had maximum error 0.03125 and 100% argmax
agreement, then the grouped SDPA fallback failed because its output could not
be flattened with `view_as`.

The fallback now copies into a preallocated destination view with matching
`[B,8,4,128]` shape. Its noncontiguous-layout CPU regression passes. The full GPU
matrix remains unvalidated pending approval for the next L4 call. The original
report is preserved as `neokernel/results/hand_rolled_unit_tests_attempt1.json`.
The existing CPU suite passed 52 tests, with five GPU tests skipped; the new
CPU layout regression also passed. Engine integration has not started.

## Second attended validation and integration

The second L4 run passed all six test methods, including 96 complete-layer
comparisons (32 CONFIG combinations at batches 1, 4, and 16). Reported GPU time
was 16.636 seconds; total task spending after both calls is approximately $0.0483.

| Component | Cases | Max absolute difference | Lowest per-case argmax agreement |
| --- | ---: | ---: | ---: |
| Norm and residual | 6 | 0 | 100% |
| Q/K norm and RoPE | 96 | 0 | 100% |
| SiLU product | 6 | 0 | 100% |
| Attention | 48 | 0.00390625 | 93.75% |
| Full layer | 96 | 0.03125 | 75% |

Argmax agreement here refers to activation coordinates, not emitted vocabulary
tokens. These results do not establish full-model token correctness.

The active engine now uses packed projections, shared native-prefill weight
views, fixed decode scratch, native precomputed RoPE tables, direct cache writes,
and a captured hand-rolled layer loop. Tested RMSNorm code was moved into
`engine/kernels/rmsnorm.py` without changing arithmetic. A CPU test confirms that
packing leaves native prefill logits exactly unchanged and that its weights
share storage. Both cache layouts are covered by the causal/reset CPU test.
The source guard passes. An existing starter-only sweep test now uses a preserved
starter fixture instead of mixing current candidate kernels with the starter.
No judge, guard, or gate was edited.

Next attended command:

```powershell
py -3.11 -m neokernel.hand_rolled_validation --integration
```

It runs active-source kernel tests plus full-model prefill/decode handoff and
graph reuse tests on all public shapes, both layouts, and both attention paths.
It checks the 120-second load/warmup target. Only if those pass does it run the
unchanged harness public L4 correctness check. Two-call maximum reservation is
approximately $0.766. H100 comparison, profiles, bench, freeze, and commit remain
pending. Nothing has been committed or pushed.
