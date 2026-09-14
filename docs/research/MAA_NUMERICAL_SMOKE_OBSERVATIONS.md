# MAA-10a first executed observations

Protocol: [maa-numerical-smoke/v1](MAA_NUMERICAL_SMOKE_PREREGISTRATION.md).
These observations are synthetic host diagnostics, not a model-quality or
performance qualification.

## Provenance

- Preregistration commit: `bad913814f99f1e8b7723bdce046e53c7eb03be4`.
- Executed source commit: `35bfa476df889889af9369c0b99fef7ea971459d`.
- GitHub Actions run: `34896731097`; job: `104152588197`.
- Numerical CSV emitted on 2026-09-14 at approximately 21:05:49 UTC.
- Runner reported x86_64 Linux, Ubuntu 24.04.5; Rust 1.89.0.
- The preparation workflow started at `855d5e08d2b32c93a1ae32b51eb8c179e09cecd4`,
  formatted the code and synchronized only its local development-dependency lock
  edge, committed the result, then compiled/tested/executed the source commit
  above. The run-head SHA is therefore NOT the executed source SHA.
- That temporary branch-only preparation workflow was removed before opening the
  product PR. The permanent read-only `maa-host` workflow must independently
  validate and execute the final PR head before merge.

Source log:
https://github.com/Memorithm/FLAT-ATTENTION/actions/runs/34896731097/job/104152588197

Observed preparation validation: strict formatting and Clippy passed; 53 MAA unit
tests, 13 provenance integration tests and 8 new numerical-example tests passed.
The release example emitted all 32 diagnostic rows. No test was ignored in those
three test groups. This record does not claim that the final PR CI has finished.

## Combined Boolean + F2 + Zhegalkin arm

The reference has eight numerical score evaluations per query. Counts below
refer to each numerical arm, excluding dense label construction, reference runs,
qualification, validation and gathering.

| Frozen case | Kept original IDs | Score evaluations | Retained dense top-1 | Output max absolute error | Absolute LSE difference |
| --- | --- | ---: | ---: | ---: | ---: |
| all_accept | 0,1,2,3,4,5,6,7 | 8 | 1 | 0 | 0 |
| constant_values | 0,4 | 2 | 0 | 0 | 1.800016284 |
| dominant_key_dropped | 0,4 | 2 | 0 | 9.999567032 | 11.30689621 |
| empty_selection | none | 0 | 0 | not produced | not produced |

All eight all-accept arms had zero observed O/LSE difference on this runner.
Empty selections reported `no_survivors`, not zero numerical errors or a dense
fallback.

## Required adverse outcome and ablations

On `dominant_key_dropped`:

| Arm | Score evaluations | Retained dense top-1 | Output max absolute error |
| --- | ---: | ---: | ---: |
| Boolean-only | 6 | 1 | 0.0001239776611 |
| Boolean + F2 | 4 | 1 | 0.0002470016479 |
| Boolean + Zhegalkin | 4 | 0 | 9.999567032 |
| Boolean + F2 + Zhegalkin | 2 | 0 | 9.999567032 |

The structural Zhegalkin predicate deliberately excludes key 2. The resulting
large error is a successful falsification control for the measurement harness,
NOT evidence that this predicate is a useful attention policy. None of these
arms is promoted to production.

The `constant_values` case also shows why diagnostics cannot be substituted for
one another: the combined arm retains no dense top-1 key and changes LSE, yet its
output matches here because every V row is identical. Neither top-1 coverage nor
output equality alone proves full numerical equivalence.

## Controls and limits retained after observation

The fixed seeded-priority control selected the same IDs `{0,4}` as the combined
arm in the nontrivial two-survivor cases. The preregistered seed is NOT changed to
produce a more favorable comparison. This coincidence provides no independent
support for algebraic superiority. First/last-position controls also lost the
dominant key in this fixture. A future protocol needs multiple independent
controls and held-out cases rather than claims from these four constructed cases.

A reduction from eight to two numerical scores is only a 75% reduction of that
per-arm logical count. It is not a measured speedup: no latency, bandwidth,
physical traffic, energy or real-model quality measurement was collected. The
pipeline still evaluates exact scores and softmax for survivors; it does not
replace Q.K or softmax. Max-plus readiness/defer semantics remain a separate
pending experiment, not a hidden relevance filter in this smoke.

## Reproduction from a checked-out qualified revision

```bash
cargo test -p flat-algebraic-attention --locked --all-targets
cargo test -p flat-algebraic-attention --locked --release --all-targets
cargo run -p flat-algebraic-attention --locked --release --example matched_numerical_smoke
```

The permanent CI runs the release example twice and compares the emitted CSV
byte for byte on the same runner. Across different toolchains/devices, consult
the preregistered numerical tolerances rather than assuming identical floating-
point bytes. The full raw CSV is retained in the execution log and job summary.
