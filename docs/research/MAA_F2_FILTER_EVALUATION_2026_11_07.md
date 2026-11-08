# MAA F2 filter evaluation launch — 2026-11-07

Status: **evaluation requested; results pending this branch's CI**.
Scope: reproducible host qualification of existing research-only Boolean/F2-assisted Q/K candidate selection, **not** a production kernel or real-device performance claim.

## Pinned provenance

- Repository: `Memorithm/FLAT-ATTENTION`
- Base branch: `main`
- Base revision at branch creation: `fe79917fa0bc1951753f78b83a640d8e2305e1fc`
- Evaluation branch: `research/maa-f2-filter-eval-20261107`
- CI source: `.github/workflows/maa-host.yml`
- Head revision evaluated: **record the exact GitHub Actions checkout SHA and final PR head**.
- Test inputs, seeds, thresholds and policies: **unchanged** in the existing preregistered harnesses.
- The CI is expected to run for this file because `maa-host.yml` listens to PR changes in `docs/research/MAA*`.

## Scientific question

Does a deterministic bit-parallel selection of candidate K/V entries based on Q/K-derived Boolean signatures reduce **logical exact-score evaluations**, while retaining enough dense-reference softmax probability mass and output/LSE fidelity to justify a subsequent hardware experiment?

A reduction in logical candidate count **does not** prove reduced physical K/V transactions, GPU latency, energy, throughput, or model quality. F2 predicates and packed Boolean control are not a proof of a fully multiplication-free attention kernel. Dense qualified FLAT remains the numerical baseline and fail-closed fallback.

## Evaluation arms and frozen executables

1. **Dense reference**: all candidates admitted, exact FLAT host attention reference.
2. **Boolean/Hamming signatures**: `qk_signature_sweep`, with frozen widths 8, 16, 32, 64, 128 and thresholds; include positive-scale collision negative control.
3. **Norm-aware Boolean/F2 selection**: `norm_aware_qk_sweep`; compare strict/relaxed Hamming and norm-only against matched-cardinality random selection.
4. **Retained softmax mass**: `retained_softmax_mass`; compare dense, frozen Hamming and norm-aware routes, matched random and the non-deployable exact mass oracle.
5. **Tiered-recall confirmatory**: `tiered_recall_confirmatory` (existing frozen independent holdout; do not retune based on this evaluation).
6. **Repair diagnostics**: `adaptive_repair_signal` and `adaptive_repair_confirmatory` are contextual; they are not to be conflated with F2 filter success.

The `maa-host` workflow performs both debug and release tests and runs each listed executable twice, comparing the emitted CSV bytes.

## Required measurements and interpretation

| Dimension | Metrics / evidence | Interpretation |
| --- | --- | --- |
| Logic | selected / total candidates, rejection fraction, dense-top2 misses | Counts are logical, not time or DRAM bytes |
| Numerical | mean/min retained softmax mass, p10/p50/p90, O max absolute error and LSE error | Judge output fidelity, not only top-k retention |
| Mathematical checks | `LSE_dense - LSE_selected = -ln(retained_mass)`, output-error bound | Validate against existing preregistered f64 diagnostic |
| Controls | dense, Boolean-only, matched random, mass upper bound, positive-scale collision, zero survivors | Prevent overstated gains and ambiguous fallback behavior |
| Reproducibility | exact checkout SHA, workflow run URL, identical CSV across two runs | Qualifies host evidence only |
| Physical | same-device latency, tokens/s, DRAM traffic, effective bits/value incl. overhead, power | **Not measured by this host workflow** |

## Decision rules

- **Reject** any malformed metadata, non-finite output, non-deterministic CSV or failed oracle parity.
- **No quality promotion** from an improved top2 hit-rate alone: use retained mass and O/LSE behavior.
- **No speedup or memory claim** from candidate reduction alone: physical measurement on a named device must be separately controlled through `Memorithm/RemoteOps`, with exactly matched dense/Boolean-only/F2 arms and precise SHA.
- **No default routing change**, stable API change, semantic relaxation or merger of this research into deployment without required independent qualification.
- No CUDA/vendor SDK is introduced.

## CI evidence to record when complete

- PR number: pending
- Evaluated exact head SHA: pending
- `maa-host` run ID / URL: pending
- Result (success/failure): pending
- F2 policy logical selection and retained mass: pending **new run**, separate from historical observations
- Numerical O/LSE diagnostics: pending **new run**
- Real-device performance: **not evaluated in this PR**

## Baseline context (historical observations only)

- MAA-13a's frozen 128-bit, half-width threshold arm selected 1,030/2,048 candidates and retained 251/256 dense-top2 occurrences on its own synthetic dataset.
- MAA-13b's frozen norm-aware arm selected 1,299/2,048 candidates and retained 255/256 dense-top2 occurrences on a **different** synthetic dataset. It had higher O/LSE error than the looser Hamming comparator at the same top2 hit count.
- These two observations are **not** a matched A/B hardware speed benchmark and are **not new results of this PR**.

Relevant source protocols: `docs/research/MAA_QK_SIGNATURE_EXPLORATORY_OBSERVATION.md`, `docs/research/MAA_NORM_AWARE_QK_EXPLORATORY_OBSERVATION.md`, `docs/research/MAA_RETAINED_SOFTMAX_MASS_PREREGISTRATION.md` and `docs/research/MULTI_ALGEBRA_ATTENTION.md`.
