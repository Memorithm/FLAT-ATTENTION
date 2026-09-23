# MAA-14b structural adaptive-repair exploratory observation

Status: preserved exploratory observation. This document records the first retained MAA-14b execution without retuning the frozen trigger family.

## Provenance

- protocol: `maa-adaptive-repair-signal/v1`
- preregistration merge: `a42a43840ed9df3d3cf7604e7fe303699beb49ed`
- matched-random freeze merge: `cee1bdc1405ef6c014a4013077fcad781427360e`
- observed implementation head: `2d16204b0b18ef51e91a6c6161d7df1d9a3dad96`
- GitHub Actions workflow: `maa-host`
- run: `35855000127`
- job: `107161261536`
- panel: 640 cases x 32 candidates, D=16
- execution: release harness executed twice with byte-identical CSV output

This observation was produced only after both preregistration commits were merged. No trigger threshold, dataset seed, projector seed, random-control seed, random universe, cardinality rule, or diagnostic retained-mass threshold was changed in response to the result.

## Preserved aggregate evidence

| arm | repaired rows | final selected | density | mass mean | mass p10 | mass p90 | output error sum | LSE error sum | trigger precision | trigger recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0 | 20,480 | 1.000000000 | 1.000000000000 | 1.000000000000 | 1.000000000000 | 0.000000000 | 0.000000000 | n/a | 0.000000000000 |
| never_repair | 0 | 14,356 | 0.700976562 | 0.819600075169 | 0.727837341090 | 0.900605730216 | 58.375667848 | 129.553893566 | n/a | 0.000000000000 |
| repair_any_gap | 534 | 15,576 | 0.760546875 | 0.866468595461 | 0.790864515706 | 0.932961262712 | 45.415096261 | 93.161193132 | 0.925093632959 | 0.860627177700 |
| coverage_below_7_8 | 118 | 14,827 | 0.723974609 | 0.838321383506 | 0.765348497851 | 0.908358432257 | 53.480457880 | 114.491113186 | 1.000000000000 | 0.205574912892 |
| coverage_below_3_4 | 0 | 14,356 | 0.700976562 | 0.819600075169 | 0.727837341090 | 0.900605730216 | 58.375667848 | 129.553893566 | n/a | 0.000000000000 |
| coverage_below_2_3 | 0 | 14,356 | 0.700976562 | 0.819600075169 | 0.727837341090 | 0.900605730216 | 58.375667848 | 129.553893566 | n/a | 0.000000000000 |
| always_repair | 640 | 15,576 | 0.760546875 | 0.866468595461 | 0.790864515706 | 0.932961262712 | 45.415096261 | 93.161193132 | 0.896875000000 | 1.000000000000 |
| matched_random_any_gap | 534 | 15,576 | 0.760546875 | 0.856945258583 | 0.782422415473 | 0.927306009980 | 48.926324401 | 100.325933695 | 0.925093632959 | 0.860627177700 |
| matched_random_7_8 | 118 | 14,827 | 0.723974609 | 0.834486501180 | 0.760067108311 | 0.906202796043 | 54.714635052 | 117.453480721 | 1.000000000000 | 0.205574912892 |
| mass_below_0_90_oracle | 574 | 15,519 | 0.757763672 | 0.864596593962 | 0.790864515706 | 0.926689304844 | 46.111700730 | 94.443813086 | 1.000000000000 | 1.000000000000 |

All arms retained 1,280 / 1,280 dense top-2 occurrences on this panel, so top-2 retention is saturated here and does not discriminate the repair policies.

The base tiered-recall arm fell below the frozen 0.90 retained-mass diagnostic on 574 / 640 rows.

## Interpretation

### Endpoint controls

`repair_any_gap` and `always_repair` produce the same final candidate count and the same numerical diagnostics. The 106 rows on which `repair_any_gap` does not fire have no medium-envelope gap, so the final numerical set is already identical. Therefore `repair_any_gap` does not provide an intermediate numerical-work point on this panel.

The `3/4` and `2/3` coverage triggers never fire. They are preserved as negative evidence and are not retuned on this dataset.

### 7/8 structural trigger

The `coverage_below_7_8` trigger is the only frozen deployable policy strictly between the base and complete hamming-medium envelope:

- 118 / 640 rows request repair;
- 471 candidates are added, versus 1,220 for complete widening;
- final density is 0.723974609, versus 0.700976562 base and 0.760546875 complete widening;
- retained-mass mean rises from 0.819600075169 to 0.838321383506;
- output-error sum falls from 58.375667848 to 53.480457880;
- LSE-error sum falls from 129.553893566 to 114.491113186.

Relative to the gap from `never_repair` to complete widening, the 7/8 trigger uses 38.61% of the additional logical candidates and recovers approximately:

- 39.94% of the mean retained-mass improvement;
- 37.77% of the output-error-sum improvement;
- 41.39% of the LSE-error-sum improvement.

These are arithmetic descriptions of this bounded synthetic panel, not model-quality or physical-performance effects.

Against the frozen diagnostic label `base retained mass < 0.90`, the trigger has precision 1.0 and recall 0.205574912892. It therefore identifies a small, clean subset of low-retained-mass rows while missing most such rows.

At identical final cardinality, the frozen matched-random 7/8 control has mean retained mass 0.834486501180, versus 0.838321383506 for structural 7/8; its output-error and LSE-error sums are also higher. This is descriptive evidence that the hamming-medium additions carry structure beyond this particular random control. No statistical-significance claim is made.

## Research decision

Under the exploratory criterion frozen before observation, `coverage_below_7_8` is the only deployable trigger that supplies a genuine intermediate work/quality point. It is therefore the sole candidate from this panel that may advance to a fresh confirmatory holdout.

This selection does not authorize:

- changing the 7/8 threshold;
- using this panel again as confirmatory data;
- WGPU implementation;
- default runtime routing;
- latency, bandwidth, memory-traffic, energy, or model-quality claims.

A next confirmatory slice must freeze a distinct dataset identity and the exact unchanged 7/8 trigger before observing that holdout.
