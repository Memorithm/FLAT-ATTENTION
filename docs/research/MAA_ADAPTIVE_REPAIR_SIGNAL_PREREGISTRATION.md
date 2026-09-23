# MAA-14b deployable repair-signal preregistration

Status: preregistered exploratory protocol. No observation, winner, physical-performance claim, or runtime promotion is made by this document.

## Separation from prior evidence

MAA-14b begins only after merged MAA-14a. MAA-13d/13e remain frozen historical evidence and MUST NOT be retuned by this work.

MAA-14b uses a fresh deterministic dataset identity and seed. The MAA-13c, MAA-13d exploratory, and MAA-13d confirmatory panels MUST NOT be reused for signal selection or threshold tuning.

The purpose of this slice is to test whether cheap deployable router metadata can decide which query rows should widen from the frozen tiered-recall candidate set toward the already-defined hamming-medium envelope.

Dense exact scores, retained-softmax mass, dense output error, and dense LSE error are evaluation oracles only. They MUST NOT enter a deployable repair trigger.

## Frozen geometry

- protocol: `maa-adaptive-repair-signal/v1`
- cases: 640
- candidates per case: 32
- head dimension: 16
- dataset seed: `0x004d_4141_2d31_3462`
- projector seed: `0x514b_5349_474e_3133`
- signature width: 128 bits
- angular thresholds: 64 / 68 / 72 Hamming distance
- key norm scales: `[0.5, 1.0, 2.0, 4.0]`
- tiered-recall rule: `d <= 68 || (d <= 72 && norm >= 1.5)`
- full repair envelope: hamming-medium `d <= 72`

The base and full-envelope policies are copied unchanged from the already-qualified MAA-13d family so MAA-14b changes only the adaptive repair decision.

## Deployable signal inputs

For each query row, the repair policy may consume only metadata available without evaluating omitted exact q.k scores:

- `base_count`: number of keys admitted by tiered-recall;
- `medium_count`: number of keys in the hamming-medium envelope;
- `gap_count = medium_count - base_count`;
- exact integer ratios derived from those counts;
- empty/non-empty structural state.

No value-vector contents, dense labels, retained mass, dense scores, dense output, or dense LSE may be consumed by these candidate triggers.

The first MAA-14b signal family is intentionally structural-only. Sparse-pass numerical residual signals are deferred to a later independently preregistered experiment so they cannot be selected after observing this panel.

## Frozen repair triggers

All triggers either keep the base tiered-recall set or widen the row completely to hamming-medium. No partial post-hoc key choice is allowed in this slice.

1. `never_repair`
   - always retain tiered-recall;
   - serves as the base sparse control.

2. `repair_any_gap`
   - repair when `gap_count > 0`.

3. `repair_if_coverage_below_7_8`
   - repair when `8 * base_count < 7 * medium_count`.

4. `repair_if_coverage_below_3_4`
   - repair when `4 * base_count < 3 * medium_count`.

5. `repair_if_coverage_below_2_3`
   - repair when `3 * base_count < 2 * medium_count`.

6. `always_repair`
   - always widen to hamming-medium;
   - serves as the deterministic full-envelope control.

Checked integer arithmetic is mandatory. A malformed geometry or `base_count > medium_count` fails closed.

No threshold may be added, removed, or changed after the first retained observation on this dataset.

## Required matched controls

For every adaptive trigger:

- dense FLAT reference;
- never-repair tiered-recall;
- always-repair hamming-medium;
- deterministic density-matched random expansion under the frozen contract below;
- a non-deployable oracle that repairs a row when the base retained-softmax mass is below 0.90.

### Frozen matched-random expansion contract

The matched-random seed is fixed at `0x4d41_4131_3462_524e`.

For each case and each deployable adaptive trigger, the random control:

1. preserves every key in the tiered-recall base set;
2. uses the complete dense complement `all_candidate_ids - base` as its sampling universe;
3. adds exactly `adaptive_final_count - base_count` distinct candidates, so its final cardinality matches that adaptive trigger exactly on the same case;
4. ranks complement candidates by a deterministic `mix64` priority keyed by the frozen seed, case ID, trigger ID and candidate ID;
5. breaks equal priorities by ascending candidate ID;
6. sorts the final candidate IDs into canonical ascending numerical order before FLAT evaluation.

Frozen trigger IDs for the random-control key are:

- `repair_any_gap = 1`;
- `repair_if_coverage_below_7_8 = 2`;
- `repair_if_coverage_below_3_4 = 3`;
- `repair_if_coverage_below_2_3 = 4`.

`never_repair` and `always_repair` are deterministic endpoint controls and do not need separate random counterparts.

Sampling from the full dense complement rather than only `hamming_medium - base` is intentional: when a repair happens, adding the complete medium gap would reproduce the hamming-medium arm exactly and would not be an independent density-matched structural control.

This seed, universe, cardinality rule, trigger-ID mapping, tie break and canonicalization are frozen before any MAA-14b observation.

The non-deployable oracle is diagnostic headroom only. The 0.90 retained-mass threshold is frozen here before observation and is not a real-model quality guarantee.

## Required evidence

For every arm record:

- rows repaired / rows total;
- initial selected pairs;
- final selected pairs;
- added pairs;
- mean final density;
- dense top-2 hits / false negatives;
- retained-softmax mass mean, minimum, p10, p50, p90;
- cases at or above 0.90 / 0.95 / 0.99 retained mass;
- output-error sum;
- LSE-error sum;
- number of base rows below the diagnostic 0.90 retained-mass threshold;
- repair-trigger precision/recall against that diagnostic label;
- empty rows;
- exact deterministic replay identity.

Candidate counts are logical algorithmic evidence, not latency, bandwidth, energy, or physical memory-traffic evidence.

## Exploratory interpretation

No single winning trigger is preregistered.

The experiment should expose a Pareto surface over:

- repaired-row fraction;
- final candidate density;
- retained-mass mean and p10;
- dense-reference output/LSE error.

A deployable trigger is interesting only if it lies strictly between `never_repair` and `always_repair` in work while recovering a measurable fraction of the quality gap under the frozen diagnostics.

A favorable exploratory result does not authorize WGPU implementation or default routing. It only justifies a fresh confirmatory holdout with the exact selected trigger frozen before observation.

## Negative result handling

If no structural-only trigger offers useful separation, preserve that result. Do not tune thresholds on this dataset.

The next valid research move would then be a new preregistration for sparse-pass numerical residual signals, using a fresh dataset identity.

## Non-claims

MAA-14b does not establish:

- model quality;
- generalization;
- attention replacement;
- physical work avoided;
- latency or throughput improvement;
- memory-bandwidth reduction;
- energy efficiency;
- production readiness.
