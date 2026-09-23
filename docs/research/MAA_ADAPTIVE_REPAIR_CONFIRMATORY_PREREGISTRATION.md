# MAA-14c coverage-7/8 confirmatory holdout preregistration

Status: preregistered confirmatory protocol. No holdout observation, promotion, physical-performance claim, or runtime change is made by this document.

## Candidate provenance

The sole candidate entering this holdout is the already-observed MAA-14b structural trigger:

`repair_if_coverage_below_7_8`:

`8 * base_count < 7 * medium_count`

where:

- `base_count` is the unchanged tiered-recall candidate count;
- `medium_count` is the unchanged hamming-medium envelope count.

The candidate was selected because it was the only frozen deployable MAA-14b arm that produced a genuine intermediate logical-work/quality point on the exploratory panel. No threshold is changed for this holdout.

The exploratory panel, its seed, and its observations MUST NOT be reused as holdout data.

## Frozen holdout identity

- protocol: `maa-adaptive-repair-confirmatory/v1`
- cases: 1,024
- candidates per case: 32
- head dimension: 16
- dataset seed: `0x004d_4141_2d31_3463`
- projector seed: `0x514b_5349_474e_3133`
- signature width: 128 bits
- angular thresholds: 64 / 68 / 72 Hamming distance
- key scales: `[0.5, 1.0, 2.0, 4.0]`
- tiered-recall base: `d <= 68 || (d <= 72 && norm >= 1.5)`
- repair envelope: hamming-medium `d <= 72`
- deployable trigger: unchanged 7/8 coverage rule above
- matched-random seed: `0x4d41_4131_3463_524e`
- diagnostic retained-mass threshold: 0.90, evaluation-only

No dense score, retained mass, output error, LSE, or V value may enter the deployable repair trigger.

## Frozen matched controls

The holdout MUST evaluate:

1. dense FLAT;
2. `never_repair` tiered-recall base;
3. unchanged `coverage_below_7_8`;
4. exact-cardinality matched-random expansion for the 7/8 trigger;
5. `always_repair` hamming-medium envelope;
6. non-deployable `base retained mass < 0.90` repair oracle.

The matched-random control preserves the base, samples from the complete dense complement, adds exactly the candidate count added by the 7/8 arm on the same case, uses trigger ID 2, breaks priority ties by candidate ID, and canonicalizes final IDs in ascending order.

## Required evidence

Record for every arm:

- repaired rows;
- initial/final/added candidate counts;
- mean final density;
- dense top-2 hits and false negatives;
- retained-softmax mass mean/min/p10/p50/p90;
- counts above 0.90/0.95/0.99 retained mass;
- output-error sum;
- LSE-error sum;
- base rows below the 0.90 diagnostic threshold;
- trigger TP/FP/FN, precision and recall;
- empty rows;
- exact source revision and deterministic replay identity.

The release harness MUST be executed twice and the CSV outputs MUST be byte-identical.

## Frozen confirmatory decision rule

The candidate receives `CONFIRMATORY_PASS` only when all conditions below hold:

1. zero empty final selections;
2. no more dense top-2 false negatives than `never_repair`;
3. repaired rows are strictly between 0 and 1,024;
4. final selected-pair count is strictly greater than `never_repair` and strictly less than `always_repair`;
5. retained-mass mean and p10 are both strictly greater than `never_repair`;
6. output-error sum and LSE-error sum are both strictly lower than `never_repair`;
7. retained-mass mean is strictly greater than the exact-cardinality matched-random 7/8 control;
8. candidate uses no more than 60% of the additional candidates required by `always_repair`, measured as:
   `candidate_added / always_repair_added <= 0.60`;
9. candidate recovers at least 20% of the retained-mass-mean gap from `never_repair` to `always_repair`, measured as:
   `(candidate_mass_mean - base_mass_mean) / (always_mass_mean - base_mass_mean) >= 0.20`.

Checked arithmetic and finite denominators are mandatory. If an endpoint gap is zero or invalid, the holdout fails closed.

No criterion may be modified after observing this holdout.

## Interpretation boundary

A confirmatory pass would establish only that the same structural trigger reproduces an intermediate synthetic host-level work/quality tradeoff on a fresh deterministic panel.

It would permit a separately preregistered MAA-14d physical qualification.

It would not establish:

- real-model quality;
- general language-model generalization;
- latency or throughput improvement;
- bandwidth or memory-traffic reduction;
- energy efficiency;
- production routing readiness.
