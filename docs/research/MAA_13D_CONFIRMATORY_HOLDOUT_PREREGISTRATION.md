# MAA-13d confirmatory holdout preregistration

Status: preregistered confirmatory protocol. No holdout observation has been made by this commit.

## Frozen candidate

The exploratory MAA-13d panel on the merged implementation identified `tiered_recall` as the candidate to carry forward because it occupies the high-recall end of the preregistered tier ladder while remaining structurally below `hamming_medium`.

This document freezes that candidate without changing any routing threshold:

- signed projection width: 128 bits;
- projector seed: `0x514b_5349_474e_3133`;
- inner-medium Hamming threshold: 68;
- medium Hamming threshold: 72;
- rule: admit d<=68, or d<=72 when key norm >=1.5;
- key scales: `[0.5, 1.0, 2.0, 4.0]`.

No alternative MAA-13d arm may replace `tiered_recall` after holdout observation.

## Fresh holdout identity

The confirmatory panel MUST use a fresh deterministic generator identity and seed distinct from MAA-13c and the MAA-13d exploratory panel. The implementation commit MUST record the seed before retaining any holdout output.

Geometry is frozen at 512 cases x 32 candidates, D=16.

## Frozen comparators

The holdout MUST report:

- dense;
- frozen `hamming_medium`;
- frozen MAA-13c norm-aware policy;
- frozen MAA-13d `tiered_recall`;
- density-matched deterministic random control for `tiered_recall`;
- non-deployable exact-score mass oracle matched to `tiered_recall` cardinality.

## Required evidence

For every arm record selected count/density, top-2 hits and false negatives, retained-mass mean/min/p10/p50/p90, counts >=0.90/0.95/0.99, dropped mass, O error, LSE error, retained-mass/LSE identity residual, output-bound excess and empty cases.

The run MUST be byte-for-byte reproducible and preserve all fail-closed MAA-13c numerical checks.

## Confirmatory decision rule

This holdout is confirmatory only for whether `tiered_recall` preserves the exploratory direction. Promotion to a physical MAA-13e experiment requires all of:

1. zero empty selections;
2. no more dense-top2 false negatives than `hamming_medium`;
3. lower density than `hamming_medium`;
4. retained-mass mean and p10 both strictly above the frozen MAA-13c norm-aware policy;
5. retained-mass mean and p10 both strictly above density-matched random.

Failure of any item is retained as a negative result and does not permit threshold retuning on this holdout.

Passing does not establish model quality, real-model generalization, latency, bandwidth, energy, GPU usefulness or production readiness. It only permits MAA-13e physical qualification work.
