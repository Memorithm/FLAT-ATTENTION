# MAA-13d mass-aware adaptive routing frontier preregistration

Status: preregistered exploratory protocol. No observation, winner, performance, or promotion claim is made by this document.

## Provenance and separation

MAA-13d follows the merged MAA-13c retained-softmax-mass experiment. MAA-13c is prior evidence only: its 256 x 32 panel MUST NOT be reused for MAA-13d parameter selection, candidate ranking, threshold selection, or acceptance.

The MAA-13d implementation MUST use a fresh deterministic dataset identity and seed, recorded in source before any observation is retained. Any later confirmatory promotion requires a second, distinct holdout identity frozen before that confirmatory run.

## Research question

Can a preregistered multi-tier angular/norm policy improve the retained-softmax-mass versus candidate-density frontier relative to the frozen MAA-13c baselines, without using exact q.k scores or relevance labels to route candidates?

The experiment tests a mechanism, not a speedup. Logical candidate removal is not latency, bandwidth, energy, memory-traffic, GPU-utilization, or end-to-end performance evidence.

## Frozen baselines

The experiment MUST reproduce, without retuning, the MAA-13c baseline families:

- dense;
- Hamming near;
- Hamming medium;
- norm-only;
- MAA-13c norm-aware policy;
- density-matched deterministic random control;
- non-deployable exact-score mass oracle at matched cardinality.

The exact-score oracle is diagnostic only and MUST NOT contribute features to a deployable policy.

## Candidate policy family

MAA-13d may evaluate a small preregistered ladder of nested angular/norm rules. Every rule MUST:

1. derive angular evidence only from the existing signed Q/K projection and Hamming distance;
2. derive magnitude evidence only from key-norm predicates that do not use dense q.k;
3. preserve every candidate in its declared innermost angular band;
4. use monotonically stronger norm requirements for progressively wider angular bands;
5. be expressible as a deterministic Boolean/F2/Zhegalkin predicate;
6. contain no label, dense score, retained-mass, output-error, or oracle feedback in routing.

The complete thresholds and all candidate arms MUST be committed before the first retained observation. Post-observation threshold search on the MAA-13d panel is forbidden.

## Frozen exploratory geometry and policy ladder

The exploratory panel is frozen at 384 cases x 32 candidates, D=16, with dataset seed `0x004d_4141_2d31_3364`. The signed-hyperplane projector remains the MAA-13c 128-bit projector with seed `0x514b_5349_474e_3133` so the experiment changes the routing policy rather than simultaneously changing the representation.

Key scales remain `[0.5, 1.0, 2.0, 4.0]`. Angular bands are frozen at Hamming distances 64 (near), 68 (inner-medium), and 72 (medium).

Three new deployable structured arms are frozen before observation:

- `tiered_strict`: admit d<=64; for 64<d<=68 require key norm >=1.5; for 68<d<=72 require key norm >=2.0.
- `tiered_balanced`: admit d<=64; for 64<d<=68 require key norm >=1.0; for 68<d<=72 require key norm >=2.0.
- `tiered_recall`: admit d<=68; for 68<d<=72 require key norm >=1.5.

These arms are nested by construction: `tiered_strict subset tiered_balanced subset tiered_recall subset hamming_medium`. The existing MAA-13c policy `d<=64 || (d<=72 && norm>=1.5)` remains a frozen baseline and is not renamed as a 13d candidate.

No threshold may be added, removed, or changed after observing this panel. A later experiment may use a different ladder only under a new preregistration and fresh dataset identity.

## Required evidence

For every arm, record at minimum:

- selected candidate count and mean density;
- dense top-2 hits and false negatives;
- retained softmax mass mean, minimum, p10, p50 and p90;
- counts at or above retained mass 0.90, 0.95 and 0.99;
- dropped-mass sum;
- output-error sum;
- LSE-error sum;
- maximum residual of the exact retained-mass/LSE identity;
- maximum excess over the retained-mass output-error bound;
- empty-selection count.

The implementation MUST preserve the MAA-13c fail-closed numerical checks and deterministic byte-for-byte evidence replay.

## Frontier analysis

No single winning arm is preregistered.

An arm is Pareto-dominated when another deployable structured arm has no greater density and no worse retained-mass p10 and mean, with at least one strict improvement. Top-2 retention alone is not a promotion criterion.

The experiment SHOULD report distance to the matched-cardinality exact-score oracle as diagnostic headroom. That distance is not a deployment acceptance threshold.

## Non-claims

MAA-13d does not establish model quality, real-model generalization, novelty, physical work avoided, latency, throughput, bandwidth, residency, energy efficiency, GPU usefulness, or production readiness.

A favorable exploratory frontier only justifies freezing a candidate for a fresh confirmatory holdout. It does not authorize default routing.

## Successor gate

A later MAA-13e physical experiment is permitted only after a candidate policy has survived a separately frozen confirmatory holdout. MAA-13e must include router construction cost, sparse execution cost, memory traffic where measurable, synchronization, fallback behavior, numerical quality, and end-to-end latency on declared hardware.
