# MAA-13c: retained softmax mass evidence preregistration

Protocol version: `maa-retained-softmax-mass/v1`.

Status: preregistered exploratory/evidence research. This document is committed
before the MAA-13c implementation and before producing MAA-13c observations.

## Motivation

MAA-13b showed that the frozen norm-aware rule can retain the same dense top-2
count as a relaxed Hamming envelope while selecting fewer candidates, but still
produce larger O/LSE error. Therefore top-k retention is not a sufficient
quality proxy for sparse exact-attention routing.

MAA-13c freezes a stronger dense-reference quantity: retained softmax
probability mass.

## Mathematical reference

For one query with finite exact scores `s_i` and dense softmax probabilities:

```text
p_i = exp(s_i) / Z
Z   = sum_j exp(s_j)
```

for a selected candidate set `S`, define retained mass:

```text
M(S) = sum_{i in S} p_i
```

The implementation must evaluate this with a deterministic stable f64 reference
by subtracting the maximum score before exponentiation.

For any non-empty `S`, exact sparse attention over the same selected scores is
the dense distribution conditioned on `S`. Therefore the mathematical
identity is:

```text
LSE_dense - LSE_S = -ln(M(S))
```

The MAA-13c harness must compare this f64 reference identity with the existing
FLAT host-oracle LSE difference and record the absolute identity residual. This
is a numerical-consistency diagnostic, not a claim of bit-exact equality
between f64 reference arithmetic and FLAT f32 arithmetic.

For output vectors `v_i`, define:

```text
Vmax = max_i ||v_i||_infinity
```

Then the exact mathematical output bound is:

```text
||O_dense - O_S||_infinity <= 2 * (1 - M(S)) * Vmax
```

The harness must record the bound and verify the FLAT observed max-absolute
output error does not exceed it beyond the frozen numerical tolerance below.

This turns retained mass into an evidence quantity connected directly to LSE
and output behavior, rather than an arbitrary retrieval metric.

## Frozen generator

MAA-13c uses a fresh deterministic panel:

- generator: SplitMix64;
- dataset seed: `0x004d_4141_2d31_3363` ("MAA-13c");
- matched-random seed: `0x4d41_4131_3343_524e`;
- signed-hyperplane projector seed: `0x514b_5349_474e_3133`;
- cases: 256;
- candidates per case: 32;
- Q/K dimension: `D=16`;
- V dimension: `D=16`;
- non-causal exact FLAT host oracle;
- softmax scale: 1.0;
- Q direction is L2-normalized;
- each K direction is L2-normalized and multiplied by one deterministic scale
  from exactly `{0.5, 1.0, 2.0, 4.0}`;
- each V component is generated independently in [-1, 1].

No dense relevance label or dense probability participates in a deployable
routing arm.

## Frozen routing arms

Use the already-merged MAA-13a/13b mechanisms without parameter retuning:

- signature width: 128 bits;
- `near = Hamming <= 64`;
- `medium = Hamming <= 72`;
- `high_norm = ||K|| >= 1.5`;
- `norm_aware_maa = near OR (medium AND high_norm)`.

Compare these arms:

1. `dense`: all candidates;
2. `hamming_near`;
3. `hamming_medium`;
4. `norm_only`;
5. `norm_aware_maa`;
6. `matched_random`: deterministic random selection with exactly the same
   per-case cardinality as `norm_aware_maa`;
7. `mass_oracle_matched`: diagnostic upper bound selecting exactly the same
   per-case cardinality as `norm_aware_maa`, but choosing candidates by
   descending exact dense score.

The `mass_oracle_matched` arm is explicitly non-deployable. It is an oracle
control that quantifies how much retained-mass headroom exists at the same
cardinality. It must never be described as a routing implementation.

Sparse candidate IDs are restored to ascending original numerical order before
FLAT evaluation.

## Frozen retained-mass diagnostics

For each arm record aggregate:

- selected candidate count;
- mean selected density;
- dense-top2 retained count and false negatives for continuity only;
- sum and mean retained softmax mass;
- minimum per-case retained mass;
- nearest-rank p10, p50 and p90 retained mass;
- number of cases with retained mass >= 0.90;
- number of cases with retained mass >= 0.95;
- number of cases with retained mass >= 0.99;
- sum of dropped mass `1-M`;
- sum of observed max-absolute O error;
- sum of observed absolute LSE error;
- maximum absolute residual of
  `(LSE_dense - LSE_sparse) - (-ln(M))`;
- maximum excess of observed O error over
  `2*(1-M)*Vmax`, clamped at zero for reporting;
- empty-selection count.

For empty selections:

- retained mass is exactly zero;
- dropped mass is exactly one;
- no sparse O/LSE is fabricated;
- no dense fallback is credited;
- LSE identity residual and output-bound excess are omitted for that case.

## Frozen numerical tolerance

The output-bound check against FLAT host output allows:

```text
1e-5 + 1e-5 * bound
```

absolute slack to account for f32/f64 numerical differences. Any larger excess
fails the harness.

The LSE identity is recorded rather than used as a strict scientific pass
threshold, but a non-finite residual fails closed.

## Required structural checks

1. Scores and V values must be finite.
2. Selected candidate IDs must be unique and in bounds.
3. Dense selection must return retained mass exactly 1 within f64 roundoff.
4. Empty selection must return mass 0 without a sparse FLAT result.
5. Retained mass must remain in [0,1] within numerical roundoff.
6. Selection order must not change retained mass.
7. `near` remains a subset of `norm_aware_maa`.
8. `norm_aware_maa` remains a subset of `hamming_medium`.
9. Matched-random and mass-oracle controls match MAA cardinality separately for
   every case.
10. The mass-oracle arm must have retained mass >= every other arm with the same
    per-case cardinality; in particular it must be >= norm-aware MAA and
    matched-random.
11. Dense labels/probabilities do not influence deployable routing arms.
12. The complete CSV is byte-identical across two executions in CI.
13. Existing MAA-10/11/12/13a/13b evidence remains unchanged.
14. No stable API, default routing, GPU kernel or performance claim changes.

## Decision boundary

MAA-13c does not preregister a winner or promotion threshold.

The result should answer three questions:

1. Does norm-aware MAA retain substantially more softmax mass than a
   density-matched random route?
2. At the same logical cardinality, how far is norm-aware MAA from the
   non-deployable mass-oracle upper bound?
3. Does retained mass explain the observed LSE and bound the observed output
   error as the mathematics predicts?

Only after this evidence exists may a new tiered angular/norm policy be
preregistered. Any confirmatory promotion still requires a new frozen policy
identity and distinct tuning/holdout datasets.

## Non-claims

This work does not establish model quality, novelty, latency, throughput,
bandwidth, energy efficiency, physical memory savings, GPU usefulness or
production readiness. Logical selected counts are not hardware performance
evidence.
