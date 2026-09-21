# MAA-13b: norm-aware Q/K multi-algebra exploratory preregistration

Protocol version: `maa-norm-aware-qk-exploratory/v1`.

Status: preregistered exploratory research. This document is committed before
the MAA-13b implementation and before producing MAA-13b observations.

## Motivation

MAA-13a produced two bounded observations on its fresh synthetic panel:

1. Q/K-derived signed-hyperplane signatures carried useful angular ordering
   information. The 128-bit, half-width Hamming arm selected 1,030/2,048
   logical candidates while retaining 251/256 dense-top2 occurrences with no
   empty selection.
2. The required positive-scale control produced identical signatures for every
   low/high pair, leaving every strict exact-score magnitude ordering unresolved.

MAA-13b tests the missing candidate-dependent term directly. For one fixed query:

```text
q.k = ||q|| * ||k|| * cos(theta)
```

The query norm is common across candidate keys, while key norm is not.
MAA-13b therefore retains the MAA-13a angular signature and adds a compact,
precomputable key-norm predicate before testing a nonlinear Boolean/F2/
Zhegalkin interaction.

This is a new exploratory dataset. It does not reuse the MAA-11b confirmatory
holdout or the MAA-12a/13a exploratory generators.

## Frozen generator

- generator: deterministic SplitMix64;
- dataset seed: `0x004d_4141_2d31_3362` ("MAA-13b");
- random matched-control seed: `0x4d41_4131_3342_524e`;
- signed-hyperplane projector seed: `0x514b_5349_474e_3133` (same projection
  mechanism as MAA-13a);
- cases: 128;
- candidates per case: 16;
- Q/K dimension: `D=16`;
- one query/head;
- non-causal exact FLAT host oracle;
- softmax scale: 1.0;
- Q direction is L2-normalized;
- each K direction is L2-normalized and then multiplied by one deterministic
  scale selected from exactly `{0.5, 1.0, 2.0, 4.0}`;
- each V vector is generated independently from the same deterministic stream
  with finite components in [-1, 1];
- exact relevance score: unscaled `q.k`;
- dense relevance target: exact top-2 with lower candidate ID tie-break.

The selected K scale is not derived from dense relevance labels.

## Frozen angular features

MAA-13b fixes the signature width to 128 bits because that was the widest and
strongest observed angular arm in the completed exploratory MAA-13a series.
This is hypothesis generation from prior exploratory evidence, not a
confirmatory reuse of MAA-13a data.

Using the actual M13B.2 Hamming contract:

```text
near   = Hamming(Qsig, Ksig) <= 64
medium = Hamming(Qsig, Ksig) <= 72
```

The medium threshold is the preregistered 9/16-width band. It is a fixed
exploratory relaxation around the half-width baseline and is not tuned after
MAA-13b observation.

## Frozen norm feature

Compute the key L2 norm as host research metadata and define:

```text
high_norm = ||K|| >= 1.5
```

The threshold 1.5 is the fixed midpoint separating generated scale classes
`{0.5, 1.0}` from `{2.0, 4.0}`.

The intended systems interpretation is precomputed K metadata: a real runtime
would need to account for generation/storage cost separately before promotion.
MAA-13b does not make such a performance claim.

## Frozen multi-algebra rule

The candidate admission rule is:

```text
admit = near OR (medium AND high_norm)
```

Use Boolean variables:

```text
x0 = near
x1 = medium
x2 = high_norm
```

The exact Zhegalkin algebraic-normal-form representation is frozen as:

```text
P(x) = x0 XOR (x1*x2) XOR (x0*x1*x2)
```

MAA-13b must:

1. evaluate the direct Boolean rule;
2. evaluate the Zhegalkin polynomial;
3. split that polynomial through the existing exact Zhegalkin->F2 cooperation
   bridge;
4. recompose the F2 affine part XOR the nonlinear residual;
5. require all three verdicts to agree for every candidate.

This is semantic equivalence evidence, not a claim that the split is faster.

## Frozen comparison arms

For every case compare:

1. **dense**: all 16 candidates;
2. **hamming_near**: `near` only;
3. **hamming_medium**: `medium` only;
4. **norm_only**: `high_norm` only;
5. **norm_aware_maa**: the frozen multi-algebra rule above;
6. **matched_random**: deterministic random-priority selection with exactly the
   same per-case cardinality as `norm_aware_maa`.

The matched-random control may use only case ID, candidate ID and the frozen
control seed. It must not use Q, K, V, dense scores or relevance labels.

All sparse candidate lists must be restored to original ascending candidate
order before exact numerical attention is evaluated.

## Frozen diagnostics

For each arm aggregate:

- selected candidate count / logical exact-score evaluations;
- dense-top2 hits retained;
- false negatives relative to dense top-2;
- empty-selection case count;
- sum of max-absolute output error versus dense FLAT on non-empty selections;
- sum of absolute LSE error versus dense FLAT on non-empty selections;
- number of non-empty comparable cases.

For `norm_aware_maa` additionally record:

- candidates admitted by norm-aware MAA but rejected by `hamming_near`;
- dense-top2 occurrences among those additional candidates.

Logical selected counts are not physical latency, bandwidth, energy or memory
measurements.

## Structural acceptance criteria

1. All generated numerical inputs are finite.
2. Generated Q directions and pre-scale K directions have valid non-zero norms.
3. MAA-13a's authoritative M13B.2 signature type and Hamming rule are reused.
4. `near => medium` holds for every candidate.
5. `hamming_near` is a subset of `norm_aware_maa`.
6. `norm_aware_maa` is a subset of `hamming_medium`.
7. Direct Boolean, Zhegalkin and F2+nonlinear recomposition verdicts are
   bit-identical for all eight truth-table assignments and every experiment
   candidate.
8. Matched-random cardinality equals norm-aware cardinality separately for every
   case.
9. Dense labels do not influence admission or random-control selection.
10. Empty sparse selections remain explicit; no hidden dense fallback is used.
11. The complete CSV is byte-identical across two executions in CI.
12. Existing MAA-10/11/12/13a evidence remains unchanged.
13. No stable API, default routing, GPU kernel or performance claim is changed.

## Decision boundary

No winning arm or confirmatory pass threshold is preregistered. MAA-13b is an
exploratory characterization of whether a compact norm side channel improves
the work/quality tradeoff beyond:

- the stricter angular baseline;
- simply relaxing the Hamming threshold;
- norm-only routing;
- a density-matched random control.

A later confirmatory series requires a separately frozen policy and fresh
tuning/holdout identities.

## Non-claims

MAA-13b does not establish real-model quality, novelty, latency, throughput,
bandwidth, energy efficiency, physical memory savings, GPU usefulness or
production readiness. It does not replace exact q.k, softmax or dense FLAT.
