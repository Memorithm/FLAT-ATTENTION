# MAA-13a: Q/K-derived Boolean signature exploratory preregistration

Protocol version: `maa-qk-signature-exploratory/v1`.

Status: preregistered exploratory research. This document is committed before
the MAA-13a implementation and before producing MAA-13a numerical observations.

## Motivation

MAA-11b and MAA-12a used Boolean/F2/Zhegalkin features that were structurally
generated but were not derived from the numerical Q/K geometry used by exact
attention scoring. Both preserved negative evidence: the strict conjunction
over-filtered, and the later support-count survivor floor did not recover the
Boolean-only work/quality frontier.

MAA-13a tests a different hypothesis: derive the binary state directly from the
same Q and K vectors that define exact `q.k`, then measure whether the existing
M13B.2 Hamming/XNOR contract carries useful relevance information.

This series does not reuse the MAA-11b holdout or the MAA-12a exploratory
generator.

## Frozen projection

For an input vector `x in R^D`, signature width `B`, and fixed projector seed,
define one deterministic signed hyperplane per output bit:

```text
s[j,i] in {-1,+1}
projection[j] = sum_i s[j,i] * x[i]
bit[j] = 1 iff projection[j] >= 0
```

The sign `s[j,i]` is derived only from the frozen projector seed, bit index and
input-coordinate index through the implementation's deterministic `mix64`
function. No dense relevance label, candidate ID, exact q.k score, V value or
post-observation tuning input participates in projection.

The research primitive returns the existing MAA `F2Vector`. The executable
experiment must then construct the authoritative M13B.2
`BooleanAttentionSignature` from the exact same packed words and verify that:

- bit width is identical;
- packed words are identical;
- MAA F2 Hamming distance equals M13B.2 Hamming distance;
- M13B.2 XNOR match count equals `B - Hamming`.

MAA-13a must not define a competing Boolean-attention signature type.

## Frozen primary generator

The primary panel is angularly controlled so that exact dot-product ranking and
directional similarity are directly comparable.

- generator: deterministic SplitMix64;
- dataset seed: `0x004d_4141_2d31_3361` ("MAA-13a");
- projector seed: `0x514b_5349_474e_3133`;
- cases: 128;
- candidates per case: 16;
- input dimension: `D=16`;
- every generated Q and K vector is L2-normalized independently before scoring
  or projection;
- exact relevance score: unscaled `q.k`;
- dense relevance target: exact top-2, lower candidate ID as deterministic
  tie-break;
- signature widths: `8, 16, 32, 64, 128` bits.

For each width, rank candidates by ascending M13B.2 Hamming distance, breaking
ties by ascending candidate ID.

## Frozen ranking diagnostics

For every width, record:

- exact top-1 match count across 128 cases;
- exact top-2 intersection count across all cases;
- Hamming-tied candidate-pair count;
- strict Hamming pairwise ordering agreements;
- strict Hamming pairwise ordering disagreements.

Dense pair ordering is defined by descending exact q.k with lower candidate ID
as tie-break. A Hamming tie is recorded separately and is not credited as an
agreement.

These are ranking diagnostics, not model-quality or performance claims.

## Frozen M13B admission sweep

For each signature width `B`, evaluate the existing
`HammingAdmissionRule` at exactly three preregistered thresholds:

```text
B / 4
3 * B / 8
B / 2
```

using integer arithmetic. For every width/threshold arm record:

- selected candidate count, interpreted only as logical exact-score evaluations
  that would remain after this host filter;
- dense-top2 hits retained;
- false negatives relative to dense top-2;
- empty-selection case count.

No latency, memory-traffic, energy or device-execution inference may be made from
these logical counts.

## Frozen scale-invariance negative control

The signed-hyperplane bit map is expected to be invariant under multiplication
by a strictly positive scalar. MAA-13a must preserve this as an explicit
limitation rather than hide it.

A separate deterministic negative-control panel uses 64 generated Q/base-K
pairs. For each base K, compare signatures of:

```text
K_low  = 0.5 * K
K_high = 2.0 * K
```

at every preregistered signature width.

Record:

- number of low/high signature collisions;
- number of pairs for which exact q.K_low and q.K_high have a strict ordering;
- number of those strict exact-score orderings that remain unresolved because
  Hamming distance is tied.

This control demonstrates what information the signature does not encode. It
must not be converted into a pass/fail threshold for the primary angular panel.

## Acceptance criteria

1. Projector dimensions and bit widths must be non-zero.
2. Input length mismatches fail closed.
3. Non-finite input values fail closed.
4. Index conversion/arithmetic used to derive deterministic hyperplanes fails
   closed rather than wrapping silently where relevant.
5. Repeated projection of identical input/configuration is bit-identical.
6. Positive scalar multiplication preserves the projected signature for finite
   non-zero scale in the explicit regression fixture.
7. F2 and M13B.2 packed representations and Hamming distances agree exactly.
8. The experiment uses the actual M13B.2 `BooleanAttentionSignature` and
   `HammingAdmissionRule`.
9. Dense q.k labels do not participate in projection or admission.
10. The full exploratory CSV is byte-identical across two executions in CI.
11. Existing MAA-10/11/12 evidence remains unchanged.
12. No stable API, default routing, GPU kernel or performance claim is changed.

## Decision boundary

MAA-13a does not preregister a winning width or admission threshold. It
characterizes whether Q/K-derived signatures carry materially more relevant
structure than the earlier unrelated feature bits.

A successor MAA-13b may combine the resulting signature/mismatch bits with F2
and Zhegalkin only if MAA-13a supplies a concrete reason to do so. Any later
confirmatory series must freeze a new policy identity plus distinct
tuning/confirmatory dataset identities before observing the holdout.

## Non-claims

This work does not establish novelty, real-model quality, latency, throughput,
bandwidth, energy efficiency, memory savings, GPU usefulness or production
readiness. It does not replace exact q.k or softmax.
