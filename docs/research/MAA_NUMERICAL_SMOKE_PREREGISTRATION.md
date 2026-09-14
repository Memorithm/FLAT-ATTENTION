# MAA-10a: bounded numerical smoke preregistration

Protocol version: `maa-numerical-smoke/v1`.
Committed before the first execution of its implementation. This is a synthetic
correctness/diagnostic smoke, NOT a confirmatory quality or performance campaign.
Do not change the cases or acceptance criteria after observing results; use a
new protocol version for a different experiment.

## Question

Can the qualified host MAA survivor machinery drive the existing numerical FLAT
oracle, preserve all-accept O/LSE behavior, and expose rather than hide quality
loss when a structurally valid filter removes an important key?

## Frozen geometry and numerical inputs

One query, one query head, one KV head, one batch, eight KV tokens, head width two.
Each candidate ID is exactly one original KV token position. Scale is explicitly
1.0. Query is `[1, 0]`. All keys belong to a completed prefix; the oracle is called
with `causal = false`, no RoPE, no ALiBi and no position-dependent term. Gathered
survivors remain in increasing original token order. Compaction is valid ONLY for
this restricted geometry; this is not a general positional-attention adapter.

Cases:

- `all_accept`: K_i = `[i/4 - 1, (i mod 3) - 1]`, V_i = `[i/8, 1 - i/8]`.
  All Boolean admissions and all three predicate features are true.
- `constant_values`: the same K, every V_i = `[2, -3]`.
- `dominant_key_dropped`: K_2 = `[12, 0]`, V_2 = `[10, -10]`; every other K/V
  row is `[0, 0]`.
- `empty_selection`: the same numerical K/V as `all_accept`, with the first
  predicate feature false for every candidate.

Except in `all_accept`, Boolean admission B_i is `i != 3 AND i != 7`.
Except in `all_accept` and the stated empty-selection override, the three
candidate features are `[i mod 2 == 0, i < 6, i != 2]`.
F2 computes the affine predicate `x_0` with zero constant; Zhegalkin computes the
explicit degree-two polynomial `x_1 * x_2`. Recomposition requires all selected
predicates to qualify. These are fixed STRUCTURAL features, not learned Q/K
projections or a claim of content-sensitive admission.

M13B masks are built with the existing canonical BooleanAttentionMask. The MAA
adapter also carries canonical packed structural policy vectors: an all-true
query-side eligibility vector and the key-side eligibility vector B. These are
explicitly fixture policy metadata, NOT numerical Q/K signatures, Hamming-search
evidence, or an attestation of the full Q/K/V input. Optional KV generation is
absent because no mutable KV cache is instantiated in this smoke.

## Frozen arms and controls

Run eight arms for each case: dense, Boolean-only, Boolean+F2, Boolean+Zhegalkin,
Boolean+F2+Zhegalkin, density-matched first positions, density-matched last
positions, and density-matched seeded-priority positions. All controls select
from the Boolean-admitted set and have exactly the combined arm's survivor
count. Seeded priorities use the specified SplitMix64 mixing function on
`candidate_id XOR 0x4d41410a`; break ties by candidate ID and restore original
position order before numerical evaluation. This is one reproducible pseudo-
randomized control, not an unbiased random experiment or an independent holdout.

Max-plus is deliberately NOT a numerical rejection filter here. Temporal
readiness/defer accounting remains a separate pending experiment; lateness must
not silently become proof of numerical irrelevance.

Dense relevance is the single maximum scaled Q.K score from the same frozen
inputs; ties prefer the smallest original key index. Label construction is
explicit dense work used offline for evaluation, never supplied to the filters.

## Execution and observations

Reuse `flat_attention::forward_reference_grouped_asymmetric` for every nonempty
arm; do not introduce another softmax recurrence. Validate selected IDs and full
input shape/finiteness before gathering. An empty arm records `no_survivors`
with zero score evaluations and NO fabricated output/LSE or dense fallback.
Non-finite numerical results are errors, not successful evidence.

Emit deterministic CSV rows containing protocol, case, arm, selection IDs,
selected-token/score count, retained dense-top1 count, and observed maximum
absolute output error / absolute LSE difference against the dense oracle.
Numbers are diagnostics, not a promotion decision. MAA's matched evidence
constructor independently checks each algebraic arm using derived reference
labels and the actual canonical survivor set; latency observations stay absent.
The count of selected tokens equals the number of scalar Q.K score evaluations
in this one-query, one-head oracle ONLY. The full suite additionally executes
reference/label/gather/validation work; counts are not runtime savings.

## Acceptance criteria frozen before execution

1. Every nonempty `all_accept` arm reproduces the dense O/LSE within 1e-6 absolute
   tolerance, with exactly eight selected keys.
2. `constant_values` combined output error is at most 1e-6, but its LSE difference
   is greater than 1e-3. Output equality alone must not be called full parity.
3. `dominant_key_dropped` combined retains zero dense-top1 keys and exhibits output
   error greater than 9.0. This is a REQUIRED negative control, not a failure to
   hide or tune away. Boolean+F2 must still retain the dominant key.
4. Empty selections have no numerical output, no LSE error and no fallback.
5. Controls are nested in Boolean admission, unique, in original position order,
   exactly density matched, deterministic for the frozen seed and independent
   of the dense relevance label.
6. Duplicate, unsorted and out-of-range selection IDs, malformed inputs and
   non-finite results fail closed.
7. Existing MAA tests and full repository qualification remain green. Run this
   example in both debug and release tests and retain the exact-head CSV in CI.

No latency timer, hardware throughput, physical traffic measurement, model
accuracy, learned policy, general sparsity superiority, stable API promotion or
GPU candidate is part of this slice. Full MAA-10/11 still need representative
inputs, independent held-out cases, repeated controls, real input/predicate
hashes, complete measurement scopes and a numerical quality budget.
