# Multi-Algebra Attention (MAA) research contract

Status: research-only. This document does not change `flat_attention::api::v1`, default routing, or any performance claim.

## Objective

Evaluate whether FLAT-ATTENTION can reduce unnecessary numerical attention work by coordinating four algebraic domains without conflating their laws:

1. classical Boolean admission and bit-packed routing;
2. linear/affine computation over the finite field `F2`;
3. algebraic-normal-form (Zhegalkin) polynomials for explicit nonlinear Boolean interactions;
4. max-plus constraints for readiness, priority, and scheduling decisions.

Dense qualified FLAT attention remains the semantic and numerical reference. A research decision layer may reject work only under an explicitly qualified policy; it must never silently weaken the existing correctness contract.

## Existing foundation

FLAT already owns Boolean masks, packed Q/K signatures, Hamming/XNOR-style admission, Boolean KV metadata, and WGPU routing before numerical K/V staging. MAA must reuse those contracts rather than duplicate them.

The first MAA implementation is therefore an algebraic foundation, not a new attention kernel.

## Algebraic domains

### Boolean control plane

Boolean logic owns discrete control decisions such as admitted/rejected, ready/not-ready, and feature predicates. Existing M13B contracts remain authoritative for bit-packed attention admission.

### `F2`

`F2 = {0, 1}` with addition `XOR` and multiplication `AND`.

For vectors `x, y in F2^n`:

```text
x + y = x XOR y
<x, y> = XOR_i (x_i AND y_i)
```

An affine predicate is:

```text
f(x) = b XOR <a, x>
```

The MAA `F2` implementation is a deterministic scalar/packed-bit oracle. It is not a performance claim and is not a replacement for the existing Boolean attention API.

### Zhegalkin / algebraic normal form

A Boolean function may be written uniquely as an algebraic-normal-form polynomial over `F2`:

```text
P(x) = a_empty XOR XOR_{S != empty} a_S * product_{i in S} x_i
```

with `x_i^2 = x_i`.

MAA explicitly decomposes:

```text
P(x) = L(x) XOR N(x)
```

where `L` contains degree 0 and 1 terms and can be evaluated through the `F2` layer, while `N` contains degree >= 2 interactions. Zhegalkin work must therefore reuse the qualified `F2` semantics rather than implement a second incompatible notion of Boolean arithmetic.

### Max-plus

The max-plus semiring uses:

```text
a (+_max) b = max(a, b)
a (*_plus) b = a + b
```

with an explicit negative-infinity element.

MAA uses max-plus only for temporal/readiness constraints unless later evidence justifies a broader role. Example:

```text
t_route = max(t_query + d_signature,
              t_metadata + d_metadata)

t_attention = max(t_route + d_route,
                  t_kv_ready + d_kv)
```

This is a scheduling model, not a claim that GPU concurrency or latency improvement exists. Such claims require trace/timing evidence under the existing M13B.4 evidence rules.

For the default MAA temporal interpretation, Max-Plus is applied **after** semantic survivor qualification. A finite readiness time may defer an already-qualified survivor but does not make it numerically irrelevant. Unreachable readiness for an already-qualified survivor is a fail-closed contract error. A separate explicitly declared policy may still use a deadline as an admissibility predicate in research code, but that policy must not be confused with the default readiness/defer model.

## Cooperation model

The intended research graph is:

```text
Q/K state
  -> existing Boolean signatures / admission metadata
  -> optional F2 linear/affine predicates
  -> optional Zhegalkin nonlinear predicates
  -> survivor set
  -> optional max-plus readiness / priority plan
  -> exact qualified FLAT attention for survivors
```

The domains are cooperative, not mandatory serial filters. A plan may legitimately use only a subset of them.

## Matched evidence model

Host qualification compares three arms under one workload identity and candidate geometry:

1. dense qualified FLAT reference;
2. Boolean-only M13B control;
3. nested multi-algebra candidate using Boolean admission plus at least one additional algebraic domain.

Logical score accounting and physical latency observations are deliberately separate. Avoided exact-score evaluations are algorithmic evidence only and must not be translated into latency, bandwidth, energy, or device-utilization claims. Latency evidence is comparable only when all three arms are present with matched sample counts. Survivor-quality evidence is recorded relative to the same dense-reference relevant set.

For this first nested matched design, the multi-algebra arm is not allowed to resurrect work rejected by the Boolean control. Its recorded survivor set must agree with the deterministic host qualification oracle.

Confirmatory MAA evidence additionally requires an explicit frozen-policy boundary. A confirmatory observation must be bound to the preregistered source revision, ordered algebraic route, recomposition policy, predicate identity, feature-schema identity and holdout dataset identity. Exploratory/tuning and confirmatory dataset identities must remain distinct. This contract prevents silent post-observation policy substitution; it does not by itself prove that a human never inspected a holdout before the freeze.

## Promotion rules

No MAA component may enter the stable public API or default runtime routing until all applicable gates are satisfied:

- mathematical definition is frozen;
- deterministic Rust oracle tests are green;
- all-accept behavior preserves the existing dense reference contract where applicable;
- malformed dimensions or metadata fail closed;
- no hidden CPU fallback is presented as GPU execution;
- portable GPU work, if added, is qualified against the Rust oracle;
- any performance claim is tied to reproducible real-device evidence;
- quality/sparsity claims include dense and matched-control baselines;
- confirmatory claims use a policy/dataset identity frozen before confirmatory observation.

## Implementation milestones

The original foundation plan was expanded into smaller reviewable slices as the repository contracts became concrete:

- **MAA-0**: mathematical contract and repository boundary.
- **MAA-1..4**: typed algebra primitives, exact bridges, max-plus oracle, cooperation routing, and canonical M13B Boolean evidence binding.
- **MAA-5**: per-candidate multi-algebra qualification with explicit policy-defined recomposition.
- **MAA-6**: deterministic survivor-set construction with per-domain rejection attribution.
- **MAA-7**: schema-versioned matched host evidence versus dense FLAT and Boolean-only routing.
- **MAA-8**: derive matched arms from the canonical mask, survivor oracle and one reference relevance set.
- **MAA-9**: bind evidence to its original route, metadata generation and candidate mapping; validate realizable intersection and observed-latency summaries. See [integrity gates and continuation protocol](MAA_EVIDENCE_INTEGRITY.md).
- **MAA-10a**: preregistered bounded numerical smoke through the existing FLAT oracle, with structural F2/Zhegalkin ablations, matched-density controls, O/LSE diagnostics and required negative/empty controls. See [protocol](MAA_NUMERICAL_SMOKE_PREREGISTRATION.md). Not a performance campaign or completion of full MAA-10.
- **MAA-10b**: separate Max-Plus readiness/defer planning from relevance rejection, require exact eventual survivor preservation, and exercise deterministic chain/fork-join/shared-node schedules. See [protocol](MAA_MAX_PLUS_READINESS_PREREGISTRATION.md).
- **MAA-10c**: require a complete Max-Plus readiness snapshot to reconstruct the exact original semantic survivor list and prove bit-for-bit O/LSE equivalence through the existing FLAT numerical oracle. See [protocol](MAA_MAX_PLUS_NUMERICAL_EQUIVALENCE_PREREGISTRATION.md).
- **MAA-11a**: freeze policy/source/route/recomposition/predicate/features/tuning-dataset/confirmatory-dataset identity before any confirmatory holdout observation. See [protocol](MAA_HOLDOUT_POLICY_FREEZE_PREREGISTRATION.md).
- **MAA-11b**: execute the frozen Boolean+F2+Zhegalkin synthetic holdout without retuning. The preserved confirmatory classification is `FRONTIER_REJECT`; this is negative evidence for that exact strict-conjunction policy, not a rejection of all MAA policies.
- **MAA-12a**: preregister and explore a batch-level survivor floor that may rescue only Boolean-qualified candidates using deterministic F2/Zhegalkin support, while keeping Max-Plus outside semantic rescue and never reusing the MAA-11b holdout for tuning. See [protocol](MAA_SURVIVOR_FLOOR_EXPLORATORY_PREREGISTRATION.md).
- **MAA-12b**: only if MAA-12a motivates a concrete successor policy, freeze a new policy identity and new tuning/confirmatory dataset identities before any confirmatory observation.
- **MAA-GPU**: portable WGPU candidates only after host confirmatory evidence justifies a specific GPU hypothesis.

## Non-goals for the foundation series

The first series does not:

- replace softmax;
- replace exact `Q.K` scoring;
- alter default kernel selection;
- claim lower latency, higher throughput, lower energy, or lower physical memory traffic;
- duplicate the M13B Boolean mask/signature contracts;
- promote any research type into `api::v1`.
