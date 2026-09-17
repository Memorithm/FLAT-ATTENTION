# MAA-11b: frozen confirmatory synthetic holdout

Status: **PRE-RESULT FREEZE**. This file is committed before any MAA-11b
confirmatory metric is computed or recorded.

Protocol version: `maa-confirmatory-holdout/v1`.

Frozen source revision:

```text
4030b229a2ff329c1263cad677812d659422b448
```

That revision contains the qualified MAA-11a policy/dataset identity contract.
The result-producing harness will be added only in a later commit on this branch.
No policy field, feature mapping, dataset generator or decision criterion below
may be changed in response to the confirmatory result.

## Frozen policy

Policy ID:

```text
maa-11b-structural-v1
```

Algebraic route, in order:

```text
Boolean,F2,Zhegalkin
```

Max-Plus is intentionally absent from semantic rejection. MAA-10b/10c already
qualified its separate readiness/defer role.

Recomposition:

```text
AllSelectedMustQualify
```

Canonical policy payload, including the final newline:

```text
policy=maa-11b-structural-v1
route=Boolean,F2,Zhegalkin
recomposition=AllSelectedMustQualify
boolean=structural_admit
f2=x0
zhegalkin=x1*x2
```

SHA-256 identity of those UTF-8 bytes:

```text
7c36df81b9d213fb5a198da5491211242d7db3ef4c02fa5be432885d03552194
```

Semantics:

- Boolean qualifies iff `structural_admit == true`;
- F2 qualifies iff `x0 == true`;
- Zhegalkin qualifies iff `x1 * x2 == true`;
- a candidate survives only when every selected semantic domain qualifies.

The policy is deliberately simple and is **not** tuned against the holdout below.

## Frozen feature schema

Canonical feature-schema payload, including the final newline:

```text
feature_schema=maa-11b-features-v1
fields=structural_admit:bool,x0:bool,x1:bool,x2:bool
```

SHA-256 identity:

```text
781cd7d2d78194b40083ce68705fe7e99a3130e1d175cdd1ce7d55c0ee8ea5fe
```

## Frozen exploratory/tuning identity

MAA-11b treats the earlier MAA-10a synthetic suite as the exploratory dataset
identity. No MAA-11b threshold or predicate may be tuned on the confirmatory
suite.

Canonical tuning-dataset identity payload, including the final newline:

```text
dataset=maa-10a-synthetic-v1
source=docs/research/MAA_NUMERICAL_SMOKE_PREREGISTRATION.md
cases=all_accept,constant_values,dominant_key_dropped,empty_selection
```

SHA-256 identity:

```text
eb122c2d24751b7c2c81547755c69889e3cdf15fca7efd7a12f8447edf37aef4
```

## Frozen confirmatory dataset generator

The confirmatory suite contains 16 generated cases, each with one query, eight
candidate K/V pairs and head dimension two. It is defined by the generator spec
below rather than by hand-selected post-result fixtures.

Canonical holdout identity payload, including the final newline:

```text
dataset=maa-11b-holdout-v1
cases=16
candidates_per_case=8
head_dim=2
seed=0x4d41412d313162
generator=splitmix64-v1
float_mapping=top24_to_unit_then_2x_minus1
draw_order=per_case:q[2],then_per_candidate:k[2],v[2],feature_word
features=structural_admit(bit0),x0(bit1),x1(bit2),x2(bit3)
relevance=dense_top2_by_dot_score_ties_lower_candidate_id
```

SHA-256 identity:

```text
87585c258c254de0a2d4878581138c8a4a58bad07d446cf9876d9ab24622b3d9
```

### Exact generator semantics

State starts at unsigned 64-bit value `0x004d41412d313162`. Each draw applies
standard SplitMix64 with wrapping unsigned arithmetic:

```text
state += 0x9e3779b97f4a7c15
z = state
z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
z = (z ^ (z >> 27)) * 0x94d049bb133111eb
z = z ^ (z >> 31)
```

For every generated floating value, use bits `z >> 40` as an integer in
`[0, 2^24-1]`, divide by `2^24-1`, then map with `2*u - 1` to `[-1,1]` and cast
to `f32`.

For each case, draw in this exact order:

1. query `q[0]`, `q[1]`;
2. for candidate IDs `0..7`, draw `k[0]`, `k[1]`, `v[0]`, `v[1]`;
3. draw one `feature_word`; bits 0,1,2,3 map respectively to
   `structural_admit,x0,x1,x2`.

The dense relevance target is the two candidate IDs with largest unscaled dot
score `q·k`. Ties are resolved by lower candidate ID. Relevance labels are
computed only inside the result-producing harness after this freeze commit.

## Frozen matched arms

For every holdout case record:

1. dense FLAT: all eight candidates;
2. Boolean-only control: candidates with `structural_admit`;
3. frozen multi-algebra candidate: Boolean + F2 + Zhegalkin under
   `AllSelectedMustQualify`.

No policy may resurrect a Boolean rejection. Empty sparse selections are retained
as negative evidence; no dense fallback or fabricated zero output is credited as
a sparse result.

## Frozen metrics

Aggregate over all 16 cases:

- exact numerical score evaluations = number of selected candidates;
- additional exact-score evaluations avoided by multi-algebra versus Boolean-only;
- retained dense-top2 hits for Boolean-only and multi-algebra;
- false negatives relative to dense top2;
- number of empty sparse selections;
- for non-empty sparse selections, maximum absolute O error versus dense FLAT;
- absolute LSE error versus dense FLAT;
- deterministic per-case and aggregate rows.

No latency, bandwidth, energy, GPU-utilization or physical-memory metric is
inferred from these host counts.

## Frozen decision rule

This first confirmatory stress holdout classifies the frozen multi-algebra policy
as `FRONTIER_PASS` only if **all** conditions hold:

1. total multi-algebra exact-score evaluations are strictly lower than the
   Boolean-only total;
2. total retained dense-top2 hits are not lower than the Boolean-only total;
3. among cases where both sparse arms are non-empty, the sum of multi-algebra
   max-absolute O errors is no greater than the Boolean-only sum plus `1e-6`;
4. among the same comparable cases, the sum of multi-algebra absolute LSE errors
   is no greater than the Boolean-only sum plus `1e-6`;
5. multi-algebra does not create more empty selections than Boolean-only.

Otherwise the classification is `FRONTIER_REJECT`. A rejection is a valid and
preserved scientific result and must not trigger policy modification in this
confirmatory series.

## Required integrity gates

Before producing any result, the later harness must construct a
`FrozenPolicyManifest` with exactly:

- source revision `4030b229a2ff329c1263cad677812d659422b448`;
- policy ID `maa-11b-structural-v1`;
- route `Boolean,F2,Zhegalkin`;
- recomposition `AllSelectedMustQualify`;
- predicate digest `7c36df81...552194` as written above in full;
- feature-schema digest `781cd7d2...8ea5fe` as written above in full;
- tuning digest `eb122c2d...7aef4` as written above in full;
- confirmatory digest `87585c25...22b3d9` as written above in full.

The confirmatory context must bind successfully before the first case is
evaluated. Any mismatch aborts the run.

## Explicit non-claims

This is a small deterministic synthetic holdout, not a real-model evaluation and
not a GPU benchmark. Passing would justify only the statement that this frozen
policy improved the declared synthetic work/quality frontier under the exact
protocol above. Failing must be reported unchanged. Neither outcome establishes
production usefulness, novelty or hardware speedup.
