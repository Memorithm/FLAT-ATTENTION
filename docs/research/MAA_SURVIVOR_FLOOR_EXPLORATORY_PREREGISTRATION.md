# MAA-12a: survivor-floor exploratory preregistration

Protocol version: `maa-survivor-floor-exploratory/v1`.

Status: preregistered exploratory research. This document is committed before
the MAA-12a implementation and before producing MAA-12a numerical observations.

## Motivation

MAA-11b preserved a negative confirmatory result for the frozen strict
Boolean + F2 + Zhegalkin conjunction: the multi-algebra arm reduced score
evaluations but also increased false negatives, numerical error, and empty
selections relative to Boolean-only control.

MAA-12a does not modify, reinterpret, or retune that frozen series. The MAA-11b
confirmatory dataset is not reused for tuning. MAA-12a tests a new batch-level
hypothesis on a distinct deterministic exploratory generator.

## Hypothesis

The existing per-candidate policy `AllSelectedMustQualify` can collapse an
otherwise non-empty Boolean survivor set to too few or zero candidates. A
batch-level survivor floor may reduce that failure mode while preserving the
following invariants:

1. candidates rejected by the canonical Boolean control are never resurrected;
2. strict multi-algebra survivors are never removed by the floor;
3. rescue is bounded by the Boolean survivor set;
4. Max-Plus readiness is not used as semantic rescue evidence;
5. final survivor order remains the original numerical candidate order;
6. rescue priority is deterministic and uses only already-computed auxiliary
   F2/Zhegalkin verdicts, never exact Q.K scores or dense relevance labels.

This is an exploratory structural hypothesis, not a claim of quality or speedup.

## Frozen rescue rule

For each candidate, the already-qualified strict decision exposes Boolean, F2,
and Zhegalkin verdicts. Define auxiliary support as:

```text
support = I(F2 qualifies) + I(Zhegalkin qualifies)
```

Only candidates that pass Boolean admission are eligible for rescue.

For a requested floor `F`:

1. retain every strict `AllSelectedMustQualify` survivor;
2. set `target = min(F, number_of_boolean_survivors)`;
3. if strict survivor count is below `target`, rank rejected Boolean survivors
   by descending auxiliary support;
4. break equal-support ties by ascending candidate ID;
5. rescue exactly enough candidates to reach `target`;
6. return the selected candidates in original input/numerical order.

A floor of zero must reproduce the strict survivor set. A floor at or above the
Boolean survivor count must reproduce the Boolean survivor set.

Max-Plus verdicts are explicitly excluded from this mechanism because MAA-10b
and MAA-10c define Max-Plus as readiness/defer semantics after semantic survivor
qualification. A floor request over decisions carrying Max-Plus evaluation must
fail closed.

## Frozen exploratory sweep

The numerical harness will use:

- deterministic SplitMix64 generator;
- seed `0x004d_4141_2d31_3261` ("MAA-12a");
- 64 cases;
- 8 candidates per case;
- head dimension `D=2`;
- one query/head;
- non-causal exact FLAT host oracle;
- Boolean structural admission from one generated bit;
- F2 predicate `x0`;
- Zhegalkin predicate `x1 * x2`;
- floors `0, 1, 2, 3, 4`;
- dense top-2 unscaled q.k with lower candidate ID tie-break for diagnostic
  relevance only.

Dense top-2 labels must never participate in Boolean/F2/Zhegalkin admission,
support scoring, rescue ranking, or survivor selection.

## Recorded metrics

For Boolean-only control and every floor arm, record aggregate:

- exact-score evaluations represented by selected candidate count;
- dense-top2 hits retained;
- false negatives relative to dense top-2;
- empty selections;
- sum of max-absolute output error on non-empty selections;
- sum of absolute LSE error on non-empty selections;
- number of rescued candidates.

The exploratory harness must emit deterministic machine-readable CSV twice in CI
and require byte-for-byte equality.

MAA-12a does not preregister a winning floor or promotion threshold. Its purpose
is to characterize the work/quality curve without post hoc labeling of one arm
as confirmatory evidence.

## Acceptance criteria

1. The survivor-floor primitive is deterministic.
2. Duplicate candidate IDs fail closed.
3. Missing Boolean verdicts fail closed.
4. Any Max-Plus verdict fails closed.
5. Boolean-rejected candidates are never selected.
6. Strict survivors are never removed.
7. Floor zero equals strict multi-algebra selection.
8. Floor >= Boolean survivor count equals Boolean-only selection.
9. Rescue ties are resolved by candidate ID while final output order remains
   original candidate order.
10. The exploratory CSV is byte-identical across two executions.
11. Existing MAA-10/11 tests and frozen MAA-11b result remain unchanged.
12. No stable API, default router, GPU kernel, or performance claim is changed.

## Next gate

If an exploratory floor or family of floors appears materially better under the
declared diagnostics, a successor MAA-12b must define a new policy identity,
new tuning/confirmatory dataset identities, and a fresh pre-result freeze before
any confirmatory observation. The MAA-11b holdout must not be used for that
tuning or confirmation.
