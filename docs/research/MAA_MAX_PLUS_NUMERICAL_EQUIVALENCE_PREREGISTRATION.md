# MAA-10c: Max-Plus final-readiness numerical equivalence preregistration

Protocol version: `maa-maxplus-numerical-equivalence/v1`.

This experiment is frozen before its first execution. It connects MAA-10b's
readiness/defer semantics to FLAT's existing deterministic numerical attention
oracle without turning an intermediate readiness snapshot into a new attention
semantic.

## Question

When Boolean/F2/Zhegalkin have already produced a semantic survivor set and
Max-Plus only schedules those survivors, does the **complete** readiness snapshot
at the critical ready time produce exactly the same FLAT numerical result as the
original survivor set?

The required invariant is:

```text
final_ready_ids == original_survivor_ids
forward_reference(final_ready_ids) == forward_reference(original_survivor_ids)
```

Both equalities are required in the original numerical candidate order.

## Frozen semantic fixture

There are six logical key/value candidates with head dimension two and one
non-causal query.

Semantic qualification uses three distinct domains before Max-Plus:

- Boolean admits candidates `0..=4` and rejects candidate `5`;
- F2 rejects candidate `1` and accepts the other Boolean-admitted candidates;
- Zhegalkin rejects candidate `4` and accepts the other surviving candidates.

The frozen semantic survivor set is therefore:

```text
[0, 2, 3]
```

Max-Plus then operates only on that survivor set. It uses a three-node chain
with feasible coordinates `[0, 2, 5]` and bindings chosen so that temporal order
is deliberately different from numerical order:

```text
candidate 3 -> node 0 -> time 0
candidate 0 -> node 1 -> time 2
candidate 2 -> node 2 -> time 5

readiness order: [3, 0, 2]
numerical order: [0, 2, 3]
```

Frozen cutoffs are `0`, `2`, `4`, and `5`.

## Acceptance criteria frozen before execution

1. The semantic qualifier produces exactly `[0, 2, 3]` and records one Boolean,
   one F2, and one Zhegalkin rejection.
2. The Max-Plus planner accepts that survivor set because its qualification route
   does not contain Max-Plus.
3. Readiness order is exactly `[3, 0, 2]` and is not reused as numerical key order.
4. At cutoffs `0`, `2`, and `4`, the readiness snapshot is incomplete and the
   final-output helper returns no complete candidate list.
5. At cutoff `5`, the snapshot is complete and returns exactly `[0, 2, 3]`.
6. FLAT `forward_reference` evaluated on the complete snapshot is bit-for-bit
   equal to `forward_reference` evaluated directly on `[0, 2, 3]`, for both O
   and LSE.
7. No dense fallback, zero output, or partial output is fabricated for an
   incomplete snapshot.
8. The emitted CSV is byte-for-byte identical across two executions on the same
   exact head.
9. Existing MAA, repository, WGPU, security and documentation gates remain green.

## Explicit non-claims

The Max-Plus coordinates are abstract readiness values, not nanoseconds. This
experiment does not demonstrate GPU overlap, asynchronous execution, first-token
latency improvement, throughput, bandwidth reduction, energy reduction, or model
quality. It does not change `flat_attention::api::v1` or the production router.

A future device experiment may bind abstract readiness to measured M13B.4 trace
and timestamp evidence, but only after this host numerical equivalence gate is
closed.
