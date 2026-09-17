# MAA-10b: Max-Plus readiness/defer preregistration

Protocol version: `maa-maxplus-readiness/v1`.

This slice tests temporal cooperation only. It does **not** use Max-Plus lateness
as evidence that a candidate is numerically irrelevant. Boolean/F2/Zhegalkin
qualification owns the survivor set; Max-Plus receives that already-qualified
set and may only classify a survivor as ready now or deferred until a reachable
feasible time.

## Question

Can Max-Plus attach deterministic readiness semantics to an existing MAA survivor
set while preserving the exact eventual candidate set and keeping temporal order
separate from numerical key order?

The required invariant is:

```text
survivors_after_all_reachable_dependencies == original_qualified_survivors
```

A candidate that is not ready at an intermediate cutoff is **deferred**, not
rejected. An unreachable survivor is a contract error and fails closed.

## Frozen semantics

- The input survivor set is produced by the existing deterministic MAA host
  qualification machinery before the readiness planner is called.
- Each survivor has exactly one explicit binding to a Max-Plus schedule node.
  Several survivors may share the same node.
- `MaxPlusSchedule::earliest_feasible_times` is the only readiness recurrence.
- A finite feasible time `t_i <= cutoff` means ready at that cutoff.
- A finite feasible time `t_i > cutoff` means deferred until `t_i`.
- `-infinity` for an already-qualified survivor is an error, never a numerical
  rejection.
- The temporal readiness ordering may differ from the survivor/numerical order.
  Snapshots returned for numerical consumption preserve the original survivor
  order.
- No latency is measured or inferred. Max-Plus values in this protocol are
  abstract deterministic readiness coordinates, not nanoseconds or device time.

## Frozen deterministic cases

The CI smoke emits CSV twice and requires byte-for-byte equality on the same
runner.

1. `chain`: survivors `[0,1,2]`; schedule `0->1 +3`, `1->2 +4`; readiness
   coordinates `[0,3,7]`; cutoffs `[-1,0,2,3,6,7]`.
2. `fork_join`: survivors `[0,1,2,3]`; edges `0->1 +2`, `0->2 +5`,
   `1->3 +7`, `2->3 +1`; readiness `[0,2,5,9]`; cutoffs `[0,2,5,8,9]`.
3. `original_order`: survivors `[10,5]`; candidate 10 becomes ready at 4 while
   candidate 5 is ready at 0. Temporal order is `[5,10]`, but the complete
   numerical snapshot must remain `[10,5]`.
4. `shared_node`: survivors `[7,4]` share a node ready at 2 and must become ready
   together without reordering the numerical survivor list.
5. `empty`: no survivors; no critical ready time exists and every snapshot is
   complete.

## Acceptance criteria frozen before execution

1. Ready sets are monotone non-decreasing as the cutoff increases.
2. Deferred sets are the complement of ready survivors at each cutoff.
3. At the critical ready time, every reachable survivor is present exactly once
   and in the original survivor order.
4. Temporal readiness order is exposed separately and may differ from numerical
   order without changing the latter.
5. Duplicate, missing or foreign candidate bindings fail closed.
6. Out-of-bounds nodes and mismatched initial-vector geometry fail closed.
7. A survivor bound to an unreachable node fails closed as
   `UnreachableSurvivor`; it is never converted into a relevance rejection.
8. Empty survivor sets remain valid and have no fabricated critical time.
9. Existing MAA tests, strict lint/documentation gates and the full repository
   qualification remain green.
10. The smoke CSV is deterministic across two executions on the same exact head.

## Explicit non-claims

This protocol does not claim GPU overlap, latency reduction, first-token speedup,
throughput, physical memory-traffic reduction, scheduling optimality or model
quality. It does not promote Max-Plus into the stable API and does not alter the
production attention router.

A later device experiment may map these abstract readiness coordinates to actual
trace/timestamp evidence, but only through the existing M13B.4 rule that overlap
must be demonstrated rather than inferred from source structure.
