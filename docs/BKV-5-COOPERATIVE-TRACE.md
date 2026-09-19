# BKV-5 bounded cooperative timing trace

Status: **research evidence infrastructure only**. This contract does not demonstrate CPU/GPU overlap, acceleration, physical traffic reduction, model quality, or a production scheduling policy.

`api::bkv5_cooperative_trace` binds caller-observed monotonic timing to one already-qualified `CooperativeSelectionTicket`. It exists so later target-backend experiments can retain first-token and steady-state timing without moving queue/scheduler ownership into FLAT-ATTENTION.

The trace records selection readiness, qualified handoff, numerical-consumer start/finish, and an optional complete interval for production of the next Boolean decision. All timestamps are unsigned nanoseconds relative to one caller-owned monotonic trace origin; no wall-clock identity is inferred.

Validation fails closed when handoff precedes selection readiness, numerical execution precedes handoff, an interval is reversed, or only one endpoint of the optional next-Boolean interval is present. A complete next-Boolean interval yields the exact interval intersection with current numerical execution. `None` means the interval was not observed; `Some(0)` means both intervals were observed but did not overlap.

The overlap value is data, not a promotion verdict. The module owns no clock, WGPU submission, CPU worker, thread, queue, transfer, synchronization primitive, scheduler, or benchmark harness. A future BKV-5 qualification must collect this contract from a real target backend, bind device/runtime/source provenance, include transfer and synchronization costs, retain dense fallback, and compare end-to-end first-token and steady-state latency before making any performance claim.
