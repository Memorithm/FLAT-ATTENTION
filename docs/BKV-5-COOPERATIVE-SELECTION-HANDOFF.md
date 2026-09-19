# BKV-5 cooperative selection handoff

Status: **research execution-contract infrastructure only**. This does not demonstrate CPU/GPU overlap, latency improvement, physical traffic reduction, quality preservation, or speedup.

`api::bkv5_cooperative_selection_handoff` closes a narrow correctness gap between the already-qualified M13B.4 first-token readiness contract and the existing Boolean-indexed numerical-KV selection record. It validates one selection at a time; it is not a scheduler or queue implementation.

Before a Boolean selection can cross the cooperation boundary, the handoff requires:

- the selection's own canonical evidence invariants to validate;
- exact generation equality with the qualified first-decode prefix;
- exact live-prefix token equality;
- exact mapped-page coverage equality;
- a non-zero declared pending-decision bound; and
- pending accounting that does not already exceed that bound.

When the pending count is exactly at capacity, the contract returns an explicit `DenseFallbackBackpressure` disposition. It does not enqueue unbounded work and does not defer the Boolean selection for later consumption, which prevents a valid-now selection from becoming a stale decision in an implicit backlog. The existing dense numerical path remains the fallback owner.

The pending count and capacity are caller observations. This module does not own threads, futures, WGPU queues, device submission, retries, leases, transport, or generic scheduling. The companion research-only [`bkv5_cooperative_trace`](BKV-5-COOPERATIVE-TRACE.md) contract now retains bounded caller-observed monotonic intervals for later target-backend qualification. It does not make those observations authoritative performance evidence: target-host measurements must still bind execution provenance and include transfer/synchronization and fallback costs before any overlap or performance claim.
