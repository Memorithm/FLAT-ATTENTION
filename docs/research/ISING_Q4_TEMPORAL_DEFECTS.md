# Q4-TOPO-2 — Longitudinal defect accounting

Research-only temporal observer for the Boolean-quaternary Ising controller.
This experiment extends the signed-graph observer from Q4-TOPO-1 without
changing production FLAT numerical attention, WGPU dispatch, SML core or
routing defaults.

## Scientific distinction

Signed-graph structural frustration is determined by the effective graph's
coupling signs on closed cycles. Changing only neuron activity or polarity
cannot change that graph invariant. Changing the structural/causal eligibility
mask or reconfiguring couplings **can** change the effective graph. A count of
negative fundamental cycles may change simply because the cycle basis changed
even when the signed graph remains unbalanced: the count is NOT a basis-free
topological defect count, frustration index or energy-barrier estimate.

A domain wall or unsatisfied active coupling is state-dependent and may move
while a negative signed cycle stays fixed. Consecutive snapshots must therefore
report topology changes separately from defect changes.

## Implemented interface

Module: crates/flat-algebraic-attention/src/ising_defect_dynamics.rs

- Every frozen frame consists of eligibility, activity and polarity word slices,
  and canonical unique signed couplings. Each frame is independently validated.
- The observer computes before/after topology with
  ising_topological_defects::analyze_topological_defects.
- A 4x4 state-transition matrix counts transitions between the four Q4 codes
  **only for vertices eligible at both instants**.
- New eligibility and new masking are counted separately; masked vertices
  are not assigned an artificial inactive or negative state.
- Common effective edges require exactly matching endpoints AND coupling.
  An edge entering because of a causal mask or rewiring is not counted as a
  newly born activity wall or unsatisfied energetic link.
- On common effective edges, activity interfaces, active-polarity walls,
  dormant/latent polarity interfaces and unsatisfied active couplings have
  separate appeared/resolved/persisted counters.
- The same_effective_topology flag is true only when the eligibility mask and
  full effective signed edge identity match exactly, not just edge counts,
  hashes, degree distributions or a matching cycle rank.
- BooleanQuaternary::bitplanes() exposes read-only state planes to capture
  real snapshots before and after strict-energy-descent sweeps.

No unstated time or physics measurement is inferred from two frames.
Specifically, an energy barrier B requires a specified path of intermediate
states and a maximum along that path (for example max E(path) - E(initial)).
Endpoint energy changes alone do not determine B, metastability, memory
capacity, protected topological encoding, or a learning effect.

## Gate and downstream opportunity

Run the targeted regression tests with:

    cargo test -p flat-algebraic-attention --lib ising_defect_dynamics

Then require fmt, Clippy, rustdoc, host tests, WGPU matrix, CodeQL, semver,
supply-chain and all required CI green on the exact PR commit before merge.

Test fixtures include a negative triangle whose *unsatisfied edge moves*
without changing negative-cycle topology; latent polarity preserved under
deactivation; causal mask growth with separate graph births; changed coupling
reclassified as structural rewiring; malformed frame fail-closed; and a real
BooleanQuaternary local energy descent with two observed snapshots.

A later benchmark must explicitly freeze density-matched randomized and
degree-preserving rewired graph controls. Only matched graph/snapshot
sequences support conclusions about dynamic persistence or recall. A signed
cycle is a necessary obstruction to simultaneous edge satisfaction, not
evidence of error correction or a useful memory by itself.

Related research contracts:
- docs/research/ISING_Q4_BOOLEAN.md
- docs/research/ISING_Q4_CAUSAL_ATTENTION_BRIDGE.md
- docs/research/ISING_Q4_TOPOLOGICAL_DEFECT_MONITOR.md
