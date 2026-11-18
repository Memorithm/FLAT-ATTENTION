# BPS-18 — Per-operation uniform compilation with mixed graph topology

Research-only, exact-output CPU optimization of the per-neuron Boolean
operator graph introduced in BPS-16 and extended in BPS-17.

BPS-17 dispatches the entire 64-neuron graph to the BPS-15 kernel when
**all four** operation channels have a single shared predecessor rotation.
Previously, rewiring just one lane of just one operation disabled that
whole-model fast path and forced every other channel through the generic
per-shift iteration loop.

BPS-18 computes `channel_uniform_shifts: [Option<u8>;4]` at topology
construction. When a graph is mixed, the generic path first checks
each operation independently:

- `Some(shift)`: a single packed rotation followed by the channel's
  input-gate mask, bypassing the grouped-mask walk.
- `None`: retain the exact per-shift grouped reference, with the BPS-17
  rule to avoid zero-contribution shift groups in sparse mode.
- When all channels are uniform, keep BPS-17's existing direct BPS-15
  **whole-model** dispatch, unchanged.

All inputs, gates, existing 256 potential individually rewired sources,
state transition order, initial-state handling and outputs are unchanged.
The new `compiled_channel_uniform_shifts()` accessor allows callers
to inspect which routes were compiled. `with_edge()` reconstructs
the optimization from the new topology, avoiding stale caching.

## Why this matters

A graph with one heterogeneous operation and three uniform operations
previously paid four generic grouped-loop setup costs. BPS-18 limits
that cost to the heterogeneous operation. It may reduce instructions
and improve throughput on such mixed graphs, but the actual outcome
must be measured; instruction-level compilation, cache behavior and
input gate density can offset the expected savings.

The preceding physical BPS-16 experiment (RemoteOps run
[44287889625](https://github.com/Memorithm/RemoteOps/actions/runs/44287889625))
quantified an excessive cost of the general grouped path under fully
shared topology. It **does not** directly quantify mixed-topology
BPS-18 performance.

## Qualification

The existing scalar per-neuron oracle is independent of the new
specialized implementation. Deterministic randomized tests now cover
fully shared, one/two heterogeneous channels, per-lane gates, sparse
and dense variants, and invalidation/restoration after rewiring.
The exact-head CI requires the new smoke:

```bash
cargo fmt --all -- --check
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
cargo test -p flat-algebraic-attention boolean_neuron_graph
cargo run -p flat-algebraic-attention --example bps18_partial_uniform_smoke
```

Next physical work, strictly controlled by Memorithm/RemoteOps, should
compare BPS-17 and BPS-18 on the same **mixed** topology with whole-output
oracle, paired workloads and a fair baseline. Shared-route BPS-17
measurements remain independent from this change. No speedup, joule
reduction, arbitrary adjacency learning or LSTM equivalence is claimed.
