# BPS-26 — Explicit per-channel cost/topology routing caps

## Reason

BPS-19 selects exact direct predecessor reads or grouped rotations using
a caller-supplied count cap and each channel's number of distinct shifts.
BPS-20/21 retain dense-input and independently full-channel shortcuts.
BPS-23/24 prepared a caller-selected global Hamming-weight dispatch;
BPS-25 introduced buffer-owned streaming.

This independent research-only BPS-26 extension addresses the mismatch
between one fixed per-channel direct-lane cap and the different shift
diversities and routing costs of four operation channels.

## Rust implementation

`RoutingCostModel::new` validates four explicit `ChannelRoutingCost`
values and four caller-set hard caps (zero means direct routing disabled).
A cost contains positive per-lane and per-shift units and nonnegative
fixed setup units. **The library does not measure or infer these costs**:
the values are externally supplied hypotheses, not embedded timing claims.

`RoutingCostModel::thresholds(unique_shift_counts)` derives a separate
direct-lane cap for each operation. Let U be its number of unique source
rotations and n its number of currently active destination lanes. A
candidate direct route may be selected only if:

```text
n <= hard_cap, n < U,
direct_setup + n * direct_lane_units
    < grouped_setup + U * grouped_shift_units
```

The largest qualifying n becomes the channel's cap. The computation is
performed once using u64 arithmetic; U is bounded by the graph's 64
neurons and cost units are u32, so products cannot overflow.

`BooleanNeuronGraph::prepare_cost_router(model)` compiles those caps once
and borrows the immutable graph. `PreparedCostRouter::decision` exposes
the global all-dense gate override or four BPS-19 adaptive decisions.
`step` reuses the existing exact BPS-20/21 hybrid executor;
`scan_into` accepts caller-owned output storage, preserves recurrent
state across chunks and rejects length mismatch atomically.

Gate masks come from the *current input*; the source state is the same
previous word for all operators, preserving retain/excite/inhibit/toggle
order. Rewiring creates a separate graph and requires fresh preparation.
There is no hidden switch of production defaults.

## Qualification

- Invalid zero cost is rejected before compilation.
- Deterministic threshold tests cover uniform channels, heterogeneous
  channels, cheap/expensive direct reads, caps and boundary conditions.
- Exhaustive Hamming weights 0..=64, five topology buckets and three
  contrasting cost models are compared with the independent scalar
  64-neuron recurrence, including **final state**, not only output.
- Chunk boundaries, empty chunks, rewiring/restoration, and mismatched
  buffers are checked.
- The example emits an exact-output/state smoke. Its cost values are
  deliberately illustrative and **are not Dell measurements**.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p flat-algebraic-attention bps26_
cargo run -p flat-algebraic-attention --example bps26_topology_cost_smoke
```

## Scientific boundary

The derived cap is a simple upper-bound cost heuristic. The grouped path
may skip disabled shifts at runtime, while this model charges its full
compiled U. Therefore there is **no claimed speedup, ideal routing
optimality, power benefit or model-quality improvement**. Empirical
costs should be trained on a development corpus only and independently
qualified with fresh paired physical trials, through **Memorithm/RemoteOps
only**. BPS-23/BPS-24 default paths and production services are untouched.
