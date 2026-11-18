# BPS-24 — Prepare the exact Boolean routing plan once per sequence

## Direct experimental motivation

The independent-seed [BPS-23 Dell comparison #44405185428](https://github.com/Memorithm/RemoteOps/actions/runs/44405185428)
successfully recorded 512 exact-output paired trials. A caller-supplied
BPS-23 density policy reduced median BPS-22 latency by approximately
70.6% for a four-heterogeneous-channel graph with input Hamming weight
32. However, the *direct* BPS-18 grouped-dense arm was approximately
2.3% faster than the routed BPS-23 arm in that panel. A small
per-sample policy dispatch also produced an observed 1.7–1.9%
latency regression at weights 0/1 for the four-heterogeneous graph.
These figures are single-host wall-clock observations, not an
energy or language-model-quality measurement.

BPS-23 recomputed the count of uniform source channels on
**every recurrent update**. That count depends on the graph topology,
not the live input word. BPS-24 removes this redundant work in
multi-step inference.

## Actual Rust implementation

`BooleanNeuronGraph::prepare_weight_router(policy)` produces a
`PreparedWeightRouter<'_>` borrowing the immutable graph and retaining
the caller's already validated policy. It resolves the graph's
uniform-source channel count and selects one of three exact dispatch
conditions at preparation time:

- `AlwaysHybrid`: no density window is configured for this graph,
  so no runtime popcount is needed.
- `AlwaysDense`: the configured inclusive window covers all
  possible Hamming weights, 0..=64, so no runtime popcount is needed.
- `WeightRange { first, last }`: only here do individual input words
  use `input.count_ones()` before choosing dense versus adaptive.

`PreparedWeightRouter::decision`, `step`, and `forward` expose
direct control over the prepared routing plan. The preexisting
`BooleanNeuronGraph::forward_weight_routed` now internally prepares
once and delegates the full trajectory; its public signature is
unchanged. `step_weight_routed` remains a valid independent,
unprepared per-step reference.

The prepared executor borrows the graph immutably: a rewired
`with_edge` result is a separate graph and requires its own
correctly resolved plan. **No source shift, gate, output, old-state
ordering or learning rule changes**, and the BPS-23 policy remains
explicitly configured and disabled by default.

## Correctness qualification

Tests compare prepared decisions against the original BPS-23 selector
for **all 65 Hamming weights and all five uniformity buckets** across
disabled, whole-range, narrow-range and research-candidate policies.
They also compare every prepared recurrent step with
`step_weight_routed` and the independent scalar-neuron oracle,
and compare whole trajectories, empty sequences, fresh topology
preparations after rewiring and restored connections.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p flat-algebraic-attention bps24_
cargo run -p flat-algebraic-attention --example bps24_prepared_router_smoke
```

## Strict performance boundary

This change has a clear logical work reduction, but **does not
establish an actual speedup on Dell** until measured with a fresh
paired benchmark. BPS-23 and BPS-24 must run the same inputs and
initial state under exactly pinned source commits. Avoid comparing
unpaired nanoseconds across the BPS-22 and BPS-23 runs, or selecting
thresholds on a held-out evaluation set. Continue to use **only
Memorithm/RemoteOps** for any trusted physical tests. No default
runtime policy is promoted or changed.
