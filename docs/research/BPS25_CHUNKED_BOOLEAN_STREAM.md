# BPS-25 — Exact chunked Boolean recurrent streaming without output allocations

## Motivation

BPS-23 (FLAT-ATTENTION PR #410) added explicitly configured, exact
density/topology-based kernel routing. BPS-24 (PR #412) prepared that plan
once per trajectory, but `forward` still returns a newly collected `Vec<u64>`
for every sequence. Applications processing live streams or fixed-size
network/control frames need a continuous recurrent state and a caller-owned
output buffer without building a full trajectory vector.

## Functional implementation

The research-only Rust module `boolean_neuron_stream` adds two methods
to `PreparedWeightRouter<'_>`:

- `scan_into(inputs, &mut state, outputs)`: consumes one chunk in order
  into an existing `&mut [u64]`, no output allocation, retaining the
  **updated state** as the next chunk's initial state. Input and output
  lengths must match exactly; a mismatch returns
  `BooleanStreamError::OutputLengthMismatch` before changing *any*
  state/output value. The method does not allocate.
- `scan_visit(inputs, &mut state, visit)`: sends each computed output
  to a caller-supplied `FnMut(u64)` sink. The recurrence is advanced
  before calling the sink, including when a sink panics. Sink behavior
  and any allocations it performs are the caller's responsibility.

The existing `forward`, `forward_weight_routed` and per-step APIs
remain **unchanged**. Every channel still reads the same previous
64-neuron word; gate evaluation, inhibition, toggling, output filtering,
input order and per-topology dispatch remain bit-exact.

The prepared plan immutably borrows its graph. Rewiring with
`with_edge` creates a new graph and requires a new plan; no stale
cache or hidden topology mutation is possible.

## Correctness qualification

- Invalid output buffer sizes (too short, too long, empty mismatch)
  fail atomically.
- Input Hamming weights **0 through 64**, five source-uniformity
  buckets, and disabled, all-dense, and mixed density policies
  are checked against independent `step_scalar` states and outputs.
- Irregular slices, repeated empty slices, topology rewiring,
  restoration, and callback delivery are compared against the
  full scalar 64-neuron recurrence, including the *final state*.
- A standalone executable smoke enforces these invariants in CI.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p flat-algebraic-attention bps25_
cargo run -p flat-algebraic-attention --example bps25_chunked_stream_smoke
```

## Limits and next qualification

This change makes a **provable API-level allocation avoidance** possible
with caller-owned buffers, but claims **no measured speedup**, lower
energy consumption, or improved language-model accuracy. The physical
BPS-23 result does not prove BPS-25 performance. Any physical
paired validation must use pinned revisions, an independent scalar
oracle, independent evaluation corpus and **Memorithm/RemoteOps as the
sole remote control plane**. No runtime defaults, production service
configuration, host settings, secrets, or GPU dependencies change.
