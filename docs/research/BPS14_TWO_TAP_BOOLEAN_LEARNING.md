# BPS-14 — Two incoming recurrent edges per Boolean neuron

**Research-only implementation.** BPS-14 adds a second, independent
recurrent predecessor connection per output bit and learns both predecessor
offsets and an OR/XOR aggregation choice. The existing BPS-1 through BPS-13
APIs and the production FLAT attention route are unchanged.

- Source: `crates/flat-algebraic-attention/src/boolean_two_tap_memory.rs`
- Runnable smoke: `crates/flat-algebraic-attention/examples/bps14_dual_tap_smoke.rs`

## Exact neural semantics

For 64 bit-packed neurons, each destination bit receives the previous
state from two different source neurons. Shared connection offsets
`a,b ∈ [0,63]` define a fixed fan-in of two per destination neuron:

```text
u[t] = mix(rotate_left(c[t-1], a), rotate_left(c[t-1], b))
c[t] = (forget(x[t]) & u[t]) | write(x[t])
h[t] = c[t] & output(x[t])
mix ∈ {OR, XOR}
```

All 64 neurons update synchronously from the **same old state**.
Truth-table gates depend only on the current input word. The
two incoming synapses are recurrent; they couple previously distinct
neurons. Only their common offset topology is trainable in this
milestone; there is no arbitrary per-neuron adjacency matrix.

Two equal offsets with OR exactly recover the previous single-parent
BPS-2 model. Two equal offsets with XOR cancel the recurrent input
entirely. Two distinct offsets produce behavior unavailable to a
single BPS-2 predecessor: an initial one-bit state with offsets 1 and
3 activates two distinct state bits after one step, and three after two.

Unlike BPS-13, this family does **not** generally admit a compact
two-shift associative representation under composition. Two OR-based
time steps already create three distinct source offsets from one active
initial bit. BPS-14 therefore deliberately uses an O(T) sequential
reference with O(1) additional state (excluding returned outputs).
**Do not substitute BPS-13's one-predecessor scan.** No performance,
energy or model-quality superiority is claimed.

## Implemented topology learning

`fit_topology` performs finite-budget deterministic coordinate descent
over both offsets and the OR/XOR choice while holding the six packed
gate truth-table masks fixed. All candidate errors are computed on
whole trajectories via supervised Hamming loss, not lane-independent
permutation tricks. Strict improvements only; ties keep the incumbent.

The caller supplies separated training and held-out validation
sequences, an output-bit mask and hard limits for epochs and
candidate evaluations. Validation is reported *before and after*
training, but never consulted for mutation selection, tie-breaking or
early stopping. Invalid input and arithmetic errors do not mutate the
model. This is not BPTT, general gate learning, globally optimal
topology discovery or statistically demonstrated generalization.

## Reproducibility and qualification

Deterministic unit tests exercise 64-bit scalar-neuron equivalence,
OR and XOR, duplicate-edge recovery of BPS-2, two-edge supervised
learning, an independent held-out trajectory, invalid inputs, bounded
search, and independence from validation labels.

```bash
cargo fmt --all -- --check
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
cargo test -p flat-algebraic-attention boolean_two_tap_memory
cargo run -p flat-algebraic-attention --example bps14_dual_tap_smoke
```

Full exact-HEAD Rust/WGPU/host-oracle/SemVer/supply-chain/CodeQL
checks are required before merge. No machine is touched and
Memorithm/RemoteOps remains the sole control plane for any subsequent
Dell/Thor/Mac physical qualification.

## Next substantive work

1. Extend topology from shared offsets to **independently gated
   per-neuron edge masks** with bounded adjacency and explicit
   observability; measure actual memory traffic.
2. Compare strict coordinate search with paired edge mutations
   (to escape loss barriers), a similarly budgeted local-plasticity
   baseline and a held-out selective-copy task.
3. Assess whether alternative finite-state algebraic representations
   permit a useful exact block scan without exponential state growth.
   Do not claim BPS-13 asymptotics for the present multi-parent update.
