# BPS-20 — Exact compiled dense-input guard for Boolean neural routing

**Research-only functional optimization.** This is a direct follow-up
to the Dell T430 measurement of BPS-19 on fully or partially
heterogeneous four-operation recurrent neuron graphs.

## What the physical experiment established

[RemoteOps run #44324663187](https://github.com/Memorithm/RemoteOps/actions/runs/44324663187)
completed 256 exact-output paired trials on 18 Nov 2026. Each timed
trial executes 12 × 2048 recurrence steps, across two source
topologies, four input-word densities, four exact-output execution
paths, and eight alternating-order passes.

Compared with BPS-18 grouped-sparse, median BPS-19 adaptive elapsed
times were lower by roughly 70–80% for zero-density input words and
24–30% for 25%-density words; **but BPS-19 adaptive was roughly 30–51%
slower** at 75–100% density. These are preliminary, single-host
wall-clock results, not energy or downstream model-quality evidence.

## Exact compiled Boolean input predicate

For each of the four operator gates and each of the 64 destination
lanes, the desired *dense* activity condition is that the gate
evaluate to 1. The gate truth table has two bits for that lane:
`when_false` and `when_true`.

At graph construction, derive these exact constraints:

```text
required_one  |= when_true  & !when_false
required_zero |= when_false & !when_true

impossible if any gate has a lane with both truth values 0
impossible if required_one and required_zero overlap
```

When possible, input satisfies the dense predicate exactly when:

```text
(input & required_one) == required_one
(input & required_zero) == 0
```

This is a *logical equivalence*, not a statistical predictor. When it
holds, every input-dependent recurrent gate word is exactly
`u64::MAX`; when it does not, the algorithm still uses the original
exact BPS-19 adaptive path. Write and output gates are handled by the
selected exact recurrence and need not be dense.

`step_activity_hybrid` and `forward_activity_hybrid` use this
precompiled predicate to dispatch:

- All four recurrent gates fully active: existing BPS-18 grouped-dense
  execution, which measured better on dense input words.
- Anything else: exact BPS-19 adaptive, including direct reads of
  active destinations where this can skip many source rotations.

No source shift, mask, update order, old-state semantics, output bit
or learning result is changed. `compiled_dense_gate_guard` and
`dense_gate_eligible` allow independent inspection of the predicate.
`with_edge` rebuilds the cached condition and the source topology.

## Tests and reproduction

- Exact guard compiler for conflicting, impossible and satisfiable
  multi-gate truth tables.
- Random truth tables compared against **actual full-word evaluation**
  of all recurrent gates, including ambiguous mixed bit patterns.
- Independent scalar 64-neuron output oracle for randomized
  heterogeneous topologies, sparse/dense/mixed input sequences and
  multiple adaptive policy thresholds.
- Rewiring checks that the compiled guard remains semantically valid.
- Smoke executable under exact-head GitHub CI.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p flat-algebraic-attention bps20_
cargo run -p flat-algebraic-attention --example bps20_dense_guard_smoke
```

**Performance claim is pending.** BPS-20 has an extra input guard and
a potential direct grouped fallback. It must be benchmarked on the
same mixed topologies and corpus as BPS-19 under the exclusively
RemoteOps-controlled Dell protocol, with exact source pin and full
output equality. On other input distributions, the guard may cost
more than it saves. Do not infer energy or ML quality from synthetic
CPU wall-clock timings.
