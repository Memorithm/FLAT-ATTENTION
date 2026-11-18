# BPS-15 — One recurrent connection per Boolean operation

**Research only; no production model routing, no hardware speedup claim.**

Source: `crates/flat-algebraic-attention/src/boolean_operator_channels.rs`.
Executable: `crates/flat-algebraic-attention/examples/bps15_operator_channels_smoke.rs`.

## Why more than two edges?

BPS-14 was a minimal fan-in-two correctness and learning experiment. Its
two inputs are aggregated by a single OR/XOR choice, which prevents
expressing multiple simultaneous recurrent operations. BPS-15 is a
distinct, **operation-specialized** 64-bit packed recurrent model with
four separate predecessor channels:

| Channel | Operator | Value per timestep |
| --- | --- | --- |
| Retain | AND | `rotate_left(state, retain_shift) & retain_gate(input)` |
| Excite | OR | `rotate_left(state, excite_shift) & excite_gate(input)` |
| Inhibit | AND NOT | `rotate_left(state, inhibit_shift) & inhibit_gate(input)` |
| Toggle | XOR | `rotate_left(state, toggle_shift) & toggle_gate(input)` |

The exact synchronous update uses one immutable previous-state
snapshot for **all four sources**, then combines them in a fixed order:

```text
base = retained | excited | write_gate(input)
next_state = (base & !inhibited) ^ toggled
output = next_state & output_gate(input)
```

The four rotations select independently trainable source-neuron
offsets (each 0..63); the gates have input-dependent per-destination
truth-table masks. No state-dependent *gate* or arbitrary adjacency
matrix has been implemented. A disabled gate contributes zero; direct
write is intentionally subject to inhibition, then XOR toggling. These
are specific model semantics and not equivalent to standard LSTM.

When only retention is active, the BPS-2 recurrent Boolean model is
recovered exactly. The four-channel model has greater expressivity
than that one-parent algebra; its arbitrary multi-parent recurrence
does not in general have BPS-13's compact associative prefix scan.

## Execution paths and possible throughput

- `step_dense` is a branchless, packed 64-bit oracle. All four
  channels are evaluated even when disabled.
- `step_sparse` skips source rotations whenever an operation's
  **64-bit input gate mask is exactly zero**; write and output masks
  are still evaluated. It is exact for any initial state and input
  sequence, because the skipped rotation could only have contributed
  zero. Suppressing a nonzero gate or skipping the input sampling
  would not preserve semantics.
- `forward_dense` and `forward_sparse` return one output per tick;
  each is O(T) sequential over T timestamps. No cross-timestamp
  parallelism is claimed.
- `logical_activity` reports enabled and skipped operation sources,
  *not* instructions retired, memory traffic, CPU cycles or joules.

More connections can increase useful parallel instruction-level work
and make a sparse activation policy worthwhile, but can also increase
rotations, gate evaluations and cache pressure, and can be slower
than BPS-14 or BPS-2. The compiler may eliminate dormant work in the
dense version too; tests of logical skip counts alone cannot establish
CPU savings. There is no physical throughput evidence in this PR.

A standalone **opt-in** timing path exists:

```bash
cargo run -p flat-algebraic-attention --release \
  --example bps15_operator_channels_smoke -- --measure
```

It prints raw wall-clock nanoseconds for dense and sparse execution
on the same sampled 8192-word workload over 64 repetitions, including
output allocation. Timings must be conducted through
`Memorithm/RemoteOps` on a resource-isolated host before any
performance statement; compare with BPS-2/BPS-14 on clearly matched
model tasks. The opt-in timing path is **not** part of ordinary CI.
For publishable results additionally require multiple sample seeds,
paired runs, CPU pinning, p50/p95, confidence intervals, NUMA
placement and an instrumented energy meter where available.

## Learning

`fit_shifts` performs budgeted, strict-improvement coordinate
descent over the **four source offsets**, evaluating every candidate
on complete recurrent training trajectories. All operation gates
stay fixed during topology learning. Held-out validation labels
are never used to select updates. Failed validation or loss overflow
does not modify the original model. This is a discrete baseline,
not BPTT, arbitrary graph learning or globally optimal search.

Unit tests cover:
- distinct retention/excitation/inhibition/toggle effects;
- exact recovery of BPS-2 when other channels are disabled;
- randomized bit-by-bit 64-lane scalar oracle for four operators;
- dense/sparse equivalence across input masks and lengths;
- exact zero-gate rotation-skip accounting;
- synthetic learning of two independent operation-specific routes;
- held-out label isolation, invalid shift rejection, zero budget and
  failure atomicity.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p flat-algebraic-attention boolean_operator_channels
cargo run -p flat-algebraic-attention --example bps15_operator_channels_smoke
```

## Next experiments

1. Introduce *sparse per-neuron* connectivity masks and at most N
   chosen incoming links, compared with four shared rotations.
2. Learn operation enable/disable masks as well as shifts, using
   held-out selective-copy and delayed-copy tasks with matched
   parameter budgets.
3. Measure whether bit-packed instruction-level parallelism,
   event-triggered skipping or CPU SIMD provide real throughput gains.
4. Prove any new compressed transition algebra **before**
   attempting parallel scan on four-parent recurrences.
