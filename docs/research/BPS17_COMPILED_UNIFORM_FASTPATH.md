# BPS-17 — Compiled uniform routing and masked destination groups

Status: research-only optimization. Source:
`crates/flat-algebraic-attention/src/boolean_neuron_graph.rs`;
smoke: `crates/flat-algebraic-attention/examples/bps17_uniform_fastpath_smoke.rs`.

## Motivation: measured physical regression

The exact-source Dell T430 experiment in
[RemoteOps run 44287889625](https://github.com/Memorithm/RemoteOps/actions/runs/44287889625)
completed 64 paired raw wall-clock trials. On *identical shared source
offsets* (four fixed rotations), the median of four timed passes at
100% active input was about **0.969 ms** per 32 × 4096 state updates
for BPS-15 dense versus **1.572 ms** for BPS-16 grouped, a roughly
**1.62×** slowdown for BPS-16 grouped. At 0% activity the median
times were about 1.034 ms and 1.762 ms respectively. BPS-16 grouped
sparse had 1.584 ms median at 100% activity.

These figures are preliminary: one host, four short paired passes per
setting, no energy meter, no NUMA/affinity controls and no statistically
independent replications. Because the BPS-16 graph in that benchmark
used four shared offsets, the timing isolates the overhead of the
general grouping kernel, not the intrinsic cost or utility of 256
independently configurable connections. No real-world model-quality
benefit can be inferred.

The original standard-only harness used a periodic rotate/XOR
fingerprint that collides for several 4096-step corpus patterns.
**Exact output equivalence was independently checked** via full
vector equality in the executable, so the collision does not
invalidate the functional parity check, but the weak corpus
fingerprint must be repaired before a publication-quality report.

## Implemented execution repair

At construction, the graph compiles the shift assignments from
all **64 destination neurons** for each of its four recurrent operator
channels. When each channel uses only one distinct shift, it stores
the equivalent BPS-15 BooleanOperatorProgram as a non-allocating
fast-path. Every `step_grouped` call executes that existing
`step_dense` implementation directly. Likewise the sparse
`step_grouped_sparse` path executes BPS-15's `step_sparse`.

No approximation, retraining, topology mutation or hidden state reset
occurs. The underlying output and update gates are unchanged.
`with_edge` reconstructs the compiled fast-path decision, restoring
the generic path as soon as any channel has multiple distinct offsets,
and restoring the shared fast-path when the offsets become uniform.

For heterogeneous graphs, grouped-sparse execution also computes
`active_destinations = group_destinations & gate(input)` **before**
rotating. If that value is zero, the per-shift rotation is skipped
even if other destination groups in the same operator remain active.
This is a stricter, exact zero-contribution test than BPS-16's
whole-channel mask check. It can add branches and is not always faster.

The method `compiled_uniform_shifts` makes dispatch selection
observable to callers. The BPS-16 fully heterogeneous topology is still
supported, with the same 256 independent source selections.

## Exact-head test gate

- Existing randomized scalar per-neuron oracle and BPS-15 equivalence.
- Explicit compiled-shift detection and invalidation/reinstatement.
- Mixed recurrent trajectories through shared and heterogeneous paths.
- Partially active output-lane masks exercising zero-contribution
  shift groups (not just fully disabled channels).
- Runnable `bps17_uniform_fastpath_smoke`, included in GitHub CI.

```bash
cargo fmt --all -- --check
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
cargo test -p flat-algebraic-attention boolean_neuron_graph
cargo run -p flat-algebraic-attention --example bps17_uniform_fastpath_smoke
```

## Required physical follow-up

After exact-head CI green and merge, extend the exclusively
RemoteOps-controlled Dell benchmark to compare BPS-15 dense/sparse,
BPS-16/BPS-17 grouped/grouped-sparse under the same shared topology
and a separate heterogeneous-routing scenario. Repair the corpus
fingerprint and preserve pairing, stop-on-busy admission, exact output
matching and traceable source SHA before drawing throughput conclusions.
