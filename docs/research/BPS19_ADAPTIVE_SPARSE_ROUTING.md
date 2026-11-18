# BPS-19 — Adaptive recurrent routing by live lane mask and topology

**Research-only exact Boolean execution.** This step builds on BPS-16
(individual predecessor edges), BPS-17 (uniform graph fast-path) and
BPS-18 (independent uniform operation fast-paths). It does not add
connections, approximate results or change the recurrence.

Source: `crates/flat-algebraic-attention/src/boolean_neuron_graph.rs`.
Smoke: `crates/flat-algebraic-attention/examples/bps19_adaptive_routing_smoke.rs`.

## New executable operation plan

Each of the 64 destination neurons has one incoming predecessor per
operation: retain, excite, inhibit, toggle. All channels read the
previous state from the same timestep. The gate masks are input-only.
For every **operation and input sample**, `adaptive_route_plan`
selects the exact work algorithm:

| Decision | Condition | Execution |
| --- | --- | --- |
| Dormant | Entire channel gate word is zero | No recurrent source reads |
| Uniform | All 64 neurons share the same predecessor shift | One rotated packed word, AND gate mask |
| DirectLanes | Active output lanes ≤ configured bound **and** fewer than compiled unique source shifts | Iterate active destination bits and read their one source bit each |
| Grouped | Remaining heterogeneous case | Group by shift and rotate only if active destination mask is nonzero |

The default bound is four active lanes for each of four operations.
`AdaptiveRoutingPolicy` exposes per-operation bounds in [0,255].
Zero disables DirectLanes; values greater than 64 are harmless since
at most 64 neurons can be active. There is no learned scheduler and
no reliance on uncertain wall-clock timings in the decision.

The independent direct lane computation uses
`source = (destination + 64 - shift[channel][destination]) % 64`,
then extracts that bit **from the previous state**. A candidate is
not selected based on the hidden state or outputs. The four channel
results retain the established exact order:

```text
base = retain | excite | write(input)
next = (base & !inhibit) ^ toggle
output = next & output_gate(input)
```

The whole-graph BPS-17 uniform dispatch still short-circuits into
BPS-15. The BPS-18 mixed-channel fast-path remains accessible and
unchanged through `step_grouped` and `step_grouped_sparse`.

## Functional test evidence

Tests exercise deterministic route plans, zero and dense gates,
operation-specific thresholds including 0, 1, 4, 16, 64 and 255,
randomly rewired per-neuron connections, 64-bit gate patterns, empty
sequences, mixed zero/max inputs, and exact equality to the
independent scalar 64-neuron reference over whole recurrent paths.
They also ensure rewiring and returning to uniform connectivity
cannot leave cached topology stale.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p flat-algebraic-attention boolean_neuron_graph
cargo run -p flat-algebraic-attention --example bps19_adaptive_routing_smoke
```

All standard exact-HEAD GitHub CI checks (Rust, WGPU, host oracle,
CodeQL, semver, supply chain, analysis) remain merge prerequisites.

## Performance hypothesis and necessary qualification

This algorithm can replace the iteration over up to 64 distinct
shift groups with an iteration over only a few destination bits for a
sparse gate. This is **a count of logical work, not a measured speedup**.
Bit extraction, branches, cache effects and compiler code generation
may outweigh the savings.

A paired, pinned RemoteOps physical Dell experiment is necessary.
Compare the adaptive route to BPS-18 `forward_grouped_sparse` on
identical mixed topologies and sequences, including low-activity,
high-activity and varying source diversity; confirm full output
equality before timing. Keep BPS-17 fully uniform comparisons
independent. Do not infer energy reduction or learning quality
from these CPU-only timings.
