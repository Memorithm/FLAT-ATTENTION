# BPS-23 — Explicit exact kernel selection by input Hamming weight and routing topology

## Motivation from **measured** BPS-22 evidence

[RemoteOps Dell run #44394321754](https://github.com/Memorithm/RemoteOps/actions/runs/44394321754)
completed 512 exact-output trials for eight independent 64-neuron
Hamming weights, two graph topologies and four historical kernels.
The BPS-22 instrumentation showed that the winner depends on both
input activation and predecessor wiring, not just raw sparsity.

For the one-heterogeneous-channel graph (3 uniform operations), the
BPS-18 grouped-dense kernel was fastest at all eight weights. For
the four-heterogeneous-channel graph (0 uniform operations), the
BPS-21 adaptive path was fastest at 0, 1 and 64 active inputs,
whereas grouped-dense led at 4, 16, 32, 48 and 63. These findings
are **one-host, eight-pass development-set observations**, not a
validated universal scheduler.

## New functional Rust API

`boolean_neuron_dispatch.rs` implements a deliberately explicit and
disabled-by-default selection mechanism. A validated
`WeightRoutingPolicy` contains five optional inclusive weight
windows, one for each number (0 through 4) of uniform predecessor
operation channels in the graph. Its decision uses only:

- The validated graph's **number of uniform predecessor channels**,
  automatically rebuilt when `with_edge()` rewires a source.
- The live input's exact number of active bits, `input.count_ones()`.
- The caller-supplied inclusive density window for that topology.

Within the window, `step_weight_routed` calls the existing exact
BPS-18 `step_grouped` kernel. Outside it, it calls the existing
BPS-20/BPS-21 `step_activity_hybrid` kernel. No new recurrence,
gate, source, output or learning rule is introduced. Both choices
read the same previous state and remain bit-exact with
`forward_scalar`. The method `weight_routing_decision` makes each
choice inspectable before executing it.

`WeightRoutingPolicy::DISABLED` delegates all samples to the
existing hybrid; no one-host threshold is silently promoted to
production. Policy construction rejects reversed weight ranges
and weights greater than 64 before any execution.

## Development-only example candidate

The CI smoke explicitly opts into a candidate calibrated on the
BPS-22 Dell development results:

| Uniform predecessor channels | Choose grouped-dense for input weights |
| --- | --- |
| 0 | 4..=63 |
| 3 | 0..=64 |
| 1, 2, 4 | No forced grouped-dense selection |

The intermediate uniformity classes **have not been physically
characterized**. The examples and tests intentionally avoid
assuming any speed improvement for those classes. Because this
candidate was selected on the same development run, the BPS-22
measurement **cannot** also serve as its independent evaluation.

## Verification and next steps

The exact-head GitHub CI gate executes a smoke and unit tests for:

- Invalid weight windows, including range reversal and weight 65.
- Every weight 0..64, all five uniformity counts, disabled fallback
  and boundary decisions.
- Rewiring followed by cache invalidation and restoration of
  uniformity (changing the selector's bucket immediately).
- Multiple random gate configurations and predecessor topologies,
  eight per-neuron exact-weight corpora and whole-trajectory
  bit-for-bit equality with the independent scalar oracle.

```bash
cargo fmt --all -- --check
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
cargo test -p flat-algebraic-attention bps23_
cargo run -p flat-algebraic-attention --example bps23_weight_dispatch_smoke
```

The next controlled RemoteOps experiment should compare this
**explicit** configured dispatcher against the original historical
BPS-21/BPS-22 adaptive algorithm and BPS-18 dense/sparse on fresh
seeds not used to choose the windows, with outputs checked against
the scalar oracle before and after each paired timed trial.
Confidence across hosts, weights, gates and topology classes remains
unknown until replicated. No joule, throughput or model-quality
claim is made without independent measurements.
