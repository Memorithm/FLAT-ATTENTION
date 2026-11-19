# BPS-27 — Live source-rotation groups for exact Boolean routing

## Functional difference from BPS-26

BPS-26, merged as FLAT-ATTENTION #414, uses the *total number of compiled
source shifts* U per operator in its cost comparison. A sparse input gate
often activates only a subset of those shifts. The grouped-sparse BPS-19
kernel skips disabled shifts, so charging all U overestimates the work
of the grouped path for sparse inputs.

BPS-27 adds an opt-in `BooleanNeuronGraph::prepare_live_cost_router`.
Its borrowed `PreparedLiveCostRouter` precomputes a 4 x 64 array of exact
destination bitmasks indexed by source rotation, using the already
validated graph. It exposes `live_shift_counts(input)`, the number of
distinct predecessor rotations touched by the gate-active neurons of
each operation. No memory state, labels, future inputs or targets enter
these counts.

For a channel with n active destinations and T live source shifts,
BPS-27 selects the existing BPS-19 direct-lane kernel only if:

```text
n > 0
n < U
n <= caller_hard_cap
direct_setup + n * direct_lane_units
    < grouped_setup + T * grouped_shift_units
```

All other routes use the existing BPS-19 grouped/short-circuit logic.
The all-dense BPS-20 gate guard takes precedence, with no live count
needed. Uniform and dormant channels retain their existing kernels.
This is a *routing policy*, not a new recurrent state transition.

## Implementation and tests

- `PreparedLiveCostRouter::adaptive_policy(input)` computes the four
  input-dependent caps; `decision(input)` reports the actual exact
  kernel routes; `step(input, previous)` delegates to the existing
  Boolean graph kernels and remains synchronous.
- `scan_into` executes arbitrary chunks into caller-owned buffers,
  maintaining recurrent state without forcing an allocation. Buffer
  length rejection is atomic.
- An adversarial case with 64 unique static shifts but only two active
  groups demonstrates a different route choice from BPS-26, while
  both still exactly match the scalar oracle.
- All 65 Hamming weights, five topology buckets, three cost models
  including u32::MAX cost arithmetic, arbitrary chunks, rewiring,
  restoration, output bits **and final state** are validated.
- The exact-head CI smoke is runnable with:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p flat-algebraic-attention bps27_
cargo run -p flat-algebraic-attention --example bps27_live_shift_cost_smoke
```

## Qualification boundary

Inputs to `RoutingCostModel` remain caller-provided estimates. The
library does **not** measure or calibrate hardware, and the extra gate
evaluation and live-group checks can themselves exceed any saved work.
No speedup, energy saving, learning improvement, model quality gain,
or global cost optimality is claimed. Empirical deployment thresholds
must be fit on DEV-only observations and validated on fresh paired
holdout trials. Any Dell physical run must be routed solely through
Memorithm/RemoteOps. No production defaults, hardware settings,
services or secrets are changed.
