# BPS-21 — Per-operation dense grouped execution inside mixed topology

This experimental, CPU-only improvement extends the BPS-19 adaptive
Boolean routing and BPS-20 exact global dense-input fallback.

## Source of the opportunity

[RemoteOps BPS-20 physical run #44347761337](https://github.com/Memorithm/RemoteOps/actions/runs/44347761337)
completed 256 exact-output trials on a Dell T430. The BPS-20 hybrid
reduced BPS-19's dense-input latency by approximately 24–38% for
25–100% density settings. It also incurred small overhead at zero
density. The benchmark used whole input words (0 or u64::MAX), so it
**did not measure mixed per-channel Boolean gate activation**.

Previously, BPS-20 chose the fully dense grouped kernel only when all
four recurrent channel gates evaluated to u64::MAX. For mixed gate
truth tables, this global predicate may be impossible even while one
or more individual operations are fully active. BPS-19's grouped path
then evaluates a per-shift destination mask and tests whether it is
nonzero on each iteration, needlessly repeating that work for a
channel already known to be fully enabled.

## Exact optimization implemented

For each operation routed through AdaptiveRoute::Grouped in
step_adaptive, BPS-21 checks the exact live bitmask:

- If the current channel's gate mask equals u64::MAX, every
  destination in that operation is enabled. Iterate compiled
  predecessor shift groups and perform one packed rotation and AND
  with the static destination mask per group, OR-reducing without a
  dynamic input-mask intersection or zero check.
- If the gate mask is only partially enabled, retain the previous
  per-shift gated group iteration, skipping zero-contribution groups.
- Dormant, Uniform and DirectLanes choices remain unchanged. BPS-17's
  whole-model uniform fast path and BPS-20's all-channel dense guard
  are also preserved.

This changes only how the same predecessor bit contributions are
computed, not the recurrent update or output. All four operations
still consume the **previous** state, applying retention, excitation,
inhibition and toggle in the same fixed order.

The new fully_active_channel_bits(input) inspector returns a four-bit
mask, with one bit for each active operation. In the CI example, the
bitmask is 0b0101 for input zero and 0b1011 for input u64::MAX, while
the global BPS-20 dense guard is unsatisfiable.

## Required exact-output qualification

The independent 64-neuron scalar reference checks both step_adaptive
and step_activity_hybrid on mixed per-operation dense gates.
Randomized heterogeneous graphs with nonuniform sources, random
truth tables and varied input sequences test full recurrent state
trajectories for several direct-lane thresholds.

    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo test -p flat-algebraic-attention bps21_
    cargo run -p flat-algebraic-attention --example bps21_dense_channels_smoke

The exact-head GitHub CI gate enforces these functional tests. There
is **no demonstrated BPS-21 throughput benefit yet**. An upcoming
RemoteOps physical comparison must vary the number of fully active
channels independently, including cases where the global all-channel
dense guard is impossible. A gain on whole-word input density alone
cannot establish this advantage. No energy or language-model quality
claims are made.
