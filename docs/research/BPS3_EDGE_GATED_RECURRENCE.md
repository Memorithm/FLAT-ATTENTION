# BPS-3: event-gated Boolean recurrence with exact temporal scan

Status: **research-only**. This prototype is not an exact LSTM, a
semantics-preserving optimization of BPS-2 for arbitrary raw inputs, or
a production routing change. It is an *explicitly different* event-driven
model whose behavior can be evaluated by three equivalent execution paths.

Source: crates/flat-algebraic-attention/src/boolean_edge_memory.rs
Shared prefix-scan core: crates/flat-algebraic-attention/src/boolean_rotating_memory.rs
Rising-edge detector: crates/flat-algebraic-attention/src/rising_edge.rs

## State and event semantics

Given an explicitly sampled 64-bit input sequence x[t], input baseline b
and a per-tick previous input x[t-1]:

    e[t] = x[t] AND NOT x[t-1]

The bitmask e[t] contains precisely the *sampled 0->1 fronts*.
Already-high baseline bits do not generate an initial event. A falling
transition re-arms a bit but is not itself a rising event.

If e[t] == 0, the state transition is the **identity**:

    c[t] = c[t-1]

Otherwise:

    c[t] = f(e[t]) AND ROTL(c[t-1], r) OR w(e[t])

The readout is static with respect to current input:

    y[t] = c[t] AND readout_mask

f and w are BPS-1 unary Boolean truth-table gates, r is a BPS-2
validated rotation in 0..63. An event in one bit may trigger the
whole packed recurrence for all 64 lanes; this is **event sparsity across
timesteps**, not an asynchronous per-neuron update network.
The output mask is time-invariant to make a fully quiet timestep
observationally identical to the previous one.

Importantly, the rule e[t]==0 -> identity is **part of the model**.
It is not an unconditional optimization of the classical LSTM or BPS-2.
If BPS-2 were to rotate or forget when its input is zero, applying a
quiet-step shortcut to that BPS-2 model would change its outputs.

## Exactness and executable oracles

BPS-3 contains three independent evaluation paths:

1. **Dense semantic oracle:** samples each tick and always calls
   RotatingTransition.apply, using an identity transition on quiet ticks.
2. **Sparse execution:** uses the qualified RisingEdgeDetector and
   bypasses the recurrent transition entirely on ticks without fronts,
   preserving the cached state and a constant readout mask.
3. **Parallel prefix scan:** constructs identity or event transitions
   for every sampled timestep, then evaluates the same associative
   BPS-2 scan algebra. This path intentionally does *not* skip the
   identity scan work.

The shared BPS-2 scan engine is reused by BPS-2 and BPS-3, preventing
two divergent composition implementations. Unit tests compare all
three paths bit-for-bit on randomized masks and nonzero baselines,
lengths including 0/1/127/128/257, all selected rotations, and
falling/rearming transitions.

The timestamped stateful runtime enforces strictly increasing **sample**
ticks and leaves detector, state and accounting untouched on invalid
samples. Gaps between supplied ticks do NOT imply observations of
zeros or absence of unobserved events.

## Complexity and energy limits

For T supplied samples and E eventful samples (e[t] != 0):

- Rising-edge detection: O(T) packed-word samples **unavoidably**
  inspected in this implementation.
- Sparse recurrent transition evaluation: O(E) packed-word updates,
  with O(1) persistent runtime memory for one 64-bit state.
- Dense semantic oracle: O(T) packed-word recurrence applications.
- Temporal prefix scan: O(T log T) work, O(log T) dependency depth,
  O(T) memory, with an opportunistic threaded CPU implementation.

Accounting reports sampled steps, eventful steps, skipped identity
steps, rising bit events, executed recurrent steps, state flips and
output flips. These are **logical counts**, not proven physical
instruction counts, CPU cycles, latency measurements or joules.
Even if E << T, detecting events, allocation and runtime dispatch may
outweigh avoided recurrences; energy reduction is unproven.

No dynamic pruning, approximate gating, speculative omitted rotations,
raw-input-dependent output masking or arbitrary sparse connectivity
is silently introduced by this prototype.

## Reproduce exact-head correctness

    cargo fmt --all -- --check
    cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
    cargo test -p flat-algebraic-attention boolean_edge_memory
    cargo test -p flat-algebraic-attention boolean_rotating_memory
    cargo run -p flat-algebraic-attention --example bps3_edge_memory_smoke

Normal FLAT-ATTENTION whole-workspace Rust, WGPU, MAA, supply-chain,
semver and CodeQL CI gates must also be green on the final PR HEAD
before merging.

## Next experimental milestones

- Measure CPU serial, sparse and scan latency on matched seeded datasets
  with frozen event densities: 0%, 1%, 10%, 50% and 100%; include
  p50/p95 and process-affinity controls on named hardware.
- Introduce genuinely event-indexed inputs to avoid full packed-word
  front detection when a trustworthy event feed already exists, while
  validating identical outputs at requested observation times.
- Evaluate selective-copy, delayed-bit and long-term memory tasks;
  compare accuracy and learning budgets against BPS-1, BPS-2,
  classical LSTM and minLSTM.
- Extend the model only with connectivity or training improvements
  that preserve explicitly proven semantics.
- Establish an energy-cost algebra calibrated with actual package
  and memory-controller energy measurements in TDI. The sole
  physical orchestration path is Memorithm/RemoteOps.

## Related research (prior-art boundary)

The idea of event-driven neural updates is not novel in itself;
nor is scan-parallel recurrence:

- Fang et al., *Deep Residual Learning in Spiking Neural Networks* (2021):
  https://arxiv.org/abs/2102.04159
- Feng et al., *Were RNNs All We Needed?* (2024):
  https://arxiv.org/abs/2410.01201
- Existing project RAE detector: crates/flat-algebraic-attention/src/rising_edge.rs

BPS-3 is a reproducible engineering experiment, not a claim of
literature-wide novelty, energy superiority, or language competence.
