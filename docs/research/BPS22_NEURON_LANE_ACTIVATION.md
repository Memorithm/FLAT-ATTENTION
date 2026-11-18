# BPS-22 — Exact-weight individual neuron activation and route telemetry

BPS-21's [first exact-source physical comparison](https://github.com/Memorithm/RemoteOps/actions/runs/44353248081)
found BPS-21 faster than BPS-20 in 64/64 paired passes across two
topologies and four whole-word input densities. This is meaningful
but incomplete: the Dell corpus set each `u64` input to either zero
or all ones, **not** to independently active neuron lanes. It did
not establish how BPS-21 behaves when 1, 4, 16 or 48 of 64 neurons
are active. No energy measurement was performed.

## Implemented research functionality

A new Rust module `boolean_neuron_activity.rs` provides two
complementary capabilities without modifying the BPS-16..21
recurrent transition.

**1. Deterministic exact-weight input corpora.**

`ExactWeightMaskGenerator` produces successive `u64` input words
with exactly `weight` active bits (Hamming weight 0..64). The
generator uses a specified SplitMix64 seed and a partial
Fisher–Yates selection of unique lane indices. Bounded index choices
use rejection sampling instead of biased modulo reduction. It
rejects weights exceeding 64; zero and full weights return the two
correct degenerate masks. This is a reproducible experiment
generator, *not a cryptographic source of randomness*.

`exact_weight_corpus(seed, weight, sample_count)` exposes the bounded
sequence to physical harnesses and training experiments. Hamming
weight is the number of active **neuron input lanes**, not a
percentage of entire words that are dense.

**2. Input-only streaming operation activation profile.**

`NeuronActivationProfile` can accumulate `observe(graph, input, policy)`
per sample or collect `from_inputs(graph, inputs, policy)`.
It records, for each of the four independent operations:

- Exact 0..64 Hamming weight histograms of its actual
  `UnaryGate::evaluate(input)` mask.
- Aggregate active destination bits, dormant/full-channel ticks.
- Candidate BPS-19 routes: Dormant, Uniform, DirectLanes, Grouped.
- Number of unique predecessor rotations touched by enabled output
  neurons, using the *validated per-neuron source offsets*.

It also reports the input-word weight histogram, total active
input bits and the BPS-20 all-operation dense-fallback count.
**Candidate** route counts do not mean that all those kernels
physically ran: BPS-20's global dense guard can override the
candidate plan, and BPS-17 uniform whole-model execution can
shortcut the generic kernel. Touched source groups are a logical
work proxy, not the number of CPU rotations executed.

All counters are checked against overflow. Profiling has no
access to targets, does not update model weights, is not invoked by
default inference, and cannot silently affect the original state
transition or outputs.

## Exactness tests

Unit tests cover all 65 possible Hamming weights, deterministic
reproduction across seeds and calls, validation failure, eight
independent exact-weight sample corpora, per-operator gate masks,
counts over incremental/whole sequences, and 64-neuron scalar
oracle equivalence for BPS-18, BPS-19 and BPS-20/BPS-21 variants.
The mandatory CI step runs both the unit suite and an executable
smoke on the exact branch HEAD.

```bash
cargo fmt --all -- --check
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
cargo test -p flat-algebraic-attention bps22_
cargo run -p flat-algebraic-attention --example bps22_lane_activity_smoke
```

## Next physical experiment

Use exclusively the existing Memorithm/RemoteOps Dell isolated
CPU control plane, with pinned **merged** FLAT source SHAs.
For `weight ∈ {0,1,4,16,32,48,63,64}`, compare the real
BPS-20/BPS-21 source implementations under exactly identical
per-neuron input masks, gate truth tables, topologies and initial
recurrent states. Require the independent full scalar-output vector
oracle before and after every timed trial.

A physical run must also record the activation profile, rather than
treating the requested input weight as equal to the operator mask
weight: arbitrary per-lane `UnaryGate` truth tables can invert or
force individual bits. Report individual per-density/operation
histograms and matched latency distributions, not just an average.
Do not claim power, throughput or model-quality generalization from
that single machine.
