# BPS-LSTM: bit-parallel scan Boolean memory (research v0)

Status: **research-only**. This is a compact Boolean recurrent-memory candidate,
**not** a classical Hochreiter--Schmidhuber LSTM, a substitute for
flat_attention::api::v1, or a demonstrated model-quality/energy improvement.

Implementation: crates/flat-algebraic-attention/src/boolean_scan_lstm.rs and
crates/flat-algebraic-attention/src/boolean_scan_training.rs

## Motivation and novelty boundary

Binary recurrent units, binary-valued LSTM gates and parallelizable minimal
LSTMs already exist. No broad claim of priority or originality is made for
Boolean recurrence, bit packing, or prefix scans alone.

Relevant literature:

- Li et al., *Towards Binary-Valued Gates for Robust LSTM Training*, ICML 2018.
  https://proceedings.mlr.press/v80/li18c.html
- *The Synthesis of XNOR Recurrent Neural Networks with Stochastic Logic*,
  NeurIPS 2019. https://papers.neurips.cc/paper_files/paper/2019
- Feng et al., *Were RNNs All We Needed?*, 2024 (minLSTM/minGRU, associative
  scan). https://arxiv.org/abs/2410.01201
- Hubara et al., *Binarized Neural Networks*, 2016.
  https://arxiv.org/abs/1602.02505

The research hypothesis is that **packed Boolean state + input-only Boolean
gating + temporal scan + discrete training + event-activity observability**
can provide an attractive design point for selected memory tasks.

## Exact semantics

A machine word holds 64 independent one-bit state lanes. A gate on each lane
has a two-entry Boolean truth table (for current input bit 0 or 1). The memory
transition at time t is:

    c[t] = (f[t] AND c[t-1]) OR w[t]
    h[t] = c[t] AND o[t]

Here f, w, and o depend only on x[t] (not c[t-1] or h[t-1]); w fuses a
conventional input gate and candidate gate into one Boolean write predicate.
OR is **not** arithmetic addition. Forget and write are not mutually exclusive;
when they both hold, write sets a bit to 1.

Represent a step as a pair (F, W) with action M(c) = (F AND c) OR W.
Sequential composition of first (F1, W1) and then (F2, W2) yields:

    F_combined = F2 AND F1
    W_combined = W2 OR (F2 AND W1)

This is associative because Boolean AND and OR distribute. The identity is
(ALL_ONES, ZERO). Therefore:

- sequential reference: O(T) work, O(1) auxiliary memory;
- Hillis-Steele prefix scan: O(T log T) work, O(log T) dependency depth,
  O(T) auxiliary memory;
- lane packing: 64 independent Boolean streams per 64-bit word.

These are **algorithmic** bounds, not a measured speedup. The current CPU
implementation uses scoped threads on sufficiently long sequences but can be
slower than the reference because it performs more work and spawns threads.
It does not yet include cross-lane synapses, sparse event queues, SIMD-specific
kernels, hardware energy readings, or a GPU implementation.

Critically, a standard LSTM with gates depending on the previous hidden state
cannot reuse this scan unchanged. Allowing that dependence breaks the
precomputed-transition assumption. Likewise, the model is not functionally
equivalent to an FP32 LSTM.

## Discrete training

The reference trainer minimizes supervised Hamming loss by deterministic,
greedy bit flips of the six packed truth-table parameters. The added packed
trainer encodes 64 per-lane Hamming counters as 64 Boolean bit planes,
increments them using XOR/AND ripple-carry operations, and compares candidate
losses with a Boolean most-significant-bit comparator. It evaluates one
candidate **per 64-lane gate mask**, then simultaneously accepts only the
lanes with strictly lower error. This reduces candidate forward evaluations
from up to 384 to 6 per epoch for the six parameter masks, at the cost of
bit-plane counter work. Equivalence to lane-wise greedy coordinate descent
is tested. The reduction is algorithmic, not a measured wall-clock benefit.

This avoids floating-point gradients entirely. It does not constitute
backpropagation-through-time, has no global-optimality guarantee, and presently
learns **per-lane unary input-to-output dynamics only**. Cross-lane
connections, multi-bit state depth and stronger learning rules are distinct
research milestones. The packed trainer processes all independent lanes in parallel using
Boolean operations, but epochs and parameter-mask directions remain sequential.
Cross-lane coupling would invalidate this independent-lane optimization until
a different learning rule is qualified.

## Qualification and falsification

Required correctness checks:

1. composition associative and order-sensitive;
2. identity transformation left and right;
3. scan exactly equals sequential reference for arbitrary gate masks,
   non-power-of-two lengths and nonzero initial states;
4. supervised toy task must reduce Hamming loss to zero;
5. invalid datasets fail closed;
6. 64-bit lane-boundary and activity accounting tests.

Run on the exact candidate SHA:

    cargo fmt --all -- --check
    cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
    cargo test -p flat-algebraic-attention boolean_scan

Run the executable smoke:

    cargo run -p flat-algebraic-attention --example bps_lstm_smoke

These tests establish **correctness**, not language-model competence.

For later matched experiments, preregister train/validation/test splits and
measure sequence accuracy, selective-copy retention, working-set bytes, p50/p95
latency, processed sequence bits/sec, actual joules on the same identified CPU,
and compute-budget/CPU-affinity controls. Compare with (a) a learned FP32
classical LSTM, (b) a minLSTM, and (c) the Boolean serial reference on the
same tasks. A fair accuracy comparison requires tuning and representation
budgets, not just equal bitwidth.

## Next research milestones

- BPS-2: bitset cross-lane connections using F2/Zhegalkin compositions with
  a strictly bounded sparse fan-in and preserved scan eligibility where possible.
- BPS-3: rising-edge/event-triggered writes and an explicit skip/wake protocol;
  verify unchanged outputs against the dense Boolean reference.
- BPS-4: compare coordinate search, local plasticity, surrogate gradient and
  evolutionary search under equal training compute, in addition to inference.
- BPS-5: include the transition/write/memory-access energy cost algebra and
  calibrate against physical power measurements, rather than assuming
  fewer bitwise operations means lower joules.
- BPS-6: TDI qualification, real-world workload baselines, and cross-project
  reusable Rust integration only after correctness and performance gates pass.

Neither model promotion nor merges should be inferred from a green unit-test
suite alone: normal FLAT-ATTENTION release and exact-head CI gates apply.
