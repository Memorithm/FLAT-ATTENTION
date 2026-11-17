# BPS-2: exact temporal scan with recurrent cross-lane wiring

Status: **research-only**, independent of the production attention API.
Implementation: crates/flat-algebraic-attention/src/boolean_rotating_memory.rs.
This is an LSTM-inspired Boolean memory, **not** a classical LSTM.

## Precisely what changed from BPS-1

BPS-1 has 64 independent recurrent state lanes. BPS-2 introduces real
state-to-state connections: for the shared offset r in {0,...,63},
destination lane j reads prior-state lane (j-r) modulo 64.

The memory transition is:

    c[t] = f(x[t]) AND ROTL(c[t-1],r) OR w(x[t])
    h[t] = c[t] AND o(x[t])

Here gates are current-input-only Boolean truth tables. ROTL is a circular
permutation of 64 packed state bits. Forget/write/output are still logically
separate. There is exactly **one predecessor-state edge per destination lane**,
not arbitrary 64x64 weighted connections. Input gates do not read prior hidden
state. r=0 recovers BPS-1 bit-for-bit.

This topology gives a bounded-fan-in recurrent neural connection candidate,
but does not establish language-modeling capability, memory generalization,
or reduced hardware consumption.

## Associative scan closure: proof sketch

Represent a transition as (F, W, r):

    T(F,W,r)(c) = F AND ROTL(c,r) OR W

In chronological order (first a, then b):

    (Fa,Wa,ra) ; (Fb,Wb,rb)
      = (Fb AND ROTL(Fa,rb),
         Wb OR (Fb AND ROTL(Wa,rb)),
         (ra+rb) modulo 64)

This follows from distributivity of ROTL over Boolean AND/OR and
ROTL(ROTL(x,ra),rb)=ROTL(x,(ra+rb) mod 64). Functional composition is
associative, and the identity transition is (ALL_ONES, ZERO, 0).
The representation remains constant size (two u64 masks and a u8 rotation)
at arbitrary scan depth, even if the composed masks lose their sparsity.
No dense recurrent matrix is materialized.

- Sequential reference: O(T) work, O(1) auxiliary state.
- Hillis-Steele inclusive scan: O(T log T) work, O(log T) dependency
  depth, O(T) auxiliary memory, and opportunistic CPU threads.
- Each step packs 64 destination lanes into one u64.
- Exactly one prior-state predecessor exists per destination lane. Unions of
  recurrent rotations would not generally remain in this closed form.

These are theoretical algorithmic properties, not physical speedup evidence.
Extra scan work, allocation, and thread startup may dominate in practice.

## Training: a necessary correction

BPS-2 enumerates all 64 shifts and retains strictly lower **global
supervised Hamming loss**; the existing shift wins ties. Gate truth tables
remain fixed. The tests include a teacher-generated rotated-state task.

After recurrent rotation, lanes are coupled across timesteps. Consequently,
BPS-1's independently accepted bit-plane per-lane updates are **not valid**
for BPS-2 as-is. Those updates could reduce individual lane losses while
raising the global loss. The 64-shift search is discrete and bounded, but it
is not jointly optimized gate/connectivity learning and not massively
parallelized training.

## Required tests and reproducibility

1. Reject shifts outside 0..63.
2. r=0 reproduces uncoupled BPS-1 exactly.
3. r>0 carries state between lanes, including wrap-around.
4. Associative, order-sensitive composition with identity and random masks.
5. Exact sequential/scan equality for empty, odd, singleton, 128+ length
   inputs, random nonzero initial states, and multiple rotations.
6. Learn a teacher-generated shift; reject malformed training datasets.
7. Count logical state events; do not convert them into joules.

Run on the exact candidate commit:

    cargo fmt --all -- --check
    cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
    cargo test -p flat-algebraic-attention boolean_rotating_memory
    cargo run -p flat-algebraic-attention --example bps2_rotating_memory_smoke

Repository-wide Rust/WGPU, MAA, security and exact-head checks remain
mandatory before merging. No test establishes energy/latency improvements.

## Follow-up research

BPS-2b: disjoint cycle permutations and sparse destination-source maps
with explicit closure/proof obligations; do not claim arbitrary fan-in.

BPS-3: rising-edge-triggered updates, with correctness-preserving skip/wake
protocols and explicit physical work accounting.

BPS-4: train gates and topology jointly on held-out shifted-copy and
selective-copy tasks, then compare with classic LSTM, minLSTM and BPS-1.

BPS-5: energy-cost algebra calibrated to actual package and memory readings,
rather than Hamming counts.

BPS-6: evidence in TDI before a reusable, opt-in Rust integration with
V888-BOOLEEN or SML-GENIUS. Physical remote qualification must use
Memorithm/RemoteOps as its control plane.

## Prior art

Parallel prefix scans of input-dependent recurrent gates already exist:
Feng et al., *Were RNNs All We Needed?* (2024)
https://arxiv.org/abs/2410.01201

The present combination of packed Boolean permutation recurrence, a
closed-form scan algebra, a Rust oracle and a bounded topology-learning
baseline is a research candidate; literature-wide novelty, comparative
quality, and energy superiority are **not established**.
