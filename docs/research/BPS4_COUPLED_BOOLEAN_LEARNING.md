# BPS-4: globally coupled Boolean recurrent learning

**Status: research-only.** This slice provides deterministic supervised
learning of the connection rotation and Boolean gate truth tables in
the already-qualified BPS-2 rotating memory and BPS-3 edge-gated memory.
It does not implement classical LSTM training, BPTT, a general sparse
synaptic matrix, or an energy-optimized hardware kernel.

Source: crates/flat-algebraic-attention/src/boolean_coupled_learning.rs

## Why BPS-1's lane-independent optimizer is insufficient

In BPS-2/BPS-3, a previous-state bit can rotate into a *different*
destination bit on a later timestep. A gate change to an unobserved
upstream lane can therefore alter a supervised output lane later.
Optimizing 64 losses separately, as in the BPS-1 independent bit-plane
trainer, does not preserve a global recurrent training objective.

The new CoupledModel enum explicitly supports BPS-2 (input-only forget,
write and output gate masks; six u64 parameter masks) and BPS-3
(front-only forget/write and static readout; five u64 masks). Their
semantics and forward oracles remain distinct. Each candidate
forward evaluation uses its correct original recurrent transition.

## Objective, training boundary and candidate moves

For an independent trajectory i, reset c[i,0] and use its supplied input
baseline for BPS-3. The supervised objective is

    L(model; D, M) =
        sum over examples i, samples t:
          popcount((prediction[i,t] XOR target[i,t]) AND M)

where M is an explicit **output** supervision mask, not a mask on
trainable hidden-state lanes. All 64 state bits remain candidate
parameters, including those outside M, since they may affect output
later after recurrent rotation.

At each bounded training epoch:

1. Search alternatives to the current global recurrent rotation
   (0..63), accepting only strictly lower **full-training** loss.
2. Search one-bit flips of each gate truth-table mask over **all 64
   state lanes**, again accepting only strict global improvements.
3. When no single-coordinate move improves L, evaluate two-parameter
   proposals: simultaneously change the rotation and flip one gate
   truth-table bit. Choose the best strictly improving pair (or stop
   upon zero loss). This escapes some non-separable loss barriers
   that ordinary coordinate search cannot cross.
4. Stop upon exact zero loss, no improving single/pair move, the
   epoch limit, or the explicit forward-evaluation budget.

With shared recurrent rotation and per-lane Boolean unary tables,
the number of *single-coordinate* candidates in a complete round is
at most 63 + 64 * P, with P=6 for BPS-2 or 5 for BPS-3.
A full pair fallback may evaluate at most 63 * 64 * P additional
candidates before its budget stops it. Each candidate performs a
sequential whole-trajectory evaluation. **Training is not
bit-sliced-parallel here and may be computationally expensive.**

Accept/reject decisions consult only the training set. The separately
provided validation dataset is evaluated at the beginning and end,
without adjusting the model, early stopping, or tie-breaks. Validation
loss can increase: overfitting is neither hidden nor prevented by this
baseline. The caller must establish a genuinely independent
train/validation split (the API cannot verify provenance).

The fit call is transactional on invalid datasets and evaluation
errors; for ordinary budget exhaustion, the best accepted parameters
are returned together with an explicit budget_exhausted flag.
Reports expose initial/final training loss, initial/final validation
loss, mutation counts, pair candidate counts, total candidate
evaluations, epochs started and initial/final recurrent rotations.

## Tests: falsification rather than cherry-picked accuracy claims

- A teacher-generated BPS-3 delayed-rotation task where a wrong gate
  makes the correct rotation **strictly worse in isolation**; only a
  coupled rotation+gate mutation can reach zero training loss.
- A distinct held-out sequence with a different initial-state lane.
- A supervised-output mask observing only lane 3, requiring learning a
  gate on hidden upstream lane 2 for correct delayed output after a
  rotation.
- A validation-target permutation test: changing only the held-out
  labels must not change the final trained parameters.
- A one-candidate budget with explicit stop reporting.
- Invalid data, zero-epoch and zero-budget refusal without changing
  the original model.
- Equivalence of the loss oracle to the original BPS-2 forward_serial
  implementation, plus BPS-3 forward_dense/event scan on unseen inputs.

Reproduction:

    cargo fmt --all -- --check
    cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
    cargo test -p flat-algebraic-attention boolean_coupled_learning
    cargo run -p flat-algebraic-attention --example bps4_joint_training_smoke

The exact PR HEAD must also pass the complete standard CI (Rust, WGPU,
MAA host, semver, CodeQL, supply-chain). There is no production API
promotion in this slice.

## What still needs research evidence

1. Held-out delayed-copy, selective-copy and long-sequence benchmarks
   with registered dataset generation, deterministic seeds, train/test
   separation, comparable parameter and training-compute budgets.
2. Baselines: BPS-1's independent trainer where applicable, BPS-2
   rotation-only search, BPS-3 event runtime, minLSTM and FP32 LSTM.
   Report tasks where BPS-4 underperforms, not just fitting toy data.
3. Different optimizers (local event plasticity, surrogate gradients,
   evolutionary search) to reduce the serial pair-search cost.
4. Larger fan-in, non-rotational connectivity and state-dependent
   gating are **not** algebraically scan-closed by this result; each
   requires new representation and proofs.
5. Actual latency and energy on explicitly identified CPU hosts via
   Memorithm/RemoteOps and TDI; gate flips and candidate counts are
   not joules.

## Prior-art / novelty boundary

This experiment does not establish broad scientific priority.
Relevant existing work:

- Li et al., *Towards Binary-Valued Gates for Robust LSTM Training*,
  ICML (2018), https://proceedings.mlr.press/v80/li18c.html
- Feng et al., *Were RNNs All We Needed?* (2024), input-only gated
  recurrences and parallel scan, https://arxiv.org/abs/2410.01201
- Hubara et al., *Binarized Neural Networks* (2016),
  https://arxiv.org/abs/1602.02505

BPS-4's contribution is a falsifiable **coupled recurrent
gate/topology learning baseline** with exact evaluators, a bounded
pairwise search, and explicit held-out bookkeeping, not a novel
training paradigm established against the literature.
