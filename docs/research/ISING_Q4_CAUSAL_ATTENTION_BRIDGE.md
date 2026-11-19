# ISING-Q4-ATT-1 — Boolean-quaternary admission before FLAT attention

Status: opt-in research-only Rust host implementation. This integration does not
modify the production attention API, default GPU execution, or SML's non-attention
candidate contract.

## Implemented numerical bridge

The QuaternaryHeadPlan represents a frozen key-state hypothesis for one
batch/head. Each query row independently reconstructs its own four-state Ising
controller from the frozen head plan, with exact structural and causal
eligibility applied **before** building the coupling-dependent local fields.

This distinction is critical: simply discarding future keys after optimizing a
shared Ising state would allow a future key to influence earlier decisions
through its signed couplings. In the new carrier, future keys have zero
contribution before the Ising descent and so cannot produce such leakage.

A Boolean-quaternary unit has two stored state bits: activation and polarity.
A separate eligibility bit gates the unit. For attention routing, BOTH active
polarities admit the original numeric key; dormant or masked states reject it.
The numeric key/value vectors are never replaced by Ising spins.

A deterministic row-wise sweep applies strict energy-lowering updates.
A zero sweep budget is an explicit frozen-state ablation control.
The output is the existing StructuralCandidateSet, consumed unchanged by
forward_reference_structural_sparse and comparable against
forward_reference_structural_dense and fully dense reference attention.

No automatic dense fallback, secret top-k repair or empty-key substitution is
allowed. A fully rejected query returns EmptyEffectiveCandidates with its row
index. There is no claim of model quality or speedup.

## Exact tests

- All-active frozen Q4 states equal ordinary dense FLAT numeric output and LSE,
  across causal/noncausal and batch/head geometries.
- Negative active spins retain the key as a valid numeric candidate; dormant
  positive spins do not.
- Energy updates produce a smaller active candidate set when activity costs
  justify deactivation.
- A changing future spin field (including a strong coupling to a past key)
  cannot change prior causal row selections.
- Incorrect head-plan count, padding, structural masking or empty candidate rows
  are explicit errors.

Run focused validation:

    cargo test -p flat-algebraic-attention --lib ising_quaternary_routing

Then apply repository-wide rustfmt, Clippy, tests, rustdoc and required CI on
the precise head SHA. Full host/workspace qualification is mandatory before
merging.

## Evidence boundaries and next experiment

Current memory includes one transient O(N+E) Ising controller per query row
and an explicit structural candidate result that may be quadratic under
all-accept. This is an oracle-first implementation, NOT a subquadratic
complexity claim and NOT evidence of reduced physical K/V traffic.

A meaningful optimization requires a sparse compiled policy, measured
end-to-end host or WGPU timings (including routing and transfers), matched
candidate density and topology controls, per-row false-negative/retained-mass
statistics, numerical O/LSE error and downstream model quality.

The design preserves exact causal semantics and prepares a controlled
comparison with V888-derived sparse structural priors. Raw V888 connectome
data must remain outside this source tree. Do not silently introduce Q/K/V,
Transformer caches or token-pair attention into SML-GENIUS.
