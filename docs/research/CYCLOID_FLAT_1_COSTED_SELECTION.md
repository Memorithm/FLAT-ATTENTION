# CYCLOID-FLAT-1 — cost/fidelity candidate decision (research only)

Status: executable host diagnostic in flat-algebraic-attention. The stable
flat_attention::api::v1, default GPU routing, numerical Q/K/V payloads and
model quality are unchanged.

## Hypothesis

A path with additional routing overhead may be cheaper overall if it avoids
enough expensive K/V reads and numerical attention operations. Geometric
shortest paths and local embedding curvature are **not** assumed to predict
that cost. There is no trained Riemannian metric or physical speedup result.

Unlike SML-GENIUS, which has no native Transformer Q/K/V attention in its
candidate architecture, FLAT owns numerical attention semantics. This
experimental selector is therefore kept within FLAT's algebraic research crate.

## Existing contracts reused

- Existing MAA Boolean/F2/ANF admission creates possible sparse key sets.
- Existing softmax_mass::retained_softmax_mass computes the reference mass of
  a specified set against caller-supplied dense QK scores.
- Existing softmax_mass::output_error_bound computes a conservative
  per-output-component bound based on an independently supplied max |V|.

Cycloid-FLAT-1 calls both existing oracles rather than creating a second
numerical softmax or structural mask schema. It does NOT construct the sparse
candidate sets, materialize K/V, run a GPU kernel, or promote any candidate.

## Decision rule

Caller supplies a finite dense score vector, up to 64 prequalified candidate
key-index lists (with the same original key identity), a dense reference
cost profile, a cost profile for each candidate, a maximum absolute value
bound, and fidelity/resource limits.

Each profile includes estimated routing time, estimated K/V read time,
estimated numerical time, and estimated bytes read.

For each candidate, exact host oracle computations yield retained softmax mass
and the existing output error bound. Selection is admissible only if:

1. mass >= frozen minimum mass;
2. conservative output error bound <= frozen threshold;
3. estimated read bytes <= frozen cap;
4. estimated routing + reading + numerical cost is **strictly lower** than
   the dense reference cost, after all declared routing overhead;
5. indices are non-empty, unique, and within the original dense key positions.

Choose the cheapest admissible candidate. Equal costs break on fewer read
bytes, then smaller candidate ID. Otherwise retain the dense reference and
report a detailed rejection reason for each candidate.

NOTE: the dense fallback can exceed the candidate read target. It is an
authoritative correctness reference, not a deployment-ready memory guarantee.
All per-candidate scores are supplied by an independent dense oracle,
so the current operation is O(number_of_candidates * dense_key_count).
Neither lower inference compute nor lower physical DRAM traffic follows from
this standalone decision.

A softmax-output bound is not evidence of downstream next-token accuracy,
perplexity parity, or long-context retrieval correctness.

## Implementation and tests

Code: [cycloid_cost.rs](../../crates/flat-algebraic-attention/src/cycloid_cost.rs).

Five deterministic tests cover an indirect candidate with lower declared
cost; a dominant-key-drop negative control; routing cost/read-budget
rejection; deterministic tie-breaking; and malformed evidence rejection.

Qualification must run all required FLAT CI checks on the final PR head.
The independent dense attention oracle remains authoritative.

## Follow-on gates

1. Connect this decision function to actual canonical MAA-generated candidate
   sets with shared batch/head/position/causality/epoch identities. This is
   currently a **caller obligation**, not a validated adapter.
2. On pinned hardware, profile complete route + K/V staging + numerical
   execution boundaries via Memorithm/RemoteOps. The profile must include
   memory-layout, residency and input identity and must report physical latency,
   actual traffic and all calibration costs separately.
3. Replace reliance on a runtime dense score vector with a prequalified cheap
   selector; compare against dense, Boolean-only, matched-density/random and
   adversarial dominant-key controls, with independent holdout.
4. Only then test the full numerical output, end-to-end model quality, tokens/s
   and memory resident/traffic on a real non-NVIDIA CPU/WGPU target. There is
   no default route change until all evidence gates are closed.

Source contracts:
- docs/research/MULTI_ALGEBRA_ATTENTION.md
- docs/research/MAA_F2_FILTER_EVALUATION_2026_11_07.md
- .agent/FLAT_ATTENTION_ECOSYSTEM_ROADMAP.yaml (agent/ecosystem-roadmap)
- .agent/ML_MATURITY_5_OF_5.yaml (agent/ecosystem-roadmap)


## CYCLOID-FLAT-2 — executable canonical sparse reference adapter

The [Cycloid numerical adapter](../../crates/flat-algebraic-attention/src/cycloid_execution.rs)
now exercises the existing authoritative structural attention oracles with
the output of the cost/fidelity selector. This is functional Rust code:

1. Validate **all** Q, K and V using the dense structural oracle.
2. For every batch/head/query row, recompute full eligible Q·K scores in f64
   from the original Q/K inputs. No sparse selection is allowed to skip
   dense-reference numerical validation in this offline diagnostic.
3. Reject out-of-range, causal-future, and other invalid route metadata.
4. Compute a conservative max-|V| bound from the **actual entire head**,
   not from a caller's underreported bound.
5. Invoke existing \`choose_costed_attention\` with the supplied, frozen
   dense/sparse cost profiles; preserve original key positions.
6. Canonicalize the chosen query rows using
   \`StructuralCandidateSet::from_rows\`.
7. Run \`forward_reference_structural_sparse\` with original numerical Q/K/V,
   and compare O/LSE against the independent all-accept dense oracle.
8. Report actual logical executed query/key pairs and maximum numerical
   output/LSE discrepancies. No physical GPU or memory metrics are inferred.

Five tests cover real sparse attention output, dominant-key-drop dense
fallback, causal-future rejection, invalid row-count rejection, and exact
causal dense fallback.

### Non-promotion boundaries

This path **recomputes the dense attention scores** and executes an
independent dense numerical oracle before choosing a candidate. It is
therefore *necessarily slower in total algorithmic work* than an
appropriately optimized dense-only kernel, all else equal. It is designed
to qualify semantics and make cost/quality errors inspectable, not to be
used for inference. The number of sparse query/key evaluations reported
excludes the deliberate dense oracle and selector overhead.

Actual speed research must replace that dense-oracle-dependent selector
with a separately qualified precomputed/sparse or Boolean predicate,
reuse original candidate identities and demonstrate a **net**
end-to-end measured advantage under frozen quality budgets. A reduction
in selected pair count alone is not that evidence. Runtime promotion
requires all normal FLAT hardware and model-quality gates.
