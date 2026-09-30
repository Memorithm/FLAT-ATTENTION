# Compact preselection execution roadmap

Date: 2026-09-30. Scope: research-only extension of existing FLAT structural routing, prompted by the user-provided Gemini discussion. See [the bootstrap](COMPACT_PRESELECTION_BOOTSTRAP.md) for exact semantics and exclusions.

No step below replaces a frozen MAA or KVLab protocol. Full/Reindex/Reuse work remains under the existing MAA-15 programme; the presence of `research_cross_layer_reuse` is not evidence of model-quality-preserving reuse.

## CPS-1 — executable host screening and robustness

Implementation: `crates/flat-algebraic-attention/src/compact_preselection.rs`.

Tests: `crates/flat-algebraic-attention/tests/compact_preselection.rs`.

Deliverables:
- bounded coordinate-projected top-k screening, before exact numerical attention;
- canonical original candidate IDs, causal filtering before ranking, explicit budget-zero behavior;
- no dense score matrix or dense Boolean mask;
- reuse of existing structural sparse online softmax and dense oracles;
- independent f64 numerical comparison, irregular dimensions, multihead/batch tests, invalid-input rejection;
- retained-mass and output-error panel with all-accept, full-coordinate, recency and matched-random controls;
- a mandatory dominant-key-drop negative control.

Gate: all required repository checks succeed on the exact head. Record concrete CI and merge identities separately; implementation existence is not test execution or performance qualification.

## CPS-2 — KVLab-owned comparative quality campaign

Not implemented by CPS-1. Before measurements, KVLab must freeze fixtures/model revisions, coordinate-selection or projection policy, selection budget, seeds, controls, primary metric, quality guard, multiplicity and holdout boundaries.

Compare compact screening plus full-dimensional survivor scoring against direct approximate-score replacement, full-score ranking, Boolean selection, recency and matched-density random selection. Preserve top-k recall, retained mass and downstream output/model error as separate metrics. Include projection construction, full validation, selection and exact scoring in cost accounting. Never calibrate on final holdout or discard failed attempts.

Destination reuse: KVLab can consume the Rust producer and frozen fixtures without reimplementing attention semantics. A separately reviewed consumer and immutable producer revision are required before calling that integration complete.

## CPS-3 — SLHAv2 fidelity and monotonic repair experiment

Not implemented by CPS-1. Use KVLab's qualified policy to test whether a compact SLHA representation is useful as a selector even when it is not acceptable as a final score replacement. Keep high-fidelity K/V for admitted keys. Compare bounded widening/repair through the existing MAA transport against simpler larger-budget controls.

Gate: same-model paired quality noninferiority and complete storage/latency accounting. Exact scores on selected keys do not repair a false negative. Dense retained mass is an oracle diagnostic, not a deployable confidence signal.

## CPS-4 — physical carrier and cross-layer composition

Not implemented by CPS-1. Qualify the existing exact structural WGPU carrier with the frozen selector before benchmarking. Treat selector work, resident K/V execution, transfers and synchronization as distinct measured phases, with end-to-end time authoritative.

Only then compose with independently qualified MAA-15 Full/Reindex/Reuse and hierarchical-pool contracts, preserving layer, representation, epoch, geometry and causal-domain identity. No reuse semantics are inferred from coordinate similarity.

## CPS-5 — ElasticXxx and SciRust handoffs

Not implemented by CPS-1. ElasticXxx may choose only among evidence-qualified budgets/representations under explicit quality invariants and rollback rules. Existing elastic u64/u128/limb-width control planes remain separate from numerical K/V precision.

SciRust is a possible destination for genuinely reusable checked selection/reduction primitives after independent review. FLAT retains attention and mask semantics; KVLab retains scientific comparison authority. No new general-purpose scheduler, codec or second KV laboratory is introduced.

## Exit conditions

A useful synthetic example is not promotion. Require an independently reproducible gain on a declared systems objective, quality within a frozen budget, all preparation/routing/repair overhead included, exact source/environment identity, and destination requalification before changing a runtime default.
