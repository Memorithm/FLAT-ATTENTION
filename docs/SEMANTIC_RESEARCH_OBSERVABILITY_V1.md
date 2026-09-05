# FLAT semantic research observability v1

Status: Phase-A research-only contract for issue #132.

This document describes `flat-semantic-observability`. The crate is opt-in and exists outside the historical `flat-attention` fast path.

## Purpose

The latest #132 / TDI research requirement is not a new optimized kernel. It is a bounded reference-mode surface that can:

- expose semantic identity independently from kernel identity;
- apply one declared one-shot intervention;
- keep reference and perturbed execution on identical semantic dynamics;
- expose downstream observations without materializing an N x N probability matrix;
- leave ITD static summaries and TDI dynamic-recovery metrics as separate downstream evidence;
- impose zero instrumentation cost when disabled.

The first concrete adapter is StandardSoftmax because it is the canonical S0 regression baseline and already has a deterministic scalar/reference oracle.

## Disabled-path guarantee

`flat-attention` does not depend on `flat-semantic-observability` and does not invoke it implicitly.

Therefore disabling research instrumentation means simply using the existing FLAT APIs. No instrumentation branch, callback, allocation, registry lookup or dynamic dispatch is added to the existing specialized path.

Research-mode allocations are explicit and acceptable only when the caller invokes this crate.

## Identity

`ResearchExecutionIdentity` carries two deterministic identities for the same invocation:

- the existing exact `StandardSoftmaxSemantic` instance fingerprint;
- the `StandardSoftmaxMechanism` fingerprint from the Phase-A mechanism decomposition.

Kernel, device, pipeline and benchmark identities are absent. The same identity is stored once for both reference and perturbed observations.

## Intervention contract

`ResearchInterventionSiteKind` defines a semantic-neutral vocabulary:

- token/input element;
- Q element;
- K element;
- V element;
- semantic-state element;
- recurrent-memory element;
- operator-parameter element.

A concrete semantic adapter must reject site kinds it does not own.

StandardSoftmax v1 accepts only Q/K/V element interventions. Recurrent/state/operator sites fail closed rather than being mapped through a fake softmax abstraction.

`ScalarResearchIntervention` supports finite `Add` and `Replace` mutations. Invalid values, out-of-range coordinates and non-finite post-mutation results fail closed.

The public surface intentionally has no task label, target, evaluation class or TDI verdict field. Those remain experiment-harness-owned metadata.

## Paired StandardSoftmax reference execution

`run_standard_softmax_paired_reference`:

1. validates the intervention contract;
2. executes the unperturbed scalar/reference semantic;
3. copies exactly the addressed Q, K or V research input tensor;
4. applies exactly one mutation to that copy;
5. executes the perturbed arm through the same `StandardSoftmaxSemantic` instance;
6. returns raw output and LSE observations for both arms under one shared semantic/mechanism identity.

The function does not construct a score matrix or probability matrix. It delegates both executions to the existing online-softmax scalar/reference implementation.

`ResearchObservationDepth` is a caller-owned layer/step/depth label. FLAT records it but does not interpret model-graph semantics.

## Evidence boundary

The returned pair is raw FLAT reference evidence only.

FLAT does not compute or merge:

- ITD localization/concentration/effective-rank/operator summaries;
- TDI recovery/future-overlap descriptors;
- task accuracy or retrieval deficit;
- statistical intervals;
- semantic-promotion verdicts.

Those remain separate evidence blocks in their owning research harnesses. This prevents a FLAT instrumentation API from silently becoming the scientific decision rule.

## Non-softmax boundary

The intervention-site vocabulary is intentionally broad enough for future stateful semantics, but this PR does not implement or authorize one.

A future recurrent semantic may expose `SemanticStateElement` or `RecurrentMemoryElement`; a structured operator may expose `OperatorParameter`. Such support requires that semantic's own frozen scalar/reference contract and ITD/TDI admission evidence.

Phase B of #132 remains blocked on research selection of a specific non-softmax candidate. No quality, novelty, FLOP, memory, latency or throughput claim is made here.
