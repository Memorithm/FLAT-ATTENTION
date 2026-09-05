# FLAT semantic mechanism decomposition v1

Status: Phase-A control-plane contract for issue #132.

This document describes `flat-semantic-mechanism`. The crate is intentionally backend-neutral and metadata-only. It does not select a kernel, execute a GPU path, register a semantic candidate, or promote any non-softmax mechanism.

## Purpose

Issue #132 requires the mathematical semantics of an attention/mixing rule to be separable from its execution strategy. Existing `flat-semantic` already provides:

- stable semantic identity;
- semantic family classification;
- state semantics;
- mask/causality semantics;
- weight semantics;
- semantic-specific saved-state contract;
- generic output that does not require LSE;
- a bit-exact `StandardSoftmaxSemantic` adapter over the historical scalar oracle.

The remaining mathematical decomposition was implicit. `flat-semantic-mechanism` adds explicit identities for:

1. projection contract;
2. score rule;
3. normalization / weighting rule;
4. mixing operator;
5. numerical policy.

Together with the existing `flat-semantic` state, mask and saved-state contracts, these fields cover the mathematical/control-plane decomposition requested by #132. Execution strategy remains deliberately outside the descriptor.

## Contract

`MechanismComponentId` carries:

- `MechanismComponentKind`;
- a stable lowercase ASCII slug;
- a non-zero revision.

`MechanismDescriptor` requires exactly one component of each kind:

- `Projection`;
- `Score`;
- `Normalization`;
- `Mixing`;
- `NumericalPolicy`.

A component supplied in the wrong slot fails closed. The descriptor retains the complete existing `SemanticDescriptor` rather than duplicating mask/state/weight/saved-state fields.

This is not an enum-to-kernel dispatch table. A component identity says what mathematical mechanism is declared. It does not say which shader, device, pipeline or benchmark result executes it.

## StandardSoftmax mapping

`StandardSoftmaxMechanism` wraps the already validated `flat_semantic::v1::StandardSoftmaxSemantic` and exposes the following revision-1 decomposition:

| Axis | Component |
| --- | --- |
| Projection | `direct-qkv@1` |
| Score | `scaled-dot-product@1` |
| Normalization | `row-softmax@1` |
| Mixing | `weighted-value-sum@1` |
| Numerical policy | `legacy-standard-softmax-reference@1` |

The wrapped semantic remains the sole execution authority. Its existing canonical record continues to bind causal/bidirectional masking and the exact score-scale bits.

The mechanism canonical record appends the explicit decomposition and deliberately contains no WGPU, device, kernel or benchmark identity.

## Compatibility / zero-regression rule

The historical `flat-attention` fast path does not depend on `flat-semantic-mechanism` and does not construct these descriptors implicitly. Therefore this Phase-A metadata cannot add allocation, virtual dispatch or registry lookup to existing specialized StandardSoftmax execution.

A public regression test executes StandardSoftmax through the wrapped semantic and requires bit-exact output and LSE equality with the historical scalar oracle.

Existing kernel selection remains a separate concern. The semantic/mechanism fingerprint is not a kernel fingerprint and must not be used as one.

## Research observability boundary

This tranche does not yet add the TDI research-observability surface requested in the latest #132 comments. That remains a separate Phase-A slice so instrumentation can be designed explicitly as zero-cost-when-disabled and can preserve the current online-softmax/no-materialized-NxN behavior.

In particular, this contract does not force:

- materialization of an attention probability matrix;
- token/K/V/recurrent intervention storage;
- trajectory recording;
- static diagnostic computation;
- dynamic recovery computation.

Those facilities must consume semantic/mechanism identity rather than redefine it.

## Phase-B boundary

No first non-softmax semantic is selected here. Issue #132 explicitly requires that choice to come from the ITD/TDI mechanistic research gate rather than implementation convenience.

Until that gate promotes a frozen candidate, this crate provides classification/provenance infrastructure only. It makes no quality, novelty, FLOP, memory, latency or throughput claim.
