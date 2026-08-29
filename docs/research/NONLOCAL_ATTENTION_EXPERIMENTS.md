# Nonlocal attention experiment contract

Status: research-only design contract. No production attention path is changed by this document.

Audited FLAT head: `4180e33674a33937b7b84199f6beb867fb16aa39` (`main`).

Research provenance: `Memorithm/nonlocal-relativity-v2` and the SciRust harvest contract `docs/NONLOCAL_RESEARCH_HARVEST.md` establish reusable engineering patterns around explicit history, true positions, transforms, contextual weights, approximation classification and evidence discipline. They do **not** establish a general-relativistic attention law.

## Boundary

The reusable idea is structured, explicitly qualified history. No Schwarzschild, Kerr, Reissner-Nordstrom, curvature, proper-time or parallel-transport formula is admitted into FLAT attention semantics merely by analogy.

Any experimental nonlocal/recurrent attention rule must have its own mathematical definition, scalar oracle, causal contract, quality criteria and measured evidence.

## Existing FLAT contracts that remain authoritative

At the audited head:

- `src/api.rs::api::v1` is the backend-neutral standard-attention API. `AttentionShape` already encodes native GQA/MQA (`q_heads` / `kv_heads`), asymmetric query/KV lengths and the absolute `query_position_offset`. `AttentionConfig` owns causal masking and softmax scale. The standard API remains unchanged by default.
- `src/paged_kv.rs` owns vendor-independent logical-to-physical KV page metadata, exact capacity accounting, deterministic append addressing, generation invalidation and telemetry. An experimental history policy must not silently alter these storage/address semantics.
- `src/rotary_grouped.rs` is the deterministic scalar oracle for fused head-local RoPE + native GQA/MQA. V is not rotated and causal masking is preserved. Experimental history must preserve the declared positional/RoPE semantics or expose a different semantic identity.
- `crates/flat-semantic-control` already supports typed recurrent state through `SemanticState::Recurrent`, caller-owned semantic selection with no implicit fallback, and distinct `SemanticFamily::RecurrentMemory` / `SemanticFamily::Experimental`. This is the appropriate control-plane rail for research semantics.
- kernel selection, device execution and semantic identity are separate layers. A research semantic is not promoted merely because one kernel is fast.

## Default-behavior invariant

The production/default rule is normative:

```text
no explicit research semantic request
-> current StandardSoftmax semantics
-> current causal/mask/GQA/RoPE/KV behavior
-> current candidate and backend selection
```

No future `history_mode`, schedule, weighting or budget field may change this path through a changed default value, implicit fallback, environment inference or device heuristic.

## Proposed typed research configuration

The first implementation should be semantic-specific rather than added to `api::v1::AttentionConfig`.

A suitable shape is:

```text
NonlocalAttentionConfig {
    history_mode,
    history_schedule,
    history_weighting,
    history_budget_policy,
}
```

with explicit enums/typed values and no unvalidated string dispatch.

### `history_mode`

Must distinguish at least:

- complete/reference history;
- explicit bounded/windowed approximation;
- future sampled/compressed representations only when their approximation semantics are defined.

A bounded/sampled/compressed mode is never relabelled as the reference behavior.

### `history_schedule`

Controls *which true logical positions* participate. It must not reconstruct a non-uniform historical position from an array index.

### `history_weighting`

Contextual weighting is separate from the history kernel. The identity weighting is exact multiplicative identity. Non-finite weights fail closed. Domain-specific formulas require their own semantic definition and evidence.

### `history_budget_policy`

A budget constrains resource use; it is not scientific correctness. Budget exhaustion must produce an explicit decision/error/approximation state rather than silently changing the semantic rule.

## Small-attention scalar reference

Before WGPU or optimized kernels, implement a deterministic scalar oracle for the experimental semantic on deliberately small problems.

The oracle must make explicit:

1. Q/K/V or recurrent-state input meaning;
2. exact causal admissibility of every historical contribution;
3. true logical/absolute positions;
4. GQA/MQA head mapping;
5. RoPE interaction, if the semantic uses RoPE;
6. history transform/weight/kernel order;
7. normalization rule and saved-state contract;
8. whether retained-history reduction is reference or approximation.

The scalar implementation is a **correctness oracle**, not a performance baseline.

## Compatibility gates

Every experimental implementation must have differential tests for unaffected FLAT semantics.

Required gates include:

- StandardSoftmax default behavior is unchanged when no research semantic is requested;
- causal masking still excludes future keys/history;
- native GQA/MQA head mapping is unchanged;
- paged-KV logical order is preserved across physical page boundaries;
- reset/generation invalidation remains authoritative for paged KV;
- RoPE absolute positions and V-not-rotated behavior remain unchanged on compatible paths;
- attention masks/bias and logits/normalization semantics remain explicit;
- repeated scalar evaluation is deterministic under identical inputs;
- complete and bounded history agree exactly when the bounded policy retains every sample, where the representation/order is otherwise identical.

## Semantic admission and candidate qualification

Research semantics enter through the semantic control plane, not through kernel autotune side effects.

A candidate must therefore pass, in order:

```text
explicit semantic identity
-> descriptor/state-contract validation
-> correctness against the semantic scalar oracle
-> approximation/evidence classification
-> only then execution/kernel qualification
```

No kernel benchmark is evidence that two semantic rules are equivalent.

The existing no-implicit-fallback rule in `SemanticSelectionPolicy` remains mandatory: if an explicitly requested research semantic is unavailable, FLAT must not silently execute StandardSoftmax and report success as though the research semantic ran.

## Evidence model

Experimental reports must distinguish at least:

- exact mathematical result;
- numerical approximation;
- empirical validation;
- phenomenological model;
- speculative model;
- rejection criterion / negative result.

Self-convergence, regression stability and speed measurements answer different questions. They are not interchangeable validation labels.

Rejected candidates and negative experimental results remain first-class records. An experiment showing that a history reduction is slower, less accurate or quality-degrading is retained rather than removed from the evidence set.

## WGPU and hardware claims

WGPU implementation begins only after the scalar semantic oracle and CPU differential tests are stable.

Claims about throughput, memory bandwidth, latency or speedup require executed hardware evidence on the named adapter/device. Software Vulkan/lavapipe remains useful for WGPU correctness but is not evidence of NVIDIA/AMD/Apple hardware performance.

No scalar/reference implementation may be presented as a competitive performance baseline.

## Progressive implementation sequence

1. define semantic-specific typed config and identity;
2. add deterministic small-attention scalar oracle;
3. add default-path and causal/GQA/RoPE/paged-KV differential tests;
4. wire semantic observer/registry admission without execution fallback;
5. add explicit history budgets/approximation reporting;
6. only then add WGPU candidates;
7. qualify WGPU against the scalar oracle before any benchmark claim;
8. retain positive and negative evidence with exact code/config/device provenance.

## Non-goals

This research program does not:

- replace standard attention by default;
- claim that fractional calculus, GR transport or curvature is beneficial for attention;
- redefine paged-KV addressing;
- bypass FLAT semantic identities or correctness gates;
- use device performance as proof of semantic correctness;
- claim asymptotic or hardware speedups before the relevant algorithm and measurements exist.
