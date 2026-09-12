# FLAT-ATTENTION Engineering Roadmap

This document is the execution plan for FLAT-ATTENTION, Memorithm's Rust-native fused attention engine for the SciRust ecosystem.

The roadmap is gate-driven. Code is not a completed milestone until correctness, CI, device qualification and any claimed performance are backed by reproducible evidence. Current completion state belongs in `docs/ROADMAP_STATUS.md`; this file defines the target architecture and acceptance order.

## 0. Non-negotiable engineering rules

### 0.1 Language and dependency sovereignty

- Rust is the host implementation language.
- No project-authored C/C++ implementation layer or C ABI bridge.
- No mandatory CUDA C++, `nvcc`, WMMA, CUTLASS, cuDNN or similar vendor SDK in the core architecture.
- Portable GPU execution uses open shader/IR paths first.
- Hardware-specific acceleration must sit behind explicit capability detection and preserve a portable fallback.
- Optimized paths must never silently substitute CPU execution.

### 0.2 Correctness hierarchy

1. mathematical definition;
2. deterministic scalar Rust oracle;
3. portable fused GPU implementation;
4. optimized GPU implementation;
5. backend-specific open-codegen specialization.

Every layer is validated against the layer above it.

### 0.3 Merge gate

Every implementation change uses a dedicated branch/PR. Inspect the exact head SHA, fix all real CI/review failures without weakening gates, and merge only when all required checks are green and the PR is mergeable. After merge, verify the new default branch before starting dependent work.

### 0.4 Performance honesty

No speedup, throughput, bandwidth, latency, memory or efficiency claim is accepted without a reproducible benchmark tied to an exact commit and identified device/backend. Record precision, shape, causal mode, warm-up, iterations, latency statistics, tokens/s or other relevant throughput, and memory/traffic evidence when measurable.

### 0.5 Priority research override — Boolean Front-End Attention

Boolean control of attention is now a priority research track.

The hypothesis is not that a bitwise instruction is universally faster than modern dense matrix hardware. The hypothesis is that compact Boolean decisions can happen early enough, and in sufficiently parallel form, to prevent expensive numerical attention work and K/V movement from being launched at all.

The governing systems inequality is:

```text
T_boolean_front_end + T_FLAT_survivors < T_FLAT_dense
```

The initial target is therefore a Boolean control plane in front of exact FLAT attention:

```text
Q/K state
  -> compact Boolean signatures / routing metadata
  -> bit-parallel block/page admission
     -> reject: do not issue exact Q·K and, where architecture permits, do not stage/read K/V
     -> accept: execute qualified FLAT numerical attention
```

Research ownership:

- **BooleanLab**: primary scientific bench for Boolean functions, signatures, routing predicates, equivalence, search and controlled experiments.
- **FLAT-ATTENTION**: mask consumption, GPU routing, kernels and end-to-end systems qualification.
- **KVLab**: cache/page selection, retention, tiering and decode-cache experiments.
- **SciRust**: promotion target for reusable packed-bit/math primitives.
- **NNIS**: optional native hardware qualification; it must not become a mandatory dependency of the portable FLAT core.

Dense qualified attention remains the reference and fallback. Boolean routing must never silently weaken quality guarantees.

## 1. Target public contract

FLAT-ATTENTION must support:

- dense MHA, GQA and MQA;
- causal and non-causal attention;
- prefill and decode;
- resident KV cache;
- asymmetric Q/KV lengths and variable-length batches;
- masks and additive biases;
- optional bitpacked Boolean block/page admission before expensive numerical work;
- forward and backward;
- deterministic reference mode;
- portable GPU mode;
- optimized device-specialized mode;
- explicit dense fallback when Boolean routing is unsupported or not beneficial.

Canonical logical layout:

```text
Q: [batch, q_heads, q_len, head_dim]
K: [batch, kv_heads, kv_len, head_dim]
V: [batch, kv_heads, kv_len, value_dim]
O: [batch, q_heads, q_len, value_dim]
```

---

# PHASE A — Mathematical and repository foundation

## M1 — Scalar oracle + fused portable forward

Deterministic online-softmax oracle, causal/non-causal semantics, O/LSE outputs, fused WGSL path, no materialized `N x N` score/probability matrix, shader validation and strict MSRV/fmt/Clippy/tests.

## M2 — WGPU executor

Real optional WGPU adapter/device/queue path, explicit buffers/bindings/dispatch, resident-buffer execution, explicit unsupported-backend errors and no disguised CPU fallback.

## M3 — Device parity matrix

GPU-vs-reference parity across head dimensions, tile-boundary sequence lengths, batches/heads, causal modes and adversarial numerical fixtures. O and LSE are checked independently.

---

# PHASE B — First performance architecture

## M4 — Multi-query-row tiled kernel

Reuse each staged K/V tile across multiple query rows while maintaining independent online-softmax state per query row.

## M5 — Subgroup reductions

Use subgroup-assisted dot reductions only when capabilities are explicitly exposed; retain deterministic fallback and never assume subgroup width.

## M6 — Vectorized memory transactions

Aligned packed Q/K/V/O transfers for common dimensions with correct scalar tails and explicit alignment contracts.

## M7 — Double-buffered K/V staging

Ping/pong workgroup tiles with documented barrier discipline and promotion only after real-device evidence.

---

# PHASE C — Precision and numerical policy

## M8 — Mixed precision

f16 where exposed, FP32 accumulation/online-softmax state, explicit conversion policy and f32 fallback.

## M9 — Numerical policy layer

Reference, fast-portable and deterministic modes with documented guarantees and regression fixtures.

---

# PHASE D — Modern attention shapes

## M10 — GQA/MQA

Independent Q/KV head counts, no physical K/V duplication, explicit invalid grouping errors.

## M11 — Asymmetric Q/KV lengths

Rectangular attention and cross-attention without square-matrix assumptions.

## M12 — Variable-length batches

Per-sequence validity metadata, padded/ragged handling and zero contribution outside declared lengths.

## M13 — Mask and bias extensibility

Causal/padding masks, additive bias, ALiBi-compatible path and clean extension points. Add an explicit bitpacked Boolean block-admission hook that does not require a dense floating-point mask.

Acceptance for Boolean admission semantics starts with an all-accept mode exactly equivalent to the existing dense oracle.

---

# PHASE E — Boolean Front-End Attention — PRIORITY

The M13B milestones are inserted without renumbering historical M14–M43 references.

## M13B.1 — Bitpacked Boolean admission contract + exact oracle

Define the boundary between the policy producing decisions and the exact attention kernel consuming them.

Deliver:

- `BooleanAttentionMask` or equivalent bitpacked contract;
- block-level admission as the primary execution representation;
- token-level mode only for oracle/research use;
- exact Rust oracle that skips rejected interactions but otherwise preserves qualified score/online-softmax semantics;
- composition rules with causal/padding/bias semantics;
- exact storage-bit accounting;
- all-accept, all-reject, structured-sparse and adversarial fixtures.

Accept only if all-accept is numerically identical to dense reference, rejected entries cannot affect O/LSE, invalid geometry fails explicitly and no `N x N` floating mask is required.

## M13B.2 — Bitpacked Q/K signatures + Boolean prefilter

Candidate signatures:

```text
q -> S_q(q) -> packed bits
k -> S_k(k) -> packed bits
similarity -> XOR/XNOR + population count
```

Deliver deterministic signature APIs, packing tests, preregistered threshold/top-k/block policies and matched controls.

Mandatory scientific metrics:

- retained density `rho`;
- recall of a declared dense-attention target set;
- false-negative rate;
- O/LSE deviation;
- downstream quality where model-level evaluation exists;
- Boolean bits per token/block;
- exact numerical Q·K work estimated as avoided.

Compare against dense FLAT, structural masks and random masks matched for density. No speed claim from operation counts alone.

## M13B.3 — Boolean block router before K/V staging

This is the central systems milestone.

Deliver a portable WGPU bitpacked block-mask path whose decision is consumed **before** K/V tile staging and exact Q·K work. Prefer block-coherent routing to irregular token-level divergence. Expose accepted/rejected block telemetry and extend logical K/V load accounting.

Acceptance:

- all-accept matches existing qualified kernel;
- partial masks match M13B.1 oracle;
- rejected blocks do not execute K/V tile staging in the explicit shader control flow;
- device parity and shader validation pass;
- Boolean overhead and surviving numerical work are timed separately;
- default promotion requires a measured real-device win for at least one preregistered workload.

## M13B.4 — Two-plane overlap + first-token readiness

Evaluate a Boolean plane preparing admission while the numerical plane processes already accepted work:

```text
time ---------------------------------------------->
Boolean:   B0 ---- B1 ---- B2 ---- B3 ---- B4
                |       |       |       |
Numerical:      N0 -----N1------N2------N3
```

Requirements:

- precompute/store K or KV signatures with resident state;
- produce the Q signature as soon as the current query representation exists;
- allow the **first generated token after prefill** to consume valid Boolean metadata;
- distinguish true overlap from serial prefiltering using timing/trace evidence;
- report first-token and steady-state decode separately;
- compare fused/same-dispatch and multi-dispatch designs where meaningful.

## M13B.5 — Boolean KV page router

Co-develop with KVLab and M14–M16. Store compact signature/metadata beside KV pages/blocks so a cheap decision can determine whether a larger K/V region should be fetched.

Record metadata bytes read, K/V bytes avoided, false negatives, downstream quality, latency and cache consistency. Reset/reuse/append must keep signatures transactionally aligned with KV ownership. No host-side routing dependency in the token loop.

## M13B.6 — 1-bit QK research gate

Only after the router path is understood, test replacing exact Q·K itself with a Boolean similarity such as:

```text
score_bool(q,k) = transform(popcount(XNOR(S_q(q), S_k(k))))
```

This is a research gate, not the assumed destination. Compare exact FLAT Q·K, low-precision numerical paths and Boolean-prefilter-plus-exact-QK. Promote only if end-to-end quality/performance is non-dominated for a declared workload.

## M13B comparison ladder

Where applicable every campaign compares:

1. qualified dense FLAT;
2. structural non-learned mask;
3. random mask matched for density;
4. Boolean router + exact FLAT;
5. Boolean router + resident/paged KV;
6. 1-bit QK only after M13B.1–M13B.3 qualification.

Systems metrics: first-token latency, steady decode, prefill latency, Boolean front-end time, retained density, surviving exact-QK time, K/V bytes requested/avoided, metadata bytes, dispatch/synchronization, temporary memory, tokens/s, and energy only when directly measurable.

Scientific metrics: dense-target recall, false-negative rate, O/LSE deviation, downstream quality, robustness across sequence/head distributions and exact provenance of the routing function/policy.

## Immediate execution order when development resumes

1. verify current `main` and existing dense/M13 paths remain green;
2. implement M13B.1;
3. run M13B.2 experiments jointly with BooleanLab;
4. choose block granularity from measured quality/sparsity evidence;
5. implement M13B.3 before K/V staging;
6. benchmark dense versus Boolean-router paths on real hardware;
7. integrate M13B.4 with decode and first-token measurement;
8. co-develop M13B.5 with M14–M16 and KVLab;
9. open M13B.6 only if the earlier evidence justifies replacing exact Q·K.

---

# PHASE F — Inference-first KV architecture

## M14 — Resident KV cache

Append-only device-resident K/V with capacity/current-length metadata, GQA/MQA-compatible indexing and an extension point for resident Boolean signatures. Boolean metadata, when enabled, must remain transactionally consistent with its K/V entry.

## M15 — Specialized decode (`q_len = 1`)

Streaming online-softmax over resident KV, no score matrix, direct resident output and optional M13B admission from the first generated token. Report dense and Boolean first-token/steady-state latency separately.

## M16 — Chunked prefill + paged KV

Chunked long-context execution, logical-to-physical page mapping, fragmentation telemetry and optional compact Boolean page metadata. Page reuse/reset must never leave stale routing metadata.

---

# PHASE G — Training support

## M17 — Backward mathematical oracle

Scalar dQ/dK/dV using saved O/LSE and finite-difference checks.

## M18 — Fused recomputation backward GPU kernel

Recompute scores/probabilities from Q/K/LSE, no stored `N x N` probability matrix and race-free dQ/dK/dV accumulation.

## M19 — Backward tiling/specialization

Tiled, subgroup/vectorized and mixed-precision variants with deterministic fallback.

---

# PHASE H — FLAT code generation and matrix engines

## M20 — FLAT Kernel IR

Represent loads/stores, dot/matrix fragments, reductions, online-softmax transitions, barriers, vector types, capabilities and optional Boolean mask/signature operations explicitly. Deterministic serialization/hash is required.

## M21 — Portable WGSL emitter

Deterministic WGSL generation, specialization constants, generated-source hashing/cache and validator integration.

## M22 — Open cooperative/subgroup matrix research

Track SPIR-V/Vulkan/WebGPU open matrix capabilities. No vendor-only SDK becomes mandatory and no acceleration is claimed without real-device proof.

## M23 — Matrix fragment scheduler

Architecture-independent fragment layout/scheduler for common D=64/128 with correctness before performance.

---

# PHASE I — Autotuning

## M24 — Device capability model

Limits, workgroup storage, subgroup properties, f16, binding limits, adapter/backend fingerprint. Boolean-routing characteristics may be recorded only from exposed capability or measurement, never marketing-name assumptions.

## M25 — Deterministic candidate generator

Candidates include tile sizes, workgroup size, vector width, subgroup/f16 choices, prefill/decode/GQA mapping and, after M13B qualification, Boolean routing enablement, signature width and block granularity.

## M26 — Benchmark-driven autotuner

Correctness gate before timing, robust measurement, persistent cache and safe invalidation. Disable Boolean routing automatically when measured front-end cost outweighs savings for the tuned workload.

---

# PHASE J — Benchmark and observability

## M27 — Benchmark harness

Sweep prefill/decode, MHA/GQA/MQA, common D values and context sizes. Add Boolean sweeps over retained density, signature width and block/page granularity.

Report latency, first-token latency, tokens/s, bandwidth/work estimates, Boolean time, accepted/rejected blocks/pages, K/V bytes avoided, metadata bytes, allocations, temporary bytes, dispatch count and optional externally measured power.

## M28 — Baseline comparison

At minimum compare scalar oracle, prior SciRust WGPU path, portable dense FLAT, optimized FLAT generations and dense FLAT versus Boolean-router+exact-FLAT under matched workload semantics.

## M29 — Runtime telemetry

Kernel ID, tile geometry, device/backend fingerprint, dispatches, temporaries, fallback reason, autotuner cache state, Boolean routing state, retained density and Boolean timing when observable without mandatory hot-path synchronization.

---

# PHASE K — SciRust and SciAgent integration

## M30 — Stable standalone API

Backend-neutral requests/configs, owned/resident variants, explicit errors and semver policy.

## M31 — SciRust WGPU adapter

Zero-unnecessary-copy integration, explicit fallback and reusable packed-bit primitives promoted from validated work rather than duplicated.

## M32 — SciAgent prefill

Model wiring, resident weights/activations and fixed-prompt/seed parity where sampling contracts permit.

## M33 — SciAgent decode/KV

Resident KV adapter, q_len=1 dispatch, no host round-trip. Boolean page routing enters only after M13B.5 qualification. Benchmark first-token and steady decode separately.

---

# PHASE L — Portability qualification

## M34 — Vulkan/Linux

Correctness on lavapipe where appropriate and performance only on real GPUs.

## M35 — Direct3D 12/Windows

Pipeline/parity qualification with explicit platform limitations.

## M36 — Metal

WGPU/Metal parity on available Apple hardware.

## M37 — Vendor diversity

Qualify NVIDIA, AMD, Intel and software Vulkan as hardware permits. Boolean-front-end results are device/backend-specific until evidence supports broader generalization.

---

# PHASE M — Robustness and reproducibility

## M38 — Property/stress tests

Randomized shapes, causal/softmax invariants, repeated dispatch, cache reset/reuse, finite extremes and Boolean all-accept/all-reject/malformed/stale-metadata cases.

## M39 — Host API/config fuzzing

Shapes, overflow, lengths, tuning-cache/kernel metadata and Boolean mask/signature geometry. No GPU-driver fuzzing claim without an isolated harness.

## M40 — Reproducible benchmark manifests

Machine-readable SHA/environment/config/results plus Boolean policy ID, signature width, block/page granularity and retained density when applicable.

---

# PHASE N — Productization

## M41 — Licensing/ownership

Maintain explicit PolyForm Noncommercial licensing/ownership and third-party inventory consistent with repository policy.

## M42 — Documentation

Architecture, algorithms, public API, SciRust integration, GPU backend, Boolean Front-End Attention evidence guide, autotuning, benchmarks and troubleshooting.

## M43 — Release discipline

Changelog, release checklist, compatibility matrix and benchmark snapshot for significant releases.

---

# PHASE O — Continuous optimization loop

Every optimization starts from a stated bottleneck and baseline, changes one coherent mechanism, preserves correctness, runs full CI and is retained only when evidence justifies it.

Optimization targets include global-memory traffic, Boolean early-admission overhead versus eliminated work, bitpacked signature layout, K/V block/page rejection rate, workgroup reuse, occupancy, register pressure, subgroup utilization, vector transfers, synchronization, tile work distribution, pipeline/cache overhead, decode launch overhead, GQA KV reuse, long-context scheduling and open matrix engines.

No mechanism is permanent merely because it is theoretically faster. The benchmark decides.

---

# Definition of Done for FLAT-ATTENTION 1.0

FLAT-ATTENTION is 1.0-ready only when the stable API, deterministic forward/backward oracles, real fused forward/backward GPU paths, MHA/GQA/MQA, asymmetric lengths, resident KV + decode, no `N x N` probability storage, supported mixed precision, portable WGPU, autotuned kernels, SciRust/SciAgent integration, reproducible benchmarks, CI discipline and documented limitations are complete.

Boolean Front-End Attention is not required to beat dense attention on every workload. If enabled in 1.0, it must have a qualified dense fallback, explicit quality semantics, reproducible device evidence and automatic/explicit disablement when it is not beneficial.