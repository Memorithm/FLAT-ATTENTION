# FLAT-ATTENTION Engineering Roadmap

This document is the execution plan for FLAT-ATTENTION, Memorithm's Rust-native fused attention engine for the SciRust ecosystem.

The roadmap is deliberately gate-driven: a milestone is not complete because code exists. It is complete only when its acceptance criteria are demonstrated by tests, CI, device validation, and—where performance is claimed—reproducible benchmarks.

## 0. Non-negotiable engineering rules

### 0.1 Language and dependency sovereignty

- Rust is the host implementation language.
- No project-authored C or C++ implementation layer.
- No project-authored C ABI bridge.
- No CUDA C/C++, `nvcc`, WMMA, CUTLASS, cuDNN, or similar vendor SDK is required by the core architecture.
- Portable GPU execution is implemented through open shader / IR paths first.
- Hardware-specific acceleration may be added only behind explicit capability detection and must preserve a portable fallback.
- Optimized paths must not silently substitute a CPU implementation.

### 0.2 Correctness hierarchy

1. mathematical definition;
2. deterministic scalar Rust oracle;
3. portable fused GPU implementation;
4. optimized GPU implementations;
5. backend-specific open-codegen specializations.

Every level is validated against the level above it.

### 0.3 Merge gate — absolute rule

No pull request may be merged until all required CI checks for its head commit are green.

For every PR:

1. create a dedicated branch;
2. implement one coherent milestone or optimization;
3. add or update tests and documentation in the same branch;
4. open the PR;
5. inspect every CI job;
6. if any check fails, fix it on the same branch and rerun CI;
7. merge only when every required check reports success;
8. verify the resulting default-branch head;
9. create the next branch from the verified default branch.

A mergeable GitHub state is not sufficient. `fmt`, Clippy, tests, shader validation, MSRV, and any milestone-specific gate must actually pass.

### 0.4 Performance honesty

No speedup, throughput, bandwidth, latency, memory, or efficiency claim is accepted without a reproducible benchmark.

Every benchmark record must include at least:

- commit SHA;
- device name;
- driver/backend;
- precision;
- batch size;
- heads / KV heads;
- sequence length;
- head dimension;
- causal/non-causal mode;
- warm-up count;
- measured iterations;
- median and percentile latency where practical;
- effective tokens/s or TFLOP/s where meaningful;
- peak allocated/intermediate memory where measurable.

## 1. Target public contract

FLAT-ATTENTION must eventually expose one stable attention contract able to serve training and inference:

- dense MHA;
- GQA;
- MQA;
- causal and non-causal attention;
- prefill;
- decode with KV cache;
- variable sequence lengths;
- masking/bias extensions;
- forward and backward;
- deterministic reference mode;
- portable GPU mode;
- optimized device-specialized mode.

Canonical logical tensor layout:

```text
Q: [batch, q_heads, q_len, head_dim]
K: [batch, kv_heads, kv_len, head_dim]
V: [batch, kv_heads, kv_len, value_dim]
O: [batch, q_heads, q_len, value_dim]
```

The initial milestone uses equal `q_len == kv_len`, equal head counts and `value_dim == head_dim`; later milestones remove those temporary restrictions explicitly.

---

# PHASE A — Mathematical and repository foundation

## M1 — Scalar oracle + first fused portable forward

Status: in progress in PR #1.

### Deliverables

- `AttentionShape` and `FlatAttentionConfig` public contract;
- deterministic Rust online-softmax oracle;
- fused WGSL forward kernel;
- causal and non-causal modes;
- saved log-sum-exp (`LSE`);
- no materialized `N x N` score or probability matrix;
- shader parsing/validation test;
- naive-attention parity tests;
- MSRV 1.89 CI;
- strict rustfmt and Clippy gates.

### Acceptance

- `cargo fmt --all -- --check` green;
- `cargo clippy --all-targets --all-features -- -D warnings` green;
- `cargo test --all-features` green;
- WGSL accepted by the selected Naga validator;
- causal/non-causal reference parity within documented tolerance;
- no allocation proportional to `q_len * kv_len` in the reference/fused contract.

---

# PHASE B — Real portable GPU execution

## M2 — WGPU executor

### Deliverables

- optional `wgpu` feature;
- adapter/device/queue initialization;
- GPU buffer contract for Q/K/V/O/LSE/config;
- bind-group and pipeline creation;
- explicit dispatch geometry;
- upload/download convenience path for validation;
- resident-buffer path for SciRust integration;
- explicit backend-unavailable errors;
- no CPU fallback disguised as GPU execution.

### Acceptance

- pipeline creation succeeds on a real WGPU adapter or lavapipe;
- fused dispatch produces O and LSE;
- malformed dimensions and unsupported head dimensions fail explicitly;
- no intermediate score matrix GPU allocation.

## M3 — Device parity matrix

### Deliverables

- GPU-vs-reference tests for dimensions 1, 8, 16, 32, 64, 80, 96, 128;
- sequence lengths covering tile boundaries: 1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129;
- multiple batches and heads;
- causal/non-causal cases;
- adversarial numerical fixtures with large score ranges;
- finite-value and shape validation.

### Acceptance

- all parity cases within explicit relative/absolute tolerances;
- causal attention has zero future-token contribution within the defined numerical semantics;
- LSE parity validated independently from O.

---

# PHASE C — First performance architecture

## M4 — Multi-query-row tiled kernel

The M1 kernel prioritizes correctness. M4 changes work mapping so one workgroup handles a tile of query rows instead of one query row.

### Deliverables

- Q tile held in workgroup/register storage;
- K/V tile reused across several query rows;
- online-softmax state per query row;
- compile-time kernel variants for key head dimensions;
- tile constants isolated in one policy module.

### Acceptance

- exact API parity with M3;
- no `N x N` matrix;
- benchmark demonstrates reduced K/V global-memory traffic versus M2 for representative prefill cases.

## M5 — Subgroup reductions

### Deliverables

- subgroup-aware dot-product reduction where backend capability exists;
- deterministic fallback reduction;
- capability probe and explicit dispatch selection;
- no reliance on undefined subgroup width assumptions.

### Acceptance

- subgroup and fallback paths both match oracle;
- optimized path selected only when required WGPU feature/capability is present;
- benchmark evidence versus M4.

## M6 — Vectorized memory transactions

### Deliverables

- aligned packed loads/stores for Q/K/V/O where legal;
- scalar tail path;
- alignment-aware layout utilities;
- specialization for common head dimensions 64 and 128.

### Acceptance

- unaligned/misaligned logical lengths remain correct through fallback;
- vector path is benchmarked independently;
- no unsafe host aliasing introduced.

## M7 — Double-buffered K/V staging

### Deliverables

- ping/pong workgroup tiles;
- overlap-friendly load/compute structure expressible in the portable backend;
- explicit barrier discipline;
- static reasoning/documentation of workgroup-memory use.

### Acceptance

- validator passes on all supported backends;
- no workgroup race detected by parity/stress tests;
- measured improvement on at least one real GPU before being selected as default.

---

# PHASE D — Precision and numerical control

## M8 — Mixed-precision input path

### Deliverables

- f16 input/output path where supported;
- FP32 score accumulation and online-softmax state;
- conversion policy isolated from algorithm logic;
- capability-based fallback to f32;
- future bf16 contract defined without pretending unsupported WGSL capability exists.

### Acceptance

- precision-specific parity tolerances documented;
- no NaN/Inf regressions on stress fixtures;
- benchmark memory-bandwidth and latency effects.

## M9 — Numerical policy layer

### Deliverables

- explicit accumulation policy;
- exact/reference mode;
- fast portable mode;
- deterministic reduction mode where feasible;
- stable exponentiation/max-update semantics;
- regression corpus for numerical edge cases.

### Acceptance

- each mode has documented guarantees;
- optimized modes never weaken validation silently;
- deterministic mode reproduces results under repeated identical runs on the same backend/device contract.

---

# PHASE E — Modern transformer attention shapes

## M10 — GQA and MQA

### Deliverables

- independent `q_heads` and `kv_heads`;
- validated head-group mapping;
- MHA as the `q_heads == kv_heads` special case;
- MQA as `kv_heads == 1`;
- no expanded K/V duplication in memory.

### Acceptance

- oracle parity for MHA/GQA/MQA;
- non-divisible head-group relationships rejected explicitly;
- memory benchmark proves no physical K/V replication.

## M11 — Asymmetric Q/KV lengths and cross-attention

### Deliverables

- independent `q_len` and `kv_len`;
- encoder-decoder/cross-attention support;
- causal semantics defined only where meaningful;
- rectangular tiling.

### Acceptance

- parity across rectangular cases;
- no square-matrix assumptions remain in public API or indexing.

## M12 — Variable-length batches

### Deliverables

- per-sequence length metadata;
- padded-batch masking without host-side score construction;
- zero contribution outside valid lengths;
- packed/ragged representation study and documented decision.

### Acceptance

- mixed sequence lengths in one batch match per-sequence oracle calls;
- invalid lengths rejected.

## M13 — Attention bias/mask extensibility

### Deliverables

- causal mask;
- padding/length mask;
- optional additive bias contract;
- ALiBi-compatible bias path if required by SciRust models;
- clean extension point for model-specific biases without forking the kernel architecture.

### Acceptance

- every supported mask/bias has an oracle;
- unsupported combinations fail explicitly.

---

# PHASE F — Inference-first KV architecture

## M14 — Resident KV cache contract

### Deliverables

- append-only K/V cache representation;
- logical capacity and current-length metadata;
- device-resident append path;
- zero host round-trip required per generated token;
- batch/head indexing compatible with GQA/MQA.

### Acceptance

- append/replay deterministic against contiguous reference tensors;
- capacity overflow explicit;
- no K/V cache copy per decode token.

## M15 — Decode kernel (`q_len = 1`)

### Deliverables

- specialized decode attention path;
- streaming over resident KV cache;
- online softmax with no score matrix;
- GQA/MQA-native mapping;
- direct resident output.

### Acceptance

- parity for growing cache lengths;
- latency measured as microseconds/token and tokens/s;
- decode specialization beats generic prefill kernel on representative decode sizes before becoming default.

## M16 — Chunked prefill and paged KV groundwork

### Deliverables

- chunked processing for long contexts;
- cache page/table abstraction independent from vendor libraries;
- stable logical-to-physical block mapping;
- fragmentation/capacity telemetry.

### Acceptance

- chunked result matches contiguous oracle;
- page-boundary stress tests;
- no stale-page reads after reset/reuse.

---

# PHASE G — Training support

## M17 — Backward mathematical oracle

### Deliverables

- scalar Rust dQ/dK/dV oracle;
- use forward O/LSE contract;
- finite-difference gradient tests on small cases;
- causal/non-causal backward semantics.

### Acceptance

- analytic gradients match finite differences within defined tolerance;
- shape/error behavior covered.

## M18 — Fused recomputation backward GPU kernel

### Deliverables

- recompute scores/probabilities from Q/K and saved LSE;
- no stored `N x N` probability matrix;
- dQ/dK/dV accumulation strategy;
- race-free reduction design;
- portable GPU implementation first.

### Acceptance

- GPU gradients match M17;
- memory profile demonstrates no probability-matrix storage;
- forward+backward benchmark captured.

## M19 — Backward tiling and specialization

### Deliverables

- tiled dQ/dK/dV kernels;
- subgroup/vectorized variants;
- mixed-precision support;
- deterministic fallback.

### Acceptance

- performance improvements separately measured;
- no gradient tolerance regression outside documented precision envelope.

---

# PHASE H — FLAT code generation and matrix engines

## M20 — FLAT Kernel IR

Create a small internal representation owned by the project rather than tying algorithm structure to WGSL source text.

### Deliverables

- operations for tile load/store;
- dot/matrix fragments;
- reductions;
- online-softmax state transitions;
- barriers;
- vector types;
- capability requirements;
- deterministic serialization/hash for generated kernels.

### Acceptance

- M4-style kernel can be represented losslessly enough to regenerate a validated portable shader;
- IR validation rejects illegal barrier/layout/capability combinations.

## M21 — Portable WGSL emitter

### Deliverables

- deterministic WGSL code generation from FLAT IR;
- specialization constants baked into generated variants;
- generated-source hashing/cache key;
- validator integration.

### Acceptance

- generated kernel parity equals handwritten kernel;
- generated output is deterministic for identical IR/config.

## M22 — Open cooperative-matrix/subgroup-matrix research gate

### Goal

Exploit exposed matrix hardware only through standards/open IR paths available to the runtime, without making a proprietary vendor programming SDK a core dependency.

### Deliverables

- capability inventory by backend/device;
- SPIR-V/Vulkan cooperative-matrix feasibility study;
- WebGPU/WGSL subgroup-matrix capability tracking;
- prototype emitter only where the full toolchain can be kept within project policy;
- fallback always retained.

### Acceptance

- no capability is claimed without real-device proof;
- no vendor-only SDK becomes mandatory;
- prototype must beat vector/subgroup fallback on target hardware before promotion.

## M23 — Matrix-core mapping and fragment scheduler

### Deliverables

- FLAT IR matrix fragment layout;
- tile decomposition for common D=64/128;
- accumulator ownership model;
- conversion between matrix result fragments and online-softmax reductions;
- architecture-independent scheduler interface.

### Acceptance

- correctness parity first;
- performance qualification separately per backend/device class.

---

# PHASE I — Autotuning

## M24 — Device capability model

### Deliverables

- limits: workgroup size, workgroup storage, subgroup properties, f16 support, binding limits;
- adapter identity and backend;
- stable capability fingerprint;
- no marketing-name heuristics as the sole selection rule.

### Acceptance

- capability model serializes deterministically;
- unsupported configurations filtered before pipeline creation.

## M25 — Deterministic candidate generator

### Deliverables

Candidate dimensions include:

- query tile rows;
- K/V tile rows;
- workgroup size;
- vector width;
- subgroup use;
- f32/f16 storage choices;
- prefill/decode specialization;
- GQA mapping.

### Acceptance

- same capability fingerprint + policy produces same ordered candidate set;
- resource limits respected statically where possible.

## M26 — Benchmark-driven autotuner

### Deliverables

- warm-up and measurement protocol;
- robust median/percentile statistics;
- correctness gate before timing acceptance;
- persistent tuning cache keyed by device/driver/kernel hash/problem class;
- invalidation rules.

### Acceptance

- tuner never selects a candidate that failed parity;
- selected candidate is reproducible from stored evidence;
- cache corruption fails safely.

---

# PHASE J — Benchmark and observability system

## M27 — Benchmark harness

### Deliverables

- prefill sweep;
- decode sweep;
- MHA/GQA/MQA sweep;
- D=32/64/80/96/128;
- short to long context sizes;
- cold/warm pipeline distinction;
- resident vs upload/download timing distinction.

### Metrics

- latency;
- tokens/s;
- effective bandwidth estimate;
- arithmetic work estimate;
- allocations;
- intermediate bytes;
- pipeline/dispatch count;
- optional power/energy hooks when available externally.

## M28 — Baseline comparison

### Baselines

At minimum:

1. scalar Rust oracle;
2. naive/multi-dispatch SciRust WGPU attention;
3. FLAT portable fused path;
4. each optimized FLAT generation.

External competitors may be measured only where environment/licensing permits, and results must be clearly labeled as external baselines rather than dependencies.

### Acceptance

- every performance claim in README/docs is tied to a benchmark artifact and commit;
- regressions are detectable across milestones.

## M29 — Runtime telemetry

### Deliverables

- selected kernel ID;
- tile geometry;
- backend/device fingerprint;
- dispatch count;
- temporary allocation count/bytes;
- fallback reason;
- autotuner cache hit/miss.

### Acceptance

- telemetry adds no mandatory synchronization in the hot path when disabled.

---

# PHASE K — SciRust integration

## M30 — Standalone stable API

### Deliverables

- backend-neutral request/config types;
- owned and borrowed/resident variants where appropriate;
- explicit errors;
- semver policy before first reusable release.

## M31 — SciRust WGPU adapter

### Deliverables

- adapter from SciRust resident GPU tensors to FLAT buffers;
- no unnecessary H2D/D2H copies;
- one fused attention dispatch for supported shapes;
- explicit fallback policy for unsupported configurations;
- integration tests in SciRust.

### Acceptance

- current SciRust multi-dispatch attention path and FLAT produce matching outputs;
- supported FLAT path reduces attention dispatch/intermediate-score storage as designed;
- SciRust CI green before integration merge.

## M32 — SciAgent prefill integration

### Deliverables

- GQA/MHA model wiring as required by current SciAgent architecture;
- resident weights and activations preserved;
- end-to-end generation parity on fixed prompts/seeds where model sampling contract permits.

### Acceptance

- model-level output parity contract documented;
- prefill latency benchmark before/after.

## M33 — SciAgent decode/KV integration

### Deliverables

- resident KV cache adapter;
- q_len=1 decode kernel dispatch;
- device-resident generation compatibility;
- no host round-trip introduced into the token loop.

### Acceptance

- real Thor benchmark;
- tokens/s and per-token latency recorded;
- correctness regression tests for cache reset/replay/EOS paths.

---

# PHASE L — Portability qualification

## M34 — Vulkan/Linux qualification

Targets include software Vulkan (lavapipe) for correctness and real Vulkan GPUs for performance.

### Acceptance

- correctness gate in CI where feasible;
- real-device benchmark reports kept separate from software-adapter runs.

## M35 — Direct3D 12/Windows qualification

### Acceptance

- pipeline creation/parity on supported Windows adapter;
- platform-specific issues documented, not hidden behind fallback.

## M36 — Metal qualification

### Acceptance

- WGPU/Metal parity on available Apple hardware;
- capability limitations recorded explicitly.

## M37 — Vendor diversity qualification

When hardware is available, qualify at least:

- NVIDIA;
- AMD;
- Intel;
- software Vulkan reference path.

The core contract remains independent from any one vendor.

---

# PHASE M — Robustness, safety, and reproducibility

## M38 — Property and stress tests

### Deliverables

- randomized shapes within supported limits;
- causal invariants;
- softmax normalization invariants through oracle checks;
- repeated dispatch stress;
- cache reset/reuse stress;
- extreme-but-finite values.

## M39 — Fuzzing of host API/config parser

### Deliverables

- malformed shapes;
- overflow cases;
- inconsistent lengths;
- invalid tuning-cache data;
- invalid serialized kernel metadata.

No GPU driver fuzzing is claimed unless an appropriate isolated harness exists.

## M40 — Reproducible benchmark manifests

### Deliverables

- machine-readable benchmark schema;
- git SHA;
- environment metadata;
- command line/config;
- result checksum where useful.

---

# PHASE N — Productization and proprietary project hygiene

## M41 — Licensing and ownership metadata

The repository currently declares no license grant. Before external distribution as a product, Memorithm should explicitly set the intended proprietary licensing terms and ownership notices.

### Deliverables

- chosen license/terms file;
- copyright notices;
- third-party dependency/license inventory;
- clear separation between FLAT-owned implementation and external test/benchmark references.

## M42 — Documentation set

### Deliverables

- architecture document;
- algorithm notes;
- public API guide;
- integration guide for SciRust;
- GPU backend guide;
- autotuning guide;
- benchmark methodology;
- troubleshooting;
- contribution/development policy if external contributions are permitted.

## M43 — Release discipline

### Deliverables

- changelog;
- release checklist;
- signed/tagged release policy if desired;
- compatibility matrix;
- benchmark snapshot per significant release.

---

# PHASE O — Continuous optimization loop

After functional completeness, FLAT-ATTENTION enters a permanent measured optimization cycle.

For each optimization PR:

1. state the hypothesized bottleneck;
2. attach the baseline benchmark;
3. change one coherent mechanism;
4. preserve/add correctness tests;
5. run CI until fully green;
6. benchmark the candidate on the target device;
7. retain the change only when the evidence justifies it;
8. record the result in benchmark history;
9. merge only with all CI checks green.

Optimization targets, in order of evidence rather than fashion, may include:

- global-memory traffic;
- workgroup-memory reuse;
- occupancy;
- register pressure;
- subgroup utilization;
- vector load/store efficiency;
- synchronization count;
- work distribution between query/KV tiles;
- pipeline creation/cache overhead;
- decode launch overhead;
- GQA KV reuse;
- long-context tile scheduling;
- open matrix-engine utilization.

No optimization mechanism is permanent merely because it is theoretically faster. The benchmark decides.

---

# Definition of Done for FLAT-ATTENTION 1.0

FLAT-ATTENTION can be considered 1.0-ready only when all of the following are true:

- stable standalone Rust API;
- deterministic scalar forward/backward oracle;
- real fused GPU forward;
- real recomputation-based backward;
- causal/non-causal MHA/GQA/MQA;
- asymmetric Q/KV lengths;
- resident KV cache and specialized decode path;
- no `N x N` probability storage in fused forward/backward architecture;
- mixed-precision path where supported;
- portable WGPU execution;
- autotuned tiled kernels;
- SciRust integration;
- SciAgent prefill/decode integration;
- reproducible benchmark suite;
- measured improvement over SciRust's previous multi-dispatch attention for supported target workloads;
- CI green on every merged PR;
- documented limitations and unsupported cases;
- proprietary licensing/ownership policy finalized by Memorithm.
