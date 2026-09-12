# Boolean KV Cache Roadmap

Status: priority research track extending `ROADMAP.md` M13B and the existing resident/paged KV work.

This document defines the first-class Boolean KV Cache design for FLAT-ATTENTION. It does not replace the qualified numerical KV cache at the start of the programme. It introduces a new memory tier whose purpose is to make extremely cheap, highly parallel decisions about which numerical KV state must be touched at all.

## 1. Architectural decision

The Boolean KV Cache is a first-class runtime object, not a temporary mask and not a debug index.

The initial hierarchy is:

```text
query state
  -> Boolean query signature
  -> Boolean KV Cache
       -> candidate pages / blocks
       -> exact numerical KV only for accepted candidates
            -> FLAT attention
```

Three modes must remain distinguishable in all APIs and experiments:

1. **NKV — Numerical KV baseline**: current/reference numerical K/V cache, no Boolean tier.
2. **BIKV — Boolean-indexed Numerical KV**: Boolean state is authoritative only for selection/routing; accepted pages are evaluated using exact numerical K/V.
3. **NBKV — Native Boolean KV**: operations whose semantics are explicitly Boolean may remain entirely in the Boolean domain without materializing or reading numerical K/V. This mode is research-only until independently qualified.

The roadmap must never silently treat BIKV as NBKV or claim that Boolean state reconstructs numerical state unless an explicit codec/reconstruction experiment proves it.

## 2. First-class data model

The runtime design should converge on explicit types equivalent to:

```text
BooleanKvSignature
  bit_width
  packed_words[]

BooleanKvPageMeta
  page_id
  token_range
  valid_count
  key_signature
  optional_value_signature
  visibility_bits
  retention_bits
  routing_bits
  age / frequency / position metadata where preregistered

BooleanKvIndex
  logical_pages
  physical_layout
  backend
  generation
  reset_epoch
```

The representation must support exact storage-bit accounting. Tail bits, alignment, page boundaries and endianness must be specified. Boolean metadata must have an explicit lifecycle coupled to numerical KV append/reset/reuse.

## 3. Core invariants

- Numerical K/V remains the correctness authority for BIKV.
- Boolean routing is allowed to skip work only under a declared policy.
- All-accept Boolean routing must reduce to the qualified dense/paged numerical path.
- Reset/reuse must invalidate stale Boolean metadata exactly when the corresponding numerical page becomes stale.
- A rejected page must not be staged/read by the qualified consumer when the architecture claims a memory-traffic saving.
- No speedup claim is valid unless Boolean signature generation, index traversal, synchronization, transfer and surviving numerical work are all included.
- The dense numerical fallback remains available whenever Boolean routing is unsupported, too dense, too inaccurate or slower.

## 4. Backend model

Boolean KV is heterogeneous by design. CPU is not a fallback; it is a first-class execution backend.

Required backend classes:

```text
scalar Rust oracle
  -> packed u64 CPU baseline
  -> CPU SIMD when available
  -> CPU multicore
  -> CPU dual-socket / NUMA
  -> portable GPU/WGPU
  -> optional hardware-specific GPU bit-matrix qualification
  -> cooperative CPU + GPU execution
```

Runtime capability discovery, not product-name assumptions, selects legal kernels.

### 4.1 CPU backend

The CPU path must benchmark:

- scalar reference;
- packed `u64` bit operations;
- XOR / XNOR / AND / OR reductions;
- population count;
- SIMD paths exposed safely by the target ISA;
- multicore partitioning by page, query, head and tile;
- cache-aware blocking;
- NUMA-aware placement.

The CPU backend is especially important for small first-token decisions, irregular sparse routing and Boolean indexes that are too large or too expensive to keep in GPU memory.

### 4.2 Dual-socket / NUMA backend

The first dedicated large-memory target is the user-reported Dell T430-class server profile with approximately 125 GiB RAM and two sockets, reported as 32 cores per CPU. The exact CPU SKU, physical/logical core count, NUMA topology, cache hierarchy and ISA support must be discovered at runtime and recorded in every evidence pack rather than hard-coded.

The harness must capture at least:

```text
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
numactl --hardware
/proc/cpuinfo feature summary
```

NUMA experiments must compare:

- local allocation/local execution;
- remote memory access;
- interleaved allocation;
- replicated read-mostly Boolean metadata;
- page sharding by socket;
- query-signature replication to both sockets;
- merge cost of per-socket candidate sets.

The preferred hypothesis is to move tiny query signatures rather than large KV state.

### 4.3 GPU backend

The portable GPU path must use bitpacked storage and open shader/IR mechanisms first. Candidate primitives include bitwise logic, shifts and population count where exposed by the backend. Hardware-specific bit-matrix acceleration may be qualified separately without becoming a mandatory dependency of portable FLAT.

### 4.4 Cooperative CPU + GPU backend

The principal heterogeneous experiment is:

```text
CPU Boolean plane:  prepare page/block admission for current/next work
GPU numerical plane: execute exact FLAT on previously admitted pages
```

Qualification requires trace/timing evidence of overlap. Source-code concurrency alone is not evidence.

## 5. Lifecycle

### Prefill

For every numerical KV append, generate or update Boolean metadata under a frozen signature policy:

```text
K_t / V_t
  -> numerical KV append
  -> Boolean signature / page aggregate
  -> Boolean KV append
```

Boolean metadata must be ready before first-token decode qualification.

### Decode

For each new query:

```text
Q_t
  -> Boolean query signature
  -> Boolean KV search / matrix equation
  -> candidate pages
  -> exact FLAT over accepted numerical K/V
  -> output
```

The first generated token after prefill is an explicit acceptance target; the system must not require several generated tokens before Boolean routing becomes usable.

### Reset / reuse

Boolean and numerical generations must advance atomically at the logical contract level. Reused physical pages must never expose stale Boolean signatures.

## 6. Boolean Matrix Equation interface

Boolean KV search should not be tied permanently to one XNOR-popcount formula. The research interface should support a Boolean Matrix Equation (BME):

```text
C_ij = R_k( Phi(A_ik, B_kj) )
```

Baselines include:

- OR-AND Boolean matrix product;
- XOR-AND / GF(2) matrix product;
- XNOR + popcount similarity;
- thresholded popcount;
- preregistered Boolean functions / reductions exported from BooleanLab.

Arbitrary searched equations remain behind research gates until the baseline families are qualified.

## 7. Milestones

### BKV-0 — Contract + scalar oracle

Deliver:

- exact packed representation contract;
- scalar Boolean KV oracle;
- deterministic append/reset/reuse semantics;
- all-accept / all-reject / structured fixtures;
- exact storage-bit accounting.

Acceptance:

- all-accept BIKV matches numerical KV reference;
- stale-page tests fail closed;
- malformed geometry is rejected.

### BKV-1 — Bitpacked page index

Deliver:

- per-token and per-page signatures;
- page aggregation policy;
- packed index layout;
- deterministic search primitives;
- candidate-set telemetry.

Acceptance:

- exact packing tests;
- stable results across repeated runs;
- metadata bytes/page measured explicitly.

### BKV-2 — CPU packed + multicore backend

Deliver:

- packed `u64` baseline;
- thread-parallel page search;
- cache-aware tile sizes;
- capability detection;
- scaling harness from 1 core to available physical cores.

Measure:

- queries/s;
- pages/s;
- bits compared/s;
- cycles per page/candidate where measurable;
- memory bandwidth;
- scaling efficiency.

### BKV-3 — Dual-socket NUMA qualification

Deliver:

- topology capture;
- page sharding by NUMA node;
- local vs remote access experiments;
- query-signature replication;
- candidate-set merge.

Acceptance:

- NUMA placement is explicit and reproducible;
- local/remote traffic is measured where tooling permits;
- best strategy is selected by evidence, not assumed.

### BKV-4 — Portable GPU Boolean search

Deliver:

- bitpacked GPU index;
- portable shader search;
- page/block admission buffer consumed before numerical staging where possible;
- dense fallback.

Acceptance:

- candidate sets match CPU/scalar oracle;
- full Boolean overhead included in timings;
- no claim based solely on operation counts.

### BKV-5 — CPU/GPU cooperative pipeline

Deliver:

- explicit queues/generations for Boolean decisions;
- overlap instrumentation;
- first-token and steady-state modes;
- backpressure/fallback policy.

Acceptance:

- no stale decision can be consumed;
- overlap is demonstrated by traces;
- latency includes transfer/synchronization costs.

### BKV-6 — Native Boolean KV research gate

Only after BIKV is qualified, test workloads that can consume Boolean K/V semantics directly.

Deliver:

- explicit native-Boolean operation contract;
- no implicit claim of equivalence to numerical KV;
- downstream-quality evaluation;
- comparison with BIKV and NKV.

### BKV-7 — Adaptive tiering

After stable evidence exists, expose placement/routing policy to ElasticXxx:

```text
OBSERVE -> FORECAST -> PLAN -> VALIDATE -> ACT -> VERIFY -> COMMIT/ROLLBACK
```

Candidate decisions include CPU/GPU ownership, NUMA shard, page promotion/demotion, signature width and dense fallback.

## 8. Required experimental matrix

Every serious candidate must be compared against:

1. numerical KV baseline;
2. paged numerical KV baseline;
3. random Boolean mask at matched density;
4. simple structural/positional mask;
5. BIKV CPU;
6. BIKV GPU where available;
7. BIKV CPU+GPU cooperative;
8. NBKV only when semantically applicable.

Sweep at least:

- context length;
- page size;
- signature width;
- retained page density;
- batch/concurrency;
- MHA/GQA/MQA where supported;
- prefill vs first-token decode vs steady-state decode;
- one socket vs both sockets on NUMA hardware.

## 9. Required metrics

Correctness / quality:

- candidate recall against declared dense-attention target;
- false-negative rate;
- output/LSE error for BIKV;
- downstream task quality where available;
- reset/reuse correctness.

Memory / traffic:

- Boolean bits per token/page;
- total Boolean index bytes;
- numerical KV bytes touched;
- numerical KV bytes avoided;
- `bytes_numerical_KV_avoided / bytes_Boolean_KV_read`;
- host/GPU transfer bytes;
- local vs remote NUMA traffic where measurable.

Performance:

- Boolean search latency;
- first-token latency;
- TPOT / inter-token latency;
- tokens/s;
- pages/s;
- bits compared/s;
- effective memory bandwidth;
- synchronization / dispatch count;
- CPU scaling efficiency.

Economics / energy only when the inputs are known:

- joules/query or joules/token when measurable;
- throughput per known hardware cost;
- throughput per watt.

No price/performance claim may use guessed acquisition prices.

## 10. Ownership across Memorithm

- **KVLab**: scientific source of truth for Boolean KV experiments, manifests, baselines, statistics and negative results.
- **BooleanLab**: Boolean functions, BME search/equivalence, Boolean signatures and routing predicates.
- **FLAT-ATTENTION**: runtime contract, page admission, numerical attention consumer, GPU/CPU cooperative scheduling and end-to-end qualification.
- **SciRust**: reusable packed-bit, SIMD-safe, matrix/statistical primitives after stabilization.
- **NNIS**: optional native NVIDIA qualification and low-level hardware experiments.
- **ElasticXxx**: adaptive placement/routing only after evidence is stable.

## 11. Promotion rule

Boolean KV becomes a default optimization for a workload only if all of the following hold on the exact tested commit/device/configuration:

```text
quality gate passes
AND correctness gate passes
AND total latency improves or another declared resource objective improves
AND Boolean metadata cost is included
AND fallback remains available
```

A Boolean KV result that only reduces arithmetic but increases end-to-end latency is a valid negative result, not a successful optimization.
