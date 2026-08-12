# FLAT-ATTENTION engineering guide

This guide is the maintained entry point for FLAT's architecture, algorithms, public API, SciRust integration, GPU backends, autotuning, benchmark methodology, troubleshooting and development policy. Milestone-specific evidence remains in the adjacent `M*.md` documents and `ROADMAP.md` remains the acceptance-plan authority.

## 1. Architecture

FLAT is a Rust-native fused attention engine. The correctness hierarchy is deliberately one-way:

1. deterministic scalar Rust mathematical oracle;
2. portable fused WGPU/WGSL implementation;
3. optimized portable variants;
4. capability-selected/tuned variants;
5. SciRust/SciAgent adapters.

Higher layers are always checked against a lower-level correctness source. GPU execution never disguises a CPU fallback as an optimized path.

The canonical logical tensors are `Q=[B,Hq,Nq,D]`, `K/V=[B,Hkv,Nkv,D]`, and `O=[B,Hq,Nq,D]`. Public resident integrations may use sequence-major physical storage, but their shape contracts explicitly describe the mapping. MHA has `Hq=Hkv`; GQA/MQA map multiple query heads to one KV head.

Forward uses online softmax and does not materialize an `Nq × Nkv` probability matrix. Training backward recomputes the required score/softmax state instead of retaining the full probability matrix. Decode uses a resident KV cache and a specialized `query_len=1` route.

## 2. Algorithm and numerical model

For each visible query/key pair, the scalar contract computes a scaled dot product, applies the configured bias/mask, and updates online-softmax state `(m,l,O)` without storing the complete score matrix. Causal visibility is defined from logical/absolute query and key positions, which allows asymmetric prefill/decode lengths and nonzero position origins.

RoPE rotates query/key pairs with explicit position offsets. The resident decode interoperability path can consume K that was already rotated when appended to the cache while still rotating Q at its current absolute position; double rotation is rejected by contract rather than hidden.

Accumulation-sensitive paths keep the numerical policy explicit. F32 is the portable reference/storage path. Packed f16 is used only where the WGPU adapter exposes the required capability and is always qualified against the higher-precision oracle with precision-specific tolerance. Unsupported precision capabilities select an explicit qualified GPU fallback, never an unreported host implementation.

Deterministic/reference mode prioritizes reproducibility over throughput. Optimized paths may change reduction order and therefore use documented tolerance rather than pretending to be bit-identical.

## 3. Public API

The backend-neutral reusable contract is versioned under `flat_attention::api::v1`; see `API_SEMVER.md` for compatibility rules.

Use backend-neutral request/config types when a caller wants validation independent of a particular GPU API. Use borrowed requests for host-owned slices, owned requests where the request must own host data, and resident request contracts when an integration owns backend buffers externally.

With feature `wgpu`, the crate exposes qualified executors/pipelines for grouped forward, asymmetric projection/RoPE attention, decode/KV use, and training forward/backward. Caller-owned APIs accept an existing `wgpu::Device`, `Queue`, buffers and/or command encoder so SciRust can preserve a single ownership/synchronization domain.

Errors are explicit. Invalid dimensions, invalid head grouping, unsupported head dimensions/capabilities, undersized buffers and arithmetic overflow must fail before unsafe dispatch. The stable host contract is fuzzed by M39.

## 4. SciRust and SciAgent integration

SciRust enables FLAT only through explicit features. It remains the owner of WGPU device/queue/tensor storage. The stable adapter validates through `api::v1` before entering the qualified resident bridge; it does not silently select legacy multi-dispatch attention.

Training integration keeps Q/K/V/dO and resulting O/LSE/dQ/dK/dV resident across the attention boundary and can record forward→backward work into a caller-owned encoder. Device-to-device packing/copies used by the backward contract do not imply a host round-trip.

SciAgent prompt prefill routes grouped GQA/MQA through FLAT when the opt-in feature is enabled. Per-layer KV caches retain RoPE-rotated K and raw V in resident WGPU allocations. Incremental decode uses `query_len=1`, consumes the active K/V cache and introduces no host K/V round-trip; final vocabulary logits retain SciAgent's existing host-visible sampling boundary.

SciRust remains responsible for higher-level fallback policy, model sampling semantics, checkpoint/model ownership and ElasticAutoTuner persisted-plan storage.

## 5. GPU backends and portability

FLAT writes WGSL and uses WGPU as the portable GPU boundary. The core architecture does not require CUDA C++, `nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or a vendor SDK.

Linux/Vulkan is continuously qualified against Mesa lavapipe for software-adapter correctness. Exact-candidate physical NVIDIA Vulkan parity has also been demonstrated on Jetson Thor. Metal is qualified through WGPU/Metal on hosted Apple hardware. Direct3D 12 qualification is tracked independently; platform/runtime limitations remain explicit rather than being hidden by another backend.

A backend qualification result means shader translation, pipeline creation, resource binding, dispatch and numerical parity succeeded for that environment. It is not automatically a performance statement.

## 6. Kernel policy and autotuning

Kernel selection is evidence-driven. Candidate dimensions include query/KV tile geometry, workgroup size, vector width, subgroup use where supported, storage precision, prefill/decode specialization and GQA mapping.

A capability fingerprint filters impossible candidates before pipeline creation. Candidate ordering must be deterministic for the same capability/problem policy. A candidate that fails correctness is never eligible for timing-based selection.

The SciRust ElasticAutoTuner integration owns persisted runtime plans/cache policy. FLAT exposes the portable kernel/capability surfaces and passive telemetry needed to explain a selection. Cache corruption must fail safely in the layer that owns serialization; FLAT does not duplicate SciRust's persisted format.

Runtime telemetry may report selected kernel ID, tile geometry, backend/device fingerprint, dispatch/allocation counts, fallback reason and cache hit/miss state. Disabled telemetry must not add mandatory synchronization to the hot path.

## 7. Benchmark methodology

Performance evidence follows four rules: identical work, correctness before timing, explicit synchronization boundary, and exact provenance.

A promoted benchmark records exact commit SHA, device, driver/backend, precision, batch/head geometry, sequence lengths, causal mode, warmups, measured iterations, median/p95 latency where practical, throughput and memory/dispatch accounting where meaningful. M40 provides a deterministic machine-readable manifest schema and result integrity checksum.

Resident timing and transfer-inclusive timing are distinct experiments. Pipeline cold/warm lifecycle is also reported separately. Uploads, readbacks and pipeline compilation may be excluded only when the benchmark says so explicitly.

Software Vulkan and hosted CI timing are qualification evidence, not a universal speedup claim. A README/product performance statement requires a reproducible real-device artifact for the exact commit and workload class.

## 8. Troubleshooting

**No WGPU adapter.** Confirm the requested backend is installed and visible to the process. In qualification jobs, `FLAT_REQUIRE_WGPU=1` intentionally turns absence into failure.

**Pipeline creation fails.** Capture backend, adapter/driver identity, failing shader/pipeline label and exact commit. Do not work around a backend validation error by silently choosing CPU execution.

**Parity failure.** Reduce to the smallest shape while preserving batch, head mapping, position offsets, mask/bias and precision. Compare O and LSE independently against the scalar oracle. For decode, verify whether cached K is raw or already RoPE-rotated.

**Unexpected decode output.** Verify active KV logical length, query absolute position, cache reset/replay semantics and EOS handling. A fixed-capacity cache may be reused after logical reset without reallocating storage.

**Benchmark regression.** First confirm the same device/driver, commit, geometry, warmup/repeat counts and timing boundary. Never infer a kernel regression from software-adapter timing or from a machine carrying unrelated GPU workload.

**Windows/D3D12 hosted failure.** Distinguish adapter/device creation failures in WGPU/platform runtime from FLAT shader/pipeline failures. A gate that never reaches FLAT pipeline creation cannot be presented as FLAT numerical evidence.

## 9. Development and contribution policy

Every engineering change uses a dedicated branch and one coherent PR. Rustfmt, Clippy `-D warnings`, tests, shader/device validation and milestone-specific gates must all succeed on the exact PR head before merge. `mergeable=true` alone is never sufficient.

Performance PRs state the hypothesized bottleneck before changing code, preserve correctness gates and retain the optimization only when target-device evidence justifies it. Negative benchmark results are valid engineering results and must not be rewritten as wins.

Host implementation stays Rust-native. Project-authored C/C++ and C ABI bridges are outside the architecture. Hardware-specific acceleration, if ever added, must be capability-gated and retain the portable path.

The repository is source-available under PolyForm Noncommercial 1.0.0 with separate written commercial licensing. External source contributions are not accepted until the copyright holder explicitly establishes contribution-rights/provenance terms; discussion and bug reports remain welcome without transferring source rights.

## 10. Evidence map

- `ROADMAP.md` — milestone gates and 1.0 definition of done.
- `API_SEMVER.md` — stable API/versioning policy.
- `M27_BENCHMARK_HARNESS.md`, `M28_*` — measurement and baseline generations.
- `M29_RUNTIME_TELEMETRY.md` — observability contract.
- `M34_VULKAN_LINUX.md`, `M36_METAL.md` — backend qualification evidence boundaries.
- `M38_PROPERTY_STRESS.md`, `M39_HOST_API_FUZZING.md` — robustness/safety gates.
- `M40_BENCHMARK_MANIFESTS.md` — reproducible machine-readable benchmark provenance.
- `LICENSE.md`, `LICENSING.md`, `THIRD_PARTY_LICENSES.md` — ownership/licensing boundary.
