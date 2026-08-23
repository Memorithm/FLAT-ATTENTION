# Changelog

All notable FLAT-ATTENTION changes are recorded here. The project does not treat a merged milestone as a released semantic version: a release is created only after the release checklist and exact-head qualification gates are satisfied.

## Unreleased — 1.0 candidate

### Supply chain and governance

- Added a `supply-chain` CI workflow running `cargo-deny` (RustSec advisories, license allow-list, source policy) against every crate manifest on push, pull request, and a weekly schedule.
- Added `deny.toml`, Dependabot updates for Cargo dependencies and GitHub Actions, and a `SECURITY.md` private-vulnerability reporting policy.
- Pinned GitHub Actions by commit SHA in the main CI workflow and added explicit least-privilege token permissions and job timeouts.
- Committed `Cargo.lock` as the authoritative version-resolution record referenced by `THIRD_PARTY_LICENSES.md`; converted the repository into a Cargo workspace (`members = ["crates/*"]`) so one lockfile and one resolution cover every crate.
- Declared `required-features = ["wgpu"]` for GPU examples so default-feature builds and tests no longer fail on host-only machines.
- Removed the stale `.ci-trigger-m24` marker and ignored local assistant scratch directories.

### EPG (Elastic Positional Geometry) crate family

- Added `epg-core`: runtime-neutral EPG contract types with exhaustive validation.
- Added `flat-epg-reference`: deterministic CPU oracle fusing hybrid SO(2)/SO(4) rotations into online-softmax grouped attention.
- Added `flat-epg-wgpu`: correctness-first one-workgroup-per-query-row vec4 GPU qualification pipeline.
- Added `flat-epg-q4-candidate`: opt-in performance candidate reusing FLAT Q4 tiling with EPG fused into K/V staging, plus a hardware-sweep benchmark harness with software-adapter refusal.
- Added physical Jetson Thor qualification evidence and CI smoke protocol for the EPG Q4 candidate.

### Stable contract and integration

- Added the versioned backend-neutral `api::v1` attention contract for borrowed, owned, and resident use.
- Added caller-owned WGPU/WGSL execution surfaces for resident forward, decode, and training integration.
- Added SciRust integration boundaries for stable FLAT ownership and SciAgent resident prefill/decode qualification.

### Attention functionality

- Added dense MHA, native GQA, and MQA without physical K/V head expansion.
- Added causal and non-causal attention, asymmetric Q/KV lengths, variable-length batches, RoPE, ALiBi/additive-bias support, resident and paged KV cache paths, chunked prefill, and specialized decode.
- Added scalar forward/backward oracles and recomputation-based GPU backward with grouped GQA/MQA support.

### Numerical and execution policy

- Added explicit numerical modes, deterministic execution policy, mixed-precision input/output support where qualified, and fixed portable fallbacks.
- Added capability-based kernel selection, subgroup/vectorized/double-buffer candidates, passive runtime telemetry, and deterministic candidate/autotuning contracts.

### Validation and portability

- Added broad scalar, shader-validation, property/stress, hostile-host-input, resident-chain, and WGPU device parity suites.
- Added Linux/Vulkan, Windows/D3D12, and macOS/Metal qualification workflows while preserving one Rust-native WGPU/WGSL implementation.
- Added exact-head physical NVIDIA Jetson Thor Vulkan evidence for the M35 shader family. This is correctness evidence only and is not a performance claim.

### Benchmark and evidence discipline

- Added resident prefill/decode sweeps, cold/warm pipeline accounting, resident-versus-transfer timing, dispatch/allocation accounting, baseline comparisons, and reproducible benchmark manifests.
- Performance remains evidence-gated: no release note may claim a speedup, throughput, latency, bandwidth, memory, or efficiency advantage without a benchmark manifest tied to an exact commit and identified device.

### Project governance

- Added PolyForm Noncommercial 1.0.0 source-available licensing metadata, copyright/ownership documentation, third-party dependency inventory, engineering documentation, and release discipline.

## Release policy

No `1.0.0` tag is authorized by this changelog alone. See `docs/RELEASE_CHECKLIST.md`, `docs/COMPATIBILITY_MATRIX.md`, `docs/RELEASE_POLICY.md`, and `docs/RELEASE_BENCHMARK_SNAPSHOT.md` for the gates that must be closed first.
