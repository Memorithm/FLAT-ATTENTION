# FLAT-ATTENTION roadmap status

This document is the authoritative reconciliation between `ROADMAP.md` milestones and the
actual repository state. It is regenerated whenever an architectural milestone lands or an
acceptance boundary changes. It deliberately separates four dimensions that must never be
conflated:

- **Implementation** — the code exists on `main`.
- **Correctness qualification** — tests/oracle parity demonstrate behavior.
- **Physical performance qualification** — reproducible real-device benchmark evidence exists.
- **Release qualification** — evidence is attached to the exact commit proposed for release.

Statuses are restricted to: `DONE`, `PARTIAL`, `MISSING`,
`IMPLEMENTED_BUT_UNQUALIFIED`, `BLOCKED_BY_AVAILABLE_HARDWARE`,
`BLOCKED_BY_PLATFORM_CAPABILITY`, `OBSOLETE_OR_SUPERSEDED`.

Baseline for this revision: `main` at commit `8cd1b257a599882b7a7bbc095465883ec38274ed`
(2026-08-26). Every status below cites its evidence; "code exists" alone never implies DONE.

## Phase A/B — foundation and portable execution (M1–M3)

| Milestone | Implementation | Correctness | Evidence |
|---|---|---|---|
| M1 scalar oracle + fused forward | DONE | DONE | `src/lib.rs` (`forward_reference`), `shaders/flat_fwd.wgsl`, `tests/parity.rs`, WGSL/Naga gate in CI `rust` job |
| M2 WGPU executor | DONE | DONE (software Vulkan CI) | `src/wgpu_backend.rs`, `tests/wgpu_device.rs`, CI `wgpu-device` job with `FLAT_REQUIRE_WGPU=1` |
| M3 device parity matrix | DONE | DONE | `tests/wgpu_device.rs` dimension/sequence matrix, adversarial fixtures in `tests/m20_adversarial_numerics.rs` |

## Phase C — first performance architecture (M4–M7)

| Milestone | Implementation | Correctness | Physical performance | Evidence |
|---|---|---|---|---|
| M4 Q4 tiled kernel | DONE | DONE | Historical Thor qualification recorded in `docs/M28_KERNEL_GENERATIONS.md` lineage | `shaders/flat_fwd.wgsl`, `WGSL_QUERY_ROWS` policy in `src/lib.rs` |
| M5 subgroup reductions | DONE | DONE (capability-gated device suite) | Measured per `docs/M28_KERNEL_GENERATIONS.md`; subgroup stays Auto-gated | `shaders/flat_fwd_subgroup.wgsl`, `WgpuSubgroupPolicy`, `tests/wgpu_subgroup.rs` |
| M6 vec4 memory path | DONE | DONE | Same-device regression gate keeps opt-in generations bounded vs baseline (`.github/workflows/ci.yml`) | `shaders/flat_fwd_vec4.wgsl`, `tests/wgpu_vectorized.rs`, `examples/regression_gate.rs` |
| M7 double buffering | DONE (opt-in, non-default) | DONE | IMPLEMENTED_BUT_UNQUALIFIED as a default: retained experimental until physical evidence justifies promotion | `shaders/flat_fwd_double_buffer.wgsl`, `tests/wgpu_double_buffer.rs`, explicit non-default in `src/wgpu_backend.rs::new` |

## Phase D — precision and numerical control (M8–M9)

| Milestone | Status | Evidence |
|---|---|---|
| M8 mixed-precision f16 I/O | DONE (packed-f16 storage with FP32 accumulation; native shader-f16 unused) | `src/f16.rs`, `src/wgpu_f16_backend.rs`, `shaders/flat_fwd_f16.wgsl`, `tests/wgpu_f16.rs`; bf16 remains explicitly undefined per roadmap |
| M9 numerical policy layer | DONE | `src/numerical.rs`, `tests/numerical_policy.rs`, `tests/wgpu_numerical_policy.rs`, regression corpus in stress/fuzz suites |

## Phase E — modern shapes (M10–M13)

| Milestone | Status | Evidence |
|---|---|---|
| M10 GQA/MQA without K/V expansion | DONE | `src/grouped.rs`, `tests/grouped.rs`, `tests/wgpu_grouped.rs`; non-divisible grouping rejected explicitly |
| M11 asymmetric Q/KV lengths + cross-attention | DONE | `src/asymmetric_grouped.rs`, `docs/m11-*.md`, `tests/m11_rectangular_wgpu.rs` |
| M12 variable-length batches | DONE | `src/wgpu_external_variable.rs`, `docs/m12-variable-length-wgpu.md` |
| M13 bias/ALiBi extensibility | DONE | `src/attention_bias.rs`, `tests/m13_*.rs` |

## Phase F — inference KV architecture (M14–M16)

| Milestone | Status | Evidence |
|---|---|---|
| M14 resident KV cache contract | DONE | `src/wgpu_kv_cache.rs`, `tests/m14_resident_kv_cache.rs` |
| M15 specialized decode (`q_len = 1`) | DONE; decode-beats-prefill default promotion evidenced by sweep history | `src/wgpu_decode.rs`, `examples/m15_decode_bench.rs`, `docs/m15-*.md` |
| M16 chunked prefill + paged KV | DONE | `src/chunked_projection_prefill.rs`, `src/paged_kv.rs`, `src/wgpu_paged_decode.rs`, `tests/m16_*.rs` |

## Phase G — training support (M17–M19)

| Milestone | Status | Evidence |
|---|---|---|
| M17 backward oracle + finite differences | DONE | `src/backward.rs`, gradient checks in `tests/` |
| M18 recomputation backward GPU | DONE | `src/wgpu_backward.rs`, `tests/m18_wgpu_backward*.rs`, no probability-matrix storage by construction |
| M19 backward tiling/specialization | DONE for grouped recompute family; further specialization tracked as Phase O candidates | `src/backward_grouped.rs`, `src/wgpu_backward_grouped.rs`, `tests/m19_*`, `tests/m20_grouped_backward_stress.rs` |

## Phase H — code generation and matrix engines (M20–M23)

| Milestone | Status | Missing elements / notes |
|---|---|---|
| M20 FLAT Kernel IR | **MISSING** | No internal kernel IR exists on `main`; kernel structure is tied to handwritten WGSL text plus runtime enums (`RuntimeKernelId`, `WgpuKernelVariant`). Required: typed IR, validation, deterministic normalization/versioning/fingerprint, capability requirements, faithful representation of a qualified forward architecture. |
| M21 portable WGSL emitter | **MISSING** | No IR-to-WGSL generator exists. Required: deterministic emission, source hashing/cache key, Naga validation, generated-vs-handwritten parity. |
| M22 open cooperative/subgroup-matrix research gate | **MISSING** (gate not yet executed) | No capability inventory or feasibility record exists in `docs/`. wgpu 30 exposes no cooperative/subgroup-matrix feature flag (verified against `wgpu-types 30.0.x`); the gate must still be documented with specification sources before any roadmap status can change. |
| M23 matrix fragment scheduler | **BLOCKED_BY_PLATFORM_CAPABILITY** (pending M22 outcome) | No executable open matrix path is currently exposed by the runtime; scheduler work has no backend semantics to target. |

## Phase I — autotuning (M24–M26)

| Milestone | Status | Evidence / missing elements |
|---|---|---|
| M24 device capability model | PARTIAL | Done: `RuntimeDeviceCapabilities` + `RuntimeDeviceFingerprint` with deterministic canonical records and FNV-1a-64 fingerprints (`src/runtime_telemetry.rs`, `docs/M56_*`, `docs/M57_*`). Missing: static candidate-resource prefilter wired **before pipeline creation**, as recorded in `docs/M57_DEVICE_CAPABILITY_LIMITS.md`. |
| M25 deterministic candidate generator | **MISSING** | No candidate generation module exists; variant selection today is fixed policy inside `src/wgpu_backend.rs` (`kernel_variant_for_head_dim`), not generated/ordered candidates. |
| M26 benchmark-driven autotuner | PARTIAL (see split below) | Tuner core MISSING: no correctness-gated measurement/ranking loop exists in-repo. Persistent tuning-cache ownership was deliberately assigned to the SciRust ElasticAutoTuner integration (`docs/FLAT_ATTENTION_GUIDE.md` §6): that portion of M26 is OBSOLETE_OR_SUPERSEDED for this repository, provided FLAT exposes the deterministic problem/candidate/capability/evidence surfaces the integration consumes. |

## Phase J — benchmarks and observability (M27–M29)

| Milestone | Status | Evidence |
|---|---|---|
| M27 benchmark harness | DONE | `docs/M27_BENCHMARK_HARNESS.md`, `examples/m27_*.rs` sweeps with resident/transfer and cold/warm distinctions |
| M28 baseline comparison | DONE | `docs/M28_BASELINE_COMPARISON.md`, `examples/m28_scalar_flat_baseline.rs`, `examples/m28_kernel_generations.rs` |
| M29 runtime telemetry | DONE (autotuner-cache fields present, tuner itself pending M25/M26) | `src/runtime_telemetry.rs`, `docs/M29_RUNTIME_TELEMETRY.md`, `tests/m29_runtime_telemetry.rs`; passive/no-sync contract preserved |

## Phase K — SciRust integration (M30–M33)

| Milestone | Status | Notes |
|---|---|---|
| M30 standalone stable API | DONE | `src/api.rs` versioned `api::v1` with semver workflow (`.github/workflows/semver.yml`, `docs/API_SEMVER.md`) |
| M31 SciRust WGPU adapter | DONE (qualified in SciRust's repo) | `docs/COMPATIBILITY_MATRIX.md` integration section; exact-head pin verification against the intended release commit remains a release-checklist item, not an implementation gap |
| M32 SciAgent prefill integration | PARTIAL | Resident prefill qualified in SciRust; prefill-latency before/after benchmark record is owned by the SciRust side and not re-attested here |
| M33 SciAgent decode/KV integration | PARTIAL | Decode/KV lifecycle gate on the intended FLAT release revision is explicitly still open per `docs/COMPATIBILITY_MATRIX.md`; FLAT-side decode surfaces are implemented and qualified |

## Phase L — portability (M34–M37)

| Milestone | Status | Evidence |
|---|---|---|
| M34 Vulkan/Linux | DONE | `.github/workflows/portability-vulkan.yml`, `docs/M34_VULKAN_LINUX.md`; physical Thor correctness via SciRust-hosted exact-head workflow |
| M35 D3D12/Windows | DONE | `.github/workflows/portability-d3d12.yml`, `docs/M35_D3D12_WINDOWS.md`; WARP software reference recorded honestly |
| M36 Metal/macOS | DONE | `.github/workflows/portability-metal.yml`, `docs/M36_METAL.md` |
| M37 vendor diversity | PARTIAL — NVIDIA physical done; AMD/Intel BLOCKED_BY_AVAILABLE_HARDWARE | `docs/RELEASE_CHECKLIST.md` records that unavailable AMD/Intel devices strengthen but do not hard-block 1.0; `docs/COMPATIBILITY_MATRIX.md` rows kept aligned |

## Phase M — robustness/reproducibility (M38–M40)

| Milestone | Status | Evidence |
|---|---|---|
| M38 property/stress | DONE | `tests/m38_property_stress.rs`, `docs/M38_PROPERTY_STRESS.md` |
| M39 host fuzzing | DONE | `fuzz/` targets (shape/oracle, paged-KV state machine, api::v1), weekly deep session + per-PR sessions, CodeQL |
| M40 benchmark manifests | DONE | `src/benchmark_manifest.rs`, `docs/M40_BENCHMARK_MANIFESTS.md` |

## Phase N — productization (M41–M43)

| Milestone | Status | Evidence |
|---|---|---|
| M41 licensing/ownership metadata | DONE (final license election remains a Memorithm business decision outside engineering) | `LICENSE*`, `LICENSING.md`, `THIRD_PARTY_LICENSES.md`, PolyForm metadata |
| M42 documentation set | DONE for current architecture; new architectural layers must extend it | `docs/FLAT_ATTENTION_GUIDE.md` plus milestone evidence documents |
| M43 release discipline | DONE artifacts; tagging intentionally withheld pending checklist closure | `CHANGELOG.md`, `docs/RELEASE_CHECKLIST.md`, `docs/RELEASE_POLICY.md`, `docs/RELEASE_BENCHMARK_SNAPSHOT.md` |

## Phase O — continuous optimization loop

Active and evidence-driven (M44 grouped vec4 through M60 Q1 direct-load; see
`docs/M44`–`M60` records and merged PR history). The loop is not a gap; it is the
standing process.

## 1.0 Definition-of-Done reconciliation

| Definition-of-Done item | Status | Evidence / blocker |
|---|---|---|
| Stable standalone Rust API | DONE | `api::v1` + semver gate |
| Deterministic forward/backward oracles | DONE | `forward_reference`, `backward_reference(_grouped)` |
| Real fused GPU forward / recomputation backward | DONE | wgpu backends + parity suites |
| Causal/non-causal MHA/GQA/MQA; asymmetric lengths | DONE | parity matrices above |
| Resident KV cache + specialized decode | DONE | M14/M15 surfaces |
| No N×N probability storage | DONE | streaming/recompute construction, allocation accounting examples |
| Mixed-precision path where supported | DONE | packed-f16 family |
| Portable WGPU execution | DONE | three-platform portability gates |
| Autotuned tiled kernels | **MISSING** | requires M20/M21/M25/M26 core |
| SciRust integration | DONE (exact-head pin check outstanding) | release checklist item |
| SciAgent prefill/decode integration | PARTIAL | decode/KV lifecycle gate on release revision outstanding (external) |
| Reproducible benchmark suite | DONE | M27/M40 harnesses |
| Measured improvement over SciRust multi-dispatch attention | PENDING exact-head manifest | `docs/RELEASE_BENCHMARK_SNAPSHOT.md` requires the accepted comparison to be cited for the release SHA |
| Documented limitations | DONE | guide + compatibility matrix |
| Licensing finalized | DONE (engineering side) | Memorithm licensing decision recorded in repo metadata |

## Current release blockers (engineering view)

1. M20 Kernel IR + M21 WGSL emitter + M25 candidate generation + M26 autotuner core
   ("autotuned tiled kernels" DoD line) — being addressed by the kernel-architecture PR
   series.
2. Exact-candidate physical/performance manifests cited from
   `docs/RELEASE_BENCHMARK_SNAPSHOT.md`.
3. SciRust-side exact-pin and SciAgent decode/KV lifecycle gates (external repository).
