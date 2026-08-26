# M20/M21/M25 — Kernel IR foundation, deterministic WGSL emitter subset, candidate generation

Status: **implemented and hardware-qualified** on the portable generated path.

Base: verified `main` commit `8cd1b25` (`Merge pull request #113 from Memorithm/perf/m60-q1-direct-kv`).

## Scope of this slice

Three Phase-H/Phase-I foundations land together because they form one vertical:

1. **FLAT Kernel IR** (`kernel_ir`, M20) — a deliberately small,
   attention-oriented representation of the qualified Q4 fused-forward
   architecture. It separates *semantic attention configuration*
   ([`AttentionProblem`]: folded batch-heads, sequence length, head dim,
   causal) from the *device execution plan* ([`ExecutionPlan`]: tile config,
   workgroup geometry, reduction strategy, precision policy, explicit op
   program with barriers). Construction APIs validate before use; illegal
   descriptions (zero tiles, non-power-of-two workgroups, subgroup operations
   without a subgroup requirement, unqualified precision combinations,
   malformed programs) cannot become IRs.
2. **Deterministic WGSL emitter subset** (`wgsl_emit`, M21) — pure function
   from validated IR to WGSL for the portable tree-reduction FP32 subset.
   Byte-identical output, fingerprint, and `KernelCacheKey`
   (= hash(normalized IR, precision tag, backend codegen version)) for
   identical inputs. Strategies outside the subset are refused explicitly;
   handwritten shaders remain the qualification reference for them.
3. **Candidate generation** (`kernel_candidate`, M25) — deterministic,
   ordered legal-candidate enumeration per `(problem, limits, policy)` across
   the Q4 families (`subgroup`, `double-buffer`, `vec4`,
   `portable-generated`), each carrying statically checkable requirements and
   an explicit pruned-reason audit trail. The capability input is the
   host-neutral [`DeviceLimitsView`]; `RuntimeDeviceCapabilities` converts to
   it behind the `wgpu` feature (the adapter seam keeps IR/generation free of
   WGPU types).

## Execution wiring

`WgpuFlatAttention::with_generated_portable_q4_kernel(ir)` creates a context
whose primary pipeline is compiled from **generated** WGSL. The canonical Q4
geometry (tiles 4/8, workgroup 64) is dispatchable through this constructor;
other geometries are representable/emittable but explicitly rejected there.
Subgroup policy is forced to Disable so the generated pipeline is exactly what
runs, and `generated_kernel_cache_key()` exposes the M21 cache identity of the
executing artifact.

## Correctness evidence

- Host: `tests/m20_kernel_ir.rs` covers IR validation, determinism,
  fingerprints, profile-dependent candidate sets, explicit pruning reasons,
  emitter refusal outside the subset, Naga validation of generated WGSL, and
  cache-key sensitivity.
- Device: `tests/m21_generated_kernel_parity.rs` executes the generated
  kernel on a real adapter (CI: software Vulkan; qualification run recorded
  below on physical hardware) and validates O **and** LSE against the scalar
  oracle for n ∈ {1, 7, 16, 33, 64}, d ∈ {64, 128}, causal and non-causal,
  within the documented tolerances (`ATOL 5e-5`, `RTOL 5e-4`). Self-skips
  only when no adapter exists and `FLAT_REQUIRE_WGPU` is unset.

## Hardware qualification record (planning/codegen slice — not a performance claim)

- Commit: this branch head.
- Device: NVIDIA Thor (aarch64), Vulkan backend via WGPU 30.
- Result: `generated_kernel_matches_oracle_on_device` PASS (debug build).
- No latency/bandwidth numbers are attached: this slice changes how kernels
  are described and produced, not routing or performance. Per roadmap §0.4,
  any future performance claim requires a dedicated benchmark protocol.

## Non-goals / next slices

- Subgroup-assisted emission (IR already distinguishes it; emitter refuses it
  until its own qualification pass lands).
- Packed-f16 emission (representable as `PrecisionPolicy`; no executable
  generated candidate exists yet — capability may exist without candidate).
- Arbitrary-tile dispatch through `with_generated_portable_q4_kernel`.
- Elastic-side selection over these candidates (separate integration layer;
  candidates already carry normalized requirements for it).
