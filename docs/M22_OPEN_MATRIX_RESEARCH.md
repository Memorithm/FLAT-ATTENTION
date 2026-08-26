# M22 — Open cooperative-matrix / subgroup-matrix research gate

Status: **BLOCKED_BY_PLATFORM_CAPABILITY** (documented honestly; no executable
open path exists through FLAT's runtime dependency today).

Date of investigation: 2026-08-26.
Investigator scope: specification state, runtime (wgpu/Naga) support state,
backend/hardware state, and the consequences for FLAT's architecture.

This document is the roadmap M22 deliverable. It makes **no capability claim**
without a real-device proof, proposes no vendor-SDK dependency, and leaves the
portable fallback fully intact.

## 1. What was investigated

| Layer | Artifact investigated | State observed |
|---|---|---|
| WebGPU/WGSL standard | gpuweb/gpuweb PR #5335 "Proposal for subgroup matrix feature" | Merged **2025-10-07** as a *proposal* into spec Milestone 2; explicitly a basis for further iteration, not a shipped feature |
| WGSL proposal maturity | gpuweb issues #4195, #5435, #5445, #5579, #5605 | Open design questions through 2026-03: address spaces for matrix types, load/store memory-layout flags, uniformity analysis interaction, `subgroup_id` splitting |
| Reference implementation | Dawn `chromium_experimental_subgroup_matrix` design doc + Chrome prototype | Guarded by `enable-unsafe-webgpu`; validation incomplete; correctness/stability "not broadly tested"; exposed only on Metal (Apple 7+) and Vulkan (Android/ChromeOS); **D3D: no support** |
| FLAT's runtime dependency | wgpu 30 / Naga as pinned in this repository | wgpu-types 30 exposes **no cooperative/subgroup-matrix feature flag or capability query** (verified against the vendored crate source: no `MATRIX` feature exists). Naga landed experimental `coop_mat` shader IR on 2025-12-22 (gfx-rs/wgpu issue #8251), but the wgpu-side API wiring ("add a feature and enable it on applicable backends") and safety/test surface were left as follow-up work, and reviewers noted hardware quirks such as devices lacking full fp32 coop multiply-add support |
| Native platform APIs | SPV_KHR_cooperative_matrix (stable), Metal simdgroup_matrix (MSL 3.1, Apple 7+), HLSL SM6.8 WaveMatrix (experimental) | Present but non-uniform across platforms; the WGSL committee itself records portability concerns as the reason subgroup work precedes matrix work |

## 2. Why no executable path exists in FLAT today

1. **Runtime exposure is missing.** FLAT executes exclusively through WGPU.
   wgpu 30 provides no feature bit, no adapter-capability query, and no
   pipeline path for cooperative matrices. Without it there is nothing to
   detect, validate, or dispatch — capability-based routing cannot even begin.
2. **The shader-language surface is not stable.** The WGSL text is a merged
   *proposal* with open committee issues on fundamental semantics (where the
   types may live, how loads declare layout, how uniformity analysis treats
   them). Building FLAT's kernel family on it would couple the project to text
   that may change shape.
3. **Naga support is experimental and incomplete end-to-end.** The compiler IR
   exists; the runtime feature plumbing, safety checks, and broad testing do
   not. A generated matrix kernel could not pass FLAT's mandatory pipeline:
   IR → WGSL → Naga validate → WGPU pipeline creation → device execution,
   because the last two steps have no portable implementation to call.
4. **Uniformity-analysis constraints currently limit usefulness.** Committee
   minutes record that without subgroup-uniformity distinctions, useful
   multi-subgroup patterns are effectively blocked; initial usable shapes are
   single-subgroup workgroups — a poor match for FLAT's Q4 multi-subgroup
   reduction structure.

An honest blocked gate beats a fictional implementation. Per the roadmap's own
acceptance rule — *"no capability is claimed without real-device proof"* — M22
closes as blocked rather than simulated.

## 3. What would unblock it

Concrete, checkable events, any of which reopens this gate:

1. **wgpu ships a cooperative/subgroup-matrix feature flag plus adapter
   capability query**, with Naga's experimental IR wired to it and runtime
   execution tests in wgpu CI.
2. **The WebGPU proposal advances from Milestone-2 proposal to an implementable
   standard feature** with CTS coverage, resolving the open address-space,
   layout-flag and uniformity issues.
3. **A FLAT-supported backend demonstrates the full chain on real hardware:**
   parse → validate → pipeline creation → numerical parity vs FLAT's scalar
   oracle on at least one physical device per relevant backend.

When that happens, FLAT's preparation is:

## 4. IR extension point preserved

The Kernel IR deliberately contains no fake matrix operations, but its shape
already anticipates the extension:

- `CapabilityRequirement` is an open enumeration: a future
  `MatrixFragments { config: … }` variant slots beside
  `SubgroupOperations` without disturbing existing identities beyond the
  documented schema-version bump (`KernelIrVersion`).
- `KernelPhase` is an ordered typed program: matrix phases
  (`FragmentLoad`, `FragmentMultiplyAccumulate`, `FragmentStore`) would extend
  the same phase list between score computation and reduction, keeping
  barrier/softmax semantics explicit.
- `KernelFamily` gains a family (e.g. `DenseQ4MatrixForward`) rather than
  overloading the dense family, so candidate generation, lifecycle states and
  fingerprints separate cleanly.
- Candidate generation already filters on capabilities before anything
  executable exists; a not-yet-exposed matrix requirement simply prunes every
  candidate on every current device — exactly the desired behavior while the
  platform capability is absent.

No code implementing these hooks lands until section 3's events occur; this
section is the design reservation required by the roadmap.

## 5. Sources

- gpuweb/gpuweb PR #5335 (merged 2025-10-07): subgroup matrix proposal.
- gpuweb/gpuweb issue #4195: subgroup matrix tracking issue with Google
  prototype status and committee minutes.
- gpuweb/gpuweb issues #5435/#5579/#5605 and WGSL meeting notes 2025-11-17,
  2026-01-06, 2026-03-24: ongoing semantic decisions.
- Dawn design doc: `docs/dawn/features/subgroup_matrix.md` (Vulkan/Metal/D3D
  support matrix, `GPUSubgroupMatrixConfig`, extension gating).
- gfx-rs/wgpu issue #8251 (merged 2025-12-22): Naga cooperative-matrix IR;
  follow-up work recorded for API exposure/safety/tests.
- wgpu-types 30.0.x source in this repository's lockfile: absence of any
  matrix feature flag (primary evidence for the runtime gap).

## 6. Roadmap consequence

- **M22**: BLOCKED_BY_PLATFORM_CAPABILITY — research outcome documented here;
  no prototype emitter is possible within project policy today.
- **M23** (fragment scheduler): remains blocked behind M22; the interface
  reservation above defines where it will attach.

FLAT's portable fallbacks remain the qualified execution paths. This document
claims no performance effect, positive or negative, from matrix engines.
