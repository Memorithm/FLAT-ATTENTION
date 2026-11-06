# FLAT-ATTENTION industrial release dossier — preflight

status: BLOCKED
release_authorized: false
assessment_date: 2026-11-06
source_of_record: Memorithm/FLAT-ATTENTION
candidate_version: 1.0.0 (not tagged or published)

## Scope and decision

This file establishes a fail-closed preflight for the **industrialization
programme**, not a successful release attestation. The actual public package
remains `0.1.0` with `publish = false`. No tag, runtime default, or published
crate is authorized by this dossier.

A 1.0 release is allowed **only after** the complete existing
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) is satisfied and re-evaluated on
the exact FLAT commit proposed for release. No earlier green PR, software
adapter timing or separately merged research result substitutes for that gate.

## P0 gate 1 — SciRust/SciAgent decode and KV lifecycle

Owner: Memorithm/scirust. Tracking:
https://github.com/Memorithm/scirust/issues/1732

Existing integration: `docs/FLAT_M33_SCIAGENT_DECODE_KV.md` in SciRust
already documents M32 prefill, M15 token-wise decode, resident K/V,
reset/replay and EOS. Do not create another model runtime.

Release evidence required:
- exact FLAT commit and corresponding SciRust lockfile source revision;
- SciAgent production path opt-in through FLAT with no hidden CPU fallback;
- real adapter required rather than a test that silently skips when absent;
- prefill, incremental decode, reset, deterministic replay and EOS tests;
- device-resident K/V contract and identified WGPU backend;
- final SciRust head CI and machine identity.

Current preflight verdict: OPEN. No newly selected 1.0 release commit exists,
so earlier M33 evidence does not close the candidate-specific requirement.

## P0 gate 2 — real physical-device benchmark

Owner: Memorithm/FLAT-ATTENTION. Tracking:
https://github.com/Memorithm/FLAT-ATTENTION/issues/359

The existing benchmark snapshot explicitly does not provide an accepted
candidate-bound comparison against SciRust's previous multi-dispatch attention.

To qualify, measure identical work on a named **physical-device benchmark**
environment with exact FLAT commit, exact SciRust revision, backend, driver,
dtype, model/problem geometry, cache state, warmups, repeats, time boundary,
median and p95, and an immutable M40 record. Include resident-only and
transfer-inclusive measurements separately. Correctness must precede timing.

Current preflight verdict: OPEN. The software Vulkan and WARP CI gates are
valuable numerical/portability evidence, never a physical GPU speedup claim.

## P1 gate 3 — qualified kernel routing

Generated WGSL and autotuning infrastructure exist, but release promotion
requires a dedicated routing qualification. No unqualified generated kernel
should silently become default. Keep no runtime promotion for research-only
compact/Boolean candidate selection or CPS-2 synthetic diagnostics.

Current preflight verdict: OPEN as documented by
`docs/ROADMAP_STATUS.md` and `docs/RELEASE_CHECKLIST.md`.

## P1 gate 4 — publication package

Required after previous gates:
- API compatibility and package licensing review;
- Rust 1.89, workspace tests, strict Clippy, rustdoc, semver, supply chain;
- Linux Vulkan, Windows D3D12 and macOS Metal correctness checks;
- physical hardware evidence with accurate availability limitations;
- release benchmark manifest, changelog and compatibility matrix all naming
  the same commit;
- explicit tag authorization only after every mandatory checklist item.

Current preflight verdict: OPEN.

## Procedure to regenerate a preflight observation

Host-only Rust policy regressions can be run on Debian without GPU:

```bash
cargo test --locked --test industrial_release_preflight
```

This tests **honest blockage**, not completion. It is intentionally a
prerelease guard: when the checklist becomes complete, release work must replace
the guard with an immutable exact-commit release attestation, and the full
release checklist still governs the final decision.

Self-hosted qualification and machine scheduling must use Memorithm/RemoteOps,
with explicit resource coordination and no disruption to other workloads.

This dossier does not claim model-quality, physical-memory, bandwidth,
latency, throughput or cross-vendor performance improvements.
