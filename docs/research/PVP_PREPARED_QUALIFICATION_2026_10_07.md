# Prepared PVP controls: exact native qualification, no latency claim

The five equally prepared arms completed three fresh Thor processes with
**19440 complete word comparisons, zero discrepancies**. This closes the
bounded synthetic correctness slice for the prepared comparison, not its speed
study. All three runs retained unknown occupancy observations; accepted
performance rows are zero.

## Implementation and fixed identity

- Source and bounded controller: `d78599ca0e589a2d578a6523c88441210290c8e1`.
- Protocol: [PVP_PREPARED_QUALIFICATION_PROTOCOL.md](PVP_PREPARED_QUALIFICATION_PROTOCOL.md).
- Pipeline: `src/pvp_prepared.rs`, with five explicit research-only arms.
- Immutable uniforms/bindings and retained buffers are prepared before encode
  for vec4, fused2, tile8 and packed1. Packed7 delegates to the existing packed
  plan. Repeated encode allocates no buffers or bind groups for any arm.
- The old APIs, old shaders and historical benchmark example remain unchanged.
  No historical benchmark executable was rebuilt.
- Packed1 changes no packed-address layout semantics; it executes each low
  logical stage separately, then uses the existing cross-vector suffix. It
  permits separating representation effects from low-stage fusion.

All preparation costs still exist. Equal preparation does not mean that
encoding, submission, completion waits or layout conversion are free.

## Native cohort

Adapter: NVIDIA Tegra NVIDIA Thor, Vulkan, installed driver `580.00`.
Compiler: Rust 1.89.0, LLVM 20.1.7, aarch64 Linux. The frozen executable is an
unoptimized correctness test, not a performance binary.

| Property | Per process | Three-process total |
|---|---:|---:|
| Arm/geometry/bank/round cases | 720 | 2160 |
| Source/forward/inverse paired observations | 2160 | 6480 |
| A/oracle, B/oracle and A/B complete comparisons | 6480 | 19440 |
| Forward checks | 720 | 2160 |
| Inverse checks | 720 | 2160 |
| Failed comparisons | 0 | 0 |
| Accepted performance rows | 0 | 0 |

The corpus contains the existing 13 small geometries plus `(16384,2048)`,
`(65536,512)` and `(262144,128)`, three bank families and three coefficient
rounds. The three large geometries each have exactly 4194304 canonical bytes
in every arm, plus a 16-byte guard outside the canonical storage binding.
Small irregular geometries retain their distinct padding and physical sizes.
There is no selective capability dropping in this qualification cohort.

Direct wordwise monomial truth is cross-checked against independent per-address
monomial evaluation on all 117 small bank/round fixtures per process. The same
immutable plan is reused across changed coefficients and both directions.
Every phase uses fresh paired staging buffers with explicit MAP_READ
transitions. Full comparisons include canonical padding and four trailing
sentinel words. All failures would be printed and aggregated before exit;
none of the three GPU processes was repeated or substituted.

The largest geometry has structural dispatch counts 18/17/16/18/12 for
vec4/fused2/tile8/packed1/packed7 respectively. These counts are not latency,
physical bandwidth or register-allocation measurements.

## Capture integrity and restoration

Reservation began `2026-10-07T07:19:36.716050849Z`; the controller completed
`07:28:44.616361973Z` and restored services at `07:28:45.500541420Z`.
An independent successful inspection at `07:29:39.406722780Z` verified:

- all four services active with original Restart and manual-start policies;
- scoped runtime guards and kernel trace instance removed;
- restoration timer inactive, cooperative device lock available;
- controller exit zero, frozen executable and historical benchmark unchanged.

The **first inspection failed its lock-availability assertion** after the
controller had released its reservation. Its empty redirected output is
preserved as `restoration-first-attempt-empty.json`; the later contention
snapshot contains no lock owner, so the transient owner is not established.
The assertion traceback was observed in the remote process output, not saved
as an original stderr file. This incident is not silently replaced: only the
read-only inspection was repeated, never GPU tests or service mutation.
The successful receipt establishes availability at its later inspection time,
not uninterrupted device exclusivity. Cooperative flock is not hostile fencing.

All 61 original capture files (9940834 bytes), including observation limitations
and this incident's retained files, are in the lossless archive. Native and
local strict verification check every original hash and all 19440 unique
comparison keys, complete word counts, source identity, exit statuses and
restoration fields. Six negative controls reject wrong source, duplicate/missing
keys, wrong word counts, mismatched words and an unmatched file hash. Original
capture data remains unchanged.

| Identity | SHA256 |
|---|---|
| Frozen test executable | `47bd271010a501bab5a469912e7093c4bea00d32efc4c74c746be895da265b1f` |
| Controller | `0b9ba4edb981d8db897a77a4e2a836d451f54988e052f6cc0bec10fbc09ba513` |
| Lossless archive | `9bceeec99d384a79e4daaeb8cf3eb7d5dfee3d2f6b6d439d4c8dfff2e596a6ab` |
| Unchanged historical benchmark | `7aef67026ad088504478885cf9f2b7c864218f6fb3b0792bd8609fd907d3f414` |

Artifacts and reproduction: [evidence README](evidence/pvp-prepared-2026-10-07/README.md).
The source after execution remains clean at the frozen revision. Later evidence
commits must not modify executed Rust/WGSL/controller code. Required CI must
be green on the final PR head before merge; earlier green heads do not suffice.

## Boundary and next slice

No new latency samples were collected. Unknown observations in runs 1, 2 and 3
remain unknown; no independence/exclusivity upgrade or performance admission is
inferred from exact outputs. Prior failed captures remain preserved.

The next slice is a separately preregistered prepared five-arm measurement
harness: fixed shapes and counts, balanced order, common capability rejection,
every timed state verified outside its timer, separate resident wall/GPU
timestamp/conversion/upload/cold-preparation costs, and independently reviewed
device observation/admission. Thor and Dell are separate cohorts.

No default-route, autotuning, physical register, driver-root-cause or SML
model-quality promotion follows. This native path remains Rust/WGPU/open
backend with only the installed driver vendor-specific; FLAT/SciRust/NNIS are
optional partners, the final SML model is self-sufficient, and SML-HARNESS
remains separate. A model-aligned ANF-bank study and destination requalification
are still required before internalization.
