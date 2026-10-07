# Five-arm prepared PVP qualification (prospective)

This slice implements `PVP_PREPARED_COMPARISON_DESIGN.md` resource parity,
not its latency study. No timing, physical register, bandwidth, SML model-quality,
default-route or autotuning claim is admitted.

## Fixed identity and controls

Arms, in order: vec4, fused2, tile8, packed1, packed7. All uniforms and
bindings are immutable and created before `encode`; pipeline/plan creation
remains a separate cost. Historical APIs, shaders and benchmark executable
are not rebuilt or changed. Packed7 delegates to the existing qualified plan.
Packed1 uses the same gate-major address-packed layout with one dispatch for
each low logical stage, and the unchanged packed cross-vector suffix.

The qualification uses the 13 existing small geometries plus
`(16384,2048)`, `(65536,512)`, `(262144,128)`; all five arms must admit all
16 geometries. Banks are boundary, sparse4 and dense32. Round zero is the
existing fixture; rounds one and two vary coefficients with the existing
nonce convention. The same plan/state is reused across changed coefficients.

Expected truth comes from independent direct monomial masks. For every
small fixture and coefficient round it is checked against per-address direct
monomial evaluation before device work. Representation conversion happens
outside GPU execution and is not hidden as a speed improvement.

Each arm/geometry/bank/round checks uploaded source without compute,
forward transform and inverse. Each phase uses two fresh staging buffers,
explicit MAP_READ transitions and complete A/oracle, B/oracle and A/B
word comparisons. Four trailing sentinels outside the canonical storage
binding must remain unchanged; canonical padding is included in all comparisons.
All discrepancies are printed, retained and aggregated before test failure.
No rerun replaces a failed process.

## Counts and execution gates

CI uses one round: 240 arm cases, 720 phase observations, 2160 comparisons.
The native correctness cohort is three fresh processes, each with exactly
three coefficient rounds: 720 cases, 2160 observations, 6480 comparisons
per process (19440 comparisons total). Native tests require an actual adapter
and compile/runtime `FLAT_SOURCE_REVISION` identity. Debug builds qualify
correctness only, never speed. No benchmark example is built.

Before native execution freeze the formatted source revision, test executable
SHA256 and controller revision; retain them in the execution report. Use the
existing scoped service guard/cooperative reservation/restoration procedure,
with its independent restoration watchdog armed before service mutation.
Use a bounded controller with per-process timeout and independent final
restoration inspection; preserve all logs, statuses, observation limitations
and historical binary hashes. Unknown occupancy remains unknown; this cohort
admits zero performance rows regardless of successful exact checks.

Linux/Metal qualification and mandatory D3D12 WARP CI must pass on the exact
final PR head before merge. Native failures block native qualification and
are published, not silently excluded. Native controller implementation and
evidence freeze are a separate gate from the source protocol above.

## Remaining performance gates

A separately preregistered harness must fix shapes, sample counts, balanced
arm order, warmups and common capability rejection before measurement; verify
every timed state outside the timer; distinguish resident encode/submit/wait
wall latency, optional public GPU timestamps, and conversion/upload/cold
preparation. Independently reviewed observation/admission is still required.
Thor and Dell measurements must never be pooled. This qualification alone
cannot confirm a performance gain.
