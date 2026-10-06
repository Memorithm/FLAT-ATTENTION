# Separate gate-major address-packed PVP candidate

Status: research correctness candidate, no performance or default-route claim.

The historical PVP2 physical layout carries 128 gates at one Boolean address in
each `vec4<u32>`. This separate candidate carries 128 consecutive addresses of
one gate. Its schema is
`flat.pvp-gate-major-address-packed-vec4/v1`, physical shape
`[G, ceil(K/128)] vec4<u32>`, little-endian address bits, gate-major ordering.
It preserves logical `pvp-bitplanes/v1` and does not replace the old kernels.

Five masked shift/XOR stages inside each u32 followed by two XOR stages between
the vector components implement up to seven low-address butterfly stages.
Remaining stages exchange whole vectors in separate dispatches. For K<128,
only the logical log2(K) stages execute, preserving all zero address-padding
bits. K=1 performs no dispatch. For K>=128, there is no address padding; G does
not need to be a multiple of any vector width.

The inspiration is the exact intra-word packed transform already present in
SML-GENIUS `crates/sml-core/src/pascal_gf2.rs`, with the portable word narrowed
from u64 to u32. This is realization research; SML owns the model semantics and
must remain self-sufficient. No vendor SDK, subgroup operation or shared-memory
barrier is required by this candidate. Actual register allocation is unknown.

`WgpuPvpPackedPipeline::prepare` binds one caller-provided canonical state and
constructs immutable uniforms/bind groups once. `WgpuPvpPackedPlan::encode`
reuses them, retaining resources for the plan lifetime. Preparation costs still
exist and must be reported. A fair resident comparison must also preprepare
the historical controls; comparing this plan against controls that allocate
per-stage uniforms would confound layout with host setup.

Correctness gates are a direct-submask oracle, independent direct-monomial ANF
truths, equality with the old layout for every logical bit, exact inverse
coefficient recovery, malformed layout/padding rejection, Naga validation and
actual portable WGPU readback. The ANF corpus has 13 geometries × 3 families,
including every prefix boundary K=1,2,4,8,16,32,64,128 and higher suffix stages.
Adapter absence fails when `FLAT_REQUIRE_WGPU` is set; software execution is
correctness evidence only.

Any future comparison must keep source coefficients and queries identical,
measure physical-layout conversions and upload/download separately, and report
both resident and complete workload cost. Full truth-table materialization is
not automatically the appropriate ANF evaluation strategy: sparse direct
monomial evaluation and precomputed lookup need a declared query/update
workload before a model-aligned comparison. No predicted speedup is evidence.
