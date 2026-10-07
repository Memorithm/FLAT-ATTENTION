# Packed-address large-domain correctness protocol

Status: prospective; no performance qualification or large-domain pass is claimed.

The research-only schema remains `flat.pvp-gate-major-address-packed-vec4/v1`.
No default route, shader, historical benchmark or SML contract is changed.

## Fixed corpus

The five `(K,G)` geometries are `(16384,2048)`, `(65536,512)`,
`(262144,128)`, `(16384,129)` and `(65536,129)`. The first three each
occupy 4,194,304 bytes. Each uses boundary, sparse4 and dense32 ANF banks.
The 129-gate shapes exercise gate ordering, not address padding.

CI uses one round. The planned native campaign uses ten fresh processes,
each with `FLAT_PVP_PACKED_STRESS_ROUNDS=3` and `FLAT_REQUIRE_WGPU=1`.
Round zero preserves the original fixtures; rounds one and two toggle
high-variable monomials on alternating gates to detect stale state.

One immutable prepared plan is reused for each geometry/bank. Each round
checks the uploaded source, forward transform and inverse. Each phase
copies the same unchanged state into two fresh staging buffers in one
submission and explicitly transitions both to MAP_READ. Every word is
compared A/oracle, B/oracle and A/B; mismatches are counted, with the first
eight retained. The forward oracle evaluates monomials by word masks,
not by a subset-zeta transform. Host tests compare it to the existing
per-address definition on all 39 small fixtures and all three rounds.

Each native process requires 15 cases, 135 paired phase observations,
405 comparisons and zero failed comparisons. Ten processes require
1,350 observations and 4,050 comparisons. A completion line alone is
insufficient: exit status, counts, source identity and zero failures must
all agree. Missing adapters, timeouts, mapping failures and incomplete
processes are retained as failures, never replaced by successful reruns.

## Execution and provenance gate

Freeze the formatted source revision, controller revision, binary hash,
commands and environment before physical-GPU execution. Use Rust 1.89,
the installed driver and public Rust/WGPU APIs; no CUDA or NVML.
Use a bounded cooperative Thor lease with an independently armed restore
action. Capture initial service policies, occupancy observations, every
process exit and raw output. Restore original policies and independently
inspect the final state before releasing the lease. Do not overwrite the
historical benchmark executable or discard failed evidence.

This is correctness stress only. Unknown occupancy can be recorded but
cannot qualify speed. There are no accepted timing samples, speedups or
claims about physical register allocation in this protocol.
