# Prepared WGPU kernel router v1

`flat-wgpu-router` provides a narrow physical execution boundary for switching among already-prepared, qualified FLAT WGPU kernel candidates.

The router exists because planning a kernel candidate and executing a kernel candidate are different operations. `flat-kernel-api` / FLAT candidate planning can identify a qualified realization, while `WgpuPreparedKernelRouter` makes a bounded set of those realizations executable before any route change is accepted.

## Transaction boundary

Preparation constructs one `WgpuFlatAttention` executor for every admitted candidate. Every candidate must already have `CandidateLifecycle::Qualified`; duplicate or unavailable candidates fail closed. All prepared executors must resolve to the same `RuntimeDeviceFingerprint`.

After successful preparation:

1. `validate_candidate(candidate, shape, config)` proves that the target is prepared, the candidate can build an exact kernel module for the requested problem, and passive runtime telemetry reports the exact `RuntimeKernelId` requested by that candidate.
2. `apply_candidate(candidate)` changes only the active route index and returns the previous candidate identity. It performs no compilation and submits no GPU work.
3. `verify_candidate(candidate)` reads the route actually stored by the router.
4. `forward(...)` revalidates the exact candidate immediately before physical WGPU execution, then dispatches through the active prepared executor.
5. `restore_candidate(previous)` restores the prior prepared route.

Shape-dependent fallback is not accepted as execution of the requested candidate. If the pinned candidate asks for a vec4 realization but the runtime telemetry would execute the portable scalar fallback for the requested shape, the router returns `CandidateSubstitution` before physical dispatch.

## Explicit v1 limitation

`WgpuPreparedKernelTransitionRequirementsV1` records the current boundary:

- target must be prepared before apply: `true`;
- route swap is live: `true`;
- host-I/O-only: `true`;
- resident buffers preserved: `false`;
- KV state preserved: `false`.

This distinction is mandatory. Schema v1 prepares a distinct WGPU context per candidate. `WgpuFlatAttention::forward` uploads host Q/K/V into the active context, so switching the route does not migrate or preserve `WgpuResidentBuffer` values created by another prepared executor.

The contract therefore does **not** authorize a model runtime to switch a device-resident attention implementation while retaining resident model/KV state. A future resident-state-preserving router must use a shared device/context/buffer ownership boundary and qualify that behavior independently.

## ElasticXxx integration boundary

ElasticXxx owns generic adaptive-control and kernel-realization lifecycle policy; FLAT owns candidate legality, exact attention semantics, WGPU compilation and physical dispatch.

This v1 router can support an external transactional adapter for host-I/O attention experiments because validate/apply/verify/restore are explicit and physical dispatch follows the selected route. It must not be promoted as the final `LiveTransactional` resident-model backend required by ElasticXxx 5/5 because resident buffers and KV state are explicitly not preserved.

No direct ElasticXxx dependency is added by this crate. The existing `flat-elastic-kernel` planning bridge remains separate and can consume the router only after its pinned Elastic contract is deliberately requalified.

## Evidence boundary

The repository CI includes a WGPU-device job using Mesa software Vulkan. That job may prove that two prepared qualified WGPU routes can be switched and executed without hidden fallback. It is software-device execution evidence only.

Lavapipe/Mesa software Vulkan results are **not** real-GPU performance evidence and cannot satisfy FLAT's real-hardware qualification gate or justify latency/throughput claims.
