# M34 — Vulkan/Linux qualification

FLAT's portable GPU contract is WGPU/WGSL-first. M34 makes the Vulkan/Linux portability gate explicit and keeps software-adapter correctness evidence separate from physical-GPU evidence.

## Software Vulkan reference

GitHub-hosted Linux CI installs Mesa lavapipe, forces the WGPU Vulkan backend and sets `FLAT_REQUIRE_WGPU=1`. The complete integration-test matrix must execute against an actual Vulkan adapter; absence of a device is a failure rather than an implicit CPU fallback.

Lavapipe validates shader translation, pipeline creation, buffer bindings, dispatch geometry and numerical parity. Its latency is never promoted as physical-GPU performance evidence.

## Physical Vulkan qualification

A separate self-hosted ARM64 job runs on the repository's Jetson Thor runner. It forces Vulkan, reports the physical adapter when host tools are available, then executes strict lint and the same WGPU parity matrix.

Correctness may be qualified while another workload is present on the machine because M34 does not infer performance from this job. Any real-device latency/throughput report remains a separate benchmark artifact with an idle-device protocol and exact commit provenance.

## Failure policy

- no WGPU adapter: failure;
- shader/pipeline creation failure: failure;
- parity failure: failure;
- platform limitation: documented explicitly, never hidden behind a CPU fallback;
- performance regression or promotion: handled only by benchmark-backed evidence.

## Sovereignty

The Vulkan path is Rust-native host code plus WGPU/WGSL. No project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK is required by FLAT.
