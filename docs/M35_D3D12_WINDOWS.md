# M35 — Direct3D 12 / Windows qualification

M35 exercises the same WGPU/WGSL attention implementation through the Direct3D 12 backend on Windows. The goal is portability and numerical parity, not a Windows-specific implementation.

## Gate

GitHub-hosted Windows CI forces `WGPU_BACKEND=dx12` and `FLAT_REQUIRE_WGPU=1`, then runs strict rustfmt/Clippy and the complete WGPU integration-test matrix. A missing Direct3D 12 adapter is a qualification failure rather than permission to substitute another backend or CPU execution.

The test matrix covers shader/pipeline creation, causal and non-causal forward, MHA/GQA/MQA, asymmetric Q/KV, resident/decode paths, packed f16 where supported, backward/recomputation, deterministic mode and public caller-owned WGPU surfaces.

## Platform limitations

Any capability that the selected Windows adapter does not expose must remain explicit. Unsupported features may select an already-qualified portable GPU path when that is part of FLAT's capability policy, but the run must never silently move attention to the CPU.

Hosted-runner timing is not a performance claim. Performance promotion requires a benchmark artifact tied to an identified physical adapter, driver and exact commit.

## Sovereignty

The Windows path remains Rust-native host code plus WGPU/WGSL. It adds no project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK.
