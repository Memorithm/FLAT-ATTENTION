# M36 — Metal qualification

M36 exercises FLAT's portable WGPU/WGSL implementation through Apple's Metal backend. No Metal-specific attention implementation is introduced.

## Gate

GitHub-hosted macOS CI forces `WGPU_BACKEND=metal` and `FLAT_REQUIRE_WGPU=1`, reports the available graphics hardware, then runs strict rustfmt/Clippy and the complete WGPU integration-test matrix on one thread.

A missing Metal adapter, pipeline-creation failure or numerical-parity failure is a qualification failure. The gate must not silently substitute a CPU implementation or a different WGPU backend.

The matrix covers the existing forward/decode/training surfaces, including MHA/GQA/MQA, causal/non-causal attention, asymmetric Q/KV lengths, resident buffers, RoPE/bias paths, mixed precision where Metal exposes the required feature, and recomputation-based backward.

## Evidence boundary

Hosted-macOS timing is not promoted as a performance claim. M36 establishes portability and correctness. Any Metal performance statement requires a separately reproducible benchmark record tied to the exact commit, identified Apple hardware, OS/driver stack, geometry, precision and measurement protocol.

Platform capability limitations remain explicit and may select an already-qualified portable GPU variant only when FLAT's capability policy permits it.

## Sovereignty

The Metal path remains Rust-native host code plus WGPU/WGSL. It adds no project-authored C/C++, C ABI bridge, CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or mandatory vendor SDK.
