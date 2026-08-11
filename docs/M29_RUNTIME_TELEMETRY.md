# M29 — Runtime telemetry

M29 exposes execution metadata without turning observability into a hidden synchronization point.

The initial public record contains:

- stable logical kernel ID;
- device name plus backend/driver fingerprint captured once during owned WGPU context construction;
- query/KV tile geometry and dispatch workgroup dimensions;
- logical dispatch count;
- source-contract temporary allocation count and bytes;
- an explicit fallback reason when a capability-selected path cannot be used;
- autotuner cache status (`NotApplicable`, `Hit`, or `Miss`).

`WgpuFlatAttention::runtime_telemetry(shape)` is a passive host query. It does not submit a command buffer, poll the device, map a buffer, perform a readback, or otherwise force GPU completion. Recording or exporting telemetry remains caller-controlled.

Temporary allocation values describe allocations made by FLAT's source-visible host path. They are not driver allocator telemetry, physical VRAM traffic, cache traffic, or peak device-memory measurements.

A cache status can be attached by an externally-owned autotuner without changing the dispatch. `NotApplicable` is the truthful value for paths not selected through a tuning cache.

The sovereignty boundary is unchanged: Rust-native host code with WGPU/WGSL device execution, no project-authored C/C++ or C ABI bridge, and no mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK.
