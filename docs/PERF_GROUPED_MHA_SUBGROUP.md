# Grouped-forward MHA subgroup candidate

Physical Thor evidence for prepared bind-group reuse showed a real host-side improvement, but the long-context gap remained dominated by kernel execution. This candidate therefore reuses the already-qualified M5 subgroup Q4 shader inside the caller-owned grouped-forward pipeline when the logical request is MHA (`q_heads == kv_heads`).

The portable grouped shader remains the mandatory native path for GQA/MQA. No K/V head expansion is introduced. Selection is capability-driven from the features enabled on the caller-owned `wgpu::Device`; adapter marketing names are not consulted. `WgpuSubgroupPolicy::Disable` forces the portable grouped path, `Auto` permits the subgroup MHA route when available, and `Require` fails explicitly if the device was not created with subgroup support enabled.

`PreparedGroupedForward::kernel_variant()` records which path a fixed resident request will use. The prepared type and variant are publicly nameable through `flat_attention::api::wgpu` while the backend-neutral `api::v1` contract remains unchanged.

This change is a performance candidate, not a speedup claim. Promotion requires same-device oracle parity plus a paired physical-device benchmark against both the portable grouped path and SciRust's previous multi-dispatch attention. Native GQA/MQA remain supported through the portable grouped kernel regardless of the benchmark result.

The sovereignty boundary is unchanged: Rust host code, WGPU/WGSL kernels, no project-authored C/C++ or C ABI bridge, and no mandatory CUDA C++/nvcc, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK.
