# FLAT reusable API compatibility policy

FLAT is currently a pre-1.0, non-published crate (`0.1.x`), but M30 introduces an explicitly versioned reusable namespace: `flat_attention::api::v1`.

## `api::v1` compatibility commitment

Within the `v1` namespace:

- patch releases and commits advertised as API-compatible may fix implementation defects, improve validation, add documentation, and add new optional APIs;
- existing public fields, enum variants, method signatures and their documented semantics are not silently changed;
- a breaking reusable-contract change must be introduced under a new namespace such as `api::v2` while `v1` remains available for a documented migration window, or as an explicitly announced crate-level breaking release;
- backend-specific WGPU types remain outside this backend-neutral compatibility surface unless explicitly re-exported by a versioned API namespace;
- no error implies an automatic fallback. Backend adapters must document any fallback policy explicitly.

## Ownership variants

`api::v1` distinguishes three ownership boundaries:

- `BorrowedAttentionRequest<'a>`: host slices borrowed without implied allocation;
- `OwnedAttentionRequest`: caller-independent host vectors;
- `ResidentAttentionRequest<'a, B>`: backend-neutral borrowed resident handles owned by the embedding runtime.

The resident contract validates only backend-independent geometry/configuration. Concrete adapters remain responsible for buffer size, device ownership, usage flags and synchronization.

## Release discipline

Before a first crates.io/reusable release is advertised:

1. CI must cover the versioned API contract;
2. public examples/docs must use the versioned namespace where stability matters;
3. any breaking change must carry migration notes;
4. performance claims remain tied to benchmark artifacts and exact commits, independently of API stability.

This policy does not change FLAT's sovereignty boundary: Rust-native host code and the existing WGPU/WGSL path, with no project-authored C/C++ or C ABI bridge and no mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK.
