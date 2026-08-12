# Third-party dependency inventory

This file records FLAT-ATTENTION's direct Rust dependency boundary. `Cargo.lock` remains the authoritative version-resolution record for transitive crates; each transitive package retains its own upstream license and notices.

| Dependency | FLAT role | Declared version | Upstream license metadata | Runtime requirement |
|---|---|---:|---|---|
| `wgpu` | Optional portable GPU API/backend abstraction | `0.20` | `MIT OR Apache-2.0` | Only with feature `wgpu` |
| `pollster` | Optional synchronous completion of WGPU futures | `0.3` | `MIT OR Apache-2.0` | Only with feature `wgpu` |
| `naga` | WGSL parser/validator for development/tests | `0.20` | `MIT OR Apache-2.0` through the gfx-rs/wgpu workspace | Development/test only |

The gfx-rs/wgpu 0.20.1 workspace declares `MIT OR Apache-2.0` for workspace packages including `wgpu` and `naga`. Pollster upstream declares `MIT OR Apache-2.0`. FLAT does not modify or replace those grants.

## Generated transitive graph

Enabling `wgpu` pulls additional transitive platform/backend crates selected by Cargo and target configuration. They are implementation dependencies of upstream Rust crates, not FLAT-owned source. Release review must inspect the exact `Cargo.lock` and dependency license metadata for the release commit before redistribution.

## External references

Research papers, external attention implementations and competitor libraries mentioned in documentation or benchmark methodology are comparison references only unless explicitly listed in `Cargo.toml`. They are not bundled into FLAT and are not mandatory runtime dependencies.

## Sovereignty note

The optional WGPU stack may internally use platform system APIs to reach Vulkan, Direct3D 12 or Metal. FLAT itself introduces no project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK.
