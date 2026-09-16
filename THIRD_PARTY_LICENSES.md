# Third-party dependency inventory

This file records FLAT-ATTENTION's **current direct Rust dependency declarations**. `Cargo.toml` is the source of truth for direct dependency constraints and `Cargo.lock` is the source of truth for the exact resolved transitive graph at a given commit. Every third-party package retains its own upstream copyright, license terms, notices, and attribution requirements; FLAT-ATTENTION does not replace or relicense those grants.

| Dependency | FLAT role | Current direct declaration | License verification status | Runtime requirement |
|---|---|---:|---|---|
| `wgpu` | Optional portable GPU API/backend abstraction | `30.0` | Verify the exact resolved package metadata/notices from `Cargo.lock` before redistribution | Only with feature `wgpu` |
| `pollster` | Optional synchronous completion of WGPU futures | `1.0` | Verify the exact resolved package metadata/notices from `Cargo.lock` before redistribution | Only with feature `wgpu` |
| `ordered-float` | MSRV-compatible transitive-range constraint used by the WGPU dependency graph | `=5.4.0` | Verify the exact resolved package metadata/notices from `Cargo.lock` before redistribution | Direct dependency; pinned for Rust 1.89 compatibility |
| `naga` | WGSL parser/validator for development/tests | `30.0` | Verify the exact resolved package metadata/notices from `Cargo.lock` before redistribution | Development/test only |

The older inventory in this file referred to the `wgpu`/`naga` 0.20 and `pollster` 0.3 generation. Those declarations no longer match the repository's current `Cargo.toml` and are intentionally not carried forward as current-version license evidence. No conclusion about the current resolved packages' copyrights or license metadata should be inferred from historical dependency versions.

## Generated transitive graph

Enabling `wgpu` pulls additional transitive platform/backend crates selected by Cargo and target configuration. They are third-party implementation dependencies, not FLAT-owned source. Release review must inspect the exact `Cargo.lock`, each resolved package's license metadata, bundled notices, source provenance, and any platform/vendor redistribution obligations for the release commit before distribution.

Git dependencies in `Cargo.lock` (including Memorithm ecosystem crates when present) must be audited at their exact pinned revision as separate provenance/license surfaces. Their presence in this dependency graph does not transfer copyright ownership to FLAT-ATTENTION and does not authorize removal or replacement of their notices.

## External references

Research papers, external attention implementations and competitor libraries mentioned in documentation or benchmark methodology are comparison references only unless explicitly declared as dependencies or bundled assets. A reference in documentation is not, by itself, evidence that third-party source code is incorporated.

## Sovereignty note

The optional WGPU stack may internally use platform system APIs to reach Vulkan, Direct3D 12 or Metal. FLAT-ATTENTION's own source does not thereby acquire ownership of those platform components or their licensing terms. Any distribution that bundles additional vendor/runtime material requires a separate release-time license and notice review.
