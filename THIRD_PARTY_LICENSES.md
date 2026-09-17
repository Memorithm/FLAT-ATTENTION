# Third-party dependency inventory

This file records the **root `flat-attention` package's current direct Rust dependency declarations only**. It is not a repository-wide dependency inventory. The root `Cargo.toml` is the source of truth for this table's direct constraints, and the root `Cargo.lock` is the source of truth for the exact graph resolved for that root/workspace context at a given commit. Every third-party package retains its own upstream copyright, license terms, notices, and attribution requirements; FLAT-ATTENTION does not replace or relicense those grants.

| Dependency | FLAT role | Root declaration | Exact root-lock resolution | Upstream package license metadata | Runtime requirement |
|---|---|---:|---:|---|---|
| `wgpu` | Optional portable GPU API/backend abstraction | `30.0` | `30.0.1` | `MIT OR Apache-2.0`; bundled MIT + Apache-2.0 license files inspected | Only with feature `wgpu` |
| `pollster` | Optional synchronous completion of WGPU futures | `1.0` | `1.0.1` | `Apache-2.0/MIT` (verbatim upstream Cargo metadata); bundled MIT + Apache-2.0 license files inspected | Only with feature `wgpu` |
| `ordered-float` | MSRV-compatible transitive-range constraint used by the WGPU dependency graph | `=5.4.0` | `5.4.0` | `MIT`; bundled MIT license file inspected | Direct root dependency; pinned for Rust 1.89 compatibility |
| `naga` | WGSL parser/validator for root-package development/tests | `30.0` | `30.0.1` | `MIT OR Apache-2.0`; bundled MIT + Apache-2.0 license files inspected | Root development/test only |

Exact root-lock evidence for the table above is retained in [`docs/licensing/root-direct-dependency-audit-2026-09-17.md`](docs/licensing/root-direct-dependency-audit-2026-09-17.md), bound to source `27dc1790635d54b6ceb0bff654b42b20f60834ab` and root `Cargo.lock` SHA-256 `7595fc78d944d5f3547c807b3e2a274bbdd66bb3bae8f09cacb73c170956b9a3`. This verifies upstream package metadata and bundled license files for these four exact direct resolutions only; it is not a transitive or repository-wide audit.

The older inventory in this file referred to the `wgpu`/`naga` 0.20 and `pollster` 0.3 generation. Those declarations no longer match the root package's current `Cargo.toml` and are intentionally not carried forward as current-version license evidence. No conclusion about the current resolved packages' copyrights or license metadata should be inferred from historical dependency versions.

## Repository-wide scope boundary

FLAT-ATTENTION contains additional Cargo manifests that are **outside the table above** and must be audited independently before any repository-wide redistribution claim. In particular:

- workspace member manifests under `crates/*` may add direct dependencies that the root package does not declare; for example `crates/flat-ada-a1-candidate/Cargo.toml` directly declares the third-party `ash = "0.38.0"` development dependency in addition to WGPU/Naga dependencies;
- `fuzz/Cargo.toml` is explicitly excluded from the root workspace and directly declares `libfuzzer-sys = "0.4"`; its dependency graph and its own lockfile, when present for the release/audit commit, are a separate licensing surface;
- any Memorithm git/path dependencies declared by workspace-member manifests remain separate provenance/license surfaces and must be checked at their exact revision rather than being inferred from the root package's PolyForm grant.

Accordingly, a release review must enumerate **every `Cargo.toml` shipped or used to build distributed artifacts**, identify the lockfile that resolves each relevant graph (including the fuzz graph where applicable), and inspect each directly or transitively resolved package's own license metadata, bundled notices, source provenance, and redistribution obligations. The singular root table above must never be used as evidence that the repository-wide graph is complete.

## Generated transitive graph

Enabling `wgpu` pulls additional transitive platform/backend crates selected by Cargo and target configuration. They are third-party implementation dependencies, not FLAT-owned source. Release review must inspect the exact applicable lockfile, each resolved package's license metadata, bundled notices, source provenance, and any platform/vendor redistribution obligations for the release commit before distribution.

Git dependencies in any applicable lockfile (including Memorithm ecosystem crates when present) must be audited at their exact pinned revision as separate provenance/license surfaces. Their presence in a dependency graph does not transfer copyright ownership to FLAT-ATTENTION and does not authorize removal or replacement of their notices.

## External references

Research papers, external attention implementations and competitor libraries mentioned in documentation or benchmark methodology are comparison references only unless explicitly declared as dependencies or bundled assets. A reference in documentation is not, by itself, evidence that third-party source code is incorporated.

## Sovereignty note

The optional WGPU stack may internally use platform system APIs to reach Vulkan, Direct3D 12 or Metal. FLAT-ATTENTION's own source does not thereby acquire ownership of those platform components or their licensing terms. Any distribution that bundles additional vendor/runtime material requires a separate release-time license and notice review.
