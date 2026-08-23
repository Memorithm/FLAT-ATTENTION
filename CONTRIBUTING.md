# Contributing to FLAT-ATTENTION

Thank you for your interest. FLAT-ATTENTION is Memorithm's Rust-native fused
attention engine for the SciRust ecosystem. This document explains the rules
that keep the project trustworthy.

## Non-negotiable design rules

- Rust is the host language. No CUDA C/C++, `nvcc`, WMMA, CUTLASS, cuDNN, or
  vendor SDK is required by the core design.
- No project-authored C ABI / C++ FFI layer. Portable GPU kernels are expressed
  with open shader/IR paths (WGSL) first.
- Every optimized kernel must be checked against a deterministic Rust oracle.
- No performance claim is accepted without a reproducible benchmark on real
  hardware (`src/benchmark_manifest.rs` provenance records).
- No pull request is merged until all required CI jobs are green on its final
  head SHA.

## Development workflow

1. Fork / branch from `main`.
2. Make your change with tests. New GPU kernels need:
   - a naga parse/validate unit test (no device required),
   - a parity test against the scalar oracle,
   - CI coverage in `.github/workflows/ci.yml`.
3. Run the local gates:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --all-features          # full matrix; uses a WGPU adapter if present
   cargo test                          # host-only subset (works on any machine)
   cargo deny --all-features check advisories licenses sources
   ```

4. Open a pull request with a description of the contract change, if any.
   API additions require documentation and a CHANGELOG entry under
   *Unreleased*.

## Validation discipline

- Violations of shape, length, usage, or device limits must surface as typed
  errors — never panics, oversized allocations, or wgpu uncaptured-error
  aborts. If you add an allocation path, validate it.
- Shader uniform/storage layouts must match host encoders field-for-field;
  cross-check both sides when touching either.
- Numerical behavior changes must update `docs/m9-numerical-policy.md` and the
  relevant oracle first.

## Licensing

By contributing you agree that your contributions are provided under the
repository's PolyForm Noncommercial 1.0.0 licensing policy (see
`LICENSE.md`, `LICENSING.md`). Do not add dependencies whose licenses are not
in the `deny.toml` allow-list.

## Security

Report vulnerabilities privately per [`SECURITY.md`](SECURITY.md); do not open
public issues for them.
