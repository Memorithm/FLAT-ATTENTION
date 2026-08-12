# FLAT-ATTENTION release and tag policy

## Release identity

A release is identified by one immutable Git commit and one semantic-version tag. The release notes, changelog entry, compatibility matrix, and benchmark snapshot must all identify that same source state.

## Tag gate

1. Select the candidate commit only after the release checklist is complete.
2. Verify every required CI and milestone-specific gate on that exact SHA.
3. Verify SciRust resolves the intended FLAT revision where integration is part of the release claim.
4. Create an annotated semantic-version tag only after those checks are complete.
5. Cryptographically sign the tag when an approved signing identity is configured. If signing infrastructure is intentionally unavailable, record that fact in the release record rather than fabricating a signature.
6. Never move or replace a published release tag. Corrections require a new semantic version.

For FLAT-ATTENTION 1.0, `v1.0.0` is forbidden until all mandatory items in `docs/RELEASE_CHECKLIST.md` are complete, including M37 physical NVIDIA/AMD/Intel diversity and the final SciAgent decode/KV integration gate.

## Release notes

Release notes must distinguish:

- correctness/portability evidence;
- physical-device performance evidence;
- software-adapter evidence such as lavapipe or WARP;
- known capability limitations;
- integrations that are optional or maintained in SciRust rather than this repository.

No performance language may be inferred from correctness qualification. A statement such as “passes on D3D12/WARP” does not imply useful D3D12 hardware performance.

## Benchmark provenance

Any numerical performance result published with a release must be backed by the M40 benchmark-manifest contract or an artifact containing equivalent required provenance. Results are immutable historical evidence for that commit/device/protocol; later driver or hardware results are new evidence, not edits to the old measurement.

## Rollback and defects

A defective release tag remains immutable. Fixes are merged normally, requalified, and released under a new semantic version. If a release is withdrawn, the release record explains why while preserving the original source/tag for auditability.

## Sovereignty

Release tooling and metadata do not relax the project architecture: Rust-native host code and WGPU/WGSL remain the portable core, with no project-authored C/C++ or C ABI bridge and no mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK.
