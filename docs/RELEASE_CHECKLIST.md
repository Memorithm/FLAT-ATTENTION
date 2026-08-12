# FLAT-ATTENTION release checklist

A release candidate is not a release until every mandatory item below is satisfied on the exact commit being tagged. A mergeable GitHub state or an older green run is insufficient.

## Source and API

- [ ] `Cargo.toml` version and release notes agree on the intended semantic version.
- [ ] `api::v1` compatibility has been reviewed against `docs/API_SEMVER.md`.
- [ ] No temporary CI trigger, patch workflow, generated corpus, local benchmark output, or debug artifact is present in the release diff.
- [ ] Sovereignty audit confirms no project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK has entered the core architecture.

## Exact-head software gates

- [ ] rustfmt succeeds at the declared MSRV/toolchain.
- [ ] strict Clippy succeeds with warnings denied.
- [ ] `cargo test --all-features` succeeds.
- [ ] WGSL/Naga validation succeeds.
- [ ] Linux/Vulkan qualification succeeds.
- [ ] Windows/D3D12 qualification succeeds.
- [ ] macOS/Metal qualification succeeds.

## Integration gates

- [ ] SciRust stable FLAT adapter is pinned to the intended release commit and its lockfile resolves that exact revision.
- [ ] SciAgent prefill qualification succeeds on the resident FLAT path.
- [ ] SciAgent decode/KV lifecycle qualification succeeds, including reset/replay/EOS behavior.
- [ ] No integration silently falls back to CPU or an older attention implementation when FLAT was explicitly requested.

## Real-device gates

- [ ] NVIDIA physical-device correctness evidence is attached to the candidate commit.
- [ ] AMD physical-device correctness evidence is attached to the candidate commit.
- [ ] Intel physical-device correctness evidence is attached to the candidate commit.
- [ ] Software Vulkan reference qualification remains green.
- [ ] Any device-specific capability limitation is recorded in `docs/COMPATIBILITY_MATRIX.md`.

M37 is deliberately blocking for a 1.0 release: software adapters or a different vendor do not count as substitute evidence for unavailable AMD or Intel hardware.

## Benchmark and claim gates

- [ ] Every performance statement in release notes has a reproducible M40 manifest with exact commit, device/driver/backend, geometry, precision, warm-up/measurement protocol, and result.
- [ ] Correctness validation precedes accepted timing for each measured candidate.
- [ ] Hosted-runner, lavapipe, WARP, or otherwise software-rendered timings are not generalized as physical-device performance.
- [ ] If no benchmark-backed performance statement is made, the release snapshot says `performance_claim=none` explicitly.

## Licensing and provenance

- [ ] `LICENSE`, `LICENSE.md`, `LICENSING.md`, and third-party notices/inventory are present and internally consistent.
- [ ] Copyright/Required Notice is preserved.
- [ ] External benchmark or research references are clearly separated from FLAT-owned implementation.

## Tag and publication

- [ ] The exact candidate SHA is recorded before tagging.
- [ ] All required checks are green on that SHA.
- [ ] `CHANGELOG.md`, compatibility matrix, benchmark snapshot, and release notes identify the same SHA/version.
- [ ] Tag policy in `docs/RELEASE_POLICY.md` is followed.
- [ ] The tag is created only after all mandatory boxes above are complete.

Until every mandatory gate is closed, the repository may describe itself as a 1.0 candidate but must not claim FLAT-ATTENTION 1.0 is released or fully qualified.
