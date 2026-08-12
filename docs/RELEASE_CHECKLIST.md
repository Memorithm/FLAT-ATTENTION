# FLAT-ATTENTION release checklist

A release candidate is not a release until every mandatory item below is satisfied on the exact commit being tagged. A mergeable GitHub state or an older green run is insufficient. This checklist follows the roadmap's explicit 1.0 Definition of Done; optional diversity evidence is recorded separately rather than promoted into an invented hard gate.

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

## Physical-device and portability evidence

- [ ] Physical NVIDIA correctness evidence is attached to the candidate or to an exact equivalent candidate revision used by the integration gate.
- [ ] Software Vulkan reference qualification remains green.
- [ ] Any unavailable hardware class or device-specific capability limitation is recorded in `docs/COMPATIBILITY_MATRIX.md`.

M37 asks for NVIDIA, AMD, Intel, and software-Vulkan diversity **when hardware is available**. AMD/Intel evidence therefore strengthens the compatibility matrix but is not fabricated or treated as a mandatory 1.0 blocker when those physical devices are unavailable. The roadmap's final 1.0 Definition of Done does not list three-vendor qualification as an absolute release condition.

## Benchmark and claim gates

- [ ] The reproducible benchmark suite is present and green.
- [ ] Measured improvement over SciRust's previous multi-dispatch attention is backed by accepted evidence for the supported target workload(s) used to satisfy the roadmap Definition of Done.
- [ ] Every additional performance statement in release notes has a reproducible M40 manifest with exact commit, device/driver/backend, geometry, precision, warm-up/measurement protocol, and result.
- [ ] Correctness validation precedes accepted timing for each measured candidate.
- [ ] Hosted-runner, lavapipe, WARP, or otherwise software-rendered timings are not generalized as physical-device performance.
- [ ] If no additional release performance statement is made beyond already-qualified evidence, the release snapshot states that boundary explicitly.

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
