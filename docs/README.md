# FLAT-ATTENTION documentation

Start with [FLAT_ATTENTION_GUIDE.md](FLAT_ATTENTION_GUIDE.md) for architecture, algorithms, API, SciRust/SciAgent integration, GPU backends, autotuning, benchmark methodology, troubleshooting and development policy.

`ROADMAP.md` at repository root is the authoritative milestone/acceptance plan. Evidence documents in this directory are narrower records tied to the milestone named in their filename; they complement rather than override the consolidated guide.

Core contracts:

- [API_SEMVER.md](API_SEMVER.md) — reusable API/versioning policy.
- [M27_BENCHMARK_HARNESS.md](M27_BENCHMARK_HARNESS.md) — benchmark scopes and metrics.
- [M28_BASELINE_COMPARISON.md](M28_BASELINE_COMPARISON.md) and [M28_KERNEL_GENERATIONS.md](M28_KERNEL_GENERATIONS.md) — baseline and optimized-generation comparison policy.
- [M29_RUNTIME_TELEMETRY.md](M29_RUNTIME_TELEMETRY.md) — passive runtime observability.
- [M34_VULKAN_LINUX.md](M34_VULKAN_LINUX.md) and [M36_METAL.md](M36_METAL.md) — portability qualification boundaries.
- [M38_PROPERTY_STRESS.md](M38_PROPERTY_STRESS.md) and [M39_HOST_API_FUZZING.md](M39_HOST_API_FUZZING.md) — robustness and hostile-input gates.
- [M40_BENCHMARK_MANIFESTS.md](M40_BENCHMARK_MANIFESTS.md) — reproducible benchmark records.

Ownership and distribution are defined at repository root by `LICENSE`, `LICENSE.md`, `LICENSING.md` and `THIRD_PARTY_LICENSES.md`.
