# M54 — M53 physical-Thor qualification

M53 is the opt-in vec4-storage candidate for the asymmetric external projection pipeline used by SciRust prefill. The portable pipeline remains the default. M53 was introduced in exact candidate commit `0ffdbde743f86feb6d67b1aeaa177bcbfe28794f` and merged through PR #86.

The development signal on Intel Iris Xe/Vulkan was positive but explicitly non-promotable: seven of eight GQA 8/2 rows favored vec4 by roughly 1.027x to 1.075x and one row was effectively neutral. That evidence does not establish performance on NVIDIA Thor.

## Qualification scope

M54 adds no kernel or public API change. It qualifies the already-merged M53 implementation on the persistent physical NVIDIA Thor runners using the existing `m53_asymmetric_vec4_bench` resident benchmark.

The exact sweep is:

- batch 1;
- `q_heads = 8`;
- `kv_heads = 2`;
- sequence lengths 128 and 512;
- head dimensions 64 and 128;
- causal and non-causal attention;
- fused Q/K RoPE and native K/V cardinality inherited from the M53 pipeline;
- portable and vec4 variants in the same WGPU context;
- 5 warmups and 20 measured repeats;
- alternating portable/vec4 order;
- upload and readback outside the timed region;
- candidate parity checked before timing by the benchmark.

## Physical-device protocol

The GitHub Actions qualification requires:

1. exact PR-head checkout and SHA provenance;
2. compilation before the GPU reservation;
3. an executable Cargo target below `$GITHUB_WORKSPACE`, not the Thor runner's `noexec` temporary mount;
4. a persistent `tarek-scirust-arm64-01..04` runner;
5. `/dev/nvidia0` as the cross-workflow exclusive lock plus an independent contention check;
6. NVIDIA Thor inventory and WGPU/Vulkan qualification;
7. 300 seconds of continuous empty GPU-compute occupancy after setup and compilation;
8. rejection if `cuda_pretrain` or any foreign GPU-compute process appears during timing;
9. exactly eight benchmark rows matching the declared geometry and protocol;
10. empty post-run GPU-compute occupancy;
11. unconditional cleanup of isolated runtime and build directories.

## Qualification result — accepted evidence

GitHub Actions run `32301107798` on the exact PR head `f4d3d066193c3c81ed8cbfc892770fa7dd652706` (PR #94, `perf/m57-m24-capability-limits`) executed successfully on persistent physical runner `tarek-scirust-arm64-01`, adapter `NVIDIA Tegra NVIDIA Thor`, backend Vulkan, driver 580.00. The full protocol was enforced: exact-head checkout, compile before GPU reservation, `/dev/nvidia0` lock with independent contention proof, Thor/Vulkan identity verification, 300 seconds of continuous empty compute occupancy, alternating portable/vec4 order, scalar-oracle correctness gate before timing, upload/readback outside timing, 8 exact rows, and empty post-run occupancy.

Recorded rows (batch 1, GQA q_heads=8 kv_heads=2, 5 warmups, 20 repeats, medians in microseconds):

| seq | dim | causal | portable median µs | portable p95 µs | vec4 median µs | vec4 p95 µs | portable / vec4 | parity max abs |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | no | 1651.359 | 1756.028 | 1495.385 | 1513.774 | 1.104304 | 0.00000000 |
| 128 | 64 | yes | 982.741 | 1124.844 | 888.092 | 897.148 | 1.106576 | 0.00000000 |
| 128 | 128 | no | 1913.824 | 1963.139 | 1649.988 | 1657.025 | 1.159902 | 0.00000000 |
| 128 | 128 | yes | 1105.974 | 1163.224 | 964.908 | 983.555 | 1.146196 | 0.00000000 |
| 512 | 64 | no | 20801.111 | 20920.667 | 18466.727 | 18581.405 | 1.126410 | 0.00000000 |
| 512 | 64 | yes | 10945.172 | 11024.525 | 10007.440 | 10162.506 | 1.093703 | 0.00000000 |
| 512 | 128 | no | 24163.413 | 24339.107 | 20444.803 | 20652.924 | 1.181885 | 0.00000000 |
| 512 | 128 | yes | 12505.419 | 12594.436 | 10839.948 | 10876.172 | 1.153642 | 0.00000000 |

The qualification confirms the earlier Intel Iris Xe development signal on physical NVIDIA Thor: vec4 preserves exact parity (max abs 0.0 against the scalar oracle) and materially improves all eight target rows by 1.09x–1.18x. The vec4 candidate is retained as the qualified opt-in path for the asymmetric projection pipeline.

This FLAT-only comparison does **not** establish model-level SciAgent throughput or the roadmap's final comparison against SciRust's previous multi-dispatch implementation. `performance_claim=none` remains the claim boundary for product routing; the measured rows are candidate-selection evidence only.

## Decision boundary

The qualification job deliberately does **not** fail merely because vec4 is slower in a measured row. Its purpose is to produce unbiased physical-device evidence, not to encode the desired result into CI.

After a clean exact-head run:

- if vec4 preserves correctness and materially improves the target Thor rows, a separate decision PR may record the evidence and define the bounded retention/promotion scope;
- if the advantage is neutral, inconsistent or negative, M53 remains opt-in or is removed according to the Phase O retention rule;
- no result from this FLAT-only comparison establishes model-level SciAgent throughput or the roadmap's final comparison against SciRust's previous multi-dispatch implementation.

`performance_claim=none` therefore remains the claim boundary until the measured evidence is reviewed and recorded.
