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

## Decision boundary

The qualification job deliberately does **not** fail merely because vec4 is slower in a measured row. Its purpose is to produce unbiased physical-device evidence, not to encode the desired result into CI.

After a clean exact-head run:

- if vec4 preserves correctness and materially improves the target Thor rows, a separate decision PR may record the evidence and define the bounded retention/promotion scope;
- if the advantage is neutral, inconsistent or negative, M53 remains opt-in or is removed according to the Phase O retention rule;
- no result from this FLAT-only comparison establishes model-level SciAgent throughput or the roadmap's final comparison against SciRust's previous multi-dispatch implementation.

`performance_claim=none` therefore remains the claim boundary until the measured evidence is reviewed and recorded.
