# M48 physical-Thor requalification

M48 is the opt-in GQA decode K/V tile-reuse candidate for the SCIAGENT decode geometry:

- batch 1;
- `query_len = 1`;
- `q_heads = 16`;
- `kv_heads = 4`;
- `head_dim = 64`;
- pre-rotated resident K;
- raw resident V;
- causal last-token decode.

The candidate reuses each staged physical K/V tile across the four query heads that share one KV head. The M15 pre-rotated-K route remains the baseline and default.

## Existing physical signal — diagnostic only

After the M48 causal/validation fix in PR #87, GitHub Actions run `31908124457`, job `95068999416`, executed successfully on `tarek-scirust-arm64-01`, adapter `NVIDIA Tegra NVIDIA Thor`, backend Vulkan. It used 5 warmups and 20 repeats and kept upload/readback outside timing.

The recorded rows were:

| KV length | M15 median us | M48 median us | M15 / M48 | M15 max abs | M48 max abs |
|---:|---:|---:|---:|---:|---:|
| 128 | 461.885 | 375.255 | 1.230856 | 0.000022888 | 0.000022888 |
| 192 | 638.964 | 535.946 | 1.192216 | 0.000024796 | 0.000024796 |
| 256 | 822.396 | 684.983 | 1.200607 | 0.000070572 | 0.000070572 |
| 512 | 1632.638 | 1325.882 | 1.231360 | 0.000110626 | 0.000110626 |
| 1024 | 3164.048 | 2552.820 | 1.239433 | 0.000261307 | 0.000261307 |
| 2048 | 6174.324 | 4966.019 | 1.243314 | 0.000444412 | 0.000444412 |
| 4096 | 12292.268 | 9916.503 | 1.239577 | 0.001361847 | 0.001361847 |
| 8192 | 24348.308 | 19598.766 | 1.242339 | 0.002237320 | 0.002237320 |

This is strong candidate-selection evidence: M48 was faster in all eight rows and both routes passed the same scalar-oracle tolerance gate. It is **not** promoted as release-grade evidence because that historical workflow did not require a continuous thermal/occupancy cooldown and measured all baseline repeats before all candidate repeats, leaving an avoidable ordering/thermal-bias channel.

## Requalification protocol

This branch changes only benchmark/CI protocol; it does not modify the M48 kernel.

The requalification requires:

1. exact PR-head checkout and source SHA reporting;
2. benchmark compilation before the GPU reservation;
3. a persistent `tarek-scirust-arm64-01..04` physical Thor runner;
4. `/dev/nvidia0` as the cross-workflow exclusive lock, with an independent-lock contention check;
5. NVIDIA Thor and Vulkan adapter qualification before cooldown;
6. 300 seconds of continuous empty compute occupancy after setup/build/device inspection;
7. rejection if `cuda_pretrain` or any foreign compute process appears during timing;
8. alternating M15/M48 order in warmups and measured repeats;
9. the existing scalar oracle before timing;
10. exactly the eight SCIAGENT-relevant KV lengths 128, 192, 256, 512, 1024, 2048, 4096 and 8192;
11. no upload or readback in the timed region;
12. empty post-run compute occupancy.

## Decision rule

The requalification itself must not assert a speedup. `performance_claim=none` remains in the benchmark output.

If the corrected protocol preserves parity and M48 has lower median latency across the target range, the result is sufficient to justify a **separate SciRust integration candidate**, not an unconditional FLAT global-default change. SciRust must then pin an exact FLAT revision and demonstrate model-level decode parity and throughput on the same physical Thor before promotion.

If the corrected protocol removes the apparent advantage or reveals a correctness regression, M48 remains opt-in and SciRust must not adopt it.
