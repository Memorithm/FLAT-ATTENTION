# Native Dell canonical end-to-end timing — 2026-10-07

The fixed native Dell campaign completed **ten exact fresh processes**, **14,400 measured intervals**,
**3,600 warmups** and **72,000 complete oracle comparisons**, with zero wrong words and no rejected geometry.
Packed7 has a descriptive paired process-median vec4/packed7 wall ratio from
**1.402556 to 15.764191** across the 360 process/panel results.
**36/36** panels have all ten process medians above 1.05.
There are **14** individual paired regressions; all are retained.
This closes the synthetic speed diagnostic. Accepted exclusive-device performance confirmations remain zero;
no SML task/model throughput result is established.

## Execution and frozen contract

- Control PRs: RemoteOps #308 (protocol/task), #310 (observer assembly), #311 (ignored benchmark invocation).
- Execution: [RemoteOps run 37641657219](https://github.com/Memorithm/RemoteOps/actions/runs/37641657219); control commit 163deaed6dbb74a3ba5f43b4b7d9c8f4eb186e9c.
- Unchanged FLAT source: b020d31711d75807eb0f8b13a6a967a8522c5d1c.
- Rust 1.89.0, release optimization, native x86_64; Dell T430, RTX 4060, Vulkan, installed driver 595.45.04.
- Unprivileged memci, process-scoped render group; systemd limits: 12G memory, 400% CPU, 45min.
- Six shapes, three banks, two arms, two modes, five warmups and twenty balanced paired measured repeats.
- Controller/protocol bytes frozen before timing, matched independently against control Git objects.
- Unchanged binary/lock hashes, successful exits and all unique sample/check keys independently verified in std-only Rust.
- Runtime binary SHA256: a85640f84890b18262ad10aa796959420a8d6ce2148f0219e70f3c7095db2707.
- Root executes only trusted RemoteOps observer code; FLAT code never runs as root.
- No service pause or permanent group change; final cooperative lock available, no persistent memci render membership.
- Only the installed driver is vendor specific; no CUDA/NVML/nvidia-smi/NVRTC/vendor SDK bridge.

The endpoint is canonical gate-major u64 input to identical host output: scalar conversion, mode preparation,
fenced upload, encode/submit/wait, explicit MAP_READ and reverse conversion.
Device startup/destruction and verification are outside timing. Prepared resources are outside the interval;
fresh_resources allocates on a warm device/compiler context, not cold startup.
Every timed state receives a second independent readback and full A/oracle, B/oracle, A/B and canonical/oracle
checks including physical padding and four outside-binding guard words.

## Paired results

The central ratio is the median of ten process medians, each from twenty paired wall ratios.
It is not a quotient of separate latency medians. Raw p50/p95 uses 200 measured intervals per arm/panel
and is descriptive; repetitions are not independent sessions. Ratio above one favors packed7.
All 7,200 measured pairs, 360 process medians and component distributions are retained as TSV.

| K | G | bank | mode | paired ratio median [process min–max] | vec4 p50 / p95 (ms) | packed7 p50 / p95 (ms) | slower pairs / 200 |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | 128 | boundary | prepared | 4.548 [4.455–4.698] | 1.683 / 1.783 | 0.371 / 0.408 | 0 |
| 4096 | 128 | boundary | fresh_resources | 1.414 [1.403–1.431] | 2.139 / 2.658 | 1.509 / 1.747 | 0 |
| 4096 | 128 | sparse4 | prepared | 4.649 [4.546–4.798] | 1.718 / 1.994 | 0.369 / 0.431 | 10 |
| 4096 | 128 | sparse4 | fresh_resources | 1.438 [1.416–1.447] | 2.170 / 2.238 | 1.508 / 1.552 | 0 |
| 4096 | 128 | dense32 | prepared | 6.116 [5.895–6.350] | 2.264 / 2.514 | 0.370 / 0.404 | 0 |
| 4096 | 128 | dense32 | fresh_resources | 1.798 [1.775–1.815] | 2.724 / 2.958 | 1.512 / 1.597 | 4 |
| 16384 | 2048 | boundary | prepared | 12.154 [10.926–12.643] | 151.492 / 160.877 | 12.373 / 17.640 | 0 |
| 16384 | 2048 | boundary | fresh_resources | 10.721 [7.978–11.205] | 153.597 / 161.572 | 14.265 / 19.560 | 0 |
| 16384 | 2048 | sparse4 | prepared | 13.116 [9.235–14.239] | 147.399 / 158.076 | 11.222 / 16.924 | 0 |
| 16384 | 2048 | sparse4 | fresh_resources | 9.846 [8.681–12.604] | 150.215 / 160.145 | 17.163 / 18.683 | 0 |
| 16384 | 2048 | dense32 | prepared | 13.199 [10.243–15.764] | 166.922 / 176.997 | 11.593 / 16.695 | 0 |
| 16384 | 2048 | dense32 | fresh_resources | 13.517 [9.439–14.210] | 164.868 / 175.128 | 12.004 / 18.064 | 0 |
| 65536 | 512 | boundary | prepared | 11.165 [7.549–11.351] | 120.064 / 127.855 | 10.740 / 16.866 | 0 |
| 65536 | 512 | boundary | fresh_resources | 9.983 [6.946–10.146] | 122.453 / 129.317 | 12.268 / 18.428 | 0 |
| 65536 | 512 | sparse4 | prepared | 9.440 [7.455–11.228] | 121.079 / 127.141 | 12.294 / 16.835 | 0 |
| 65536 | 512 | sparse4 | fresh_resources | 7.729 [6.880–10.016] | 122.034 / 127.171 | 17.242 / 18.421 | 0 |
| 65536 | 512 | dense32 | prepared | 7.929 [7.745–11.740] | 127.257 / 132.164 | 16.116 / 17.707 | 0 |
| 65536 | 512 | dense32 | fresh_resources | 10.328 [7.233–10.504] | 127.177 / 134.599 | 12.316 / 18.287 | 0 |
| 262144 | 128 | boundary | prepared | 8.375 [5.729–8.529] | 91.194 / 99.419 | 10.665 / 16.972 | 0 |
| 262144 | 128 | boundary | fresh_resources | 7.487 [5.383–7.847] | 91.732 / 99.051 | 11.994 / 18.064 | 0 |
| 262144 | 128 | sparse4 | prepared | 8.086 [5.375–8.199] | 85.880 / 93.251 | 10.578 / 17.050 | 0 |
| 262144 | 128 | sparse4 | fresh_resources | 7.095 [5.070–7.226] | 86.829 / 93.399 | 12.097 / 18.468 | 0 |
| 262144 | 128 | dense32 | prepared | 8.027 [5.675–8.682] | 91.227 / 99.382 | 11.121 / 17.208 | 0 |
| 262144 | 128 | dense32 | fresh_resources | 7.156 [5.336–7.664] | 91.754 / 99.908 | 12.234 / 18.467 | 0 |
| 16384 | 129 | boundary | prepared | 8.467 [4.033–9.749] | 6.692 / 8.760 | 0.816 / 2.166 | 0 |
| 16384 | 129 | boundary | fresh_resources | 3.596 [2.808–3.817] | 7.305 / 8.983 | 2.015 / 3.304 | 0 |
| 16384 | 129 | sparse4 | prepared | 8.048 [4.902–9.578] | 6.684 / 8.179 | 0.911 / 2.008 | 0 |
| 16384 | 129 | sparse4 | fresh_resources | 3.341 [2.780–3.756] | 7.575 / 8.754 | 2.296 / 3.099 | 0 |
| 16384 | 129 | dense32 | prepared | 7.721 [5.762–10.974] | 8.053 / 9.230 | 1.008 / 1.786 | 0 |
| 16384 | 129 | dense32 | fresh_resources | 3.645 [3.156–4.304] | 8.669 / 9.799 | 2.605 / 3.031 | 0 |
| 65536 | 129 | boundary | prepared | 8.469 [7.049–14.368] | 27.284 / 29.137 | 3.289 / 4.154 | 0 |
| 65536 | 129 | boundary | fresh_resources | 5.790 [5.203–8.581] | 28.332 / 30.335 | 4.890 / 5.693 | 0 |
| 65536 | 129 | sparse4 | prepared | 9.887 [6.397–13.878] | 25.540 / 28.293 | 3.271 / 4.339 | 0 |
| 65536 | 129 | sparse4 | fresh_resources | 6.242 [5.297–8.301] | 26.408 / 29.856 | 4.691 / 5.498 | 0 |
| 65536 | 129 | dense32 | prepared | 9.292 [7.213–14.789] | 27.635 / 30.920 | 2.848 / 4.116 | 0 |
| 65536 | 129 | dense32 | fresh_resources | 6.300 [5.673–9.074] | 28.509 / 31.496 | 4.813 / 5.449 | 0 |

Component-p50 conversion sums divided by wall-p50 range:
vec4 **58.9–96.6%**;
packed7 **3.5–65.8%**.
These fractions summarize distinct timers; they are not a decomposition of one median interval.
The advantage concerns these complete carrier implementations, not optimal SIMD transpose,
pure GPU kernel speed, physical bandwidth or register allocation.

## Occupancy and admission

Retained **822** scans: **114** partial,
**6719** unreadable entries, **0** scan-limit events.
Preflight already sees PIDs **[1890, 4285, 2152880]** holding device descriptors before FLAT starts.
This establishes other visible device users, not their compute activity.
One-second snapshots are not continuous lifecycle fencing; empty/complete scans do not prove idle/exclusivity.
Independent occupancy/session sampling review is missing. No confirmatory p-value or default/autotuned winner.

## Retained invalid setup attempts

Run 37640508307 failed at observer compilation before FLAT launch: zero latency samples.
Run 37641009420 built the benchmark but omitted --ignored: ten ignored tests, zero timings.
Both attempts and workflow logs remain in invalid-setup-attempts.tar.xz.
The successful cohort uses the corrected prospectively frozen controller; no failed measured process was replaced.

## Reproducibility and next stage

Evidence directory: evidence/pvp-dell-e2e-20261007. Native archive SHA256: d6ac7b314ea9c5273415458ce273205e9947642245a144fcc18644cf1f10b62d.
The archive retains 79 original runtime files plus derived summaries; FILES.json hashes the originals.
The standalone Rust verifier has 14 tests, including full valid corpus and missing/duplicate/false-output/order/source/software-device controls.
CI checks both archive hashes and replays the verifier on the native corpus without a GPU.

Next: the prospectively declared SML-owned ANF-bank task study, with activation-plus-query costs,
competent sparse/submask CPU controls and resident-table lookup, explicit Q/reuse/lifecycle boundaries.
GPU performance promotion requires controlled admission; internalization requires destination requalification.
Final SML remains self sufficient; FLAT/SciRust/NNIS optional, future SML-HARNESS separate.
No attention replacement, model quality, universal speed, default route or SML throughput claim follows.
