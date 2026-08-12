# Phase O optimization cycle — GQA-specific K/V reuse

Status: **M47 candidate rejected by M48 physical qualification; no routing change**
Performance claim: **M47 is slower than M45 only for the eight exact Thor/Vulkan cases recorded below; no broader claim**

This cycle follows the Phase O continuous-optimization rules in `ROADMAP.md`: state one bottleneck hypothesis and exact baseline first, isolate one candidate, preserve the same correctness oracle, benchmark before/after under the same conditions, and reject the candidate if it is slower or regresses correctness.

## Baseline

The accepted physical evidence is the M45 native grouped-vec4 qualification performed from SciRust source revision `2b47374c8b7ce5f12d3074950729a779cb8778a2` against FLAT revision `d789823d5605399b03157b8a57b4b6f73184b17a` on NVIDIA Tegra NVIDIA Thor through Vulkan.

Protocol:

- resident same-context prepared execution;
- H2D upload and D2H readback excluded from timing;
- `q_heads=8`, `kv_heads=2`;
- sequence lengths 128 and 512;
- head dimensions 64 and 128;
- causal and non-causal attention;
- 3 warmups and 12 measured repetitions;
- 300 seconds of continuous GPU idleness before timing;
- physical-device exclusion through `/dev/nvidia0` and contamination monitoring;
- correctness parity required before timing acceptance.

Measured medians (`portable grouped` versus `Q4Vec4Grouped`):

| seq | dim | causal | portable µs | grouped vec4 µs | portable / vec4 | max abs parity |
|---:|---:|:---:|---:|---:|---:|---:|
| 128 | 64 | no | 1374.488 | 1314.672 | 1.045499 | 4.2e-7 |
| 128 | 64 | yes | 799.124 | 759.873 | 1.051655 | 4.8e-7 |
| 128 | 128 | no | 1516.617 | 1441.506 | 1.052106 | 4.8e-7 |
| 128 | 128 | yes | 886.560 | 834.800 | 1.062003 | 5.4e-7 |
| 512 | 64 | no | 17245.575 | 16539.996 | 1.042659 | 7.7e-7 |
| 512 | 64 | yes | 9216.772 | 8769.157 | 1.051044 | 6.0e-7 |
| 512 | 128 | no | 19102.951 | 18083.152 | 1.056395 | 7.2e-7 |
| 512 | 128 | yes | 10166.330 | 9546.227 | 1.064958 | 8.3e-7 |

These measurements establish only that the opt-in grouped vec4 candidate was faster than the portable grouped path for these eight Thor/Vulkan cases. They do **not** establish improvement versus SciRust's legacy multi-dispatch attention and therefore do not, by themselves, close the 1.0 Definition-of-Done performance comparison.

## Bottleneck hypothesis

For native GQA (`q_heads > kv_heads`), several query heads in one KV group consume the same physical K/V head. The next candidate will test whether increasing K/V tile reuse across query heads in the same group reduces repeated global-memory traffic relative to the current `Q4Vec4Grouped` implementation.

This is a hypothesis, not a performance claim. The candidate must keep physical K/V cardinality unchanged and must not materialize expanded K/V copies.

## Candidate boundary

The implementation candidate must remain additive and opt-in until qualification. It must:

- preserve the existing public grouped-forward contract;
- preserve the portable grouped fallback;
- preserve native GQA/MQA K/V storage without replication;
- use Rust-native host code and WGPU/WGSL only;
- introduce no project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK;
- fail explicitly for unsupported capability/geometry rather than silently selecting CPU or a legacy attention path.

## Acceptance protocol

A candidate may advance only if all of the following hold:

1. scalar/oracle parity remains within the already documented numerical envelope;
2. MHA/GQA/MQA selection and fallback tests remain green;
3. WGSL validation, rustfmt, Clippy, tests, and supported backend CI remain green;
4. the same physical-device benchmark protocol compares the candidate against `Q4Vec4Grouped` on the exact device/workload class used above;
5. the candidate is rejected if the measured medians do not improve the accepted baseline or if correctness regresses;
6. no default-routing change is made from these measurements alone;
7. any claim used for the 1.0 Definition of Done must additionally include a reproducible comparison against SciRust's previous multi-dispatch attention on the supported target workload.

## M47 implementation slice

M47 implements one isolated WGSL/host-selection candidate for cross-query-head K/V tile reuse inside a native GQA group. `Q4Vec4GroupedKvReuse` is independently selectable from `Q4Vec4Grouped`, so the physical qualification can compare both variants on identical resident buffers and command-submission boundaries. No routing or performance claim changes before that evidence is accepted.

## M48 physical qualification and decision

SciRust PR #1216 qualified FLAT revision `8fc226a1ba80117e080ded24c8a428321a984ab5` from exact SciRust source revision `c8890cd1c617139ad3f982d8af8f93a95f94d9a1`. The successful rerun used a persistent physical NVIDIA Thor runner through Vulkan after acquiring `/dev/nvidia0`, proving cross-process lock contention, and observing 300 seconds of continuous GPU idleness. The benchmark completed without SciAgent contamination.

Both variants used the same resident Q/K/V buffers, prepared dispatch path, device, context, and submission boundary. Upload and readback were excluded from timing. Every O/LSE result passed the existing scalar-oracle tolerance before its timing evidence was accepted.

Measured medians:

| seq | dim | causal | M45 grouped vec4 µs | M47 K/V reuse µs | M45 / M47 | M47 time / M45 | max abs parity |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 128 | 64 | no | 1329.688 | 2258.725 | 0.588690 | 1.698688 | 4.2e-7 |
| 128 | 64 | yes | 780.705 | 1328.455 | 0.587679 | 1.701609 | 4.8e-7 |
| 128 | 128 | no | 1433.447 | 2832.365 | 0.506095 | 1.975912 | 4.8e-7 |
| 128 | 128 | yes | 1009.214 | 1837.290 | 0.549295 | 1.820516 | 5.4e-7 |
| 512 | 64 | no | 16532.103 | 26267.107 | 0.629384 | 1.588855 | 7.7e-7 |
| 512 | 64 | yes | 8776.400 | 14119.055 | 0.621600 | 1.608752 | 6.0e-7 |
| 512 | 128 | no | 18078.570 | 33309.432 | 0.542746 | 1.842482 | 7.2e-7 |
| 512 | 128 | yes | 9715.615 | 18047.301 | 0.538342 | 1.857556 | 8.3e-7 |

M47 is therefore rejected: its median is slower in all eight qualified cases, taking 1.59–1.98 times the M45 baseline time. `Q4Vec4Grouped` remains the accepted native GQA candidate, and `Q4Vec4GroupedKvReuse` must not enter default routing or support a performance claim. The opt-in implementation remains available only to keep the negative experiment reproducible while a separate cleanup decision is made.

These measurements do not compare against SciRust's previous multi-dispatch attention and do not advance the 1.0 Definition-of-Done performance claim. `performance_claim=none` remains in force for product routing.
