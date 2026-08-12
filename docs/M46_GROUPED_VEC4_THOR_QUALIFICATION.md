# M46 grouped vec4 physical-Thor qualification

M46 records the first clean physical-device qualification of the M45 native GQA/MQA vec4 candidate. It is an evidence milestone, not a global default-routing change.

## Immutable provenance

- FLAT-ATTENTION candidate: `d789823d5605399b03157b8a57b4b6f73184b17a` (merged FLAT PR #79).
- SciRust evidence source: `2b47374c8b7ce5f12d3074950729a779cb8778a2` (merged SciRust PR #1213 as `520ca9055be76223ca883a5c561f25364abc86ab`).
- GitHub Actions run: SciRust `FLAT M45 grouped vec4 Thor candidate`, run `31633926356`, completed successfully on 2026-08-12.
- Physical adapter: `NVIDIA Tegra NVIDIA Thor` through WGPU `Vulkan`.
- NVIDIA inventory: `NVIDIA Thor`, driver `580.00`.
- Runner: persistent physical runner `tarek-scirust-arm64-01`.

The qualification held the cooperative `/dev/nvidia0` lock, proved independent-file-description lock contention, required 300 seconds of continuous idle time before timing, watched for `cuda_pretrain` contamination once per second during timing, and required empty post-run compute occupancy.

## Measured scope

The compared paths share one resident WGPU context. Upload and readback are excluded. Each timing row is correctness-gated against the grouped scalar oracle before acceptance.

Fixed geometry/protocol:

- `q_heads = 8`;
- `kv_heads = 2`;
- GQA group size = 4;
- sequence lengths = 128 and 512;
- head dimensions = 64 and 128;
- causal and non-causal;
- 3 warmups;
- 12 rotated-order timed repeats.

`portable_over_grouped_vec4` is the portable prepared median divided by the native grouped vec4 median. Values above 1 mean the grouped vec4 candidate was faster for that exact row.

| causal | seq_len | head_dim | portable median (us) | grouped vec4 median (us) | portable/grouped vec4 | grouped parity max abs |
|---|---:|---:|---:|---:|---:|---:|
| false | 128 | 64 | 1374.488 | 1314.672 | 1.045499 | 0.00000042 |
| true | 128 | 64 | 799.124 | 759.873 | 1.051655 | 0.00000048 |
| false | 128 | 128 | 1516.617 | 1441.506 | 1.052106 | 0.00000048 |
| true | 128 | 128 | 886.560 | 834.800 | 1.062003 | 0.00000054 |
| false | 512 | 64 | 17245.575 | 16539.996 | 1.042659 | 0.00000077 |
| true | 512 | 64 | 9216.772 | 8769.157 | 1.051044 | 0.00000060 |
| false | 512 | 128 | 19102.951 | 18083.152 | 1.056395 | 0.00000072 |
| true | 512 | 128 | 10166.330 | 9546.227 | 1.064958 | 0.00000083 |

For this exact Thor/Vulkan matrix, the native grouped vec4 candidate had a lower median latency than the portable prepared grouped path in every measured row. The observed median ratio range is `1.042659..=1.064958`.

## Claim boundary

This evidence does **not** establish a universal speedup, does not cover other GPUs/backends, and does not compare against every SciRust attention implementation. It does not by itself justify making grouped vec4 the unconditional global default.

The M45 constructor remains an explicit candidate-selection surface. A later routing change must preserve capability/fallback behavior and must be justified by evidence for the device/workload class on which that routing is enabled.

`performance_claim=none` remains the repository-wide claim boundary outside the exact measurements recorded above.

## Sovereignty

The qualified path remains Rust-native host code plus WGPU/WGSL. No project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK is introduced.
