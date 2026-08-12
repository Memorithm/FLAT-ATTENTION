# FLAT-ATTENTION 1.0 candidate benchmark snapshot

Status: **pre-release evidence index**  
Performance claim: **none**

This file is the release-level index for benchmark evidence. It deliberately contains no generalized speedup or throughput claim until exact-device M40 manifests have been accepted for the release candidate.

## Required benchmark families

The candidate release carries harnesses/evidence plumbing for:

- resident grouped forward across MHA/GQA/MQA and representative head dimensions/context sizes;
- resident decode and long-context decode;
- scalar-Rust versus portable FLAT forward baseline;
- SciRust historical/multi-dispatch versus FLAT paired comparison where maintained by SciRust;
- optimized FLAT kernel-generation comparisons;
- resident-only versus transfer-inclusive timing;
- cold versus warm pipeline lifecycle;
- visible resident-buffer, allocation, dispatch, and intermediate-byte accounting;
- forward/backward training-chain qualification.

## Promotion rule

A measured result may enter release notes only when its record identifies at least:

- exact FLAT commit SHA;
- device and driver;
- WGPU backend;
- precision;
- batch, query heads, KV heads, query/KV lengths, and head dimension;
- causal/non-causal mode;
- warm-up count and measured iteration count;
- median and percentile latency where applicable;
- tokens/s or other throughput unit only where semantically meaningful;
- measurement scope, including whether upload/download and pipeline creation are excluded or included.

Correctness/oracle validation must pass before a timing sample is accepted.

## Software-adapter policy

Lavapipe and D3D12/WARP are valuable portability/correctness gates. Their timings are not used as physical-GPU performance evidence. Hosted macOS timing is likewise not promoted without a release manifest that identifies the physical adapter and exact protocol.

## Physical-device snapshot status

- NVIDIA Jetson Thor: correctness evidence exists for the FLAT Vulkan path; a release performance claim still requires an idle-device benchmark manifest on the exact release candidate.
- AMD: physical M37 evidence pending.
- Intel: physical M37 evidence pending.

Because the 1.0 release candidate still has mandatory integration/vendor-diversity gates open, this snapshot intentionally remains `performance_claim=none`. When those gates close, accepted immutable benchmark manifests should be linked from the release record rather than replacing historical results with unproven summaries.
