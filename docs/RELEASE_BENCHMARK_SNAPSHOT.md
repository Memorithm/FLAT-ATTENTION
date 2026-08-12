# FLAT-ATTENTION 1.0 candidate benchmark snapshot

Status: **pre-release evidence index**  
Performance claim: **none beyond accepted benchmark evidence explicitly cited by the release record**

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

## 1.0 Definition-of-Done benchmark gate

The roadmap requires a reproducible benchmark suite and measured improvement over SciRust's previous multi-dispatch attention for supported target workload(s). The release record must cite the exact accepted comparison used to satisfy that gate; correctness-only runs do not satisfy it.

## Software-adapter policy

Lavapipe and D3D12/WARP are valuable portability/correctness gates. Their timings are not used as physical-GPU performance evidence. Hosted macOS timing is likewise not promoted without a release manifest that identifies the physical adapter and exact protocol.

## Physical-device snapshot status

- NVIDIA Jetson Thor: correctness evidence exists for the FLAT Vulkan path; any release performance claim on Thor requires an idle-device benchmark manifest on the exact candidate being cited.
- AMD: additional physical M37 diversity evidence pending hardware availability.
- Intel: additional physical M37 diversity evidence pending hardware availability.

AMD/Intel M37 evidence expands the compatibility matrix when hardware is available; the roadmap's final 1.0 Definition of Done does not make unavailable three-vendor hardware a mandatory release gate. Missing hardware remains disclosed rather than fabricated.
