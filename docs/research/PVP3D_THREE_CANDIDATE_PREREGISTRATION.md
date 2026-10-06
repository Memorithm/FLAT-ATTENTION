# PVP3d — vec4 / fused2 / tile8 same-device protocol

Status: prospectively frozen measurement protocol; no measurements or runtime
promotion. This is a separate experiment and does not change PVP3b.

## Question and candidates

Does the correctness-qualified PVP3c eight-address workgroup prefix reduce
resident wall time compared with PVP2 vec4 and PVP3a fused2 on the same adapter?
Use the existing kernels unchanged. Their layout is `[K, ceil(G/128)] vec4<u32>`.
The baseline performs one stage per dispatch, fused2 combines strides 1/2 in
registers, and tile8 combines strides 1/2/4 with 128 bytes of workgroup storage.
The unfused suffix and small-K fallbacks retain their qualified semantics.

## Frozen confirmatory protocol

- K = 256, 1024, 4096, 16384, 65536, 262144, 1048576.
- G = 128, 512, 2048; nested ascending K then G.
- Five warmups and thirty measured repetitions per candidate per geometry.
- Repeat these six orders: ABC, ACB, BAC, BCA, CAB, CBA, with A=vec4,
  B=fused2, C=tile8. Each order occurs five times in the measured panel.
- Initial words use the PVP3b xorshift32 seed and update rule. Zero unused gate
  bits for the distinct smoke panel only; confirmatory widths have no padding.
- One immutable resident source and three separate equal-size resident states.
- Compare every candidate's full readback with the deterministic CPU vec4
  oracle before warmups and again after the last wall sample. A mismatch aborts
  the run; an incomplete or failed run cannot authorize any timing claim.
- Reset from the source with a resident copy and complete that copy before
  starting each timer. Primary time includes encoder creation, parameter and
  bind-group creation, kernel encoding, submit and completed device wait.
- Pipeline creation, initial upload, reset, oracle and readback stay outside the
  timer. This is resident wall latency, not device-only kernel latency.
- Retain all thirty raw nanosecond samples per candidate. Report nearest-rank
  p50/p95, logical dispatches, state bytes, and four-state resident payload
  bytes separately. Uniforms, bind groups, readback and driver allocations are
  not included in the payload count; no total VRAM claim follows.
- Check buffer, storage-binding, WGSL u32 indexing, invocation, workgroup-size,
  workgroup-storage and dispatch limits before allocating a case. If any
  candidate fails, skip the entire three-candidate geometry with a reason.
  No partial cohort and no post-observation geometry retuning.
- Require a full 40-hex source revision; retain adapter name, backend, device
  type, vendor/device IDs, driver and driver-info metadata.
- Measurement accepts only WGPU IntegratedGpu or DiscreteGpu device types.
  CPU/software or unknown adapter classifications are not physical evidence.
- A physical execution must additionally retain exclusive-device admission,
  occupancy/contamination evidence and exact binary/source identity through
  RemoteOps. The executable cannot establish exclusivity by itself.

## Software smoke panel

`--smoke` is a separate correctness/protocol exercise on Vulkan: (K,G) =
(1,31), (4,129), (8,257), (64,129), (256,512), one warmup and one repetition.
Rows say `smoke`; `hardware_measurement_eligible=false` even on hardware.
Software timing must never be merged into the confirmatory panel.

## Commands

```bash
FLAT_SOURCE_REVISION="$(git rev-parse HEAD)" \
  cargo run --locked --release --features wgpu \
  --example pvp3d_three_candidate_bench -- --smoke

# Only in an exclusive RemoteOps execution with retained admission evidence:
FLAT_SOURCE_REVISION="$(git rev-parse HEAD)" \
  cargo run --locked --release --features wgpu \
  --example pvp3d_three_candidate_bench -- --measure
```

## Decision boundary and reuse

This experiment produces device-scoped measurements only. Preserve neutral,
slower and limit-rejected cases. No universal winner, autotuning/default route,
attention replacement, SML model-quality or 25B throughput claim is authorized.
NNIS and SciRust may reuse the balanced-order, exact-state and raw-evidence
protocol after their own destination review; SML retains semantic authority and
its self-sufficiency boundary. Only the installed GPU driver may be
vendor-specific; this implementation uses Rust/WGPU and no vendor SDK.
