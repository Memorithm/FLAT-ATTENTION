# M28 — FLAT kernel-generation comparison

This benchmark closes the optimized-generation comparison slice of M28 for the MHA forward kernels that are already implemented in FLAT.

Candidates are selected through the public runtime constructors:

- `Q4Portable` — subgroup disabled, vectorization disabled;
- `Q4Vec4Portable` — subgroup disabled, vectorization enabled for D64/D128;
- `Q4Vec4DoubleBuffered` — explicit experimental M7 candidate;
- `Auto` — the production capability-driven selection, including subgroup when available.

All candidates receive the same deterministic Q/K/V values, shape and causal configuration. Each candidate must match the scalar grouped oracle for O and LSE before timing is accepted.

The current public `WgpuFlatAttention` API owns its WGPU device and synchronization boundary, so the comparable timing scope is deliberately end-to-end `forward(...)`: input upload + fused dispatch + output/LSE readback. The benchmark does **not** reinterpret this as kernel-only latency. Resident/caller-owned timing is already covered separately by the M27/M28 grouped-forward harnesses.

Run with:

```bash
cargo run --release --features wgpu --example m28_kernel_generations
```

Optional environment overrides are `FLAT_M28_GENERATIONS_WARMUP`, `FLAT_M28_GENERATIONS_ITERATIONS`, `FLAT_M28_GENERATIONS_SEQ_LEN`, `FLAT_M28_GENERATIONS_HEADS`, and `FLAT_M28_GENERATIONS_HEAD_DIM`.

`performance_claim=none` is emitted deliberately. Results are evidence for the concrete adapter/backend/driver and exact timing scope; no candidate is promoted based on lavapipe or unreported measurements.

The sovereignty boundary remains Rust-native host + WGPU/WGSL only, with no project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK.
