# M58 — Q1 vec4 MHA kernel candidate

M58 is a Phase O optimization candidate that addresses the M28 physical-Thor baseline evidence: the FLAT fused grouped-forward pipeline was slower than SciRust's naive multi-dispatch attention in all eight measured MHA rows (naive/FLAT 0.35–0.72 on `08f7cfe`, see `FLAT_M28_SCIRUST_BASELINE.md`).

## Bottleneck hypothesis

The M28 evidence shows the FLAT Q4 tiled kernel (4 query rows × 8 KV rows per 64-lane workgroup) pays workgroup-staging and reduction-multiplexing costs that exceed the naive per-row GEMM composition at MHA `q_heads == kv_heads` with small head counts. With no GQA head-group amortization to gain, the 4-row tiling only adds overhead.

## Candidate

`FLAT_FWD_Q1_VEC4_WGSL` (`shaders/flat_fwd_q1_vec4.wgsl`) is an opt-in MHA-only kernel:

- one workgroup computes exactly one query row (`x = seq_len`, `y = batch * q_heads`);
- Q is register-resident: each lane owns one `vec4<f32>` slice of the query row, no Q workgroup staging;
- K/V tiles are staged through workgroup memory with `vec4<f32>` storage loads (D64/D128 only);
- per-key-row dot product uses the register Q slice against the staged K slice;
- the output accumulator stays register-resident per lane.

The portable grouped kernel remains the fallback for every other shape, and the Q4 vec4 MHA and grouped vec4 GQA routes remain unchanged. The candidate is selected only when `WgpuGroupedForwardPipeline::with_q1_vec4_mha(device, true)` is used.

## Correctness

`tests/m58_q1_vec4_mha.rs` verifies, on a real WGPU adapter:

- output and LSE parity against the scalar grouped oracle for D64 and D128, causal and non-causal;
- GQA shapes and non-vec4 head dimensions fall back to `Q4PortableGrouped`.

The suite passes on physical NVIDIA Thor/Vulkan.

## Benchmark protocol

The next step is a paired benchmark against the SciRust naive multi-dispatch baseline using the same `flat_m28_naive_vs_fused` harness with the Q1 route enabled, on physical Thor under the established idle/lock/alternating protocol. The candidate is retained only if the measured medians improve the accepted M28 baseline.

No routing change is made by this PR. `performance_claim=none` remains in force.
