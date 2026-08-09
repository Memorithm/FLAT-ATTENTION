# M6 — Vectorized Q/K/V memory transactions

M6 specializes the qualified portable Q4 kernel for the two common head dimensions `64` and `128`.

## Scope

The M6 shader views Q, K and V storage as `array<vec4<f32>>`. Each source-level storage access therefore brings four adjacent `f32` values into the kernel before they are unpacked into the same scalar workgroup-memory layout used by the qualified Q4 implementation.

The online-softmax algorithm, causal semantics, packed `[O | LSE]` result, workgroup size, Q4 query reuse and K/V tile size remain unchanged.

This is a source-level transaction model. It is not a claim that a given driver performs one physical DRAM transaction per WGSL `vec4` expression. Caches, lowering, coalescing and physical bus transactions are backend/device properties and require hardware measurement.

## Selection policy

M5 subgroup reduction has priority when a context selected `Q4Subgroup`.

Otherwise, with vectorization enabled:

- `head_dim = 64` -> `Q4Vec4Portable`;
- `head_dim = 128` -> `Q4Vec4Portable`;
- every other supported dimension -> qualified `Q4Portable` scalar-storage kernel.

The public constructor `with_subgroup_policy_and_vectorization(..., false)` disables M6 and exists to reproduce the M4 scalar baseline on the same device.

`kernel_variant()` retains its M5 context-level meaning. `kernel_variant_for_head_dim(d)` reports the effective dispatch generation after M6 specialization.

## Alignment argument

For `D=64` and `D=128`, every row contains a multiple of four `f32` elements. Since Q/K/V tensors are contiguous `[batch, heads, sequence, head_dim]`, every row begins at a `vec4<f32>` element boundary in the shader's logical view. No scalar tail is needed for these two specializations.

All non-specialized dimensions use the scalar fallback rather than relying on an alignment assumption.

## Validation

M6 adds mandatory device parity for:

- D64 causal and non-causal;
- D128 causal and non-causal;
- a non-specialized D80 scalar fallback;
- vectorization-disabled D64 baseline;
- subgroup-priority selection where subgroup support exists.

The existing M3 qualification matrix remains active and therefore continues to cover the full portable dimension and sequence-boundary set.

## Reproducible timing harness

Run:

```bash
cargo run --release --features wgpu --example vector_bench
```

The harness compares Q4 scalar storage and Q4 vec4 storage with subgroup use disabled in both contexts. It reports median end-to-end latency for `B1 H2 N128 D64` causal attention, including upload, fused dispatch and readback.

No universal performance claim is derived from that example. Device, driver/backend, commit SHA and the exact measurement mode must accompany any published result.

## Merge gate

M6 may be merged only when the same final PR head SHA has:

1. `rust` — rustfmt, Clippy `-D warnings`, all-feature tests, including Naga validation of the vec4 shader;
2. `wgpu-device` — the full mandatory WGPU/lavapipe suite, including the new M6 device tests.
