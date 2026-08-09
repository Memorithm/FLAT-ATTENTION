# M7 — Double-buffered K/V staging

M7 adds an **opt-in** Q4 `vec4<f32>` forward kernel that uses two workgroup-memory banks for K/V staging. It is intentionally not selected by `WgpuFlatAttention::new()` until a physical-GPU benchmark demonstrates an improvement over the qualified M6 path.

## Scope

The M7 candidate keeps the existing FLAT-ATTENTION forward contract:

- fused `QK^T + online softmax + PV`;
- no materialized `N x N` score or probability matrix;
- four query rows per workgroup;
- FP32 score, softmax state and output accumulation;
- causal and non-causal execution;
- packed `[O | LSE]` output;
- D64/D128 `vec4<f32>` Q/K/V storage view inherited from M6.

Only K/V staging changes.

## Ping/pong layout

M6 uses one K/V workgroup tile of 8 rows. M7 divides the same logical storage capacity into two banks of 4 rows:

```text
K workgroup storage: 1024 f32
  bank 0: 4 rows × 128 dims = 512 f32
  bank 1: 4 rows × 128 dims = 512 f32

V workgroup storage: 1024 f32
  bank 0: 4 rows × 128 dims = 512 f32
  bank 1: 4 rows × 128 dims = 512 f32
```

Therefore the K/V workgroup allocation does not grow relative to the qualified M6 kernel.

The remaining workgroup arrays are unchanged:

- Q: `4 × 128 = 512 f32`;
- reduction scratch: `4 × 64 = 256 f32`;
- online-softmax state: four arrays of `4 f32`.

Total explicit workgroup storage is:

```text
Q             512 f32 = 2048 B
K            1024 f32 = 4096 B
V            1024 f32 = 4096 B
reduction     256 f32 = 1024 B
softmax state  16 f32 =   64 B
--------------------------------
total                    11328 B
```

This remains below the 16 KiB portable workgroup-storage floor used by the project.

## Barrier discipline

1. Q and the first K/V bank are seeded before the tiled loop and followed by `workgroupBarrier()`.
2. At each iteration, the next K/V tile is written into the **inactive** bank.
3. No barrier is required between those writes and reads from the current bank because the address ranges are disjoint.
4. The first reduction `workgroupBarrier()` executed while consuming the current bank is also a completion point for all prior inactive-bank workgroup writes.
5. Every later read of the prefetched bank occurs only after that barrier and after the current tile has completed.
6. The final accumulator update for each query row is followed by a workgroup barrier, preventing bank reuse from racing unfinished reads.

The implementation does not claim hardware asynchronous copy semantics. WGSL expresses an overlap-friendly software pipeline; whether a backend overlaps memory latency with arithmetic is a device/compiler property and must be measured.

## Selection policy

`WgpuFlatAttention::new()` keeps M7 disabled.

The experimental path is enabled only through:

```rust
WgpuFlatAttention::with_subgroup_vectorization_and_double_buffering(
    policy,
    true,
    true,
)
```

Selection priority remains:

1. M5 subgroup kernel, when explicitly/automatically selected;
2. M7 double-buffered vec4 kernel for D64/D128 when the M7 flag is enabled;
3. M6 vec4 kernel for D64/D128;
4. M4 scalar Q4 fallback.

## Qualification

The permanent tests cover:

- D64 and D128;
- causal and non-causal attention;
- sequence lengths crossing the 4-row ping/pong boundary;
- partial final K/V tiles;
- default-constructor non-selection;
- vectorization-disabled fallback;
- subgroup priority;
- Naga WGSL parse/validation.

The full pre-existing WGPU parity matrix remains mandatory in CI.

## Performance gate

`examples/double_buffer_bench.rs` compares M6 vec4 against the M7 candidate with subgroup use disabled in both contexts.

A software Vulkan/lavapipe result is useful for regression detection but is **not** accepted as proof that M7 is faster. M7 may become a default path only after a reproducible benchmark on at least one physical GPU reports a positive result together with device, driver/backend, commit SHA, shape and measurement mode.
