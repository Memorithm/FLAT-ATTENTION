# M7 — Double-buffered K/V staging

M7 introduces an experimental D64/D128 Q4 vec4 kernel with two workgroup-memory K/V banks. It is intentionally **not selected by `WgpuFlatAttention::new()`** until benchmark evidence on a physical GPU demonstrates an improvement over M6.

## Bank geometry and workgroup memory

M6 uses one K tile and one V tile of eight rows:

```text
1 bank × 8 rows × 128 f32 = 1024 f32 per tensor
```

M7 uses two banks of four rows:

```text
2 banks × 4 rows × 128 f32 = 1024 f32 per tensor
```

Therefore K and V consume the same maximum workgroup storage as M6 despite the ping/pong structure.

At the maximum D128 specialization, the statically declared workgroup arrays are:

- Q: `512 f32` = 2048 bytes;
- K ping/pong: `1024 f32` = 4096 bytes;
- V ping/pong: `1024 f32` = 4096 bytes;
- reduction scratch: `256 f32` = 1024 bytes;
- online-softmax state: 4 arrays × 4 f32 = 64 bytes.

Total declared algorithm workgroup state: **11,328 bytes**, below the 16 KiB conservative workgroup-storage floor used by the WGPU context. This excludes implementation bookkeeping outside these WGSL workgroup variables.

## Pipeline discipline

1. Q4 is staged exactly as in M6.
2. K/V tile 0 is loaded into bank 0 and synchronized.
3. For each current tile, the next tile is written into the inactive bank.
4. No dedicated barrier is emitted immediately after the inactive-bank prefetch because current computation reads the disjoint active bank.
5. The first workgroup reduction barrier for current-tile computation is also a completion point for all prior inactive-bank writes.
6. The current tile completes with the same explicit reduction/online-softmax barriers as the qualified Q4 path.
7. Banks swap and the prefetched bank becomes active.

WGSL 0.20 does not expose an explicit asynchronous workgroup-copy primitive. Therefore this structure is described as **overlap-friendly software pipelining**, not as a guarantee of asynchronous copy/compute overlap. Physical scheduling is backend/device dependent.

## Selection

The explicit constructor is:

```text
with_subgroup_vectorization_and_double_buffering(subgroup, vectorization, double_buffering)
```

Selection priority is:

1. `Q4Subgroup` when subgroup was selected;
2. `Q4Vec4DoubleBuffered` for D64/D128 when vectorization and double buffering are both enabled;
3. `Q4Vec4Portable` for D64/D128 with vectorization only;
4. `Q4Portable` fallback.

Default construction sets `double_buffering = false`.

## Qualification

The mandatory WGPU device suite validates:

- D64 causal and non-causal across K/V bank boundaries;
- D128 with a partial final four-row tile;
- explicit opt-in selection;
- vectorization-disabled fallback;
- subgroup-priority regression;
- all pre-existing M3–M6 tests.

The new shader is also parsed and validated by Naga.

## Performance gate

Run:

```bash
cargo run --release --features wgpu --example double_buffer_bench
```

The harness compares M6 vec4 against M7 double buffering with subgroup disabled for both contexts. It reports end-to-end median latency for `B1 H2 N128 D64` causal attention.

A software Vulkan result is not accepted as evidence that M7 should become the default. The roadmap requires a measured improvement on at least one physical GPU before default selection. Until such evidence exists, M7 remains an opt-in experimental kernel even if all correctness CI is green.
