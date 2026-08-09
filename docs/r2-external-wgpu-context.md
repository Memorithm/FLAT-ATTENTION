# FLAT-R2 — caller-owned WGPU graph primitive

R2 is the integration boundary between FLAT-ATTENTION and SciRust.

It solves two independent integration costs discovered by auditing SciRust:

1. FLAT's standalone executors own their own WGPU device/queue, which is useful
   for qualification but unsuitable for a resident framework graph;
2. SciRust's Q/K/V GEMMs produce sequence-major projection matrices, while the
   canonical M10/R1 API uses head-major tensors.

R2 removes both costs.

## Direct SciRust projection layout

R2 consumes:

```text
Q   [batch, seq_len, q_heads  * head_dim]
K/V [batch, seq_len, kv_heads * head_dim]
```

and writes:

```text
O   [batch, seq_len, q_heads * head_dim]
LSE [batch, q_heads, seq_len]  // packed after O in the same buffer
```

This is intentionally the row-major layout already produced by SciRust's Q/K/V
projection GEMMs and expected by its output-projection GEMM.

Therefore the intended SciRust path is:

```text
x_norm
  ├── GEMM Wq ── Q projection buffer ─┐
  ├── GEMM Wk ── K projection buffer ─┼─ FLAT-R2 ─ O projection-layout buffer ─ GEMM Wo
  └── GEMM Wv ── V projection buffer ─┘
```

No head-major transpose is inserted. No per-head Q/K/V slice is created. No
per-head output placement/addition is required.

## Ownership contract

`ExternalProjectionRotaryGroupedPipeline` owns only a compiled
`wgpu::ComputePipeline`.

The caller owns:

- `wgpu::Device`;
- `wgpu::Queue`;
- Q/K/V buffers;
- O/LSE buffer;
- `wgpu::CommandEncoder`;
- the command submission boundary;
- synchronization and readback policy.

`encode` is deliberately forbidden by architecture from performing:

- `queue.submit`;
- `device.poll`;
- buffer mapping/readback;
- host copies of Q/K/V/O;
- creation of a second WGPU instance/device.

The only transient resource created by `encode` is the small uniform parameter
buffer and its bind group. Parameters are initialized through mapped-at-creation
memory, so a Queue is not required.

## Composability gate

The permanent WGPU test records two independent FLAT passes into one
caller-owned command encoder and performs one caller-controlled submission.
Both outputs are then checked against the projection-layout scalar oracle.

This catches accidental submission or synchronization ownership creeping back
into FLAT.

## Buffer safety

Before encoding, R2 validates:

- grouped-head divisibility;
- even RoPE head dimension;
- positive finite theta;
- position range;
- portable head-dimension maximum;
- WGPU u32 index space;
- dispatch limits;
- minimum physical byte size of Q/K/V/O buffers.

Buffers may be physically larger than the logical matrix. This is intentional:
SciRust can wrap the first `O` region as a `(batch*seq_len) × d_model` matrix
while retaining LSE in the tail for future backward/training support.

## Relation to R1

R1 remains the canonical standalone fused RoPE+GQA implementation and numerical
reference for head-major data.

R2 adds no new attention mathematics. It changes only:

- memory indexing to projection-major layout;
- ownership/submission boundaries.

A CPU gate converts between both layouts and requires bit-exact equality with
R1 after the layout transform.

## SciRust changes required after R2 qualification

The current SciRust WGPU types keep their resources private. Integration should
add only minimal `pub(crate)` accessors:

```text
WgpuContext::device() -> &wgpu::Device
WgpuContext::queue()  -> &wgpu::Queue
GpuMatrix::buffer()   -> &wgpu::Buffer
```

plus a crate-internal constructor for wrapping a caller-created buffer as a
`GpuMatrix` with explicit rows/cols.

The existing composed GQA implementation must remain available as the parity
oracle and fallback until FLAT is benchmarked on physical hardware.

## Performance honesty

R2 eliminates structural layout/slicing/submission overhead from the proposed
integration path. No runtime speedup is claimed by this document. The
production switch remains benchmark-gated on physical hardware.

No CUDA C++, nvcc, WMMA, CUTLASS, cuDNN, vendor SDK or project-authored C ABI is
introduced.
