# FLAT-R1 — fused head-local RoPE + GQA/MQA

FLAT-R1 is an integration-driven extension discovered while auditing SciRust's
resident GQA path. It is deliberately outside the generic M1–M10 progression:
it removes a sequence of model-specific intermediate GPU operations that a
standalone attention benchmark does not expose.

## Problem observed in SciRust

SciRust's current resident GQA path is mathematically correct and keeps data in
VRAM, but it composes attention head-by-head:

1. apply head-local RoPE to Q;
2. apply head-local RoPE to K;
3. slice one Q head;
4. slice the corresponding grouped K and V head;
5. score GEMM `Q·K^T`;
6. scale / causal mask;
7. row softmax;
8. value GEMM `P·V`;
9. place the head result back into the model-width tensor;
10. add placed head outputs.

The important distinction is **residency versus fusion**. All of those buffers
may remain on device and still incur separate dispatches and materialize the
`t×t` score/probability representation.

## R1 execution

R1 accepts raw projected Q/K/V in native grouped layouts:

```text
Q   [batch, q_heads,  seq_len, head_dim]
K/V [batch, kv_heads, seq_len, head_dim]
```

and performs in one compute dispatch:

```text
stage raw Q rows
    ↓
head-local RoPE(Q) in q_shared
    ↓
for each physical KV tile:
    stage raw K/V once
    RoPE(K) in k_shared
    Q·K reduction
    online max / exp / normalization state
    weighted V accumulation
    ↓
write O + LSE
```

There is no global Q-RoPE tensor, no global K-RoPE tensor, no N×N score tensor,
and no N×N probability tensor.

## Exact SciRust RoPE convention

For interleaved pair `j` in a head of width `D`:

```text
freq_j = theta^(-2*j/D)
angle  = (token_position + position_offset) * freq_j
q'[2j]   = q[2j]   * cos(angle) - q[2j+1] * sin(angle)
q'[2j+1] = q[2j]   * sin(angle) + q[2j+1] * cos(angle)
```

K uses the same head-local frequency schedule. V is not rotated.

The scalar R1 oracle computes this pairwise inside each dot product and retains
M10's scalar accumulation order. A permanent test compares it bit-for-bit with
an independently materialized RoPE followed by the M10 grouped oracle.

## Why this is FLAT-specific

The optimization is not just "fuse RoPE". FLAT's tile is the unit of both
rotation and attention:

- Q rotation is paid once per staged Q tile;
- K rotation is paid once per staged K tile;
- rotated values never cross the workgroup/global-memory boundary;
- online softmax immediately consumes the score;
- GQA retains physical KV-head cardinality.

This makes positional transformation another part of the IO-aware attention
schedule rather than a preprocessing layer.

## Correctness gate

Permanent coverage includes:

- CPU bit-exact parity against materialized head-local RoPE + native GQA;
- MHA, GQA and MQA;
- causal and non-causal attention;
- D32, D64, D80 and D128;
- theta `10000` and `500000`;
- zero and non-zero position offsets;
- Naga WGSL validation;
- mandatory WGPU execution against the scalar fused oracle;
- resident-buffer cardinality checks.

## Performance honesty

R1 structurally removes intermediate tensors and dispatch stages. That is an
architectural property of the implementation, not a measured speedup claim.

No latency, bandwidth, token/s or energy improvement will be claimed until the
same workload is benchmarked reproducibly on physical hardware against the
current SciRust composed GQA path.

## Next integration step

After R1 is qualified in FLAT-ATTENTION, SciRust should not copy the shader. The
clean integration is an adapter at the resident WGPU ownership boundary:

```text
SciRust projected resident q/k/v
        │
        └── FLAT-R1 encode/dispatch using SciRust's existing device + queue
                │
                └── resident context tensor returned to GpuChain
```

That requires FLAT to accept externally-owned `wgpu::Device`, `wgpu::Queue` and
resident buffers. The standalone R1 executor remains useful for independent
qualification, but it is not the final SciRust integration API.

No CUDA C++, WMMA, vendor SDK or project-authored C ABI is introduced.
