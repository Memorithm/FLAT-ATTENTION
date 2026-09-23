# MAA-14d Gate A2 — correctness-first structural WGPU carrier

Status: implementation candidate. Correctness/parity only; no timing or speedup claim.

## Input contract

Gate A2 consumes only the exact u32 CSR representation qualified by Gate A1:

- one offsets entry per query-row boundary;
- canonical ascending key IDs per row;
- exact seq_len/query_rows geometry;
- no page/block/density surrogate.

The uploaded candidate buffers are immutable for one dispatch and retain COPY_SRC
only so qualification can read them back and prove exact identity.

## Shader contract

The first carrier is deliberately simple:

- old self-attention AttentionShape only;
- f32;
- one WGPU invocation per output coordinate;
- exact CSR candidate traversal in ascending key order;
- stable online softmax with O and LSE;
- optional causal filtering;
- no N x N score/probability matrix;
- one row-status word per query row.

This implementation is not expected to be fast. Its role is to prove the
portable numerical/device semantics before any optimized segmented kernel.

## Empty-row boundary

A device row with no effective candidate writes status=0. The host must pass the
complete status readback through validate_structural_sparse_row_status() before
accepting O/LSE.

A zero status or non-binary status fails closed. The zero-valued shader output
for such a row is diagnostic transport only and is never a valid attention
result.

## Qualification tests

The Gate-A2 integration tests require:

1. arbitrary sparse multi-head candidate identity read back byte-for-byte and
   O/LSE parity against forward_reference_structural_sparse;
2. causal sparse parity against the same host oracle;
3. all-accept device execution matching dense forward_reference;
4. an intentionally empty causal row producing status=0 while the host oracle
   returns EmptyEffectiveCandidates;
5. malformed row-status readback rejection.

Software Vulkan may establish this correctness/plumbing evidence. It is not
physical performance evidence.

## Promotion boundary

Only after Gate A2 is green on exact-head WGPU qualification may MAA-14d add an
optimized carrier or physical benchmark harness. Timing from this correctness
kernel is not a performance claim and must not influence the frozen 7/8 routing
threshold.
