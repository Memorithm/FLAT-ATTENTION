# M15 external resident KV decode

## Purpose

The specialized M15 `q_len = 1` kernel must be usable by frameworks that already own a fixed-capacity resident KV cache. Requiring callers to copy that cache into FLAT's `WgpuResidentKvCache` would duplicate memory and per-token traffic, defeating the resident zero-copy boundary.

This slice therefore adds a non-owning entry point to `WgpuResidentDecodePipeline` while preserving the existing M14-owned-cache API.

## Public entry point

```rust
WgpuResidentDecodePipeline::encode_external_pre_rotated_k(
    device,
    encoder,
    pass: ExternalAsymmetricProjectionPass<'_>,
    kv_capacity,
)
```

The method reuses the already-public `ExternalAsymmetricProjectionPass` rather than introducing another public buffer-view type.

## Storage contract

The external buffers are interpreted as:

- Q: `[batch, 1, q_heads * head_dim]`;
- K: `[batch, kv_capacity, kv_heads * head_dim]`;
- V: `[batch, kv_capacity, kv_heads * head_dim]`;
- O: `[batch, 1, q_heads * head_dim]` followed by LSE `[batch, q_heads]` in the same output allocation.

`pass.shape.kv_len` is the logical live prefix. `kv_capacity` is the physical per-batch K/V stride and may be larger than `kv_len`.

K is already RoPE-rotated. V remains raw. Q RoPE is fused by the M15 shader from `pass.rotary.query_position_offset`. `pass.rotary.kv_position_offset` is intentionally irrelevant to this pre-rotated-K path.

## Ownership, bindings and synchronization

FLAT does not own or allocate external Q/K/V buffers. The method does not:

- copy or compact the K/V prefix;
- allocate a shadow cache;
- submit the command encoder;
- poll or synchronize the device;
- map Q/K/V through the host.

The caller owns the device, command encoder, queue and all framework data buffers.

Each Q/K/V/O storage binding is explicitly range-limited to the validated tensor byte extent rather than binding the caller's entire backing allocation. This allows a valid FLAT tensor to live at the beginning of a larger over-allocated resident buffer without making the arena's total size the WGPU binding size. The required logical range is also checked against the device's `max_storage_buffer_binding_size` before bind-group creation, so an intrinsically oversized tensor fails with a typed `StorageBindingTooLarge` error instead of an uncaptured WGPU validation failure.

The existing `encode(ResidentDecodePass)` API is retained. Internally it is reduced to the same non-owning storage view over `WgpuResidentKvCache`, so both owned and external-cache paths execute the same shader and validation core.

## Specialized semantics

This is intentionally a dedicated decode path rather than a second generic rectangular API:

- `query_len` must equal 1;
- native MHA/GQA/MQA mapping is preserved; K/V are never expanded to query-head cardinality;
- `head_dim` must be positive, even and within the portable M15 limit;
- `0 < kv_len <= kv_capacity`;
- the physical K and V buffers must be large enough for the full capacity-strided layout;
- Q and O|LSE must satisfy the specialized output geometry;
- each logical storage range must fit the device storage-binding limit;
- theta must be finite and positive;
- index and dispatch limits are checked before encoding.

The shader streams every live KV row. Therefore, when `causal = true`, the single query's absolute logical position must be able to see the entire live prefix (`query_position_offset + 1 >= kv_len`). Standard autoregressive decode satisfies this with `query_position_offset = kv_len - 1`. Geometries that would require masking some live rows are rejected instead of silently producing a generic-attention result.

## Qualification

The mandatory external-resident WGPU test uses:

- batch 2;
- GQA (`q_heads = 4`, `kv_heads = 2`);
- `kv_len = 3` with physical `capacity = 7`;
- head dimension 32;
- independently pre-rotated K;
- large finite poison values in every inactive capacity row.

The specialized device output and LSE are compared with the deterministic asymmetric projection-layout scalar oracle. Poisoned padding proves that the kernel indexes batches by physical capacity while reading only the logical live prefix.

Additional tests reject non-`q_len=1`, causal visibility mismatch and `kv_len > kv_capacity` before dispatch.

## Rotation versus causal position domains

`ResidentDecodePass` carries two independent absolute positions:

- `q_rope_position` drives only the fused query rotation;
- `q_causal_position` is the sole input of the causal visibility precondition
  (`q_causal_position + 1 >= live_tokens` under `causal = true`).

This mirrors `AsymmetricGroupedAttentionShape::query_position_offset` versus
`AsymmetricRotaryEmbeddingConfig::query_position_offset`, so deployments with a
rotation origin shifted relative to the causal origin (cross-attention,
continued pretraining with an offset RoPE schedule) are expressible without
conflating the two domains. The pre-rotated-K external path keeps deriving its
causal domain from `shape.query_position_offset`.

## SciRust integration boundary

This entry point is specifically sufficient for SciRust to keep its existing `WgpuDenseKvCache` as the sole KV owner. A SciRust bridge can borrow the underlying fixed-capacity K/V buffers, pass the cache's logical length and capacity, and record the specialized FLAT kernel on the existing `GpuChain` device/queue.

No second FLAT-owned cache is required.

## Performance policy

This milestone proves the zero-copy interoperability architecture and correctness only. It makes no latency or throughput claim. Generic-vs-specialized and legacy-vs-FLAT promotion remain gated by paired measurements on the target physical adapter.
