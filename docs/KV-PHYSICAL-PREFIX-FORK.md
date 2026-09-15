# M16 physical KV prefix fork

## Purpose

`WgpuPagedKvCache::fork_prefix_and_submit` creates an independently allocated copy of a live numerical K/V prefix. It supplies the preserved control state needed before destructive experimental interventions. It does not apply a DA-LUC transition plan or select a compression policy.

Unlike a logical checkpoint, the child owns different K/V buffers. Reusing or dropping the parent cannot overwrite the submitted child copies through the managed cache API. This is an eager copy, **not copy-on-write or prefix sharing**.

## API and example

```rust,no_run
use flat_attention::paged_kv::{PagedKvConfig, WgpuPagedKvCache, WgpuPagedKvCacheError};

fn preserve_control(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &mut WgpuPagedKvCache,
) -> Result<WgpuPagedKvCache, WgpuPagedKvCacheError> {
    let (control, _submission) = source.fork_prefix_and_submit(
        device,
        queue,
        source.len(),
        source.config(),
    )?;
    source.reset()?;
    Ok(control)
}
```

The destination geometry is explicit. A different page size is supported; the implementation resolves both page tables and splits each device-to-device copy at both source and destination page boundaries. Original logical positions and already-rotated K values are preserved byte-for-byte; no RoPE correction or positional rebasing is performed. GQA/MQA KV heads are not expanded.

For example, a seven-token prefix can be copied from four-token pages into three-token pages with `PagedKvConfig { page_size: 3, physical_pages: 3 }`. The child has seven live rows and capacity for nine. The last two rows are not copied from the source. Passing zero prefix length creates an independent empty cache; its destination configuration must still declare nonzero capacity.

## Lifecycle and ordering

The new instance begins at generation zero and branch epoch zero with a fresh checkpoint token. A parent's checkpoint is foreign to the child and vice versa. Parent metadata and its checkpoint validity are unchanged by a successful fork.

The source must use the managed append path. A cache tainted by externally recorded appends is rejected, including after the caller independently submits the encoder: the cache does not guess that all external writes have been consumed.

The supplied device, queue and buffers must belong to one WGPU device. Source writes must already have been submitted to that queue. The fork owns its encoder through submission, so no delayed fork command buffer can escape into the caller. Subsequent managed appends or decode work on the same queue are ordered after the fork without a host wait. The returned submission index does not imply device completion.

Direct writes using raw buffer handles are outside this managed lifecycle contract. The caller must order them explicitly and avoid concurrent writes. WGPU validation, out-of-memory and device-loss errors retain the existing WGPU error model; this API is not a recovery or memory-attestation mechanism.

## Bounds and cost

Before destination allocation, a prefix beyond the current live source, insufficient destination capacity and invalid/overflowing page geometry are rejected. The cache constructor checks tensor byte arithmetic and the device's declared buffer-size limit. The source is never mutated. On a returned error no fork encoder is submitted.

For `H` KV heads, head dimension `D`, prefix length `T` and destination capacity `C`, f32 storage requests are:

- destination K allocation: `4 * C * H * D` bytes;
- destination V allocation: `4 * C * H * D` bytes;
- total bytes addressed by K/V copy commands: `8 * T * H * D`.

These are allocation-request/copy-span counts, not physical DRAM traffic, peak HBM residency or latency measurements. The child requires additional storage. Buffer creation/initialization costs and driver overhead are not represented by the copy-span formula. No memory-saving or throughput claim follows.

## Qualification

`tests/m16_paged_kv_fork.rs` exercises exact payload copying across page sizes and head geometries, empty and partial prefixes, stale-tail exclusion, special f32 bit patterns, bidirectional mutation isolation, nested forks surviving dropped parents, new checkpoint identity, invalid bounds/capacity/device limits, and external-write rejection.

A separate end-to-end test consumes a fork with the existing paged M16 kernel for MHA/GQA/MQA after overwriting the source. Both O and LSE are compared with a scalar softmax calculation using the preserved vectors. That test uses query RoPE position zero; it is not a new qualification of arbitrary positional transformations.

The dedicated `M16 physical KV prefix fork qualification` workflow requires a Vulkan adapter and executes these tests, doctests, strict Clippy, strict rustdoc and rustfmt on the exact PR head. Mesa software Vulkan qualifies functionality, not physical GPU performance. Tests upload/read back fixtures for verification; production fork has no host payload round-trip.

## KVLab and remaining work

KVLab's tiering/replay protocol can use an independently preserved prefix as a control for future paired interventions without copying FLAT's allocator or page-table code into the lab. A fork is only the per-layer K/V portion of state: a complete model replay also needs every layer, model/tokenizer revisions, token positions, sampling/RNG state and any model-specific recurrent state.

Next gates remain explicit: integrate preserved K/V with a bounded representation executor, establish materialization/codebook identity, measure conversion cost and numerical drift, then qualify an actual model experiment. Device offload, L2 policy, copy-on-write sharing, arbitrary token surgery and cross-model transfer are separate capabilities and are not implemented by this fork.
