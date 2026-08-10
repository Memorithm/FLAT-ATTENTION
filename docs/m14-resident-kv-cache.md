# M14 — resident KV cache contract

M14 introduces a fixed-capacity K/V cache owned by FLAT's portable WGPU layer. The cache is intended for inference paths where projected K/V already live on the same device.

## Physical layout

Each tensor is allocated once with sequence-major projection layout:

```text
K/V: [batch, capacity, kv_heads * head_dim]
```

`len` is the number of live sequence rows. Native `kv_heads` storage is preserved, so GQA and MQA do not replicate K/V to query-head cardinality.

## Append contract

`record_append` consumes caller-owned resident K/V source buffers laid out as:

```text
[batch, append_len, kv_heads * head_dim]
```

and records only device-to-device `copy_buffer_to_buffer` commands into the caller-owned command encoder. For each batch element, only the newly appended rows are written into their final capacity-strided location. Existing cached rows are never copied, compacted, mapped, or uploaded again.

The cache itself does not submit, poll, map, or synchronize. The caller remains responsible for submitting the encoder before consuming the advanced logical length.

## Reset/reuse

`reset()` changes only `len`. Old bytes may remain resident but are outside the live prefix. New appends overwrite reused rows before they become live again. This avoids a device clear or whole-cache copy on sequence reset.

## Failure behavior

The host rejects zero dimensions/appends, arithmetic overflow, cache-capacity overflow, undersized append buffers, and allocations larger than the device's reported maximum buffer size.

## Decode follow-up

The physical batch stride is `capacity`, not the current `len`. The next M15 integration slice must therefore pass the resident cache stride explicitly to the decode kernel rather than pretending a fixed-capacity multi-batch cache is tightly packed at its current length.

## Performance policy

M14 makes no latency, bandwidth, tokens/s, or speedup claim. The architectural property established here is narrower: appending a token never copies the existing K/V prefix and requires no host round-trip. Decode performance remains benchmark-gated.

## Sovereignty

The host implementation is Rust and the device operation is standard WGPU buffer copy. No project-authored C/C++, C ABI bridge, CUDA C++, `nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or mandatory vendor SDK is introduced.
