# M13 — portable ALiBi WGPU path

This slice lifts the qualified M13 ALiBi reference semantics onto the existing caller-owned rectangular WGPU pipeline without increasing the storage-binding count.

## Binding contract

Q, K, V and packed O|LSE remain the same four storage buffers used by M11/M15. ALiBi slopes are small per-query-head model metadata, so up to 256 slopes are packed into the existing uniform parameter block as 64 `vec4<f32>` values. No Q/K/V buffer is copied or mapped by the encoder.

The score update is:

```text
score = scaled_qk + slope[q_head] * (absolute_key_position - absolute_query_position)
```

Query and KV bias position origins are independent from the causal-mask position origin and from the RoPE position origins.

Both raw-K and pre-rotated-resident-K entry points are qualified. Existing no-bias `encode` and `encode_pre_rotated_k` semantics remain selected with `bias_mode = 0`.

## Validation

The host rejects:

- slope cardinality different from `q_heads`;
- non-finite slopes;
- more than 256 query heads in the portable ALiBi uniform representation;
- query/KV bias position ranges that exceed the WGPU `u32` index domain.

Mandatory WGPU tests compare output and LSE against the deterministic M13 scalar oracle for raw K and independently pre-rotated K.

## Performance policy

This milestone makes no latency, throughput, bandwidth, or speedup claim. It adds one small uniform payload and no additional storage binding. Any performance claim requires a paired benchmark on a named adapter and driver.

## Sovereignty

The path remains Rust-native on the host and WGSL on the portable device side. It introduces no project-authored C/C++, C ABI bridge, CUDA C++, `nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or mandatory vendor SDK.
