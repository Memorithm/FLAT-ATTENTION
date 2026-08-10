# M11 — rectangular projection-layout WGPU

This qualification extends M11 from the scalar asymmetric contract to a real caller-owned WGPU dispatch while preserving the already-qualified equal-length FLAT-R2 path unchanged.

## GPU contract

The M11 pipeline consumes sequence-major projection buffers directly:

```text
Q:   [batch, query_len, q_heads * head_dim]
K/V: [batch, kv_len, kv_heads * head_dim]
O:   [batch, query_len, q_heads * head_dim]
LSE: [batch, q_heads, query_len]
```

`query_len` and `kv_len` are independent. K/V remain physically stored at `kv_heads`; there is no GQA/MQA head expansion.

The kernel keeps the Q4/KV8 fused online-softmax structure and never allocates a score or probability matrix. RoPE is fused into staged Q/K values.

## Position domains

Causal masking and RoPE position origins are deliberately independent:

- `AsymmetricGroupedAttentionShape::query_position_offset` selects which resident K/V rows a local query may see;
- `AsymmetricRotaryEmbeddingConfig::query_position_offset` selects the RoPE position of local query row zero;
- `AsymmetricRotaryEmbeddingConfig::kv_position_offset` selects the RoPE position of resident KV row zero.

This is required for decode. A one-row query at logical token `N-1` can attend a cache of `N` rows while the cache keeps its original rotary origin.

## Qualification cases

The device test matrix includes:

- causal GQA decode with `query_len = 1`, `kv_len = 17`, D64;
- non-causal rectangular MQA with `query_len = 5`, `kv_len = 9`, D80 and batch 2;
- caller-owned Q/K/V/output buffers and command encoder;
- explicit short-K buffer rejection;
- Naga/WGSL validation of the rectangular shader;
- CPU oracle parity for O and LSE.

The existing FLAT-R2 pipeline is not modified, so SciRust's current pinned integration remains stable while the new path is qualified.

## Performance honesty

This milestone proves the rectangular execution mechanism and numerical parity. It does not claim a latency or throughput improvement. Decode benchmarking and selection against the existing SciRust KV-cache path belong to the following integration/benchmark milestone.

No CUDA C/C++, `nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, vendor SDK, project-authored C/C++, or C ABI bridge is introduced.
