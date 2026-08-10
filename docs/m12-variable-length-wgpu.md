# M12 — portable variable-length WGPU qualification

This step lifts the M12 padded variable-length reference contract onto the caller-owned WGPU path without changing FLAT's four-storage-buffer portability envelope.

## Physical layout

The GPU keeps dense projection-layout buffers at padded extents:

```text
Q:   [batch, padded_q_len,  q_heads  * head_dim]
K/V: [batch, padded_kv_len, kv_heads * head_dim]
O:   [batch, padded_q_len,  q_heads  * head_dim]
LSE: [batch, q_heads, padded_q_len]
```

K/V remain physically stored at `kv_heads`; GQA/MQA never expand them to `q_heads`.

## Per-sequence metadata

Each batch element supplies:

- active query length;
- active KV length;
- causal query-position origin;
- query RoPE position origin.

The K/V RoPE origin is shared by the dispatch because the dense padded K/V rows use one physical key-position domain. A later KV-cache/page contract may widen this if caches with independent physical origins are batched together.

The portable path supports up to `WGSL_VARIABLE_MAX_BATCH = 256` entries per dispatch.

## Why metadata stays in the uniform

M11 already consumes the four minimum storage bindings needed by the fused contract:

1. Q;
2. K;
3. V;
4. packed O|LSE.

Adding a fifth storage buffer only for lengths would weaken compatibility with conservative/downlevel WGPU limits. M12 therefore packs the fixed-size metadata table into the existing uniform binding. The table occupies only a few KiB and leaves the four storage bindings unchanged.

## Device semantics

The kernel dispatches over physical padded query tiles but:

- never stages K/V rows at or beyond each sequence's active KV length;
- never uses padded Q rows in dot products;
- writes padded output rows explicitly as zero;
- writes padded LSE entries explicitly as negative infinity;
- applies causal masking using each sequence's absolute query-position origin;
- applies Q RoPE using each sequence's query RoPE origin;
- keeps FP32 score accumulation and online-softmax state;
- never materialises an attention score/probability matrix.

Host validation rejects zero active lengths, active lengths beyond physical extents, metadata cardinality mismatches, unsupported head dimensions, invalid RoPE parameters, index/dispatch overflows and causal-exclusive position overflow before encoding.

## Qualification

The mandatory WGPU tests compare mixed-length batches against independent compact M11 scalar projection-layout oracles, covering:

- causal GQA;
- non-causal MQA;
- different active Q/KV lengths in the same dispatch;
- different query causal/RoPE origins in the same dispatch;
- deliberately poisoned finite values in padded Q/K/V rows;
- exact padded-row contract (`O = 0`, `LSE = -∞`);
- malformed metadata and causal position overflow.

The WGSL is also parsed and validated directly by Naga in the normal Rust test gate.

## Performance honesty

This milestone qualifies correctness and the portable binding architecture. It does **not** claim a speedup over M11, dense SciRust attention, or any external implementation. Dense-padded versus packed/ragged performance remains a separate benchmark-driven decision.

No project-authored C/C++, C ABI bridge, CUDA C++, `nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or mandatory vendor SDK is introduced.
