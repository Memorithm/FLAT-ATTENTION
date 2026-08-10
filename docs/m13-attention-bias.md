# M13 additive attention-bias reference contract

This step starts M13 from the verified post-#21 `main` head without changing any qualified WGPU kernel.

## Scope

The deterministic scalar contract adds optional score bias to rectangular projection-layout RoPE + GQA/MQA attention. Bias is applied after Q·K scaling and before the online-softmax update.

Supported reference forms:

- no bias;
- dense additive bias with logical layout `[batch, q_heads, query_len, kv_len]`;
- ALiBi-compatible per-query-head slopes with explicit query and K/V absolute position origins.

The ALiBi path accepts slopes supplied by the caller. FLAT does not prescribe a model-specific slope-generation policy.

## Correctness rules

- causal masking remains authoritative and is evaluated before bias;
- dense bias cardinality must match the full logical rectangular score domain;
- bias values and ALiBi slopes must be finite;
- ALiBi position arithmetic must not overflow;
- `AttentionBias::None` must remain bitwise identical to the qualified M11 scalar oracle;
- ALiBi must match an independently materialized equivalent dense additive bias.

The scalar implementation streams over K/V and does not allocate a score or probability matrix. Dense bias itself is caller-owned input data; this reference contract does not imply that a production GPU path must materialize dense bias.

## GPU follow-up gate

This PR intentionally does not modify the qualified M11/M12/M15 WGPU paths. A subsequent M13 GPU slice must choose an explicit binding/encoding architecture, validate it with Naga and device parity, and preserve the existing caller-owned command-stream contract. Existing pipelines remain fallback/oracles until that path is independently qualified.

## Performance honesty

No latency, throughput, bandwidth, allocation, or memory-efficiency improvement is claimed here. This is a correctness/reference-contract step only.

## Sovereignty

The implementation remains Rust-native and introduces no project-authored C/C++, C ABI bridge, CUDA C++, `nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or mandatory vendor SDK.
