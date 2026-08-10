# M11 — asymmetric Q/KV contract

M11 removes the equal-length assumption from the mathematical GQA/MQA contract before the portable GPU kernel is widened to rectangular tiles.

## Contract

`AsymmetricGroupedAttentionShape` represents:

```text
Q:   [batch, q_heads, query_len, head_dim]
K/V: [batch, kv_heads, kv_len, head_dim]
O:   [batch, q_heads, query_len, head_dim]
LSE: [batch, q_heads, query_len]
```

K/V remain physically stored at `kv_heads`; no expansion to `q_heads` is permitted.

`query_position_offset` maps local query row `q` to absolute causal position `query_position_offset + q`. Key positions are absolute `[0, kv_len)`. This makes the decode contract explicit:

```text
query_len = 1
kv_len = cache_len
query_position_offset = cache_len - 1
```

The causal mask therefore admits exactly the same keys as the corresponding last row of a full equal-length causal forward.

## Numerical contract

`forward_reference_grouped_asymmetric` keeps the same scalar online-softmax update order as the existing M10 oracle. It does not allocate an attention score/probability matrix and does not materialise repeated K/V heads.

Qualification in this change proves:

- equal-length asymmetric calls are bitwise identical to the M10 oracle;
- single-query causal decode is bitwise identical to the last row of the full causal M10 oracle;
- large KV-cache shapes keep physical storage proportional to `kv_heads * kv_len`, not `q_heads * kv_len`;
- zero dimensions and invalid GQA grouping fail explicitly.

## What this does not claim yet

This change establishes the reference/public shape contract. It does **not** claim a GPU speedup and does not yet select a rectangular WGPU kernel. The next M11 step is to qualify a portable fused rectangular kernel and then extend the caller-owned R2/SciRust encoder so a resident `Q=1` decode can consume a long resident K/V cache directly.

No performance claim is valid until that device path is benchmarked under the repository benchmark policy.
