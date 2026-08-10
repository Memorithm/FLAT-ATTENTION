# M12 — variable-length padded batches

M12 removes the assumption that every sequence in a batch has the same active query and KV length while preserving a dense padded storage contract.

## Reference contract

`AsymmetricGroupedAttentionShape::forward_reference_variable_lengths` interprets `query_len` and `kv_len` as physical padded extents. The caller supplies one tuple per batch element:

```text
(active_query_len, active_kv_len, query_position_offset)
```

Logical storage remains:

```text
Q:   [batch, q_heads, padded_query_len, head_dim]
K/V: [batch, kv_heads, padded_kv_len, head_dim]
O:   [batch, q_heads, padded_query_len, head_dim]
LSE: [batch, q_heads, padded_query_len]
```

Only active prefixes participate in attention. Padded K/V rows are mathematically invisible. Padded query rows produce zero output and `LSE = -∞`.

## GQA/MQA and causal semantics

K/V remain physically stored at `kv_heads`; no expansion to `q_heads` is introduced. In causal mode, local query row `q` in batch element `b` has absolute position:

```text
query_position_offset[b] + q
```

and may only observe active key rows at or before that position.

## Numerical and memory policy

The implementation uses the same scalar FP32 online-softmax update order as the qualified M11 oracle and never allocates a score or probability matrix. This milestone is a mathematical/reference qualification only; it makes no GPU performance claim.

Permanent tests prove:

- a mixed-length padded batch is bitwise identical, on every active row, to independent M11 asymmetric calls;
- poisoned finite values in padded Q/K/V regions cannot influence active outputs;
- padded output/LSE semantics are deterministic;
- malformed metadata, zero active lengths and active lengths larger than the physical padded extent are rejected explicitly.

## GPU follow-up

The next M12 step is to pass per-sequence active lengths to the portable rectangular WGPU kernel without host-side score construction. Only after device parity is established should a packed/ragged representation be evaluated against the dense padded form.

No project-authored C/C++, C ABI bridge, CUDA C++, `nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or mandatory vendor SDK is introduced.
