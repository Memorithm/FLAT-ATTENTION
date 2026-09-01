# FDAL2 — DA-LUC compressed q_len=1 decode oracle

FDAL2 is a research-only scalar correctness layer above the versioned FDAL0 view contract and the deterministic FDAL1 representation payload. It does not change FLAT's stable `api::v1`, routing defaults, WGPU kernels, NNIS integration, or SLHAv2 ownership.

## Direct-consumption property

`DalucOraclePayload::q_len1_attention_direct` consumes the validated FDAL1 payload without calling `decode_keys()` or `decode_values()` and without constructing a full dense K or V cache.

For each `(batch, q_head)` it:

1. maps the query head to its native KV head using the FDAL0 GQA/MQA grouping;
2. converts the stored K codebook to scalar reference values and builds a query-to-codebook LUT per K subspace;
3. reads each packed K index, accumulates the selected LUT entries, and adds sparse K residual dot-product corrections;
4. performs the same online-softmax recurrence used by FLAT's scalar attention semantics;
5. reads V scalars directly from either the dense stored plane or the groupwise-affine packed representation, including scale/zero-point conversion;
6. adds the sparse V correction for that coordinate and accumulates the weighted scalar into the output.

The scratch state is one query-local LUT plus sparse residual entries for the current KV row. This is not a dense KV materialization.

The implementation still converts stored codebook, scale, residual, and low-bit V values into scalar `f32` arithmetic. Therefore **FDAL2 does not claim zero dequantization**.

## Dense semantic comparator

`q_len1_attention_dense_reference` intentionally reconstructs K and V through FDAL1 first, then evaluates the q_len=1 mathematical attention loop. It exists only as the semantic comparator for the direct path.

`q_len1_attention_equivalence_report` reports output and LSE error between these two paths. The payload is identical in both cases; any reported difference is floating-point accumulation-order error caused by the LUT partitioning, not representation/quantization error against the original pre-encode tensors.

The FDAL1 reconstruction report remains the place to measure representation error against original dense K/V.

## Geometry and masking

The query layout is:

```text
Q: [batch, q_heads, key_head_dim]
```

The output layout is:

```text
O:   [batch, q_heads, value_head_dim]
LSE: [batch, q_heads]
```

`key_head_dim` and `value_head_dim` may differ. Native GQA/MQA mapping is `kv_head = q_head / (q_heads / kv_heads)`.

`DalucQlen1DecodeConfig::query_position` is the absolute position of the single query. In causal mode, keys with `key_position > query_position` are not consumed. `for_last_token` selects the conventional autoregressive position `kv_len - 1`.

## Structural trace

The direct output carries `DalucQlen1DecodeTrace` with deterministic logical counts for LUT entry dot products, K index lookups, sparse corrections, attended rows, V scalar reads, and low-bit scalar conversions.

These counters are **not** physical DRAM transactions, cache behavior, bandwidth, latency, throughput, or performance evidence.

## Reproducible host gates

```bash
cargo test --test fdal2_da_luc_decode_oracle -- --nocapture
cargo run --release --example fdal2_da_luc_decode_sweep
```

The dedicated workflow is also manually invokable after merge:

```bash
gh workflow run fdal2-da-luc-decode.yml --ref main
```

It runs the public integration contract and executes the deterministic sweep twice, requiring byte-for-byte identical output.

## Promotion boundary

FDAL2 proves scalar compressed-consumption semantics only. It makes no DA-LUC latency, throughput, bandwidth, memory-reduction, model-quality, novelty, or production-routing claim.

The DA-LUC-specific physical-Thor trial remains deferred to FDAL3, where a portable/open-codegen direct-compressed GPU candidate must first exist and pass scalar-oracle parity before any performance evidence is considered.
