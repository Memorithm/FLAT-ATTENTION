# BKV-K6 qualification gate

Status: research-only measurement contract. No runtime routing change and no performance claim.

## Purpose

BKV-K6 established a correctness-first Boolean-selected paged numerical decode. The next requirement is to decide whether a Boolean-indexed numerical KV (BIKV) path is actually worth promoting after all Boolean overheads are included.

This gate intentionally comes before BKV-7 adaptive tiering. Elastic placement/routing must not consume an optimization whose benefit has not been demonstrated under a declared quality and correctness gate.

## Exact accounting

`research_bkv_qualification.rs` records one qualification sample with exact integer inputs for:

- live numerical KV tokens;
- selected live tokens;
- mapped and selected page counts;
- page size;
- KV heads and head dimension;
- numerical scalar width;
- Boolean index bytes actually read;
- Boolean signature-generation latency;
- Boolean search latency;
- synchronization latency;
- selected numerical attention latency;
- dense numerical attention latency.
- non-zero measured candidate latency; an all-zero candidate timing is treated as missing evidence.

The record derives:

```text
kv_bytes_per_token = 2 * kv_heads * head_dim * scalar_bytes

dense_numerical_kv_bytes = live_tokens * kv_bytes_per_token
selected_numerical_kv_bytes = selected_live_tokens * kv_bytes_per_token
avoided_numerical_kv_bytes = dense_numerical_kv_bytes - selected_numerical_kv_bytes

total_bikv_latency =
    signature_generation
  + boolean_search
  + synchronization
  + selected_numerical_attention
```

The K/V byte values are logical payload-byte accounting. They are **not** physical DRAM-traffic claims. A physical traffic claim still requires hardware or backend evidence.

## Promotion decision

For a latency objective the candidate may be marked `Promote` only when:

```text
correctness_gate_passed
AND quality_gate_passed
AND total_bikv_latency < dense_attention_latency
```

Otherwise the result is an explicit fallback reason:

- correctness gate failed;
- quality gate failed;
- no end-to-end latency win.

An all-reject selection can be represented for control experiments, but it cannot become a successful candidate merely because its numerical work is zero; the independent quality gate must still pass.

## Required benchmark scope

The next WGPU qualification harness must feed this record from measured data and compare the BIKV K6 path against M16 paged numerical decode on the same resident buffers. It must report at least:

- exact commit and adapter identity;
- page size, context length, GQA/MQA geometry and selected density;
- signature-generation, Boolean search, synchronization and selected-attention timings separately;
- dense M16 timing;
- Boolean index bytes read;
- logical numerical K/V bytes touched and avoided;
- O/LSE correctness for all-accept and declared sparse fixtures;
- quality/recall gate for approximate selection policies;
- fallback disposition.

Uploads or readbacks may be excluded from the timing only when both paths operate on the same already-resident buffers and that scope is stated explicitly.

## Non-claims

This gate does not show that:

- rejected K/V avoided physical DRAM transactions;
- K6 is faster than M16;
- any approximate Boolean policy preserves model quality;
- BKV-7 adaptive tiering is ready for default use.

Those claims require measured evidence on the exact candidate and hardware configuration.
