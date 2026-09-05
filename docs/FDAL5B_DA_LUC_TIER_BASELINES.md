# FDAL5b — deterministic DA-LUC tier baseline qualification

Status: research-only qualification layer. This does not start FDAL6 and does not change production routing.

## Purpose

FDAL5 already assigns canonical logical KV segments to caller-declared precision tiers through deterministic recency and attention-mass policies. FDAL5b turns those assignments into reproducible host evidence before any semantic router is allowed to compete with them.

The implementation lives in `flat-da-luc-tier-qualification` and deliberately does **not** define a second representation codec. Every assigned segment is encoded by the existing FDAL1 `DalucOraclePayload::encode`, decoded by the same FDAL1 payload, and compared against the original logical dense K/V fixture.

## Controls

The qualification layer supports four controls:

- existing FDAL5 recency routing;
- existing FDAL5 attention-mass routing, including its stable lower-segment tie break;
- deterministic random assignment, versioned as `DA_LUC_RANDOM_CONTROL_VERSION = 1` and parameterized by an explicit `u64` seed;
- explicit fixed assignment, accepted only when every canonical segment is present exactly once and the declared tier quotas are preserved.

Random v1 uses SplitMix64 plus Fisher-Yates over canonical segment indices. It is only a reproducibility control; no statistical-optimality claim is attached to it.

## Materialization and logical reconstruction

For each assignment:

1. derive the exact segment bounds from the FDAL5 plan;
2. derive a segment FDAL0 contract by changing only `kv_len`, the assigned K/V representation descriptors, and the segment-local storage capacity/page count;
3. slice the original canonical `[batch, kv_heads, kv_len, feature]` K/V tensors;
4. call FDAL1 `DalucOraclePayload::encode` with the tier's explicit codebook;
5. validate and decode that exact payload;
6. place the decoded segment back into one canonical dense logical K/V comparator.

This preserves GQA/MQA head mapping, K/V representation asymmetry, distinct K/V feature widths, and final partial segments. Unsupported descriptor/payload combinations fail through FDAL0/FDAL1 validation rather than being repaired.

## Exact storage accounting

Each segment's FDAL1 `DalucOracleStorageReport` remains visible as evidence. The composite report includes:

- K codebook payload;
- packed K indices;
- K residual values and index/bitmap payload;
- V payload;
- V scales and zero points;
- V residual values and index/bitmap payload;
- page metadata;
- byte-tail packing padding;
- plane alignment padding;
- caller-declared shared metadata;
- caller-declared serialized segment metadata.

FDAL5b has one explicit shared-overhead rule: **one physical K codebook plane is charged once per used tier**. Segment materialization internally constructs FDAL1 payloads independently so FDAL1 remains the single codec/oracle, but duplicate per-segment codebook planes are removed from the composite scientific storage budget. Their payload bytes, alignment padding, and packing-tail padding are removed together. No other payload is silently shared.

`effective_bits_per_value` is computed from the resulting exact composite bytes over the full logical K+V scalar count. No nominal index-width compression claim is produced.

## Equal-budget rule

`qualify_equal_budget` requires both:

1. identical segment counts per exact tier id; and
2. identical `total_representation_bytes` after FDAL1 materialization and the shared-codebook rule.

The second check matters when `kv_len` ends in a partial segment. Two controls can have identical tier quotas but assign different representations to the shorter final segment, yielding different exact byte budgets. Such a comparison fails closed instead of being labeled equal-budget.

## Quality evidence

For every control FDAL5b reports:

- canonical segment assignment;
- exact effective bits/value;
- K reconstruction error;
- V reconstruction error;
- q_len=1 output error against the original dense scalar comparator;
- q_len=1 LSE error against the original dense scalar comparator.

The q_len=1 comparator preserves native GQA/MQA mapping and permits distinct K and V head dimensions. It uses the same online-softmax recurrence shape as the existing FDAL2 dense comparator, but compares the complete reconstructed tiered cache against the original dense fixture.

These are host/reference correctness metrics only. They are not model-quality, physical memory, bandwidth, latency, tokens/s, or real-device performance evidence.

## Fail-closed cases

FDAL5b rejects:

- malformed quotas;
- missing, duplicate, non-canonical, or unknown segment assignments;
- invalid attention-mass evidence through the existing FDAL5 validator;
- unsupported random-control versions;
- duplicate or missing tier materialization specs;
- codebook length/non-finite payload mismatches;
- invalid FDAL0 tier descriptors;
- FDAL1 payload validation failures;
- non-finite logical fixtures;
- unequal tier quotas or unequal exact byte budgets in a requested equal-budget comparison.

## Boundary to FDAL6

This layer is the non-semantic baseline qualification prerequisite. It does not authorize PHR-Lite or any learned/semantic router. FDAL6 remains blocked until these controls are reproducible and any later semantic router is compared against them under identical exact storage and declared quality workloads.
