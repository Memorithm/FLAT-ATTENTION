# BKV-K6.3 machine-readable evidence envelope

Status: research-only evidence contract. No runtime routing change and no performance claim.

## Purpose

BKV-K6.2 established an executable WGPU comparison between Boolean-selected K6 decode and dense M16 paged decode. Its logs are useful for inspection, but promotion decisions need a deterministic record that can be retained, compared and later ingested by an evidence-normalization layer without converting a synthetic microbenchmark into an end-to-end product claim.

`research_bkv_evidence.rs` therefore binds two existing contracts:

- the M40 `BenchmarkManifest` provenance schema for the BIKV candidate and dense M16 baseline;
- the BKV-K6 `BikvQualificationRecord` for Boolean-index and numerical-KV accounting plus diagnostic phase medians.

The candidate and dense manifests must use the exact same commit, device/backend/driver environment, attention problem, warmup count and measured-iteration count.

## Promotion latency semantics

The BKV-K6.2 review established an important distinction:

```text
phase_median_sum =
    median(signature_generation)
  + median(Boolean_search)
  + median(selected_attention_submit)
  + median(synchronization)
```

is a diagnostic and is **not** the median of an end-to-end candidate iteration.

The evidence envelope therefore makes the promotion/fallback decision from:

```text
candidate BenchmarkManifest.result.median_latency_ns
versus
dense BenchmarkManifest.result.median_latency_ns
```

while retaining the exact phase medians in the qualification record for diagnosis.

## Gates

Promotion remains fail-closed and requires all of:

```text
all_accept_K6_matches_M16
AND sparse_K6_matches_restricted_numerical_oracle
AND independent_quality_gate_passes
AND candidate_end_to_end_median < dense_end_to_end_median
```

A synthetic matched-density fixture may exercise the machinery, but a `promote` disposition from such a fixture is not sufficient to enable BKV-7 or make a production claim.

## Scope honesty

The evidence schema records whether Q and K/V are device-resident, whether a host Q mirror is retained, whether uploads/readbacks are excluded, and whether GPU timestamps, physical DRAM traffic, model quality or resident-only production are actually claimed.

A record that retains a host Q mirror while claiming resident-only production scope is rejected as contradictory.

## Deterministic serialization

`BikvEvidenceManifest::canonical_json()` emits schema-versioned deterministic JSON with:

- candidate and dense M40 manifests;
- signature policy and Hamming threshold;
- timing/residency claim scope;
- exact page/token/byte accounting;
- diagnostic phase medians;
- correctness and quality gates;
- promotion/fallback disposition;
- FNV-1a checksum for accidental-corruption/reproducibility detection.

The checksum is not a cryptographic authenticity primitive.

## Exact page-selection binding

The measured BKV-K6 envelope now has an additive fail-closed binding layer,
`flat.bikv-selection-binding.v1`. It embeds the exact canonical
`flat.boolean-kv-selection.v1` decision together with the canonical BKV-K6
qualification envelope, then rejects any drift in signature width, live or
selected token counts, page counts, Boolean bytes read, numerical bytes per
token, or full/selected/avoided numerical byte accounting.

The WGPU qualification harness constructs this binding from the actual page
selection used for the measured selected decode. `FLAT_BKV_SELECTION_BINDING_OUT`
may retain the resulting canonical envelope alongside `FLAT_BKV_EVIDENCE_OUT`.
Both trailing FNV-1a values are reproducibility/accidental-corruption aids, not
cryptographic attestation. This closes a provenance gap; it does not establish a
latency win, physical DRAM avoidance, or model-quality preservation.

## Non-claims

This milestone does not demonstrate that:

- BIKV is faster than M16 on representative hardware or workloads;
- rejected numerical K/V avoided physical DRAM traffic;
- synthetic Boolean signatures preserve model quality;
- the host-mirrored K6.2 harness is a resident-only production path;
- BKV-7 adaptive placement is ready.

Those conclusions require representative workload/model-quality evidence and reproducible real-device campaigns on the exact tested commit and configuration.
