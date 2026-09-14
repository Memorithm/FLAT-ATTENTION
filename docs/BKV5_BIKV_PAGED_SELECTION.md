# BKV-K5 — Boolean-indexed numerical KV paged handoff

Status: research-only correctness contract; no production routing or speed claim.

## Motivation

KVLab BKV-K4 qualified a dual-socket CPU Boolean search architecture on the Dell T430: replicate the compact Boolean query, scan distinct socket-local Boolean-KV shards, then merge global candidate page IDs deterministically. The exact T430 distinct-shard campaign at KVLab commit `7e02edc1e3ac3b53cfb352cfad560db439755a0f` produced the same 2,283 global candidate IDs as the monolithic oracle and measured a 1.245513× speedup over the 32-physical-core interleaved monolithic comparator for that frozen workload.

BKV-K5 does **not** infer an end-to-end attention gain from that result. Its next responsibility is to hand Boolean candidate pages to authoritative numerical K/V without losing page identity, generation, sequence position, or exact byte accounting.

## Architecture boundary

This slice implements BIKV, not NBKV:

```text
Boolean page signatures
        |
        v
exact Boolean candidate search
        |
        v
logical page IDs + physical page mappings
        |
        v
authoritative numerical K/V pages
```

The Boolean plane selects. Numerical K/V remains the data used by exact FLAT attention.

## Required invariants

The host selection contract fails closed unless:

- Boolean KV generation equals paged numerical KV generation;
- there is exactly one Boolean signature per mapped numerical page;
- every selected logical page is still mapped and live;
- numerical geometry (`kv_heads`, `head_dim`, scalar byte width) is non-zero;
- all byte accounting fits in `usize`;
- selected pages retain original logical page identity.

Boolean search may rank candidates by Hamming distance. Before numerical handoff, selected pages are sorted by original logical page. This prevents a relevance ranking from silently becoming a new sequence-position convention.

## Traffic accounting

The contract reports exactly:

- Boolean pages scanned;
- packed Boolean key bytes read by the exhaustive Boolean search;
- full numerical K/V bytes represented by the live paged cache;
- numerical K/V bytes represented by selected pages, including the exact live-token count of the final partial page;
- numerical K/V bytes avoided;
- candidate-page density;
- `numerical_KV_bytes_avoided / Boolean_KV_bytes_read`.

These are logical/storage-accounting quantities. They are not automatically DRAM, PCIe, cache-line, or GPU-transfer measurements.

## Why this slice does not modify M16 paged decode yet

The current M16 portable paged decode contract represents a contiguous logical KV sequence. Replacing its table with a sparse list of selected pages would implicitly renumber positions unless the numerical consumer also receives explicit original-position metadata.

Therefore this BKV-K5 slice deliberately stops at a portable host handoff. A later sparse-page numerical consumer must preserve original token positions (for RoPE/causal semantics) and qualify its output/LSE against a declared dense target before any runtime promotion.

## Next qualification

The next FLAT slice should consume `BooleanIndexedKvSelection` with one of two explicit mechanisms:

1. sparse paged decode with original logical-position metadata; or
2. deterministic survivor compaction that carries original positions alongside compacted K/V.

Required comparisons remain:

- full numerical paged KV;
- random matched-density page selection;
- structural/positional matched-density selection;
- Boolean page selection + exact numerical FLAT.

Required measurements include Boolean selection latency, candidate density, Boolean metadata bytes, numerical K/V bytes selected/avoided, staging/transfer/synchronization, first-token latency, TPOT, output/LSE error and quality/recall gates.

Dense numerical FLAT remains the fallback authority until those gates pass.