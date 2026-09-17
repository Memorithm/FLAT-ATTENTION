# M13B.5 Boolean KV page-router research API

Status: candidate research API; dense numerical FLAT remains the fallback authority.

This slice exposes the already merged Boolean-KV index and paged-selection
contracts through the crate's research API so experiments do not need to
path-import implementation files. It does not move the contracts into
`api::v1`, change default dispatch, or claim a speedup or quality result.

## Exposed contracts

- `api::boolean_kv`: versioned packed Boolean page signatures, append/reset
  generation semantics, exact storage accounting and deterministic Hamming
  search.
- `api::boolean_kv_paged_selection`: fail-closed mapping of admitted Boolean
  pages onto the authoritative `PagedKvTable`, preserving original logical page
  identity and exact live-token accounting.

The selection report retains Boolean key bytes read, full/selected/avoided
numerical K/V bytes, candidate density and
`numerical_KV_bytes_avoided / Boolean_KV_bytes_read`. These are logical/storage
accounting quantities, not DRAM, PCIe, cache-line or device-transfer
measurements.

The accounting boundary is deliberate. `BooleanKvCache::accounting()` reports
packed signature payload bytes: key payload plus any optional Boolean value-signature
payload. This excludes the Rust object/`Vec`/`Option` metadata, allocation capacity,
alignment and allocator overhead, so it is not a measurement of total host/device
residency.
The current CPU Hamming router scans every key signature before ranking and
applying an optional result limit, so `boolean_key_bytes_read` records the key
signature payload bytes inspected by that search and excludes optional value-signature
payload bytes. A result limit therefore reduces the numerical survivor set but does not
pretend that the current full-scan Boolean search read fewer keys. The regression
`bkv5_accounting_boundary` freezes this distinction. A future indexed search
that genuinely inspects fewer key signatures must update the implementation,
metric semantics and qualification evidence together rather than reusing a
packed-storage payload number as a read measurement.

## Invariants

A selection is rejected unless Boolean and numerical generations match, the
Boolean index has exactly one signature for every mapped numerical page, each
selected page has a live authoritative mapping, numerical geometry is non-zero,
and all accounting is representable. Boolean relevance order is never reused
as sequence order: the numerical handoff is restored to original logical-page
order.

## Matched-density controls

BKV-K5 now exposes selection-only matched-density controls through
`api::bkv5_matched_density_baselines`. The random-like control uses the
versioned `splitmix64-page-ranking-v1` rule and the positional control uses
`tail-window-v1`; both select exactly the Boolean candidate count. Full/paged
numerical controls retain every logical page. Shared reference vectors match
KVLab's BKV-K5 control contract.

These controls select logical page IDs only. They do not turn candidate density
into measured traffic, latency, quality, residency or speedup, and dense
numerical FLAT remains authoritative.

## Evidence boundary

The public integration test consumes these contracts through the actual
`flat_attention` crate export. Existing BKV-K6 code remains research-only and
separately responsible for sparse numerical decode correctness. M13B.5 is not
qualified for promotion until exact-head CI, false-negative/quality evidence and
real-device traffic/latency measurements satisfy the preregistered roadmap.
