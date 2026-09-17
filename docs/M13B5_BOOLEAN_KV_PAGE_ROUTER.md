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
KVLab's BKV-K5 control contract merged at `7b2b9a07f0ad17e3f0942a9f920f633f7a50f483`.

These controls select logical page IDs only. They do not turn candidate density
into measured traffic, latency, quality, residency or speedup, and dense
numerical FLAT remains authoritative.

## Canonical page-selection evidence

`BooleanIndexedKvSelection::canonical_evidence_json()` emits the research schema
`flat.boolean-kv-selection.v1`. BKV-K6 qualification additionally uses the compatible `flat.boolean-kv-selection.v2` evidence surface, which retains the exact Hamming threshold used by the authoritative search alongside the v1 fields. V1 remains available for pre-existing consumers. It retains the exact logical/physical page IDs,
live-token counts, Hamming/XNOR accounting, exact numerical K+V bytes/token and
the already-computed Boolean/numerical byte accounting in deterministic field
order. Evidence export recomputes packed Boolean bytes as
`mapped_pages * ceil(signature_bits / 64) * 8`, recomputes full/selected/avoided
numerical totals from retained bytes/token geometry, and also revalidates
ordering, uniqueness and signature accounting so a mutated transparent research
record fails closed.

The trailing FNV-1a checksum is an accidental-corruption/content-identity aid for
KVLab ingestion; it is not cryptographic attestation. The envelope deliberately
does not contain a physical-DRAM, latency, model-quality or promotion field, so
a retained routing decision cannot be mistaken for a performance result.

## BKV-K6 measured-evidence binding

The research-only `flat.bikv-selection-binding.v1` envelope binds this exact
canonical page-selection decision to the canonical BKV-K6 qualification record.
The binding fails closed if the declared Hamming threshold, token/page counts,
Boolean bytes read, or numerical byte accounting drift from the selection that
was actually executed. This is provenance hardening only; it adds no latency,
quality, bandwidth, or promotion claim.

## Evidence boundary

The public integration test consumes these contracts through the actual
`flat_attention` crate export. Existing BKV-K6 code remains research-only and
separately responsible for sparse numerical decode correctness. M13B.5 is not
qualified for promotion until exact-head CI, false-negative/quality evidence and
real-device traffic/latency measurements satisfy the preregistered roadmap.
