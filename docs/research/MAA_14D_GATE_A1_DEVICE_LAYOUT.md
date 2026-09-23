# MAA-14d Gate A1 — exact structural device layout

Status: implementation candidate. No WGPU shader execution and no timing claim.

## Purpose

Gate A1 converts the canonical host StructuralCandidateSet into an exact portable
u32 CSR representation before any shader is allowed to consume it.

The representation contains:

- seq_len as u32;
- query_rows as u32;
- offsets[query_rows + 1];
- key_positions[admitted_pairs].

Each row retains the exact canonical ascending key order from the host oracle.
No page/block approximation or density-only surrogate is accepted.

## Invariants

- host->device conversion fails when any index exceeds WGSL u32 space;
- offsets start at zero, are non-decreasing, and terminate at key_positions.len();
- keys are strictly increasing per row;
- every key is < seq_len;
- round-trip reconstruction against an explicit AttentionShape reproduces the
  exact StructuralCandidateSet;
- causal-intersected candidate sets round-trip unchanged;
- all-accept candidate sets round-trip unchanged.

## Accounting

The plan reports offsets bytes, key-position bytes and total device metadata
bytes exactly. canonical_bytes() is little-endian and includes schema/geometry.
fingerprint_fnv1a64() is a deterministic structural fingerprint only, not
cryptographic attestation.

## Boundary

This slice does not:

- create WGPU buffers;
- compile WGSL;
- run sparse numerical attention on a device;
- claim fewer physical reads;
- claim latency/bandwidth/energy improvement.

The next Gate-A slice may create a WGPU sparse numerical carrier only by consuming
this exact representation or proving an equivalent representation candidate by
candidate.
