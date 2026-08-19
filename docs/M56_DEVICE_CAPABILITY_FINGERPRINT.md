# M56 — Phase I device fingerprint foundation

M56 resumes the roadmap's Phase I autotuning work from verified `main` commit `f8178167410fa5442a23d233d31248544c64905d`.

The repository already exposes M29 runtime telemetry with device/driver provenance and an autotuner cache-status field, but that identity record previously had no deterministic serialization suitable for an autotuning cache key.

## This slice

`RuntimeDeviceFingerprint` now provides:

- deterministic canonical serialization with an explicit field order;
- delimiter escaping that preserves UTF-8 adapter/driver text;
- a stable FNV-1a-64 fingerprint over the canonical record;
- unit coverage proving repeatability and driver-sensitive invalidation.

The hash is a deterministic cache-key component only. It is not a cryptographic authenticity mechanism.

## M24 boundary

This is the first M24 foundation slice, not completion of the whole milestone. A subsequent Phase I slice must add the explicit resource/capability limits required by ROADMAP M24 (workgroup size/storage, binding limits, subgroup properties and f16 support) and use those limits to reject unsupported candidates before pipeline creation. M25 deterministic candidate generation and M26 benchmark-driven selection remain later work.

Adapter marketing names remain provenance only and must not become the sole optimization selector.

No WGSL, kernel routing, numerical behavior, `api::v1` contract, benchmark result or performance claim changes in this PR.
