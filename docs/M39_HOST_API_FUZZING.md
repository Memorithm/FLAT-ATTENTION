# M39 — Host API/config fuzzing

M39 treats every untrusted host-side geometry/configuration as data that must fail closed rather than panic, overflow, or trigger an implicit execution fallback.

## Standalone FLAT fuzz surface

`m39_host_api_fuzz` drives the stable backend-neutral `api::v1` contract with a deterministic 4,096-case corpus containing zero dimensions, valid dimensions, `usize` overflow edges, invalid head grouping, position overflow, invalid softmax scales, inconsistent Q/K/V lengths, and non-finite tensor values.

Each case exercises shape validation, checked element counts, group-size conversion, stable-to-core conversion, borrowed/owned request validation, and resident-contract validation inside `catch_unwind`. Every arbitrary input must return a defined `Ok`/`Err`; a panic is a test failure.

Explicit regression cases additionally cover shape/position overflow, each Q/K/V length mismatch class, and NaN/+Inf/-Inf in every input tensor class.

The corpus is fixed-seed and dependency-free so failures are reproducible under normal `cargo test` without a native fuzzing runtime or FFI.

## Tuning-cache and serialized-kernel ownership

Standalone FLAT currently exposes passive autotuner-cache telemetry but does **not** parse a persisted tuning cache or serialized executable kernel metadata. The persisted ElasticAutoTuner plan/cache is owned by the SciRust integration layer. Fabricating a second FLAT cache format solely to satisfy a fuzz target would create a competing source of truth.

Therefore M39's standalone gate fuzzes every serialized/untrusted input that FLAT actually owns today. If a persisted FLAT cache or serialized kernel-metadata parser is introduced later, its malformed/corrupt corpus becomes a mandatory extension of this gate before merge. SciRust's existing persisted-plan validation remains responsible for its own cache corruption tests.

## Sovereignty

The fuzz corpus is ordinary safe Rust. It adds no libFuzzer C ABI, project-authored C/C++, C bridge, mandatory CUDA toolchain, or vendor SDK.
