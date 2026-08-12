# M40 — Reproducible benchmark manifests

M40 adds a dependency-free, machine-readable provenance contract for benchmark results. It does not change any attention kernel and does not promote any new performance claim.

## Schema v1

`BenchmarkManifest::canonical_json` emits deterministic JSON containing:

- schema version;
- exact 40-hex Git commit SHA;
- benchmark identifier and complete invocation command;
- device, WGPU backend, driver, operating system and architecture;
- precision, batch size, query/KV head counts, query/KV lengths, head dimension and causal mode;
- warm-up and measured iteration counts;
- median latency, p95 latency and effective token throughput;
- deterministic result checksum.

Latency is stored as integer nanoseconds and throughput as thousandths of a token per second. Avoiding floating-point JSON fields removes locale and display-rounding ambiguity from the canonical representation.

## Validation and checksums

A promotable record fails closed when its revision is not an exact 40-hex SHA, required provenance is empty, dimensions are zero, GQA grouping is invalid, measured iteration count is zero, or p95 is below the median.

The result checksum is FNV-1a-64 over the canonical result object. It detects accidental result corruption and makes repeated manifests directly comparable; it is explicitly **not** a cryptographic authenticity/signature mechanism.

Canonical JSON field order is part of schema version 1. Any future incompatible serialization change must increment the schema version rather than silently changing checksums.

## Benchmark integration rule

Existing M27/M28 and integration benchmarks may emit or archive this manifest once they have collected their measurements. A benchmark artifact intended to support a README/documentation performance statement must provide the exact commit, environment and command represented by the manifest. Software Vulkan timing remains qualification evidence only.

The manifest layer records evidence; it never fabricates a speedup, chooses a kernel, synchronizes the GPU or modifies the measured region.

## Sovereignty

The schema and checksum implementation are safe Rust using only the standard library. No C/C++, C ABI bridge, native serialization dependency, CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK is introduced.
