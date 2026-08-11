# M27 benchmark harness

This document describes the currently qualified M27 benchmark-harness slices.

## Resident grouped-forward sweep

`examples/m27_resident_grouped_forward_sweep.rs` exercises the public `WgpuGroupedForwardPipeline` with caller-owned, already-resident Q/K/V/output buffers. It sweeps:

- MHA (`q_heads=4, kv_heads=4`), GQA (`4,2`) and MQA (`4,1`);
- sequence lengths 32, 128 and 512;
- head dimensions 32, 64 and 128;
- causal and non-causal attention;
- FP32 storage/compute under the existing grouped-forward contract.

For each case it evaluates `forward_reference_grouped`, executes the public WGPU path once, reads O/LSE back outside the timed region, and rejects the case unless parity is within explicit tolerances.

Each timing sample includes command-encoder creation, public pipeline encoding, queue submission, and blocking device completion. Input upload and output readback are excluded and labeled as such.

Run:

```bash
cargo run --release --features wgpu --example m27_resident_grouped_forward_sweep
```

## Resident decode sweep

`examples/m27_resident_decode_sweep.rs` adds the M27 decode matrix through the public specialized `WgpuResidentDecodePipeline`. It keeps the resident M14/M15 fixed-capacity cache contract and sweeps:

- MHA (`q_heads=4, kv_heads=4`), GQA (`4,2`) and MQA (`4,1`);
- live KV lengths 32, 128, 512 and 2048;
- head dimensions 32, 64, 80, 96 and 128;
- causal single-token decode (`query_len=1`, absolute query position `kv_len-1`);
- FP32 storage/compute under the existing resident-decode contract.

K is independently RoPE-rotated before it is appended to the resident cache, matching the M15 physical cache semantics. Before collecting timing samples for every problem case, the harness evaluates `forward_reference_projection_grouped_rope_asymmetric` from the original raw K fixture, performs one resident decode, reads O/LSE back outside the timed region, and requires numerical parity. A failed correctness gate prevents timing evidence from being accepted.

The cache allocation, input upload, K rotation, cache append, output readback and pipeline construction are outside the timed region. Every timing sample includes:

- command-encoder creation;
- `WgpuResidentDecodePipeline::encode`;
- queue submission;
- blocking `device.poll` completion.

The output records adapter name, backend, driver provenance, warm-up count, measured iteration count, median latency, p95 latency and single-token decode tokens/s.

Run:

```bash
cargo run --release --features wgpu --example m27_resident_decode_sweep
```

## Dispatch and allocation accounting

`examples/m27_dispatch_allocation_accounting.rs` adds deterministic source-contract accounting for the two timed resident paths. It deliberately does not pretend to be allocator telemetry. Instead, it records quantities that are exact from the current host implementation and logical tensor contract:

- grouped forward: caller-owned resident Q/K/V plus combined O/LSE, four resident storage buffers;
- resident decode: resident Q, source K/V staging buffers, K/V cache buffers and combined O/LSE, six live storage buffers in the current benchmark lifetime;
- one transient uniform GPU-buffer allocation per timed encode;
- 32 transient uniform bytes for grouped forward (eight `u32` parameters);
- 48 transient uniform bytes for resident decode (twelve `u32` parameters);
- one bind-group creation and one compute dispatch per timed encode;
- zero pipeline creations inside the timed region because both benchmark harnesses construct the pipeline before timing;
- zero materialized score/probability-matrix bytes under the fused attention contract.

The resident-decode source K/V buffers are currently kept alive after their contents have been appended into the fixed-capacity K/V cache, so their bytes are included in the live-storage accounting. These values are an accounting model tied to the source contract and benchmark object lifetimes, not measurements of driver-internal allocations, memory traffic, cache activity or physical bandwidth. The example prints `accounting_scope=source_contract_not_allocator_telemetry` and `performance_claim=none` to keep that distinction machine-readable.

Run:

```bash
cargo run --release --example m27_dispatch_allocation_accounting
```

## Current M27 coverage

These slices now cover resident prefill and resident decode timing, MHA/GQA/MQA, all roadmap head dimensions `D=32/64/80/96/128` across the combined harnesses, context lengths through 2048 on decode, explicit pipeline/dispatch-count reporting, caller-visible live storage accounting, timed transient uniform-buffer allocations/bytes, and the fused no-score-matrix intermediate-byte contract.

The complete M27 milestone still requires explicit cold-versus-warm pipeline measurements, resident-versus-upload/download measurements, longer-context expansion where device limits permit, and optional external power/energy hooks. Driver-internal allocator telemetry remains intentionally distinguished from source-contract accounting and must not be inferred from these counters.

## Performance policy

All M27 harnesses print `performance_claim=none`. Their output is evidence tied to the exact FLAT commit and, for device benchmarks, the adapter/driver on which it is executed; no universal speedup, latency, throughput, bandwidth, memory-efficiency or runtime-selection claim follows from the existence of the harnesses alone.

## Sovereignty

The benchmark system uses Rust host code and the existing WGPU/WGSL execution paths. It adds no project-authored C/C++, C ABI bridge, CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or mandatory vendor SDK.
