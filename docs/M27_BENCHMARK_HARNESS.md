# M27 benchmark harness — resident grouped-forward sweep

This document describes the first M27 benchmark-harness slice implemented by `examples/m27_resident_grouped_forward_sweep.rs`.

## Scope

The harness exercises the public `WgpuGroupedForwardPipeline` with caller-owned, already-resident Q/K/V/output buffers. It sweeps:

- MHA (`q_heads=4, kv_heads=4`), GQA (`4,2`) and MQA (`4,1`);
- sequence lengths 32, 128 and 512;
- head dimensions 32, 64 and 128;
- causal and non-causal attention;
- FP32 storage/compute under the existing grouped-forward contract.

This is a prefill/resident timing slice. It does not yet implement the complete M27 matrix (decode, upload/download timing, long-context expansion, allocation/intermediate-byte accounting, or external power hooks).

## Correctness gate

Before collecting timing samples for each problem case, the harness:

1. evaluates `forward_reference_grouped` on the same deterministic fixture;
2. executes the public WGPU grouped-forward path once;
3. reads O/LSE back outside the timed region;
4. requires O and LSE parity within the same explicit tolerances used by the qualified grouped-forward device tests.

A case that fails parity panics before benchmark samples are accepted.

## Timing boundary

Inputs are uploaded once before timing. Output readback is also outside the timed region. Each sample includes:

- command-encoder creation;
- public `WgpuGroupedForwardPipeline::encode`;
- queue submission;
- blocking `device.poll` completion.

The harness reports warm-up count, measured iteration count, adapter name, backend, driver provenance, median latency, p95 latency, and logical query tokens/s.

## Performance policy

The harness prints `performance_claim=none`. Its output is evidence tied to the adapter/driver on which it is executed; it is not a universal speedup or throughput claim and does not change runtime selection policy.

## Sovereignty

The benchmark uses the existing Rust host API plus WGPU/WGSL execution path. It adds no project-authored C/C++, C ABI bridge, CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or mandatory vendor SDK.

## Run

```bash
cargo run --release --features wgpu --example m27_resident_grouped_forward_sweep
```
