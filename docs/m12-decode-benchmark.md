# M12 — decode benchmark protocol

M11 established numerical/device qualification for rectangular caller-owned WGPU attention. M12 starts the performance-qualification phase without changing the kernel or making an unmeasured speed claim.

## What is measured

`examples/m11_decode_bench.rs` measures one causal decode attention dispatch with:

- `batch = 1`;
- `query_len = 1`;
- `q_heads = 8`;
- `kv_heads = 2`;
- `head_dim = 64`;
- resident K/V lengths of 16, 64, 256, 1024 and 4096 rows;
- pre-created Q/K/V/output buffers and a pre-created M11 pipeline.

Each timed sample creates the caller-owned command encoder, records the M11 dispatch, submits it and waits for device completion with `Device::poll(Maintain::Wait)`. Input upload, pipeline compilation and output readback are outside the timed region.

The harness reports minimum, median, mean and p95 wall-clock microseconds after 20 warmup iterations and 200 measured iterations. These numbers include host submission and synchronization overhead. They are not GPU timestamp-query measurements and must not be described as kernel-only latency.

Run it with:

```text
cargo run --release --features wgpu --example m11_decode_bench
```

## Selection gate for SciRust

This harness characterizes M11 itself. It does **not** establish that M11 is faster than SciRust's current resident KV-cache attention path.

The SciRust integration milestone must pin an exact reviewed FLAT commit and run a paired benchmark on the same machine, adapter, model geometry, cache lengths, warmup policy and synchronization boundary. Selection may only claim a latency or throughput advantage when those paired measurements are recorded with the exact SciRust and FLAT commit SHAs.

Until that paired benchmark exists, the existing qualified SciRust path remains the performance baseline and M11 remains an opt-in integration candidate.

## Sovereignty

The benchmark uses the same Rust + WGPU path as M11. It introduces no project-authored C/C++, C ABI bridge, CUDA C++, `nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK dependency.
