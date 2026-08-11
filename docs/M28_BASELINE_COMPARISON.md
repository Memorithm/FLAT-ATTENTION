# M28 baseline comparison — scalar Rust vs resident FLAT

This slice starts the roadmap M28 baseline-comparison phase with two in-tree baselines that can be exercised without introducing any new dependency:

1. the deterministic scalar Rust grouped-forward oracle;
2. the public resident `WgpuGroupedForwardPipeline` path.

Run:

```bash
cargo run --release --features wgpu --example m28_scalar_flat_baseline
```

The harness covers representative MHA, GQA, and MQA shapes at sequence lengths 32 and 128, head dimension 64, and both causal and non-causal modes. Before timing, the resident FLAT output and LSE are checked against the scalar oracle with the same tolerances used by the M27 grouped-forward sweep.

## Timing contracts

The two baselines intentionally report their timing scopes separately:

- `scalar_timing_scope=forward_reference_grouped_including_return_allocation`
- `flat_timing_scope=command_encoder+public_encode+queue_submit+device_poll`

The resident FLAT timing excludes H2D upload and D2H readback. The scalar timing includes construction of the returned `GroupedForwardOutput`. Because these scopes are not identical work contracts, the harness does not emit or promote a speedup claim. It records medians and p95 values for both baselines side by side so later M28 work can preserve the provenance of each measurement.

The output also records the GitHub commit SHA when available at compile time, adapter name, backend, driver, driver information, precision, warm-up count, iteration count, and the correctness gate. `performance_claim=none` remains explicit.

## Sovereignty

This slice keeps the existing project constraints unchanged: Rust-native host code and WGPU/WGSL only, with no project-authored C/C++ implementation or C ABI bridge and no mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK.

## Remaining M28 work

The roadmap still requires additional baselines, including SciRust's naive/multi-dispatch WGPU attention and comparisons across optimized FLAT generations. External competitors remain optional and may be measured only when environment and licensing permit. No external baseline is made a dependency by this slice.
