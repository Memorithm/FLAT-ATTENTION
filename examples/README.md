# Examples

Examples are split by whether they need a GPU / the `wgpu` feature.
Presence in this table is documentation, not default routing and not a
performance claim.

## Host-only (default features)

| Example | Purpose |
|---------|---------|
| `hello_attention` | Scalar oracle smoke: tiny causal MHA, prints O/LSE. |
| `io_model` | Analytical K/V staging-load comparison of single-row vs Q4 tiling. |

```bash
cargo run --example hello_attention
cargo run --example io_model
```

## GPU (`--features wgpu`)

These examples declare `required-features = ["wgpu"]` in the root
`Cargo.toml` so a host-only `cargo test` / `cargo build` does not compile
them.

| Example | Purpose |
|---------|---------|
| `subgroup_bench` | M5 Q4 vs subgroup median latency on the selected adapter. |
| `vector_bench` | M6 vec4 storage path timing on supported head dims. |
| `double_buffer_bench` | M7 double-buffered staging timing (evidence-gated). |
| `f16_bench` | M8 packed-binary16 I/O path timing. |
| `regression_gate` | Same-device kernel regression vs qualified Q4 portable. |
| `m11_decode_bench` | Rectangular / decode-oriented M11 harness. |
| `m15_decode_bench` | M15 resident decode harness (one query token). |
| `m20_grouped_backward_bench` | M19/M20 grouped backward recomputation bench. |
| `m20_grouped_backward_host_overhead` | Host overhead around grouped backward. |
| `m27_cold_warm_pipeline` | Cold vs warm pipeline accounting. |
| `m27_dispatch_allocation_accounting` | Dispatch / allocation counters. |
| `m27_resident_decode_sweep` | Resident decode sweep. |
| `m27_resident_grouped_forward_sweep` | Resident grouped forward sweep. |
| `m27_resident_vs_host_io` | Resident vs host-transfer timing split. |
| `m28_kernel_generations` | Kernel-generation comparison harness. |
| `m28_scalar_flat_baseline` | Scalar FLAT baseline comparison. |
| `m48_decode_kv_reuse_sweep` | Decode that reuses projected/rotated K. |
| `m53_asymmetric_vec4_bench` | M53 rectangular vec4 candidate bench. |
| `m60_q1_direct_ab` | M60 Q1 direct vec4 A/B candidate. |
| `fdal1_da_luc_oracle_sweep` | Research DA-LUC oracle sweep. Not api v1. |
| `fdal2_da_luc_decode_sweep` | Research DA-LUC decode sweep. Not api v1. |

```bash
cargo run --release --features wgpu --example subgroup_bench
```

A timing example is evidence for the adapter it ran on. It is not a
universal speedup claim. See `docs/M27_BENCHMARK_HARNESS.md` and
`docs/M40_BENCHMARK_MANIFESTS.md`.
