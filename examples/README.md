# Examples

Examples are split by whether they need a GPU / the `wgpu` feature.

## Host-only (default features)

| Example | Purpose |
|---------|---------|
| `hello_attention` | Scalar oracle smoke: tiny causal MHA, prints O/LSE. Not a performance claim. |
| `io_model` | Analytical K/V staging-load comparison of single-row vs Q4 tiling. Architectural count only. |

```bash
cargo run --example hello_attention
cargo run --example io_model
```

## GPU (`--features wgpu`)

The remaining examples under this directory are device benches, sweeps, or
qualification harnesses. They declare `required-features = ["wgpu"]` in the
root `Cargo.toml` so a host-only `cargo test` / `cargo build` does not try to
compile them.

```bash
cargo run --release --features wgpu --example subgroup_bench
```

A timing example is evidence for the adapter it ran on. It is not a universal
speedup claim. See `docs/M27_BENCHMARK_HARNESS.md` and `docs/M40_BENCHMARK_MANIFESTS.md`.
