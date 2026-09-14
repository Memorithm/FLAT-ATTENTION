# First 10 minutes

This is the shortest path from a clone to a verified host-only oracle run.
It does not require a GPU.

## 1. Build and smoke

```bash
cargo test --test host_oracle_smoke
cargo run --example hello_attention
cargo run --example io_model
```

`hello_attention` prints a tiny causal MHA result from `forward_reference`.
It is a contract demo, not a performance claim.

## 2. Read next

1. Root [`README.md`](../README.md) — status, license (PolyForm Noncommercial 1.0.0, same as SciRust), layout.
2. [`FLAT_ATTENTION_GUIDE.md`](FLAT_ATTENTION_GUIDE.md) — architecture and public contract.
3. [`m9-numerical-policy.md`](m9-numerical-policy.md) — ExactReference / FastPortable / DeterministicPortable.
4. [`API_SEMVER.md`](API_SEMVER.md) — what may change before 1.0.

## 3. Maps

- Crates: [`CRATES.md`](CRATES.md)
- Examples: [`EXAMPLES.md`](EXAMPLES.md)
- Shaders: [`SHADERS.md`](SHADERS.md)

## 4. GPU later

Only after the host path is green:

```bash
cargo test --features wgpu
FLAT_REQUIRE_WGPU=1 cargo test --features wgpu --tests -- --nocapture
```

A missing adapter is an error under `FLAT_REQUIRE_WGPU=1`. There is no silent CPU fallback behind a GPU mode.
