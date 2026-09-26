# Host inventory gates

These tests require no GPU. They keep maps honest. They do not change routing
and they are not a performance claim.

| Gate | Command | Contract |
|------|---------|----------|
| Shaders | `cargo test --test shader_inventory` | every `shaders/*.wgsl` is named in `shaders/README.md` |
| Examples | `cargo test --test example_inventory` | every `examples/*.rs` stem is named in `examples/README.md` |
| Example manifest | `cargo test --test example_manifest` | every example has a `Cargo.toml` `[[example]]` row |
| Crates | `cargo test --test crate_inventory` | every `crates/*` package is named in `crates/README.md` |
| Workspace | `cargo test --test workspace_inventory` | each `crates/*` directory name matches `package.name` |
| Oracle smoke | `cargo test --test host_oracle_smoke` | ExactReference matches `forward_reference` |

Related: [`ONBOARDING.md`](ONBOARDING.md), [`COMPETITIVE.md`](COMPETITIVE.md).
