# Examples index

See [`examples/README.md`](../examples/README.md) for the authoritative
host-only vs GPU inventory. Presence in that table is documentation, not
default routing and not a performance claim.

Start host-only with:

```bash
cargo run --example hello_attention
cargo run --example io_model
cargo test --test example_inventory
```

GPU examples require `--features wgpu`. A timing example is evidence for the
adapter it ran on. Quote backend and device; never a universal speedup.
