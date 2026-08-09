# M3 Merge Gate

M3 is accepted only when the PR head passes both GitHub Actions jobs on the same commit:

- `rust`: `cargo fmt --all -- --check`, strict Clippy, and all-feature tests;
- `wgpu-device`: Mesa Vulkan/lavapipe with `FLAT_REQUIRE_WGPU=1`, executing the complete M3 device parity suite.

No skipped device validation is accepted as a successful M3 qualification.
