# M3 Device Parity Qualification

M3 turns the M2 real-device smoke test into a permanent numerical qualification gate for the portable fused forward kernel.

## Required matrix

The WGPU implementation is checked independently against the scalar Rust online-softmax oracle for:

- head dimensions: `1, 8, 16, 32, 64, 80, 96, 128`;
- sequence lengths: `1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129`;
- causal and non-causal execution;
- multiple batches and heads;
- large-score adversarial fixtures;
- causal future-token isolation;
- non-finite input rejection;
- explicit rejection above the portable `head_dim = 128` limit;
- independent `O` and `LSE` comparisons.

The dimension and sequence lists are covered as two orthogonal sweeps rather than a full Cartesian product. This preserves complete boundary coverage while keeping the mandatory software-Vulkan CI gate practical.

## Numerical tolerances

For the regular deterministic fixture:

- output `O`: `atol = 2e-5`, `rtol = 2e-4`;
- log-sum-exp `LSE`: `atol = 3e-5`, `rtol = 3e-4`.

For the deliberately large-score stress fixture, the tolerance is widened only for cross-device transcendental differences:

- output `O`: `atol = 8e-5`, `rtol = 8e-4`;
- `LSE`: `atol = 2e-4`, `rtol = 8e-4`.

Every compared scalar must also be finite.

## Causal isolation invariant

For query position zero, every key/value position greater than zero is masked. M3 mutates all future K/V values by very large amounts and requires the first output row and its LSE to remain unchanged. The first output row must additionally equal `V[0]`, because the causal softmax contains exactly one admissible key at that position.

## CI contract

The existing `wgpu-device` job runs with `FLAT_REQUIRE_WGPU=1` on Mesa Vulkan/lavapipe. Adapter absence or device-test skipping is therefore a hard failure in the mandatory device job.

M3 may be merged only when both jobs are green on the same PR head SHA:

1. `rust`: rustfmt, Clippy `-D warnings`, all-feature tests;
2. `wgpu-device`: mandatory real WGPU execution of the complete M3 parity suite.
