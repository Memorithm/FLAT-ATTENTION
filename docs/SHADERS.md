# Shader map

See [`shaders/README.md`](../shaders/README.md) for the handwritten WGSL inventory.
`cargo test --test shader_inventory` fails if a `shaders/*.wgsl` file is missing
from that README, or if the README names a file that is gone.

Rules:

1. The scalar oracle (`forward_reference` and grouped variants) is the correctness source.
2. A shader is executable only after Naga validation and device parity against that oracle.
3. Capability-selected variants (subgroup, vec4, f16, double-buffer) must fail with a typed error when unavailable. They must not fall back to the CPU oracle.
4. Research shaders (DA-LUC and similar) stay outside `api::v1` until an explicit promotion gate.
5. Inventory presence is documentation, not default routing and not a performance claim.
