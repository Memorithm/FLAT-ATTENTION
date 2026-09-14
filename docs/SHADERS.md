# Shader map

See [`shaders/README.md`](../shaders/README.md) for the handwritten WGSL inventory.

Rules:

1. The scalar oracle (`forward_reference` and grouped variants) is the correctness source.
2. A shader is executable only after Naga validation and device parity against that oracle.
3. Capability-selected variants (subgroup, vec4, f16, double-buffer) must fail with a typed error when unavailable. They must not fall back to the CPU oracle.
4. Research shaders (DA-LUC and similar) stay outside `api::v1` until an explicit promotion gate.
