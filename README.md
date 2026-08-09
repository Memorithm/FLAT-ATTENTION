# FLAT-ATTENTION

**FLAT-ATTENTION** is Memorithm's Rust-native fused attention engine for the SciRust ecosystem.

The project targets the same systems problem class as IO-aware attention kernels: compute
`softmax(QKᵀ / sqrt(d))V` without materializing the full `N × N` score/probability matrix,
while keeping the implementation under SciRust architectural control.

## Non-negotiable design rules

- Rust is the host language.
- No CUDA C/C++, `nvcc`, WMMA, CUTLASS, cuDNN, or vendor SDK is required by the core design.
- No project-authored C ABI / C++ FFI layer.
- Portable GPU kernels are expressed with open shader/IR paths first.
- Every optimized kernel must be checked against a deterministic Rust oracle.
- No performance claim is accepted without a reproducible benchmark on real hardware.
- No pull request is merged until all required CI jobs are green.

> GPU execution inevitably crosses the operating system / device-driver ABI. "No FFI" here
> means FLAT-ATTENTION does not introduce its own C/C++ bridge or vendor SDK dependency.

## Implemented foundation

The current foundation contains:

- `forward_reference`: scalar Rust oracle using streaming online softmax;
- `shaders/flat_fwd.wgsl`: one-dispatch fused `QKᵀ + online-softmax + PV` forward kernel;
- causal and non-causal modes;
- log-sum-exp (`LSE`) output for recomputation-based backward;
- no `N × N` score/probability storage allocation;
- runtime head dimensions from 1 to 128 in the first portable WGSL kernel;
- four storage bindings (`Q`, `K`, `V`, packed `[O | LSE]`) to fit the portable downlevel contract;
- an 8-row K/V tile keeping workgroup storage below the 16 KiB portable floor;
- optional real WGPU execution behind the `wgpu` feature;
- resident Q/K/V execution with one fused compute dispatch and no implicit readback;
- explicit adapter/device errors — no fake CPU fallback;
- MSRV 1.89, matching SciRust.

This is still the correctness/portable-execution generation, not a matrix-core-equivalent performance kernel. Performance claims begin only after measured optimization phases.

## Tensor layout

Q, K, V and O are contiguous `f32` tensors in:

```text
[batch, heads, sequence, head_dim]
```

The portable dispatch flattens `batch × heads` and uses:

```text
x = sequence
y = batch × heads
z = 1
```

The GPU output storage is packed as:

```text
[ O tensor | LSE vector ]
```

Packing O and LSE preserves the backward statistic while keeping the shader to four storage bindings.

## Algorithm

For each query row, FLAT-ATTENTION streams over K/V tiles and updates:

```text
m_new = max(m_old, score)
alpha = exp(m_old - m_new)
p     = exp(score - m_new)
l_new = alpha * l_old + p
O_new = alpha * O_old + p * V
```

Final output is `O / l`; the saved statistic is `LSE = m + log(l)`.

## Build and validate

Core/reference validation:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Real WGPU device parity:

```bash
FLAT_REQUIRE_WGPU=1 cargo test --features wgpu --test wgpu_device -- --nocapture
```

CI installs Mesa Vulkan on its WGPU device job, so this device test is mandatory rather than silently skipped there.

## Engineering roadmap

The authoritative phase-by-phase plan, acceptance criteria, benchmark policy, backward/KV-cache work, open matrix-codegen path, autotuning and SciRust/SciAgent integration plan are maintained in [`ROADMAP.md`](ROADMAP.md).

## Licensing

No license grant is declared in this repository. Memorithm can set the project's final proprietary licensing policy independently of the technical architecture.
