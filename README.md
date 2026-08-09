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
- No pull request is merged until all required CI jobs are green on its final head SHA.

> GPU execution inevitably crosses the operating system / device-driver ABI. "No FFI" here
> means FLAT-ATTENTION does not introduce its own C/C++ bridge or vendor SDK dependency.

## Current architecture

The implementation currently contains:

- `forward_reference`: scalar Rust oracle using streaming online softmax;
- `shaders/flat_fwd.wgsl`: M4 Q4 fused forward kernel;
- `shaders/flat_fwd_single.wgsl`: qualified M2/M3 single-query baseline source;
- causal and non-causal modes;
- log-sum-exp (`LSE`) output for recomputation-based backward;
- no `N × N` score/probability storage allocation;
- head dimensions from 1 to 128 in the portable WGSL generation;
- four storage bindings (`Q`, `K`, `V`, packed `[O | LSE]`);
- 8-row K/V tiles;
- optional real WGPU execution behind the `wgpu` feature;
- resident Q/K/V execution with one fused compute dispatch and no implicit readback;
- mandatory WGPU/lavapipe device qualification in CI;
- MSRV 1.89, matching SciRust.

### M4: four query rows per workgroup

The M4 kernel maps one workgroup to up to four consecutive query rows. A K/V tile is loaded into workgroup memory once, then reused by those four queries while each query keeps an independent online-softmax state.

```text
Q rows q..q+3 ─┐
               ├─ one workgroup
K/V tile ──────┤  loaded once
               └─ reused by up to 4 Q rows
```

Dispatch geometry is now:

```text
x = ceil(sequence / 4)
y = batch × heads
z = 1
```

For causal attention, a query tile stops staging K/V after the last valid query position in that tile, so wholly future K/V tiles are not loaded.

The kernel's declared workgroup arrays occupy about 11 KiB of scalar `f32` storage, remaining below the 16 KiB portable floor used by this project.

## IO model: what is and is not claimed

`single_row_io_model` and `tiled_q4_io_model` count the logical scalar K/V storage loads implied by the explicit WGSL staging loops. This is useful for verifying the architectural reuse transformation.

It is **not** a measurement of physical DRAM transactions, cache behavior, bandwidth, latency, or throughput. Runtime speed claims require real-device timing benchmarks.

For example:

```bash
cargo run --example io_model
```

For non-causal sequence lengths divisible by four, Q4 performs one quarter of the baseline logical K/V scalar loads because each staged tile serves four query rows. Causal Q4 additionally omits fully-future rows.

## Tensor layout

Q, K, V and O are contiguous `f32` tensors in:

```text
[batch, heads, sequence, head_dim]
```

The GPU output storage is packed as:

```text
[ O tensor | LSE vector ]
```

Packing O and LSE preserves the backward statistic while keeping the shader to four storage bindings.

## Online softmax

For each active query and streamed key, FLAT-ATTENTION updates:

```text
m_new = max(m_old, score)
alpha = exp(m_old - m_new)
p     = exp(score - m_new)
l_new = alpha * l_old + p
O_new = alpha * O_old + p * V
```

Final output is `O / l`; the saved statistic is `LSE = m + log(l)`.

## Build and validate

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Required real-WGPU qualification:

```bash
FLAT_REQUIRE_WGPU=1 cargo test --features wgpu --tests -- --nocapture
```

CI installs Mesa Vulkan and requires this device suite to pass before merge.

## Engineering roadmap

The authoritative phase-by-phase plan, acceptance criteria, benchmark policy, backward/KV-cache work, open matrix-codegen path, autotuning and SciRust/SciAgent integration plan are maintained in [`ROADMAP.md`](ROADMAP.md).

## Licensing

No license grant is declared in this repository. Memorithm can set the project's final proprietary licensing policy independently of the technical architecture.
