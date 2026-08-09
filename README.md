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
- `shaders/flat_fwd.wgsl`: qualified M4 Q4 portable fused kernel;
- `shaders/flat_fwd_subgroup.wgsl`: M5 subgroup-assisted Q4 reduction kernel;
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

The qualified M4 kernel maps one workgroup to up to four consecutive query rows. A K/V tile is loaded into workgroup memory once, then reused by those four queries while each query keeps an independent online-softmax state.

```text
Q rows q..q+3 ─┐
               ├─ one workgroup
K/V tile ──────┤  loaded once
               └─ reused by up to 4 Q rows
```

Dispatch geometry is:

```text
x = ceil(sequence / 4)
y = batch × heads
z = 1
```

For causal attention, a query tile stops staging K/V after the last valid query position in that tile, so wholly future K/V tiles are not loaded.

The Q4 kernel's declared workgroup arrays occupy about 11 KiB of scalar `f32` storage, remaining below the 16 KiB portable floor used by this project.

### M5: subgroup-assisted dot reductions

M5 keeps the Q4 memory architecture and changes only the Q·K reduction mechanism on devices that expose WGPU's native `Features::SUBGROUP` capability.

Each 64-lane dot-product reduction is split into two levels:

1. `subgroupAdd(partial)` produces one sum per hardware subgroup;
2. subgroup leaders write those totals to workgroup memory;
3. a deterministic tree reduces only the subgroup totals.

The shader uses `num_subgroups`, `subgroup_id` and `subgroup_invocation_id`, so it does **not** assume a particular subgroup/warp width.

Runtime selection is explicit:

- `WgpuSubgroupPolicy::Auto`: use M5 when the adapter reports subgroup support, otherwise M4 Q4;
- `WgpuSubgroupPolicy::Disable`: force the qualified M4 Q4 GPU path;
- `WgpuSubgroupPolicy::Require`: require M5 or return an explicit error.

`Auto` never falls back to CPU. If subgroup shader validation fails despite an advertised capability, it uses the qualified M4 **GPU** path; `Require` reports the failure.

The selected path and adapter-reported subgroup range are observable with `kernel_variant()` and `subgroup_size_range()`.

## IO model: what is and is not claimed

`single_row_io_model` and `tiled_q4_io_model` count the logical scalar K/V storage loads implied by the explicit WGSL staging loops. This is useful for verifying the architectural reuse transformation.

It is **not** a measurement of physical DRAM transactions, cache behavior, bandwidth, latency, or throughput. Runtime speed claims require real-device timing benchmarks.

```bash
cargo run --example io_model
```

For non-causal sequence lengths divisible by four, Q4 performs one quarter of the baseline logical K/V scalar loads because each staged tile serves four query rows. Causal Q4 additionally omits fully-future rows.

## M5 benchmark harness

A reproducible end-to-end comparison of the qualified Q4 path and required subgroup path is available when the selected adapter exposes subgroup support:

```bash
cargo run --release --features wgpu --example subgroup_bench
```

The M5 harness reports median latency after warm-up for `B1 H2 N128 D64`, causal attention. It intentionally includes upload, fused dispatch and readback and labels that fact in its output. It is evidence for comparing the two executable paths on a particular adapter; it is **not** a universal GPU performance claim.

If the adapter exposes no subgroup feature, the harness reports that fact and makes no subgroup timing claim.

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

A device gate can additionally require subgroup capability with:

```bash
FLAT_REQUIRE_WGPU=1 FLAT_REQUIRE_SUBGROUP=1 cargo test --features wgpu --test wgpu_subgroup -- --nocapture
```

CI installs Mesa Vulkan and requires the normal WGPU device suite to pass before merge. A subgroup-specific CI requirement is enabled only after the CI adapter itself has been verified to expose `Features::SUBGROUP`.

## Engineering roadmap

The authoritative phase-by-phase plan, acceptance criteria, benchmark policy, backward/KV-cache work, open matrix-codegen path, autotuning and SciRust/SciAgent integration plan are maintained in [`ROADMAP.md`](ROADMAP.md).

## Licensing

No license grant is declared in this repository. Memorithm can set the project's final proprietary licensing policy independently of the technical architecture.
