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

> GPU execution inevitably crosses the operating system / device-driver ABI. "No FFI" here
> means FLAT-ATTENTION does not introduce its own C/C++ bridge or vendor SDK dependency.

## Milestone 1: fused portable forward

This initial implementation contains:

- `forward_reference`: scalar Rust oracle using streaming online softmax;
- `shaders/flat_fwd.wgsl`: one-dispatch fused `QKᵀ + softmax + PV` forward kernel;
- causal and non-causal modes;
- log-sum-exp (`LSE`) output for a later recomputation-based backward pass;
- no `N × N` storage allocation;
- runtime head dimensions from 1 to 128 in the first WGSL kernel;
- MSRV 1.89, matching SciRust.

The WGSL milestone is a correctness/architecture kernel, not yet a Tensor-Core equivalent.
Its purpose is to establish the fused memory behavior and public contract before adding
backend-specific code generation and autotuning.

## Tensor layout

Q, K, V and O are contiguous `f32` tensors in:

```text
[batch, heads, sequence, head_dim]
```

The WGSL dispatch flattens `batch × heads` and uses:

```text
x = sequence
y = batch × heads
z = 1
```

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

## Validate

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Roadmap

1. **M1 — portable fused forward:** Rust oracle + WGSL fused online-softmax kernel.
2. **M2 — WGPU executor:** buffer contract, dispatch, parity on lavapipe and real GPUs.
3. **M3 — tiled multi-row kernel:** increase arithmetic intensity, vectorized loads, subgroup reductions.
4. **M4 — mixed precision:** BF16/F16 inputs with FP32 online-softmax state where the open backend exposes it reliably.
5. **M5 — backward:** recompute scores from Q/K + saved LSE; no probability matrix storage.
6. **M6 — GQA/MQA + KV cache:** native inference path for SciAgent.
7. **M7 — open matrix-core codegen:** capability-gated cooperative-matrix/subgroup-matrix path when available through an open IR/runtime path; scalar/vector fallback remains valid.
8. **M8 — autotuner:** deterministic tile selection from measured device characteristics and benchmark evidence.
9. **M9 — SciRust integration:** replace the current multi-dispatch resident attention chain behind a stable FLAT-ATTENTION adapter.

## Licensing

No license grant is declared in this initial repository scaffold. Memorithm can set the
project's final licensing policy independently of the technical architecture.
