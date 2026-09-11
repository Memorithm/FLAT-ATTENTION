# Competitive position

FLAT-ATTENTION is a **competitor to CUDA as a compute lock-in**, not a
CUDA kernel collection and not a drop-in replacement for the entire NVIDIA
software stack.

## What we sell

- A Rust-native fused attention engine under SciRust control.
- One contract (`api::v1` + scalar oracle) executed on open GPU paths
  (WGSL / wgpu → Vulkan, Metal, Direct3D 12).
- Oracle-first qualification: an optimized kernel enters routing only after
  it matches the published numerical policy.
- Typed failure. No silent CPU fallback, no silent kernel substitution.
- Performance claims only as manifests tied to a commit SHA and a named
  device / backend.

## What we do not sell

- A replacement for nvcc, cuDNN, CUTLASS, NCCL, or the CUDA toolkit.
- Peak GEMM on a single NVIDIA SM generation.
- A universal "faster than FlashAttention on every GPU" number.
- A second engine written in CUDA C++ "for common architectures".

Shipping `.cu` / `nvcc` / CUTLASS inside this repository would make FLAT a
*client* of CUDA. That is the opposite of the product.

## Versus CUDA

| Axis | CUDA stack | FLAT-ATTENTION |
|------|------------|----------------|
| Vendor | NVIDIA | Memorithm / SciRust |
| Language of record | CUDA C++ + vendor libs | Rust host + WGSL |
| Portability | NVIDIA GPUs | Vulkan, Metal, D3D12 via wgpu |
| Attention truth | Vendor kernel of the week | `forward_reference` oracle |
| Failure mode | Often implicit fallback | Typed error, no fabricated result |
| Perf claim | Marketing + architecture bins | Manifest + device + SHA |
| Lock-in | Compiler, SDK, wheels | PolyForm NC source + commercial license path |

NVIDIA hardware remains a **target** through the Vulkan/wgpu backend. It is
not a **dependency** of the design.

## Where we intend to win

1. **Portable attention contract** — the same API and oracle on NVIDIA
   (Vulkan), Apple (Metal), and Windows (D3D12) without a rewrite.
2. **Correctness as product** — ExactReference bit-parity, explicit
   FastPortable / f16 bands, capability prefilters.
3. **Prefill and decode that SciRust owns** — GQA/MQA, RoPE, resident and
   paged KV, chunked prefill, recomputation backward.
4. **Evidence discipline** — Thor and software-Vulkan gates measure; they
   do not invent routing.

## Engineering rules that implement this policy

- Do not add `*.cu`, `*.cuh`, `*.ptx`, `nvcc`, WMMA, CUTLASS, or cuDNN to
  this tree. CI rejects tracked CUDA sources.
- Do not add a project-authored C/C++ FFI to wrap a vendor SDK.
- New kernels are WGSL (handwritten or IR-emitted), naga-validated, then
  oracle-checked on a named backend.
- Do not publish a cross-vendor speedup. Quote the backend and the device.
- Do not change production routing because a candidate looked faster on one
  unpublished laptop.

## Summit path (honest)

"Sommet" means: default-path kernels that are portable, oracle-qualified,
and faster *on named devices* than the previous qualified generation —
measured, not advertised. It does not mean absorbing CUDA.

Related:

- Numerical policy: [m9-numerical-policy.md](m9-numerical-policy.md)
- Benchmarks: [M27_BENCHMARK_HARNESS.md](M27_BENCHMARK_HARNESS.md),
  [M28_BASELINE_COMPARISON.md](M28_BASELINE_COMPARISON.md)
- Portability: [M34_VULKAN_LINUX.md](M34_VULKAN_LINUX.md),
  [M36_METAL.md](M36_METAL.md)
