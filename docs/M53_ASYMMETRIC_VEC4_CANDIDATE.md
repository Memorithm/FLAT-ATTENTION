# M53 asymmetric projection vec4 candidate

M53 adds an **opt-in** vec4-storage candidate to the rectangular external
projection pipeline used by SciRust prefill. The existing portable pipeline
remains the default, and unsupported head dimensions continue to fall back to
it.

`ExternalAsymmetricProjectionRotaryGroupedPipeline::with_vectorization(device,
true)` compiles both variants. `kernel_variant_for_shape` selects vec4 only for
head dimensions 64 and 128. The candidate keeps the existing sequence-major
projection layout, native GQA/MQA K/V cardinality, fused Q/K RoPE, optional
pre-rotated K interoperability, causal offsets, ALiBi, online softmax, and
combined output/LSE layout.

The WGSL change is deliberately narrow: Q, K, and V staging use
`array<vec4<f32>>` storage loads, while the qualified M11/M13/M15 reduction and
accumulation structure is preserved. The implementation remains Rust-native
host code plus WGPU/WGSL and introduces no mandatory C/C++, C ABI, CUDA C++,
vendor SDK, or vendor-specific shader extension.

## Correctness and fallback gates

The WGPU integration suite covers causal and non-causal GQA at D64 and D128,
checks output and LSE against the scalar oracle, and verifies that D80 uses the
portable fallback. The full all-target/all-feature compile also remains green.

## Benchmark protocol

`m53_asymmetric_vec4_bench` compares the portable and vec4 pipelines in one
resident WGPU context using the same input buffers. Upload and readback are
outside the timed region. Each row is parity-gated before timing, execution
order alternates, and the output records medians, p95, and
`performance_claim=none`.

The development-machine run on Intel Iris Xe/Vulkan used 3 warmups and 12
timed repeats for GQA 8/2, sequence lengths 128 and 512, D64/D128, causal and
non-causal. Exact output parity was observed. Seven rows favored vec4 by
1.027x to 1.075x; the remaining row was effectively neutral at 0.9997x.
These measurements are only a candidate-selection signal and support no
product or Thor performance claim.

Promotion requires a clean physical-Thor qualification on the exact candidate
commit and a full SciRust model-level comparison. Until then the portable path
remains the default and `performance_claim=none` remains in force.
