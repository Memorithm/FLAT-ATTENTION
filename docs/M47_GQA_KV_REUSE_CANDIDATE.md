# M47 GQA K/V tile-reuse candidate

M47 implements the isolated Phase O candidate defined after M46. It remains opt-in and makes no performance claim.

`WgpuGroupedForwardPipeline::with_grouped_kv_reuse(device, true)` selects `Q4Vec4GroupedKvReuse` only for native GQA/MQA shapes with group size at least two and head dimension 64 or 128. MHA and unqualified dimensions remain on `Q4PortableGrouped`. Existing constructors and their selection semantics are unchanged.

The dispatch Y axis addresses `(batch, kv_head, query-head pair within the KV group)`. One workgroup stages each physical K/V tile once and consumes it for up to two query heads from the same native group. Query storage retains query-head cardinality; K/V storage remains exactly `batch * kv_heads * seq_len * head_dim` scalars. Odd group sizes use a one-head tail workgroup without expanding or duplicating K/V.

The candidate preserves the existing Q4 online-softmax order within each query head and row. It uses fixed iteration counts around workgroup barriers and requires no subgroup feature, vendor extension, or proprietary SDK.

Correctness coverage includes:

- Naga 0.20 parse and validation;
- GQA group sizes two and three, including the one-head tail workgroup;
- MQA group size four;
- head dimensions 64 and 128;
- causal and non-causal attention;
- batch size two;
- output and LSE parity against the scalar grouped oracle;
- explicit physical K/V cardinality checks;
- selection/fallback independence from the existing MHA vec4 and grouped vec4 candidates.

The next gate is a same-context physical benchmark against `Q4Vec4Grouped` using the M46 Thor/Vulkan protocol. The candidate must be rejected if correctness regresses or if the measured medians do not improve the accepted baseline. No default-routing change follows from this implementation.

Sovereignty remains Rust-native host code plus WGPU/WGSL. No project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK is introduced. `performance_claim=none` remains in force.
