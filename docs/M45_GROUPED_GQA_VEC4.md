# M45 native GQA/MQA vec4 candidate

M45 extends the grouped-forward experimentation surface with an **opt-in** vec4 memory-staging kernel for native GQA and MQA at head dimensions 64 and 128. It does not change `WgpuGroupedForwardPipeline::new`, and it does not alter the M44 MHA-only `with_vectorization` contract.

`with_grouped_vectorization(device, true)` enables the candidate only when `q_heads != kv_heads` and `head_dim` is 64 or 128. Query storage keeps query-head cardinality. K/V storage keeps exactly `batch * kv_heads * seq_len * head_dim` scalars; no query-head expansion or duplicate cache is created. MHA and unqualified dimensions remain on `Q4PortableGrouped` under this constructor.

The WGSL shader uses the same Q4 online-softmax/reduction structure as the portable grouped kernel while staging Q/K/V through `vec4<f32>` loads. Query and KV bases are computed independently from the existing native grouping map `kv_head = q_head / (q_heads / kv_heads)`. Fixed iteration counts are used for vec4 staging before workgroup barriers to keep cross-backend control flow uniform.

Correctness qualification covers GQA (`4/2`) and MQA (`4/1`), D64/D128, causal and non-causal attention, batch size 2, output/LSE parity against the scalar grouped oracle, Naga 0.20 validation, and explicit physical K/V cardinality checks. The M44 MHA vec4 opt-in remains independent.

The original M45 change made **no performance claim** and required clean physical evidence before any promotion. That evidence is now recorded in `docs/M46_GROUPED_VEC4_THOR_QUALIFICATION.md`: on its exact NVIDIA Thor/Vulkan matrix, the native grouped vec4 candidate had lower median latency than the portable prepared grouped path in all eight measured rows. This remains a device/workload-scoped qualification, not a universal speedup claim or an unconditional global-default justification.
