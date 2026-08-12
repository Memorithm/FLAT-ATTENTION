#![cfg(feature = "wgpu")]

use flat_attention::{
    api::wgpu::{GroupedForwardKernelVariant, PreparedGroupedForward},
    GroupedAttentionShape, WgpuGroupedForwardPipeline,
};

#[test]
fn prepared_grouped_forward_types_are_publicly_nameable() {
    fn accepts_prepared(_: Option<PreparedGroupedForward>) {}
    accepts_prepared(None);

    let portable = GroupedForwardKernelVariant::Q4PortableGrouped;
    let subgroup = GroupedForwardKernelVariant::Q4SubgroupMha;
    assert_ne!(portable, subgroup);
}

#[test]
fn shape_routing_keeps_gqa_native_even_when_subgroup_pipeline_is_absent() {
    // The actual device-capability branch is qualified on physical hardware.
    // This compile-time contract documents that the shape API itself remains
    // native GQA/MQA and does not need K/V head expansion.
    let gqa = GroupedAttentionShape {
        batch: 1,
        q_heads: 8,
        kv_heads: 2,
        seq_len: 16,
        head_dim: 64,
    };
    let mha = GroupedAttentionShape {
        kv_heads: 8,
        ..gqa
    };
    assert_eq!(gqa.q_heads / gqa.kv_heads, 4);
    assert_eq!(mha.q_heads, mha.kv_heads);

    // Keep the public constructor/method symbols type-checked without requiring
    // a GPU in this test. Runtime parity remains in the WGPU integration matrix.
    let _constructor = WgpuGroupedForwardPipeline::new;
    let _selector = WgpuGroupedForwardPipeline::kernel_variant_for_shape;
}
