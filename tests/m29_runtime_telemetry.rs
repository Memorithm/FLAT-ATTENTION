#![cfg(feature = "wgpu")]

use flat_attention::{
    AttentionShape, AutotunerCacheStatus, RuntimeKernelId, WgpuFlatAttention, WgpuKernelVariant,
    WgpuSubgroupPolicy,
};

#[test]
fn passive_snapshot_matches_selected_portable_dispatch() {
    let attention = match WgpuFlatAttention::with_subgroup_policy(WgpuSubgroupPolicy::Disable) {
        Ok(attention) => attention,
        Err(flat_attention::WgpuFlatAttentionError::Unavailable) => return,
        Err(error) => panic!("M29 WGPU setup failed: {error}"),
    };
    let shape = AttentionShape {
        batch: 2,
        heads: 3,
        seq_len: 17,
        head_dim: 64,
    };
    let selected = attention.kernel_variant_for_head_dim(shape.head_dim);
    let telemetry = attention.runtime_telemetry(shape).unwrap();

    assert_eq!(selected, WgpuKernelVariant::Q4Vec4Portable);
    assert_eq!(telemetry.kernel_id, RuntimeKernelId::Q4Vec4Portable);
    assert_eq!(telemetry.tile.query_rows, 4);
    assert_eq!(telemetry.tile.kv_rows, 8);
    assert_eq!(telemetry.tile.workgroups, [5, 6, 1]);
    assert_eq!(telemetry.dispatch_count, 1);
    assert_eq!(telemetry.temporary_allocation_count, 1);
    assert_eq!(telemetry.temporary_allocation_bytes, 32);
    assert_eq!(telemetry.autotuner_cache, AutotunerCacheStatus::NotApplicable);
    assert!(!telemetry.device.name.is_empty());
    assert!(!telemetry.device.backend.is_empty());
}

#[test]
fn unsupported_vec4_shape_reports_selection_reason_without_dispatch() {
    let attention = match WgpuFlatAttention::with_subgroup_policy(WgpuSubgroupPolicy::Disable) {
        Ok(attention) => attention,
        Err(flat_attention::WgpuFlatAttentionError::Unavailable) => return,
        Err(error) => panic!("M29 WGPU setup failed: {error}"),
    };
    let telemetry = attention
        .runtime_telemetry(AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 5,
            head_dim: 32,
        })
        .unwrap();

    assert_eq!(telemetry.kernel_id, RuntimeKernelId::Q4Portable);
    assert_eq!(
        telemetry.fallback_reason.as_deref(),
        Some("vec4 specialization unavailable for head_dim=32")
    );
}
