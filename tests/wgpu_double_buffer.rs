#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, WgpuFlatAttention,
    WgpuFlatAttentionError, WgpuKernelVariant, WgpuSubgroupPolicy,
};

const ATOL: f32 = 5.0e-5;
const RTOL: f32 = 5.0e-4;

fn context(double_buffering: bool) -> Option<WgpuFlatAttention> {
    match WgpuFlatAttention::with_subgroup_vectorization_and_double_buffering(
        WgpuSubgroupPolicy::Disable,
        true,
        double_buffering,
    ) {
        Ok(context) => Some(context),
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional M7 test skipped");
            None
        }
        Err(error) => panic!("M7 WGPU context creation failed: {error}"),
    }
}

fn fixture(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| ((i as f32) * 0.041 - 0.5).sin() * 0.9)
        .collect();
    let k = (0..len)
        .map(|i| ((i as f32) * 0.033 + 0.4).cos() * 0.8)
        .collect();
    let v = (0..len)
        .map(|i| ((i as f32) * 0.021 - 0.1).sin() * 1.2)
        .collect();
    (q, k, v)
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(actual.is_finite(), "{name}[{index}] is not finite");
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

fn check_parity(context: &WgpuFlatAttention, shape: AttentionShape) {
    let (q, k, v) = fixture(shape);
    for causal in [false, true] {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let expected = forward_reference(&q, &k, &v, shape, config).unwrap();
        let actual = context.forward(&q, &k, &v, shape, config).unwrap();
        assert_close("O", &actual.output, &expected.output);
        assert_close("LSE", &actual.lse, &expected.lse);
    }
}

#[test]
fn m7_is_not_selected_by_existing_m6_constructor() {
    let Some(context) = WgpuFlatAttention::with_subgroup_policy_and_vectorization(
        WgpuSubgroupPolicy::Disable,
        true,
    )
    .ok()
    else {
        assert!(std::env::var_os("FLAT_REQUIRE_WGPU").is_none());
        return;
    };
    assert!(!context.double_buffering_enabled());
    assert_eq!(
        context.kernel_variant_for_head_dim(64),
        WgpuKernelVariant::Q4Vec4Portable
    );
}

#[test]
fn opt_in_selects_double_buffer_only_for_d64_d128() {
    let Some(context) = context(true) else {
        return;
    };
    assert!(context.double_buffering_enabled());
    assert_eq!(
        context.kernel_variant_for_head_dim(64),
        WgpuKernelVariant::Q4Vec4DoubleBuffered
    );
    assert_eq!(
        context.kernel_variant_for_head_dim(128),
        WgpuKernelVariant::Q4Vec4DoubleBuffered
    );
    assert_eq!(
        context.kernel_variant_for_head_dim(80),
        WgpuKernelVariant::Q4Portable
    );
}

#[test]
fn double_buffer_d64_matches_reference_across_tile_boundaries() {
    let Some(context) = context(true) else {
        return;
    };
    check_parity(
        &context,
        AttentionShape {
            batch: 1,
            heads: 2,
            seq_len: 17,
            head_dim: 64,
        },
    );
}

#[test]
fn double_buffer_d128_matches_reference_with_partial_final_tile() {
    let Some(context) = context(true) else {
        return;
    };
    check_parity(
        &context,
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 9,
            head_dim: 128,
        },
    );
}

#[test]
fn vectorization_off_disables_double_buffer_candidate() {
    match WgpuFlatAttention::with_subgroup_vectorization_and_double_buffering(
        WgpuSubgroupPolicy::Disable,
        false,
        true,
    ) {
        Ok(context) => {
            assert!(context.double_buffering_enabled());
            assert_eq!(
                context.kernel_variant_for_head_dim(64),
                WgpuKernelVariant::Q4Portable
            );
        }
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() => {}
        Err(error) => panic!("M7 scalar fallback context failed: {error}"),
    }
}

#[test]
fn subgroup_keeps_priority_over_double_buffer_candidate() {
    match WgpuFlatAttention::with_subgroup_vectorization_and_double_buffering(
        WgpuSubgroupPolicy::Require,
        true,
        true,
    ) {
        Ok(context) => {
            assert_eq!(context.kernel_variant(), WgpuKernelVariant::Q4Subgroup);
            assert_eq!(
                context.kernel_variant_for_head_dim(64),
                WgpuKernelVariant::Q4Subgroup
            );
        }
        Err(WgpuFlatAttentionError::RequiredSubgroupUnavailable) => {
            assert!(std::env::var_os("FLAT_REQUIRE_SUBGROUP").is_none());
        }
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() => {}
        Err(error) => panic!("M7 subgroup-priority context failed: {error}"),
    }
}
