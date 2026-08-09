#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, WgpuFlatAttention,
    WgpuFlatAttentionError, WgpuKernelVariant, WgpuSubgroupPolicy,
};

const ATOL: f32 = 5.0e-5;
const RTOL: f32 = 5.0e-4;

fn context(vectorization_enabled: bool) -> Option<WgpuFlatAttention> {
    match WgpuFlatAttention::with_subgroup_policy_and_vectorization(
        WgpuSubgroupPolicy::Disable,
        vectorization_enabled,
    ) {
        Ok(context) => Some(context),
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional M6 device test skipped");
            None
        }
        Err(error) => panic!("M6 WGPU context creation failed: {error}"),
    }
}

fn fixture(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| ((i as f32) * 0.029 - 0.6).sin() * 0.85)
        .collect();
    let k = (0..len)
        .map(|i| ((i as f32) * 0.037 + 0.3).cos() * 0.75)
        .collect();
    let v = (0..len)
        .map(|i| ((i as f32) * 0.023 - 0.2).sin() * 1.4)
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
fn vec4_specializes_64_and_128_only() {
    let Some(context) = context(true) else {
        return;
    };
    assert_eq!(context.kernel_variant(), WgpuKernelVariant::Q4Portable);
    assert!(context.vectorization_enabled());
    assert_eq!(
        context.kernel_variant_for_head_dim(64),
        WgpuKernelVariant::Q4Vec4Portable
    );
    assert_eq!(
        context.kernel_variant_for_head_dim(128),
        WgpuKernelVariant::Q4Vec4Portable
    );
    for head_dim in [1, 8, 16, 32, 80, 96] {
        assert_eq!(
            context.kernel_variant_for_head_dim(head_dim),
            WgpuKernelVariant::Q4Portable
        );
    }
}

#[test]
fn vec4_d64_matches_reference() {
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
fn vec4_d128_matches_reference_across_q4_boundary() {
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
fn scalar_fallback_remains_qualified_for_non_specialized_dimension() {
    let Some(context) = context(true) else {
        return;
    };
    assert_eq!(
        context.kernel_variant_for_head_dim(80),
        WgpuKernelVariant::Q4Portable
    );
    check_parity(
        &context,
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 17,
            head_dim: 80,
        },
    );
}

#[test]
fn vectorization_can_be_disabled_for_baseline_measurement() {
    let Some(context) = context(false) else {
        return;
    };
    assert!(!context.vectorization_enabled());
    assert_eq!(
        context.kernel_variant_for_head_dim(64),
        WgpuKernelVariant::Q4Portable
    );
    assert_eq!(
        context.kernel_variant_for_head_dim(128),
        WgpuKernelVariant::Q4Portable
    );
    check_parity(
        &context,
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 9,
            head_dim: 64,
        },
    );
}

#[test]
fn subgroup_selection_keeps_priority_over_vec4() {
    match WgpuFlatAttention::with_subgroup_policy_and_vectorization(
        WgpuSubgroupPolicy::Require,
        true,
    ) {
        Ok(context) => {
            assert_eq!(context.kernel_variant(), WgpuKernelVariant::Q4Subgroup);
            assert_eq!(
                context.kernel_variant_for_head_dim(64),
                WgpuKernelVariant::Q4Subgroup
            );
            assert_eq!(
                context.kernel_variant_for_head_dim(128),
                WgpuKernelVariant::Q4Subgroup
            );
        }
        Err(WgpuFlatAttentionError::RequiredSubgroupUnavailable) => {
            assert!(
                std::env::var_os("FLAT_REQUIRE_SUBGROUP").is_none(),
                "FLAT_REQUIRE_SUBGROUP=1 but selected adapter exposes no subgroup support"
            );
        }
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional subgroup priority test skipped");
        }
        Err(error) => panic!("subgroup-priority context failed: {error}"),
    }
}
