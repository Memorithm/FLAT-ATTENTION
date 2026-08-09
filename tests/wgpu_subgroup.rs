#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, WgpuFlatAttention,
    WgpuFlatAttentionError, WgpuKernelVariant, WgpuSubgroupPolicy,
};

const ATOL: f32 = 5.0e-5;
const RTOL: f32 = 5.0e-4;

fn context(policy: WgpuSubgroupPolicy) -> Option<WgpuFlatAttention> {
    match WgpuFlatAttention::with_subgroup_policy(policy) {
        Ok(context) => Some(context),
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional subgroup test skipped");
            None
        }
        Err(error) => panic!("WGPU context creation failed for {policy:?}: {error}"),
    }
}

fn fixture(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| ((i as f32) * 0.031 - 0.4).sin() * 0.9)
        .collect();
    let k = (0..len)
        .map(|i| ((i as f32) * 0.043 + 0.2).cos() * 0.8)
        .collect();
    let v = (0..len)
        .map(|i| ((i as f32) * 0.017 - 0.6).sin() * 1.3)
        .collect();
    (q, k, v)
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

fn check_parity(context: &WgpuFlatAttention) {
    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 17,
        head_dim: 64,
    };
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
fn disable_policy_forces_qualified_portable_q4() {
    let Some(context) = context(WgpuSubgroupPolicy::Disable) else {
        return;
    };
    assert_eq!(context.kernel_variant(), WgpuKernelVariant::Q4Portable);
    check_parity(&context);
}

#[test]
fn auto_policy_matches_adapter_capability() {
    let Some(context) = context(WgpuSubgroupPolicy::Auto) else {
        return;
    };
    eprintln!(
        "adapter={} subgroup_supported={} subgroup_range={:?} selected={:?}",
        context.adapter_name(),
        context.subgroup_supported(),
        context.subgroup_size_range(),
        context.kernel_variant()
    );

    if context.subgroup_supported() {
        assert_eq!(context.kernel_variant(), WgpuKernelVariant::Q4Subgroup);
        let (minimum, maximum) = context
            .subgroup_size_range()
            .expect("subgroup-capable adapter must expose subgroup limits");
        assert!(minimum > 0);
        assert!(maximum >= minimum);
    } else {
        assert_eq!(context.kernel_variant(), WgpuKernelVariant::Q4Portable);
        assert_eq!(context.subgroup_size_range(), None);
    }
    check_parity(&context);
}

#[test]
fn require_policy_is_explicit_and_never_falls_back_silently() {
    match WgpuFlatAttention::with_subgroup_policy(WgpuSubgroupPolicy::Require) {
        Ok(context) => {
            assert_eq!(context.kernel_variant(), WgpuKernelVariant::Q4Subgroup);
            assert!(context.subgroup_supported());
            check_parity(&context);
        }
        Err(WgpuFlatAttentionError::RequiredSubgroupUnavailable) => {
            assert!(
                std::env::var_os("FLAT_REQUIRE_SUBGROUP").is_none(),
                "FLAT_REQUIRE_SUBGROUP=1 but the selected WGPU adapter exposes no subgroup feature"
            );
        }
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional subgroup test skipped");
        }
        Err(error) => panic!("required subgroup context failed unexpectedly: {error}"),
    }
}
