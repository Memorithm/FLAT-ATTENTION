#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference_grouped, FlatAttentionConfig, FlatAttentionError, GroupedAttentionShape,
    WgpuFlatAttentionError, WgpuGroupedAttention,
};

const ATOL: f32 = 6.0e-5;
const RTOL: f32 = 6.0e-4;

fn context() -> Option<WgpuGroupedAttention> {
    match WgpuGroupedAttention::new() {
        Ok(context) => Some(context),
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional M10 grouped test skipped");
            None
        }
        Err(error) => panic!("M10 grouped WGPU context creation failed: {error}"),
    }
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i as f32 * 0.021 + phase;
            x.sin() * 2.0 + (x * 0.73).cos() * 0.35
        })
        .collect()
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "{name}[{index}] is not finite");
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

#[test]
fn grouped_wgpu_matches_oracle_for_mha_gqa_and_mqa() {
    let Some(context) = context() else {
        return;
    };
    eprintln!("M10 grouped adapter: {}", context.adapter_name());

    for shape in [
        GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 4,
            seq_len: 17,
            head_dim: 32,
        },
        GroupedAttentionShape {
            batch: 1,
            q_heads: 8,
            kv_heads: 2,
            seq_len: 17,
            head_dim: 64,
        },
        GroupedAttentionShape {
            batch: 2,
            q_heads: 8,
            kv_heads: 1,
            seq_len: 9,
            head_dim: 80,
        },
        GroupedAttentionShape {
            batch: 1,
            q_heads: 16,
            kv_heads: 4,
            seq_len: 5,
            head_dim: 128,
        },
    ] {
        let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
        let k = fixture(shape.kv_tensor_len().unwrap(), 0.9);
        let v = fixture(shape.kv_tensor_len().unwrap(), 1.7);

        for causal in [false, true] {
            let config = FlatAttentionConfig {
                causal,
                softmax_scale: None,
            };
            let expected = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
            let actual = context.forward(&q, &k, &v, shape, config).unwrap();
            assert_close("grouped O", &actual.output, &expected.output);
            assert_close("grouped LSE", &actual.lse, &expected.lse);
        }
    }
}

#[test]
fn resident_gqa_keeps_physical_kv_head_cardinality() {
    let Some(context) = context() else {
        return;
    };
    let shape = GroupedAttentionShape {
        batch: 2,
        q_heads: 12,
        kv_heads: 3,
        seq_len: 17,
        head_dim: 64,
    };
    let q = fixture(shape.q_tensor_len().unwrap(), 0.3);
    let k = fixture(shape.kv_tensor_len().unwrap(), 1.1);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.9);

    let q_gpu = context.upload(&q).unwrap();
    let k_gpu = context.upload(&k).unwrap();
    let v_gpu = context.upload(&v).unwrap();
    assert_eq!(q_gpu.len(), shape.q_tensor_len().unwrap());
    assert_eq!(k_gpu.len(), shape.kv_tensor_len().unwrap());
    assert_eq!(v_gpu.len(), shape.kv_tensor_len().unwrap());
    assert_eq!(q_gpu.len(), 4 * k_gpu.len());

    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: Some(0.125),
    };
    let resident = context
        .forward_resident(&q_gpu, &k_gpu, &v_gpu, shape, config)
        .unwrap();
    assert_eq!(resident.output_len(), shape.q_tensor_len().unwrap());
    assert_eq!(resident.lse_len(), shape.lse_len().unwrap());

    let actual = context.download_attention(&resident).unwrap();
    let expected = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
    assert_close("resident GQA O", &actual.output, &expected.output);
    assert_close("resident GQA LSE", &actual.lse, &expected.lse);
}

#[test]
fn invalid_grouping_fails_before_dispatch() {
    let Some(context) = context() else {
        return;
    };
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 6,
        kv_heads: 4,
        seq_len: 2,
        head_dim: 8,
    };
    let error = context
        .forward(&[], &[], &[], shape, FlatAttentionConfig::default())
        .unwrap_err();
    assert!(matches!(
        error,
        WgpuFlatAttentionError::Core(FlatAttentionError::InvalidHeadGrouping {
            q_heads: 6,
            kv_heads: 4
        })
    ));
}
