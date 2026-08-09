#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, WgpuFlatAttention,
    WgpuFlatAttentionError, WGSL_MAX_HEAD_DIM,
};

const ATOL: f32 = 5.0e-5;
const RTOL: f32 = 5.0e-4;

fn context() -> Option<WgpuFlatAttention> {
    match WgpuFlatAttention::new() {
        Ok(context) => Some(context),
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional matrix test skipped");
            None
        }
        Err(error) => panic!("required WGPU context failed: {error}"),
    }
}

fn fixture(shape: AttentionShape, phase: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| ((i as f32) * 0.037 + phase).sin() * 0.85)
        .collect();
    let k = (0..len)
        .map(|i| ((i as f32) * 0.053 - phase * 0.7).cos() * 0.72)
        .collect();
    let v = (0..len)
        .map(|i| ((i as f32) * 0.029 + phase * 1.3).sin() * 1.15)
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

fn check_case(context: &WgpuFlatAttention, shape: AttentionShape, causal: bool, phase: f32) {
    let (q, k, v) = fixture(shape, phase);
    let config = FlatAttentionConfig {
        causal,
        softmax_scale: None,
    };
    let expected = forward_reference(&q, &k, &v, shape, config).unwrap();
    let actual = context.forward(&q, &k, &v, shape, config).unwrap();
    assert_close("O", &actual.output, &expected.output);
    assert_close("LSE", &actual.lse, &expected.lse);
}

#[test]
fn every_supported_head_dimension_matches_reference() {
    let Some(context) = context() else {
        return;
    };
    eprintln!("FLAT-ATTENTION WGPU adapter: {}", context.adapter_name());

    for (case, head_dim) in [1usize, 8, 16, 32, 64, 80, 96, 128].into_iter().enumerate() {
        let shape = AttentionShape {
            batch: 1,
            heads: 2,
            seq_len: 9,
            head_dim,
        };
        check_case(&context, shape, false, 0.11 + case as f32 * 0.07);
        check_case(&context, shape, true, 0.19 + case as f32 * 0.05);
    }
}

#[test]
fn sequence_tile_boundaries_match_reference() {
    let Some(context) = context() else {
        return;
    };

    for (case, seq_len) in [
        1usize, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129,
    ]
    .into_iter()
    .enumerate()
    {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len,
            head_dim: 16,
        };
        let causal = case % 2 == 0;
        check_case(&context, shape, causal, 0.23 + case as f32 * 0.031);
    }
}

#[test]
fn multiple_batches_and_heads_match_reference() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 3,
        heads: 4,
        seq_len: 17,
        head_dim: 32,
    };
    check_case(&context, shape, false, 0.41);
    check_case(&context, shape, true, 0.67);
}

#[test]
fn high_dynamic_range_scores_remain_finite_and_match_reference() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 17,
        head_dim: 32,
    };
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q: Vec<f32> = (0..len)
        .map(|i| if i % 2 == 0 { 8.0 } else { -8.0 })
        .collect();
    let k: Vec<f32> = (0..len)
        .map(|i| match i % 4 {
            0 => 7.0,
            1 => -7.0,
            2 => -6.5,
            _ => 6.5,
        })
        .collect();
    let v: Vec<f32> = (0..len)
        .map(|i| ((i as f32) * 0.017 - 0.3).sin() * 3.0)
        .collect();

    for causal in [false, true] {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: Some(0.125),
        };
        let expected = forward_reference(&q, &k, &v, shape, config).unwrap();
        let actual = context.forward(&q, &k, &v, shape, config).unwrap();
        assert!(actual.output.iter().all(|value| value.is_finite()));
        assert!(actual.lse.iter().all(|value| value.is_finite()));
        assert_close("dynamic O", &actual.output, &expected.output);
        assert_close("dynamic LSE", &actual.lse, &expected.lse);
    }
}

#[test]
fn causal_first_query_uses_only_first_value_row() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 9,
        head_dim: 8,
    };
    let tensor_len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = vec![0.0; tensor_len];
    let k = vec![0.0; tensor_len];
    let mut v = vec![0.0; tensor_len];
    for key in 0..shape.seq_len {
        for dim in 0..shape.head_dim {
            v[key * shape.head_dim + dim] = key as f32 * 1000.0 + dim as f32;
        }
    }

    let actual = context
        .forward(
            &q,
            &k,
            &v,
            shape,
            FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
        )
        .unwrap();

    assert_eq!(&actual.output[..shape.head_dim], &v[..shape.head_dim]);
}

#[test]
fn excessive_head_dimension_is_rejected_before_dispatch() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 2,
        head_dim: WGSL_MAX_HEAD_DIM + 1,
    };
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let data = vec![0.0; len];
    let error = context
        .forward(&data, &data, &data, shape, FlatAttentionConfig::default())
        .unwrap_err();
    assert_eq!(
        error,
        WgpuFlatAttentionError::UnsupportedHeadDim {
            actual: WGSL_MAX_HEAD_DIM + 1,
            maximum: WGSL_MAX_HEAD_DIM,
        }
    );
}
