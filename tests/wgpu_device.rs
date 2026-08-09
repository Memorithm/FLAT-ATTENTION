#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionError, WgpuFlatAttention,
    WgpuFlatAttentionError,
};

const O_ATOL: f32 = 2.0e-5;
const O_RTOL: f32 = 2.0e-4;
const LSE_ATOL: f32 = 3.0e-5;
const LSE_RTOL: f32 = 3.0e-4;

fn fixture(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| ((i as f32) * 0.071 - 0.8).sin() * 0.75)
        .collect();
    let k = (0..len)
        .map(|i| ((i as f32) * 0.113 + 0.2).cos() * 0.65)
        .collect();
    let v = (0..len)
        .map(|i| ((i as f32) * 0.047 - 0.4).sin() * 1.25)
        .collect();
    (q, k, v)
}

fn adversarial_fixture(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * (7.0 + (i % 11) as f32 * 0.125)
        })
        .collect();
    let k = (0..len)
        .map(|i| {
            let sign = if (i / 3) % 2 == 0 { -1.0 } else { 1.0 };
            sign * (6.5 + (i % 13) as f32 * 0.1875)
        })
        .collect();
    let v = (0..len)
        .map(|i| ((i as f32) * 0.03125).sin() * 3.0)
        .collect();
    (q, k, v)
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite(), "{name}[{index}] is not finite: {a}");
        assert!(b.is_finite(), "{name} reference[{index}] is not finite: {b}");
        let tolerance = atol + rtol * b.abs();
        let error = (a - b).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={a}, expected={b}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

fn assert_forward_parity(
    context: &WgpuFlatAttention,
    shape: AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    causal: bool,
) {
    let config = FlatAttentionConfig {
        causal,
        softmax_scale: None,
    };
    let expected = forward_reference(q, k, v, shape, config).unwrap();
    let actual = context.forward(q, k, v, shape, config).unwrap();
    assert_close("O", &actual.output, &expected.output, O_ATOL, O_RTOL);
    assert_close(
        "LSE",
        &actual.lse,
        &expected.lse,
        LSE_ATOL,
        LSE_RTOL,
    );
}

fn context() -> Option<WgpuFlatAttention> {
    match WgpuFlatAttention::new() {
        Ok(context) => Some(context),
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional device test skipped");
            None
        }
        Err(error) => panic!("required WGPU context failed: {error}"),
    }
}

#[test]
fn m3_head_dimension_matrix_matches_reference() {
    let Some(context) = context() else {
        return;
    };
    eprintln!("FLAT-ATTENTION WGPU adapter: {}", context.adapter_name());

    for head_dim in [1, 8, 16, 32, 64, 80, 96, 128] {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 17,
            head_dim,
        };
        let (q, k, v) = fixture(shape);
        for causal in [false, true] {
            assert_forward_parity(&context, shape, &q, &k, &v, causal);
        }
    }
}

#[test]
fn m3_sequence_tile_boundaries_match_reference() {
    let Some(context) = context() else {
        return;
    };

    for seq_len in [1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129] {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len,
            head_dim: 32,
        };
        let (q, k, v) = fixture(shape);
        for causal in [false, true] {
            assert_forward_parity(&context, shape, &q, &k, &v, causal);
        }
    }
}

#[test]
fn m3_multiple_batches_and_heads_match_reference() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 2,
        heads: 3,
        seq_len: 31,
        head_dim: 16,
    };
    let (q, k, v) = fixture(shape);
    for causal in [false, true] {
        assert_forward_parity(&context, shape, &q, &k, &v, causal);
    }
}

#[test]
fn m3_large_score_range_remains_finite_and_matches_reference() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 31,
        head_dim: 64,
    };
    let (q, k, v) = adversarial_fixture(shape);

    for causal in [false, true] {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: Some(0.125),
        };
        let expected = forward_reference(&q, &k, &v, shape, config).unwrap();
        let actual = context.forward(&q, &k, &v, shape, config).unwrap();
        assert_close("adversarial O", &actual.output, &expected.output, 8.0e-5, 8.0e-4);
        assert_close("adversarial LSE", &actual.lse, &expected.lse, 2.0e-4, 8.0e-4);
    }
}

#[test]
fn m3_causal_future_tokens_have_zero_contribution() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 9,
        head_dim: 8,
    };
    let (q, k, v) = fixture(shape);
    let mut k_mutated = k.clone();
    let mut v_mutated = v.clone();

    for position in 1..shape.seq_len {
        let base = position * shape.head_dim;
        for dim in 0..shape.head_dim {
            k_mutated[base + dim] = if dim % 2 == 0 { 1000.0 } else { -1000.0 };
            v_mutated[base + dim] = 500.0 + position as f32 * 10.0 + dim as f32;
        }
    }

    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let baseline = context.forward(&q, &k, &v, shape, config).unwrap();
    let mutated = context
        .forward(&q, &k_mutated, &v_mutated, shape, config)
        .unwrap();

    assert_close(
        "causal first-row invariance",
        &mutated.output[..shape.head_dim],
        &baseline.output[..shape.head_dim],
        1.0e-6,
        1.0e-6,
    );
    assert_close(
        "causal first-row equals V0",
        &mutated.output[..shape.head_dim],
        &v[..shape.head_dim],
        1.0e-6,
        1.0e-6,
    );
    assert_close(
        "causal first-row LSE invariance",
        &mutated.lse[..1],
        &baseline.lse[..1],
        1.0e-6,
        1.0e-6,
    );
}

#[test]
fn m3_validation_rejects_non_finite_input_and_unsupported_head_dim() {
    let Some(context) = context() else {
        return;
    };

    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 2,
        head_dim: 8,
    };
    let (mut q, k, v) = fixture(shape);
    q[3] = f32::NAN;
    let error = context
        .forward(&q, &k, &v, shape, FlatAttentionConfig::default())
        .unwrap_err();
    assert!(matches!(
        error,
        WgpuFlatAttentionError::Core(FlatAttentionError::NonFiniteInput {
            tensor: "Q",
            index: 3
        })
    ));

    let oversized = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 1,
        head_dim: 129,
    };
    let q = vec![0.0; 129];
    let k = vec![0.0; 129];
    let v = vec![0.0; 129];
    let error = context
        .forward(&q, &k, &v, oversized, FlatAttentionConfig::default())
        .unwrap_err();
    assert!(matches!(
        error,
        WgpuFlatAttentionError::UnsupportedHeadDim {
            actual: 129,
            maximum: 128
        }
    ));
}

#[test]
fn resident_forward_is_one_packed_linear_memory_result() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 2,
        heads: 1,
        seq_len: 9,
        head_dim: 16,
    };
    let (q, k, v) = fixture(shape);
    let q_gpu = context.upload(&q).unwrap();
    let k_gpu = context.upload(&k).unwrap();
    let v_gpu = context.upload(&v).unwrap();
    let resident = context
        .forward_resident(
            &q_gpu,
            &k_gpu,
            &v_gpu,
            shape,
            FlatAttentionConfig {
                causal: true,
                softmax_scale: Some(0.25),
            },
        )
        .unwrap();

    let tensor_len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let lse_len = shape.batch * shape.heads * shape.seq_len;
    assert_eq!(resident.output_len(), tensor_len);
    assert_eq!(resident.lse_len(), lse_len);
    assert_eq!(resident.combined().len(), tensor_len + lse_len);

    let actual = context.download_attention(&resident).unwrap();
    let expected = forward_reference(
        &q,
        &k,
        &v,
        shape,
        FlatAttentionConfig {
            causal: true,
            softmax_scale: Some(0.25),
        },
    )
    .unwrap();
    assert_close(
        "resident O",
        &actual.output,
        &expected.output,
        O_ATOL,
        O_RTOL,
    );
    assert_close(
        "resident LSE",
        &actual.lse,
        &expected.lse,
        LSE_ATOL,
        LSE_RTOL,
    );
}
