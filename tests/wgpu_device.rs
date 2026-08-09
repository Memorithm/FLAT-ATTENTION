#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, WgpuFlatAttention,
    WgpuFlatAttentionError,
};

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

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * b.abs();
        let error = (a - b).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={a}, expected={b}, abs_error={error}, tolerance={tolerance}"
        );
    }
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
fn fused_wgpu_matches_reference_causal_and_non_causal() {
    let Some(context) = context() else {
        return;
    };
    eprintln!("FLAT-ATTENTION WGPU adapter: {}", context.adapter_name());

    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 17,
        head_dim: 32,
    };
    let (q, k, v) = fixture(shape);

    for causal in [false, true] {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let expected = forward_reference(&q, &k, &v, shape, config).unwrap();
        let actual = context.forward(&q, &k, &v, shape, config).unwrap();
        assert_close("O", &actual.output, &expected.output, 2.0e-5, 2.0e-4);
        assert_close("LSE", &actual.lse, &expected.lse, 2.0e-5, 2.0e-4);
    }
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
        2.0e-5,
        2.0e-4,
    );
    assert_close("resident LSE", &actual.lse, &expected.lse, 2.0e-5, 2.0e-4);
}
