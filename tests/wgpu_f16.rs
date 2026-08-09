#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, WgpuF16Attention,
    WgpuF16AttentionError, WgpuIoPrecision, WgpuPreferredAttention, F16,
};

const O_ATOL: f32 = 2.0e-3;
const O_RTOL: f32 = 5.0e-3;
const LSE_ATOL: f32 = 8.0e-4;
const LSE_RTOL: f32 = 2.0e-3;

fn fixture(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| ((i as f32) * 0.031 - 0.7).sin() * 0.85)
        .collect();
    let k = (0..len)
        .map(|i| ((i as f32) * 0.047 + 0.2).cos() * 0.75)
        .collect();
    let v = (0..len)
        .map(|i| ((i as f32) * 0.019 - 0.4).sin() * 1.2)
        .collect();
    (q, k, v)
}

fn quantize(values: &[f32]) -> Vec<F16> {
    values.iter().copied().map(F16::from_f32).collect()
}

fn expand(values: &[F16]) -> Vec<f32> {
    values.iter().copied().map(F16::to_f32).collect()
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "{name}[{index}] is not finite: {actual}"
        );
        assert!(
            expected.is_finite(),
            "{name} reference[{index}] is not finite"
        );
        let tolerance = atol + rtol * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

fn context() -> Option<WgpuF16Attention> {
    match WgpuF16Attention::new() {
        Ok(context) => Some(context),
        Err(WgpuF16AttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional packed-f16 test skipped");
            None
        }
        Err(error) => panic!("required M8 packed-f16 context failed: {error}"),
    }
}

fn check_raw_f16_parity(context: &WgpuF16Attention, shape: AttentionShape) {
    let (q, k, v) = fixture(shape);
    let q16 = quantize(&q);
    let k16 = quantize(&k);
    let v16 = quantize(&v);
    let q32 = expand(&q16);
    let k32 = expand(&k16);
    let v32 = expand(&v16);

    for causal in [false, true] {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let reference = forward_reference(&q32, &k32, &v32, shape, config).unwrap();
        let expected_o: Vec<f32> = reference
            .output
            .iter()
            .copied()
            .map(F16::from_f32)
            .map(F16::to_f32)
            .collect();
        let actual = context.forward(&q16, &k16, &v16, shape, config).unwrap();
        let actual_o = expand(&actual.output);
        assert_close("packed-f16 O", &actual_o, &expected_o, O_ATOL, O_RTOL);
        assert_close(
            "packed-f16 LSE",
            &actual.lse,
            &reference.lse,
            LSE_ATOL,
            LSE_RTOL,
        );
    }
}

#[test]
fn packed_f16_path_is_baseline_wgsl_and_matches_reference() {
    let Some(context) = context() else {
        return;
    };
    eprintln!("FLAT M8 packed-f16 adapter: {}", context.adapter_name());
    assert_eq!(context.io_precision(), WgpuIoPrecision::PackedF16);
    check_raw_f16_parity(
        &context,
        AttentionShape {
            batch: 1,
            heads: 2,
            seq_len: 17,
            head_dim: 64,
        },
    );
    check_raw_f16_parity(
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
fn preferred_router_uses_packed_f16_only_for_qualified_shapes_and_never_cpu() {
    let context = match WgpuPreferredAttention::new() {
        Ok(context) => context,
        Err(WgpuF16AttentionError::F32(_)) | Err(WgpuF16AttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            return;
        }
        Err(error) => panic!("preferred WGPU context failed: {error}"),
    };

    assert_eq!(context.io_precision_for_head_dim(80), WgpuIoPrecision::F32);
    assert_eq!(
        context.io_precision_for_head_dim(64),
        WgpuIoPrecision::PackedF16
    );
    assert_eq!(
        context.io_precision_for_head_dim(128),
        WgpuIoPrecision::PackedF16
    );

    for head_dim in [64, 80, 128] {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 9,
            head_dim,
        };
        let (q, k, v) = fixture(shape);
        let config = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };
        let actual = context.forward(&q, &k, &v, shape, config).unwrap();

        match context.io_precision_for_head_dim(head_dim) {
            WgpuIoPrecision::F32 => {
                let reference = forward_reference(&q, &k, &v, shape, config).unwrap();
                assert_close(
                    "preferred f32 O",
                    &actual.output,
                    &reference.output,
                    5e-5,
                    5e-4,
                );
                assert_close(
                    "preferred f32 LSE",
                    &actual.lse,
                    &reference.lse,
                    5e-5,
                    5e-4,
                );
            }
            WgpuIoPrecision::PackedF16 => {
                let q16 = quantize(&q);
                let k16 = quantize(&k);
                let v16 = quantize(&v);
                let reference =
                    forward_reference(&expand(&q16), &expand(&k16), &expand(&v16), shape, config)
                        .unwrap();
                let expected_o: Vec<f32> = reference
                    .output
                    .iter()
                    .copied()
                    .map(F16::from_f32)
                    .map(F16::to_f32)
                    .collect();
                assert_close(
                    "preferred packed-f16 O",
                    &actual.output,
                    &expected_o,
                    O_ATOL,
                    O_RTOL,
                );
                assert_close(
                    "preferred packed-f16 LSE",
                    &actual.lse,
                    &reference.lse,
                    LSE_ATOL,
                    LSE_RTOL,
                );
            }
        }
    }
}

#[test]
fn resident_f16_output_keeps_binary16_o_and_fp32_lse_lengths() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 9,
        head_dim: 64,
    };
    let (q, k, v) = fixture(shape);
    let q = quantize(&q);
    let k = quantize(&k);
    let v = quantize(&v);
    let q_gpu = context.upload_f16(&q).unwrap();
    let k_gpu = context.upload_f16(&k).unwrap();
    let v_gpu = context.upload_f16(&v).unwrap();
    let resident = context
        .forward_resident(
            &q_gpu,
            &k_gpu,
            &v_gpu,
            shape,
            FlatAttentionConfig::default(),
        )
        .unwrap();
    let tensor_len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let lse_len = shape.batch * shape.heads * shape.seq_len;
    assert_eq!(resident.output_len(), tensor_len);
    assert_eq!(resident.lse_len(), lse_len);
    assert_eq!(resident.packed_words(), tensor_len / 2 + lse_len);
    let output = context.download_attention(&resident).unwrap();
    assert_eq!(output.output.len(), tensor_len);
    assert_eq!(output.lse.len(), lse_len);
}

#[test]
fn raw_f16_rejects_non_finite_input_before_dispatch() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 1,
        head_dim: 64,
    };
    let mut q = vec![F16::ZERO; 64];
    let k = vec![F16::ZERO; 64];
    let v = vec![F16::ZERO; 64];
    q[3] = F16::from_bits(0x7c00);
    let error = context
        .forward(&q, &k, &v, shape, FlatAttentionConfig::default())
        .unwrap_err();
    assert!(matches!(
        error,
        WgpuF16AttentionError::Core(flat_attention::FlatAttentionError::NonFiniteInput {
            tensor: "Q",
            index: 3
        })
    ));
}

#[test]
fn packed_upload_rejects_odd_scalar_counts() {
    let Some(context) = context() else {
        return;
    };
    let error = context.upload_f16(&[F16::ZERO; 3]).unwrap_err();
    assert!(matches!(
        error,
        WgpuF16AttentionError::OddPackedLength { actual: 3 }
    ));
}
