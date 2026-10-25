//! FML1 packed-bfloat16 device parity against `forward_reference_bf16`.
//!
//! Correctness qualification only; no performance claim.

#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference_bf16, AttentionShape, Bf16, FlatAttentionConfig, FlatAttentionError,
    WgpuBf16Attention, WgpuBf16AttentionError,
};

/// Portable FP32 budget, mirroring `numerical_tol::FAST_PORTABLE_*`.
const FAST_PORTABLE_ABS_ATOL: f32 = 1.0e-4;
const FAST_PORTABLE_REL_RTOL: f32 = 1.0e-3;

/// One bfloat16 ulp relative to |O|: the device and the oracle round FP32
/// values that already differ by the portable reduction-order budget, so
/// their RNE results may land on adjacent bfloat16 values.
const BF16_ULP_RTOL: f32 = 1.0 / 128.0;

fn fixture(shape: AttentionShape) -> (Vec<Bf16>, Vec<Bf16>, Vec<Bf16>) {
    let len = shape.tensor_len().unwrap();
    let make = |step: f32, phase: f32, amplitude: f32| -> Vec<Bf16> {
        (0..len)
            .map(|i| Bf16::from_f32(((i as f32) * step + phase).sin() * amplitude))
            .collect()
    };
    (
        make(0.031, -0.7, 0.85),
        make(0.047, 0.2, 0.75),
        make(0.019, -0.4, 1.2),
    )
}

fn context() -> Option<WgpuBf16Attention> {
    match WgpuBf16Attention::new() {
        Ok(context) => Some(context),
        Err(WgpuBf16AttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional packed-bf16 test skipped");
            None
        }
        Err(error) => panic!("required packed-bf16 context failed: {error}"),
    }
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "{name}[{index}] is not finite: {actual}"
        );
        let tolerance = atol + rtol * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

fn check_parity(context: &WgpuBf16Attention, shape: AttentionShape) {
    let (q, k, v) = fixture(shape);
    for causal in [false, true] {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let reference = forward_reference_bf16(&q, &k, &v, shape, config).unwrap();
        let actual = context.forward(&q, &k, &v, shape, config).unwrap();
        assert_eq!(actual.output.len(), reference.output.len());

        let decode = |values: &[Bf16]| values.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        let actual_o = decode(&actual.output);
        let expected_o = decode(&reference.output);
        let exact = actual
            .output
            .iter()
            .zip(&reference.output)
            .filter(|(a, b)| a == b)
            .count();
        eprintln!(
            "packed-bf16 {shape:?} causal={causal}: {exact}/{} O values bit-identical to oracle",
            actual.output.len()
        );
        assert_close(
            "packed-bf16 O",
            &actual_o,
            &expected_o,
            FAST_PORTABLE_ABS_ATOL,
            FAST_PORTABLE_REL_RTOL + BF16_ULP_RTOL,
        );
        assert_close(
            "packed-bf16 LSE",
            &actual.lse,
            &reference.lse,
            FAST_PORTABLE_ABS_ATOL,
            FAST_PORTABLE_REL_RTOL,
        );
    }
}

#[test]
fn packed_bf16_matches_bf16_oracle() {
    let Some(context) = context() else {
        return;
    };
    eprintln!("FLAT FML1 packed-bf16 adapter: {}", context.adapter_name());
    for (batch, heads, seq_len, head_dim) in [(1, 2, 17, 64), (2, 1, 9, 128), (1, 1, 1, 64)] {
        check_parity(
            &context,
            AttentionShape {
                batch,
                heads,
                seq_len,
                head_dim,
            },
        );
    }
}

#[test]
fn packed_bf16_keeps_wide_exponent_range() {
    // Magnitudes far outside binary16 range (max 65504) stay representable.
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 5,
        head_dim: 64,
    };
    let (q, k, v) = fixture(shape);
    let v_wide: Vec<Bf16> = v
        .iter()
        .map(|x| Bf16::from_f32(x.to_f32() * 1.0e30))
        .collect();
    let q_small: Vec<Bf16> = q
        .iter()
        .map(|x| Bf16::from_f32(x.to_f32() * 1.0e-20))
        .collect();
    let config = FlatAttentionConfig::default();
    let reference = forward_reference_bf16(&q_small, &k, &v_wide, shape, config).unwrap();
    let actual = context
        .forward(&q_small, &k, &v_wide, shape, config)
        .unwrap();
    let decode = |values: &[Bf16]| values.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let expected = decode(&reference.output);
    let atol = 1.0e30 * FAST_PORTABLE_ABS_ATOL;
    assert_close(
        "wide-range O",
        &decode(&actual.output),
        &expected,
        atol,
        FAST_PORTABLE_REL_RTOL + BF16_ULP_RTOL,
    );
}

#[test]
fn packed_bf16_rejects_inputs_before_dispatch() {
    let Some(context) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 3,
        head_dim: 64,
    };
    let (q, mut k, v) = fixture(shape);
    let config = FlatAttentionConfig::default();

    let unsupported = AttentionShape {
        head_dim: 80,
        ..shape
    };
    assert!(matches!(
        context.forward(&q, &k, &v, unsupported, config),
        Err(WgpuBf16AttentionError::UnsupportedHeadDim { actual: 80 })
    ));

    let huge: Vec<Bf16> = q.iter().map(|_| Bf16::from_f32(1.0e20)).collect();
    assert!(matches!(
        context.forward(&huge, &huge, &v, shape, config),
        Err(WgpuBf16AttentionError::ScoreBoundExceeded { .. })
    ));

    k[7] = Bf16::from_f32(f32::INFINITY);
    assert!(matches!(
        context.forward(&q, &k, &v, shape, config),
        Err(WgpuBf16AttentionError::Core(
            FlatAttentionError::NonFiniteInput {
                tensor: "K",
                index: 7
            }
        ))
    ));

    assert!(matches!(
        context.upload_bf16(&q[..3]),
        Err(WgpuBf16AttentionError::OddPackedLength { actual: 3 })
    ));
}

#[test]
fn resident_bf16_rejects_foreign_buffers_and_keeps_fp32_lse() {
    let Some(first) = context() else {
        return;
    };
    let Some(second) = context() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 6,
        head_dim: 64,
    };
    let (q, k, v) = fixture(shape);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let q_gpu = first.upload_bf16(&q).unwrap();
    let k_gpu = first.upload_bf16(&k).unwrap();
    let v_gpu = first.upload_bf16(&v).unwrap();
    let resident = first
        .forward_resident(&q_gpu, &k_gpu, &v_gpu, shape, config)
        .unwrap();
    assert_eq!(resident.output_len(), shape.tensor_len().unwrap());
    assert_eq!(resident.lse_len(), shape.lse_len().unwrap());

    assert!(matches!(
        second.forward_resident(&q_gpu, &k_gpu, &v_gpu, shape, config),
        Err(WgpuBf16AttentionError::ForeignBuffer)
    ));
    assert!(matches!(
        second.download_attention(&resident),
        Err(WgpuBf16AttentionError::ForeignBuffer)
    ));

    let downloaded = first.download_attention(&resident).unwrap();
    let direct = first.forward(&q, &k, &v, shape, config).unwrap();
    assert_eq!(
        downloaded, direct,
        "resident and slice paths must agree bit-exactly"
    );
}
