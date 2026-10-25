//! Host-only qualification of the bfloat16-I/O scalar oracles.
//!
//! These tests require no GPU and no `wgpu` feature. They qualify the storage
//! contract and FP32-core composition only; no device or performance claim.

use flat_attention::{
    forward_reference, forward_reference_bf16, forward_reference_grouped,
    forward_reference_grouped_bf16, AttentionShape, Bf16, FlatAttentionConfig, FlatAttentionError,
    GroupedAttentionShape,
};

fn fixture(len: usize, phase: f32) -> Vec<Bf16> {
    (0..len)
        .map(|i| {
            let x = i as f32 * 0.041 + phase;
            Bf16::from_f32(x.sin() * 1.5 + (x * 0.37).cos() * 0.25)
        })
        .collect()
}

fn decode(values: &[Bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn mha_shape() -> AttentionShape {
    AttentionShape {
        batch: 2,
        heads: 3,
        seq_len: 17,
        head_dim: 16,
    }
}

#[test]
fn mha_bf16_core_is_bit_identical_to_fp32_oracle_on_decoded_inputs() {
    let shape = mha_shape();
    let len = shape.tensor_len().unwrap();
    let (q, k, v) = (fixture(len, 0.2), fixture(len, 0.8), fixture(len, 1.4));
    for causal in [false, true] {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let bf16 = forward_reference_bf16(&q, &k, &v, shape, config).unwrap();
        let fp32 = forward_reference(&decode(&q), &decode(&k), &decode(&v), shape, config).unwrap();

        // LSE never leaves FP32.
        assert_eq!(bf16.lse, fp32.lse);
        // O is exactly the RNE encoding of the FP32 oracle output...
        let expected: Vec<Bf16> = fp32.output.iter().copied().map(Bf16::from_f32).collect();
        assert_eq!(bf16.output, expected);
        // ...so its only error versus FP32 is the final half-ulp rounding.
        for (encoded, reference) in bf16.output.iter().zip(&fp32.output) {
            let error = (encoded.to_f32() - reference).abs();
            assert!(
                error <= reference.abs() * 2f32.powi(-8),
                "{encoded:?} vs {reference}"
            );
        }
    }
}

#[test]
fn grouped_bf16_core_is_bit_identical_to_fp32_oracle_on_decoded_inputs() {
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 8,
        kv_heads: 2,
        seq_len: 9,
        head_dim: 32,
    };
    let q = fixture(shape.q_tensor_len().unwrap(), 0.1);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.6);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.1);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let bf16 = forward_reference_grouped_bf16(&q, &k, &v, shape, config).unwrap();
    let fp32 =
        forward_reference_grouped(&decode(&q), &decode(&k), &decode(&v), shape, config).unwrap();
    assert_eq!(bf16.lse, fp32.lse);
    let expected: Vec<Bf16> = fp32.output.iter().copied().map(Bf16::from_f32).collect();
    assert_eq!(bf16.output, expected);
}

#[test]
fn bf16_rejects_non_finite_inputs_with_typed_errors() {
    let shape = mha_shape();
    let len = shape.tensor_len().unwrap();
    let (q, mut k, v) = (fixture(len, 0.2), fixture(len, 0.8), fixture(len, 1.4));
    k[5] = Bf16::from_f32(f32::NAN);
    let error = forward_reference_bf16(&q, &k, &v, shape, FlatAttentionConfig::default())
        .expect_err("NaN K must fail");
    assert_eq!(
        error,
        FlatAttentionError::NonFiniteInput {
            tensor: "K",
            index: 5
        }
    );

    let error =
        forward_reference_bf16(&q, &k[..len - 1], &v, shape, FlatAttentionConfig::default())
            .expect_err("short K must fail");
    assert!(matches!(
        error,
        FlatAttentionError::LengthMismatch { tensor: "K", .. }
    ));
}

#[test]
fn bf16_range_scores_that_overflow_fp32_are_rejected() {
    // bfloat16 shares binary32's exponent range, so finite bf16 inputs can
    // produce scores beyond f32::MAX; they must fail closed, not yield NaN.
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 2,
        head_dim: 2,
    };
    let huge = Bf16::from_f32(1.0e20);
    assert!(huge.is_finite());
    let q = vec![huge; 4];
    let v = vec![Bf16::from_f32(1.0); 4];
    let error = forward_reference_bf16(&q, &q, &v, shape, FlatAttentionConfig::default())
        .expect_err("overflowing score must fail");
    assert!(matches!(error, FlatAttentionError::NonFiniteScore { .. }));
}
