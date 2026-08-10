pub use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
};

#[path = "../src/attention_bias.rs"]
mod attention_bias;

use attention_bias::{
    forward_reference_projection_grouped_rope_asymmetric_biased, AttentionBias,
};

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| ((index as f32 * 0.071) + phase).sin() * 0.625)
        .collect()
}

fn shape() -> AsymmetricGroupedAttentionShape {
    AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        query_len: 3,
        kv_len: 5,
        head_dim: 8,
        query_position_offset: 2,
    }
}

fn rotary() -> AsymmetricRotaryEmbeddingConfig {
    AsymmetricRotaryEmbeddingConfig {
        theta: 10_000.0,
        query_position_offset: 17,
        kv_position_offset: 13,
    }
}

#[test]
fn no_bias_is_bitwise_identical_to_m11_oracle() {
    let shape = shape();
    let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
    let config = FlatAttentionConfig {
        causal: false,
        softmax_scale: None,
    };
    let expected = forward_reference_projection_grouped_rope_asymmetric(
        &q,
        &k,
        &v,
        shape,
        config,
        rotary(),
    )
    .unwrap();
    let actual = forward_reference_projection_grouped_rope_asymmetric_biased(
        &q,
        &k,
        &v,
        shape,
        config,
        rotary(),
        AttentionBias::None,
    )
    .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn dense_bias_changes_softmax_without_score_matrix() {
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 1,
        kv_heads: 1,
        query_len: 1,
        kv_len: 2,
        head_dim: 2,
        query_position_offset: 0,
    };
    let q = [0.0, 0.0];
    let k = [0.0, 0.0, 0.0, 0.0];
    let v = [1.0, 2.0, 3.0, 4.0];
    let bias = [0.0, 3.0f32.ln()];
    let output = forward_reference_projection_grouped_rope_asymmetric_biased(
        &q,
        &k,
        &v,
        shape,
        FlatAttentionConfig {
            causal: false,
            softmax_scale: None,
        },
        AsymmetricRotaryEmbeddingConfig {
            theta: 10_000.0,
            query_position_offset: 0,
            kv_position_offset: 0,
        },
        AttentionBias::Dense(&bias),
    )
    .unwrap();
    assert!((output.output[0] - 2.5).abs() <= 1.0e-6);
    assert!((output.output[1] - 3.5).abs() <= 1.0e-6);
    assert!((output.lse[0] - 4.0f32.ln()).abs() <= 1.0e-6);
}

#[test]
fn alibi_matches_equivalent_dense_bias() {
    let shape = shape();
    let q = fixture(shape.q_tensor_len().unwrap(), 0.3);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.9);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.5);
    let slopes = [0.25, 0.5, 0.75, 1.0];
    let query_origin = 31usize;
    let kv_origin = 29usize;
    let mut dense =
        Vec::with_capacity(shape.batch * shape.q_heads * shape.query_len * shape.kv_len);

    for _batch in 0..shape.batch {
        for &slope in &slopes {
            for query_pos in 0..shape.query_len {
                for key_pos in 0..shape.kv_len {
                    let delta =
                        (kv_origin + key_pos) as f32 - (query_origin + query_pos) as f32;
                    dense.push(slope * delta);
                }
            }
        }
    }

    let config = FlatAttentionConfig {
        causal: false,
        softmax_scale: None,
    };
    let dense_output = forward_reference_projection_grouped_rope_asymmetric_biased(
        &q,
        &k,
        &v,
        shape,
        config,
        rotary(),
        AttentionBias::Dense(&dense),
    )
    .unwrap();
    let alibi_output = forward_reference_projection_grouped_rope_asymmetric_biased(
        &q,
        &k,
        &v,
        shape,
        config,
        rotary(),
        AttentionBias::Alibi {
            slopes: &slopes,
            query_position_offset: query_origin,
            kv_position_offset: kv_origin,
        },
    )
    .unwrap();
    assert_eq!(alibi_output, dense_output);
}

#[test]
fn malformed_biases_fail_explicitly() {
    let shape = shape();
    let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
    let error = forward_reference_projection_grouped_rope_asymmetric_biased(
        &q,
        &k,
        &v,
        shape,
        FlatAttentionConfig::default(),
        rotary(),
        AttentionBias::Dense(&[0.0]),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        FlatAttentionError::LengthMismatch {
            tensor: "attention bias",
            ..
        }
    ));

    let slopes = [0.0, f32::NAN, 0.0, 0.0];
    let error = forward_reference_projection_grouped_rope_asymmetric_biased(
        &q,
        &k,
        &v,
        shape,
        FlatAttentionConfig::default(),
        rotary(),
        AttentionBias::Alibi {
            slopes: &slopes,
            query_position_offset: 0,
            kv_position_offset: 0,
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        FlatAttentionError::NonFiniteInput {
            tensor: "ALiBi slopes",
            index: 1
        }
    ));
}
