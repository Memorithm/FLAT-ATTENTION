//! Host-only regression: finite Q/K whose scaled scores overflow FP32 must be
//! rejected explicitly by every scalar oracle instead of returning `Ok(NaN)`.
//!
//! These tests require no GPU and no `wgpu` feature.

use flat_attention::{
    backward_reference, backward_reference_grouped, forward_reference, forward_reference_grouped,
    forward_reference_grouped_asymmetric, AsymmetricGroupedAttentionShape, AttentionShape,
    FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput, GroupedAttentionShape,
};

/// Finite magnitude whose pairwise products overflow FP32 (`1e20 * 1e20`).
const HUGE: f32 = 1.0e20;

fn unit_scale() -> FlatAttentionConfig {
    FlatAttentionConfig {
        causal: false,
        softmax_scale: Some(1.0),
    }
}

fn assert_non_finite_score(
    result: Result<impl core::fmt::Debug, FlatAttentionError>,
    query: usize,
    key: usize,
) {
    match result {
        Err(FlatAttentionError::NonFiniteScore {
            query_position,
            key_position,
        }) => {
            assert_eq!((query_position, key_position), (query, key));
        }
        other => panic!("expected NonFiniteScore at ({query}, {key}), got {other:?}"),
    }
}

fn mha_shape() -> AttentionShape {
    AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 2,
        head_dim: 2,
    }
}

#[test]
fn mha_forward_rejects_positive_score_overflow() {
    let q = vec![HUGE; 4];
    let k = vec![HUGE; 4];
    let v = vec![1.0, 2.0, 3.0, 4.0];
    assert_non_finite_score(
        forward_reference(&q, &k, &v, mha_shape(), unit_scale()),
        0,
        0,
    );
}

#[test]
fn mha_forward_rejects_negative_score_overflow() {
    let q = vec![HUGE; 4];
    let k = vec![-HUGE; 4];
    let v = vec![1.0, 2.0, 3.0, 4.0];
    assert_non_finite_score(
        forward_reference(&q, &k, &v, mha_shape(), FlatAttentionConfig::default()),
        0,
        0,
    );
}

#[test]
fn mha_forward_rejects_overflow_on_a_later_key() {
    // Key 0 is harmless; key 1 overflows during the dot-product accumulation.
    let q = vec![1.8e19, 1.8e19, 1.0, 1.0];
    let k = vec![0.0, 0.0, 1.0e19, 1.0e19];
    let v = vec![1.0, 2.0, 3.0, 4.0];
    assert_non_finite_score(
        forward_reference(&q, &k, &v, mha_shape(), unit_scale()),
        0,
        1,
    );
}

#[test]
fn large_finite_scores_remain_accepted_and_finite() {
    // Scores of order 1e30 are finite and must still go through the stable
    // max-subtracted update without being rejected.
    let q = vec![1.0e15, 1.0e15, -1.0e15, 1.0e15];
    let k = vec![1.0e15, 1.0e15, 1.0e15, -1.0e15];
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let out = forward_reference(&q, &k, &v, mha_shape(), unit_scale()).unwrap();
    assert!(out.output.iter().all(|x| x.is_finite()));
    assert!(out.lse.iter().all(|x| x.is_finite()));
    // Query 0 scores are (2e30, 0): softmax collapses onto key 0.
    assert_eq!(&out.output[..2], &[1.0, 2.0]);
}

#[test]
fn grouped_forward_rejects_score_overflow() {
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 2,
        kv_heads: 1,
        seq_len: 2,
        head_dim: 2,
    };
    let q = vec![HUGE; 8];
    let k = vec![HUGE; 4];
    let v = vec![1.0; 4];
    assert_non_finite_score(
        forward_reference_grouped(&q, &k, &v, shape, unit_scale()),
        0,
        0,
    );
}

#[test]
fn asymmetric_forward_rejects_score_overflow() {
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 1,
        kv_heads: 1,
        query_len: 1,
        kv_len: 3,
        head_dim: 2,
        query_position_offset: 2,
    };
    let q = vec![HUGE, HUGE];
    let k = vec![0.0, 0.0, 0.0, 0.0, HUGE, HUGE];
    let v = vec![1.0; 6];
    assert_non_finite_score(
        forward_reference_grouped_asymmetric(&q, &k, &v, shape, unit_scale()),
        0,
        2,
    );
}

#[test]
fn variable_length_forward_rejects_score_overflow() {
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 1,
        kv_heads: 1,
        query_len: 2,
        kv_len: 2,
        head_dim: 2,
        query_position_offset: 0,
    };
    let q = vec![HUGE; 4];
    let k = vec![HUGE; 4];
    let v = vec![1.0; 4];
    assert_non_finite_score(
        shape.forward_reference_variable_lengths(&q, &k, &v, &[(2, 2, 0)], unit_scale()),
        0,
        0,
    );
}

#[test]
fn backward_rejects_score_overflow() {
    // The forward record is finite and well-formed; only the recomputed
    // backward score overflows.
    let shape = mha_shape();
    let q = vec![HUGE; 4];
    let k = vec![HUGE; 4];
    let v = vec![1.0; 4];
    let d_out = vec![1.0; 4];
    let forward = FlatAttentionOutput {
        output: vec![1.0; 4],
        lse: vec![0.0; 2],
    };
    assert_non_finite_score(
        backward_reference(&q, &k, &v, &d_out, shape, unit_scale(), &forward),
        0,
        0,
    );

    let grouped = GroupedAttentionShape {
        batch: 1,
        q_heads: 1,
        kv_heads: 1,
        seq_len: 2,
        head_dim: 2,
    };
    assert_non_finite_score(
        backward_reference_grouped(&q, &k, &v, &d_out, grouped, unit_scale(), &forward),
        0,
        0,
    );
}

#[test]
fn non_finite_score_error_is_descriptive() {
    let message = FlatAttentionError::NonFiniteScore {
        query_position: 3,
        key_position: 7,
    }
    .to_string();
    assert!(message.contains("query 3"));
    assert!(message.contains("key 7"));
}
