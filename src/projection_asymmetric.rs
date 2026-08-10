//! Asymmetric sequence-major projection-layout attention for decode/cross-attention.
//!
//! This is the rectangular counterpart of FLAT-R2. Q, K and V stay in the
//! row-major layout emitted by framework projection GEMMs, but Q and K/V may
//! have different sequence lengths. RoPE is evaluated inside the dot product;
//! no rotated tensor or score/probability matrix is materialised.

use super::{
    validate_input, AsymmetricGroupedAttentionShape, FlatAttentionConfig, FlatAttentionError,
    FlatAttentionOutput, RotaryEmbeddingConfig,
};

/// Independent RoPE position domains for rectangular attention.
///
/// Equal-length prefill normally uses the same offset for Q and K/V. Cached
/// decode uses the absolute position of the new query for `query_position_offset`
/// while the resident cache can keep its original `kv_position_offset`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsymmetricRotaryEmbeddingConfig {
    pub theta: f32,
    pub query_position_offset: usize,
    pub kv_position_offset: usize,
}

impl AsymmetricRotaryEmbeddingConfig {
    pub fn validate(
        self,
        head_dim: usize,
        query_len: usize,
        kv_len: usize,
    ) -> Result<(), FlatAttentionError> {
        RotaryEmbeddingConfig {
            theta: self.theta,
            position_offset: self.query_position_offset,
        }
        .validate(head_dim, query_len)?;
        RotaryEmbeddingConfig {
            theta: self.theta,
            position_offset: self.kv_position_offset,
        }
        .validate(head_dim, kv_len)
    }
}

#[inline]
fn projection_index(
    batch: usize,
    position: usize,
    head: usize,
    dim: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<usize, FlatAttentionError> {
    let width = heads
        .checked_mul(head_dim)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    batch
        .checked_mul(seq_len)
        .and_then(|row| row.checked_add(position))
        .and_then(|row| row.checked_mul(width))
        .and_then(|base| {
            head.checked_mul(head_dim)
                .and_then(|head_base| base.checked_add(head_base))
        })
        .and_then(|base| base.checked_add(dim))
        .ok_or(FlatAttentionError::ShapeOverflow)
}

#[inline]
fn rotated_pair(
    even: f32,
    odd: f32,
    pair: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) -> (f32, f32) {
    let exponent = -2.0 * pair as f32 / head_dim as f32;
    let frequency = theta.powf(exponent);
    let angle = position as f32 * frequency;
    let (sin, cos) = angle.sin_cos();
    (even * cos - odd * sin, even * sin + odd * cos)
}

/// Scalar oracle for rectangular projection-layout RoPE + GQA/MQA attention.
///
/// Logical storage:
///
/// - Q: `[batch * query_len, q_heads * head_dim]`
/// - K/V: `[batch * kv_len, kv_heads * head_dim]`
/// - O: `[batch * query_len, q_heads * head_dim]`
///
/// `shape.query_position_offset` belongs to causal masking. RoPE offsets are
/// intentionally independent because a decode query can sit at absolute token
/// `N-1` while its K/V cache remains indexed from an earlier origin.
pub fn forward_reference_projection_grouped_rope_asymmetric(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AsymmetricGroupedAttentionShape,
    config: FlatAttentionConfig,
    rotary: AsymmetricRotaryEmbeddingConfig,
) -> Result<FlatAttentionOutput, FlatAttentionError> {
    shape.validate()?;
    rotary.validate(shape.head_dim, shape.query_len, shape.kv_len)?;
    let q_len = shape.q_tensor_len()?;
    let kv_len = shape.kv_tensor_len()?;
    validate_input("Q", q, q_len)?;
    validate_input("K", k, kv_len)?;
    validate_input("V", v, kv_len)?;
    let scale = config.resolved_scale(shape.head_dim)?;
    let group_size = shape.q_heads / shape.kv_heads;

    let mut output = vec![0.0f32; q_len];
    let mut lse = vec![0.0f32; shape.lse_len()?];

    for batch in 0..shape.batch {
        for q_head in 0..shape.q_heads {
            let kv_head = q_head / group_size;
            let lse_base = (batch * shape.q_heads + q_head) * shape.query_len;

            for query_pos in 0..shape.query_len {
                let causal_query_position = shape
                    .query_position_offset
                    .checked_add(query_pos)
                    .ok_or(FlatAttentionError::PositionOverflow)?;
                let query_rotary_position = rotary
                    .query_position_offset
                    .checked_add(query_pos)
                    .ok_or(FlatAttentionError::PositionOverflow)?;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;

                for key_pos in 0..shape.kv_len {
                    if config.causal && key_pos > causal_query_position {
                        break;
                    }
                    let key_rotary_position = rotary
                        .kv_position_offset
                        .checked_add(key_pos)
                        .ok_or(FlatAttentionError::PositionOverflow)?;
                    let mut dot = 0.0f32;
                    for pair in 0..shape.head_dim / 2 {
                        let dim = 2 * pair;
                        let qe_idx = projection_index(
                            batch,
                            query_pos,
                            q_head,
                            dim,
                            shape.q_heads,
                            shape.query_len,
                            shape.head_dim,
                        )?;
                        let ke_idx = projection_index(
                            batch,
                            key_pos,
                            kv_head,
                            dim,
                            shape.kv_heads,
                            shape.kv_len,
                            shape.head_dim,
                        )?;
                        let (qe, qo) = rotated_pair(
                            q[qe_idx],
                            q[qe_idx + 1],
                            pair,
                            shape.head_dim,
                            query_rotary_position,
                            rotary.theta,
                        );
                        let (ke, ko) = rotated_pair(
                            k[ke_idx],
                            k[ke_idx + 1],
                            pair,
                            shape.head_dim,
                            key_rotary_position,
                            rotary.theta,
                        );
                        dot += qe * ke;
                        dot += qo * ko;
                    }

                    let score = dot * scale;
                    let new_max = running_max.max(score);
                    let alpha = if running_max.is_infinite() {
                        0.0
                    } else {
                        (running_max - new_max).exp()
                    };
                    let probability_numerator = (score - new_max).exp();

                    for dim in 0..shape.head_dim {
                        let out_idx = projection_index(
                            batch,
                            query_pos,
                            q_head,
                            dim,
                            shape.q_heads,
                            shape.query_len,
                            shape.head_dim,
                        )?;
                        let v_idx = projection_index(
                            batch,
                            key_pos,
                            kv_head,
                            dim,
                            shape.kv_heads,
                            shape.kv_len,
                            shape.head_dim,
                        )?;
                        output[out_idx] =
                            output[out_idx] * alpha + probability_numerator * v[v_idx];
                    }
                    running_sum = running_sum * alpha + probability_numerator;
                    running_max = new_max;
                }

                let inv_sum = running_sum.recip();
                for dim in 0..shape.head_dim {
                    let out_idx = projection_index(
                        batch,
                        query_pos,
                        q_head,
                        dim,
                        shape.q_heads,
                        shape.query_len,
                        shape.head_dim,
                    )?;
                    output[out_idx] *= inv_sum;
                }
                lse[lse_base + query_pos] = running_max + running_sum.ln();
            }
        }
    }

    Ok(FlatAttentionOutput { output, lse })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        forward_reference_projection_grouped_rope, GroupedAttentionShape, RotaryEmbeddingConfig,
    };

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.071) + phase).sin() * 0.625)
            .collect()
    }

    #[test]
    fn equal_length_is_bitwise_identical_to_r2_projection_oracle() {
        let equal = GroupedAttentionShape {
            batch: 2,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 5,
            head_dim: 8,
        };
        let q = fixture(equal.q_tensor_len().unwrap(), 0.2);
        let k = fixture(equal.kv_tensor_len().unwrap(), 0.8);
        let v = fixture(equal.kv_tensor_len().unwrap(), 1.4);
        let config = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };
        let old_rotary = RotaryEmbeddingConfig {
            theta: 10_000.0,
            position_offset: 17,
        };
        let expected =
            forward_reference_projection_grouped_rope(&q, &k, &v, equal, config, old_rotary)
                .unwrap();
        let shape = AsymmetricGroupedAttentionShape {
            batch: equal.batch,
            q_heads: equal.q_heads,
            kv_heads: equal.kv_heads,
            query_len: equal.seq_len,
            kv_len: equal.seq_len,
            head_dim: equal.head_dim,
            query_position_offset: 0,
        };
        let rotary = AsymmetricRotaryEmbeddingConfig {
            theta: old_rotary.theta,
            query_position_offset: old_rotary.position_offset,
            kv_position_offset: old_rotary.position_offset,
        };
        let actual =
            forward_reference_projection_grouped_rope_asymmetric(&q, &k, &v, shape, config, rotary)
                .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn decode_projection_matches_last_row_of_full_r2_forward() {
        let equal = GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 7,
            head_dim: 8,
        };
        let q = fixture(equal.q_tensor_len().unwrap(), 0.3);
        let k = fixture(equal.kv_tensor_len().unwrap(), 0.9);
        let v = fixture(equal.kv_tensor_len().unwrap(), 1.5);
        let config = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };
        let old_rotary = RotaryEmbeddingConfig {
            theta: 10_000.0,
            position_offset: 11,
        };
        let full = forward_reference_projection_grouped_rope(&q, &k, &v, equal, config, old_rotary)
            .unwrap();

        let width = equal.q_heads * equal.head_dim;
        let last_row = (equal.seq_len - 1) * width;
        let decode_q = q[last_row..last_row + width].to_vec();
        let shape = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: equal.q_heads,
            kv_heads: equal.kv_heads,
            query_len: 1,
            kv_len: equal.seq_len,
            head_dim: equal.head_dim,
            query_position_offset: equal.seq_len - 1,
        };
        let rotary = AsymmetricRotaryEmbeddingConfig {
            theta: old_rotary.theta,
            query_position_offset: old_rotary.position_offset + equal.seq_len - 1,
            kv_position_offset: old_rotary.position_offset,
        };
        let decode = forward_reference_projection_grouped_rope_asymmetric(
            &decode_q, &k, &v, shape, config, rotary,
        )
        .unwrap();
        assert_eq!(decode.output, full.output[last_row..last_row + width]);

        let mut expected_lse = Vec::with_capacity(equal.q_heads);
        for head in 0..equal.q_heads {
            expected_lse.push(full.lse[head * equal.seq_len + equal.seq_len - 1]);
        }
        assert_eq!(decode.lse, expected_lse);
    }

    #[test]
    fn independent_rotary_domains_reject_position_overflow() {
        let rotary = AsymmetricRotaryEmbeddingConfig {
            theta: 10_000.0,
            query_position_offset: usize::MAX,
            kv_position_offset: 0,
        };
        assert_eq!(
            rotary.validate(64, 2, 8).unwrap_err(),
            FlatAttentionError::PositionOverflow
        );
    }
}
