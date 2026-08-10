//! Asymmetric grouped-query attention contract for decode and KV-cache paths.
//!
//! Unlike the equal-length M10 contract, this module represents query and KV
//! sequence lengths independently. That is the shape needed by autoregressive
//! decode: one (or a small batch of) new query row(s) can attend directly over a
//! longer resident K/V cache without materialising or replaying older queries.

use super::{
    validate_input, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
    GroupedAttentionShape,
};

/// Native GQA/MQA shape with independent query and key/value sequence lengths.
///
/// Logical layouts:
///
/// - Q: `[batch, q_heads, query_len, head_dim]`
/// - K/V: `[batch, kv_heads, kv_len, head_dim]`
/// - O: `[batch, q_heads, query_len, head_dim]`
/// - LSE: `[batch, q_heads, query_len]`
///
/// `query_position_offset` maps local query row `q` to absolute causal position
/// `query_position_offset + q`. Key positions are absolute `[0, kv_len)`. For a
/// single-token decode over an `N`-row cache, use `query_len = 1`, `kv_len = N`
/// and `query_position_offset = N - 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsymmetricGroupedAttentionShape {
    pub batch: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub query_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
    pub query_position_offset: usize,
}

impl AsymmetricGroupedAttentionShape {
    pub fn q_tensor_len(self) -> Result<usize, FlatAttentionError> {
        self.batch
            .checked_mul(self.q_heads)
            .and_then(|n| n.checked_mul(self.query_len))
            .and_then(|n| n.checked_mul(self.head_dim))
            .ok_or(FlatAttentionError::ShapeOverflow)
    }

    pub fn kv_tensor_len(self) -> Result<usize, FlatAttentionError> {
        self.batch
            .checked_mul(self.kv_heads)
            .and_then(|n| n.checked_mul(self.kv_len))
            .and_then(|n| n.checked_mul(self.head_dim))
            .ok_or(FlatAttentionError::ShapeOverflow)
    }

    pub fn lse_len(self) -> Result<usize, FlatAttentionError> {
        self.batch
            .checked_mul(self.q_heads)
            .and_then(|n| n.checked_mul(self.query_len))
            .ok_or(FlatAttentionError::ShapeOverflow)
    }

    pub fn group_size(self) -> Result<usize, FlatAttentionError> {
        self.validate()?;
        Ok(self.q_heads / self.kv_heads)
    }

    pub(crate) fn validate(self) -> Result<(), FlatAttentionError> {
        if self.batch == 0
            || self.q_heads == 0
            || self.kv_heads == 0
            || self.query_len == 0
            || self.kv_len == 0
            || self.head_dim == 0
        {
            return Err(FlatAttentionError::ZeroDimension);
        }
        if self.q_heads % self.kv_heads != 0 {
            return Err(FlatAttentionError::InvalidHeadGrouping {
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
            });
        }
        self.query_position_offset
            .checked_add(self.query_len - 1)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        self.q_tensor_len()?;
        self.kv_tensor_len()?;
        self.lse_len()?;
        Ok(())
    }
}

impl From<GroupedAttentionShape> for AsymmetricGroupedAttentionShape {
    fn from(shape: GroupedAttentionShape) -> Self {
        Self {
            batch: shape.batch,
            q_heads: shape.q_heads,
            kv_heads: shape.kv_heads,
            query_len: shape.seq_len,
            kv_len: shape.seq_len,
            head_dim: shape.head_dim,
            query_position_offset: 0,
        }
    }
}

/// Deterministic asymmetric GQA/MQA oracle using online softmax.
///
/// The function never allocates an attention score/probability matrix and never
/// expands K/V to the query-head count. In causal mode, masking is evaluated in
/// absolute positions, so decode (`Q=1`, `KV=N`) is mathematically identical to
/// the corresponding last row of an equal-length causal forward.
pub fn forward_reference_grouped_asymmetric(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AsymmetricGroupedAttentionShape,
    config: FlatAttentionConfig,
) -> Result<FlatAttentionOutput, FlatAttentionError> {
    shape.validate()?;
    let q_tensor_len = shape.q_tensor_len()?;
    let kv_tensor_len = shape.kv_tensor_len()?;
    validate_input("Q", q, q_tensor_len)?;
    validate_input("K", k, kv_tensor_len)?;
    validate_input("V", v, kv_tensor_len)?;
    let scale = config.resolved_scale(shape.head_dim)?;
    let group_size = shape.q_heads / shape.kv_heads;

    let mut output = vec![0.0f32; q_tensor_len];
    let mut lse = vec![0.0f32; shape.lse_len()?];
    let q_head_stride = shape.query_len * shape.head_dim;
    let kv_head_stride = shape.kv_len * shape.head_dim;

    for batch in 0..shape.batch {
        for q_head in 0..shape.q_heads {
            let kv_head = q_head / group_size;
            let q_bh = batch * shape.q_heads + q_head;
            let kv_bh = batch * shape.kv_heads + kv_head;
            let q_head_base = q_bh * q_head_stride;
            let kv_head_base = kv_bh * kv_head_stride;
            let lse_base = q_bh * shape.query_len;

            for query_pos in 0..shape.query_len {
                let absolute_query_pos = shape.query_position_offset + query_pos;
                let q_base = q_head_base + query_pos * shape.head_dim;
                let out_base = q_base;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;

                for key_pos in 0..shape.kv_len {
                    if config.causal && key_pos > absolute_query_pos {
                        break;
                    }

                    let kv_base = kv_head_base + key_pos * shape.head_dim;
                    let mut dot = 0.0f32;
                    for dim in 0..shape.head_dim {
                        dot += q[q_base + dim] * k[kv_base + dim];
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
                        output[out_base + dim] = output[out_base + dim] * alpha
                            + probability_numerator * v[kv_base + dim];
                    }
                    running_sum = running_sum * alpha + probability_numerator;
                    running_max = new_max;
                }

                let inv_sum = running_sum.recip();
                for dim in 0..shape.head_dim {
                    output[out_base + dim] *= inv_sum;
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
    use crate::forward_reference_grouped;

    fn deterministic_values(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.173) + phase).sin() * 0.7)
            .collect()
    }

    #[test]
    fn equal_length_contract_is_bitwise_identical_to_m10_oracle() {
        let equal = GroupedAttentionShape {
            batch: 2,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 5,
            head_dim: 8,
        };
        let asymmetric = AsymmetricGroupedAttentionShape::from(equal);
        let q = deterministic_values(equal.q_tensor_len().unwrap(), 0.1);
        let k = deterministic_values(equal.kv_tensor_len().unwrap(), 0.7);
        let v = deterministic_values(equal.kv_tensor_len().unwrap(), 1.3);

        for causal in [false, true] {
            let config = FlatAttentionConfig {
                causal,
                softmax_scale: None,
            };
            let expected = forward_reference_grouped(&q, &k, &v, equal, config).unwrap();
            let actual =
                forward_reference_grouped_asymmetric(&q, &k, &v, asymmetric, config).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn single_query_decode_matches_last_row_of_full_causal_forward() {
        let equal = GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 6,
            head_dim: 8,
        };
        let q = deterministic_values(equal.q_tensor_len().unwrap(), 0.2);
        let k = deterministic_values(equal.kv_tensor_len().unwrap(), 0.8);
        let v = deterministic_values(equal.kv_tensor_len().unwrap(), 1.4);
        let config = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };
        let full = forward_reference_grouped(&q, &k, &v, equal, config).unwrap();

        let mut decode_q = Vec::with_capacity(equal.q_heads * equal.head_dim);
        for head in 0..equal.q_heads {
            let head_base = head * equal.seq_len * equal.head_dim;
            let row_base = head_base + (equal.seq_len - 1) * equal.head_dim;
            decode_q.extend_from_slice(&q[row_base..row_base + equal.head_dim]);
        }
        let decode_shape = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: equal.q_heads,
            kv_heads: equal.kv_heads,
            query_len: 1,
            kv_len: equal.seq_len,
            head_dim: equal.head_dim,
            query_position_offset: equal.seq_len - 1,
        };
        let decode =
            forward_reference_grouped_asymmetric(&decode_q, &k, &v, decode_shape, config).unwrap();

        let mut expected_output = Vec::with_capacity(equal.q_heads * equal.head_dim);
        let mut expected_lse = Vec::with_capacity(equal.q_heads);
        for head in 0..equal.q_heads {
            let out_head_base = head * equal.seq_len * equal.head_dim;
            let out_row_base = out_head_base + (equal.seq_len - 1) * equal.head_dim;
            expected_output
                .extend_from_slice(&full.output[out_row_base..out_row_base + equal.head_dim]);
            expected_lse.push(full.lse[head * equal.seq_len + equal.seq_len - 1]);
        }

        assert_eq!(decode.output, expected_output);
        assert_eq!(decode.lse, expected_lse);
    }

    #[test]
    fn lengths_keep_kv_cache_physical_and_unexpanded() {
        let shape = AsymmetricGroupedAttentionShape {
            batch: 2,
            q_heads: 16,
            kv_heads: 2,
            query_len: 1,
            kv_len: 4096,
            head_dim: 64,
            query_position_offset: 4095,
        };
        assert_eq!(shape.q_tensor_len().unwrap(), 2 * 16 * 64);
        assert_eq!(shape.kv_tensor_len().unwrap(), 2 * 2 * 4096 * 64);
        assert_eq!(shape.lse_len().unwrap(), 2 * 16);
        assert_eq!(shape.group_size().unwrap(), 8);
    }

    #[test]
    fn rejects_non_divisible_groups_and_zero_lengths() {
        let invalid_group = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 6,
            kv_heads: 4,
            query_len: 1,
            kv_len: 8,
            head_dim: 32,
            query_position_offset: 7,
        };
        assert_eq!(
            invalid_group.group_size().unwrap_err(),
            FlatAttentionError::InvalidHeadGrouping {
                q_heads: 6,
                kv_heads: 4,
            }
        );

        let zero_query = AsymmetricGroupedAttentionShape {
            query_len: 0,
            ..invalid_group
        };
        assert_eq!(
            zero_query.group_size().unwrap_err(),
            FlatAttentionError::ZeroDimension
        );
    }
}
