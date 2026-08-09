//! Native grouped-query and multi-query attention contracts.
//!
//! GQA/MQA are represented without expanding K/V to the query-head count.
//! Query head `qh` reads KV head `qh / group_size`, where
//! `group_size = q_heads / kv_heads`.

use super::{validate_input, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput};

/// Canonical equal-length GQA/MQA shape for M10.
///
/// Logical layouts:
///
/// - Q: `[batch, q_heads, seq_len, head_dim]`
/// - K/V: `[batch, kv_heads, seq_len, head_dim]`
/// - O: `[batch, q_heads, seq_len, head_dim]`
/// - LSE: `[batch, q_heads, seq_len]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedAttentionShape {
    pub batch: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
}

impl GroupedAttentionShape {
    pub fn q_tensor_len(self) -> Result<usize, FlatAttentionError> {
        self.batch
            .checked_mul(self.q_heads)
            .and_then(|n| n.checked_mul(self.seq_len))
            .and_then(|n| n.checked_mul(self.head_dim))
            .ok_or(FlatAttentionError::ShapeOverflow)
    }

    pub fn kv_tensor_len(self) -> Result<usize, FlatAttentionError> {
        self.batch
            .checked_mul(self.kv_heads)
            .and_then(|n| n.checked_mul(self.seq_len))
            .and_then(|n| n.checked_mul(self.head_dim))
            .ok_or(FlatAttentionError::ShapeOverflow)
    }

    pub fn lse_len(self) -> Result<usize, FlatAttentionError> {
        self.batch
            .checked_mul(self.q_heads)
            .and_then(|n| n.checked_mul(self.seq_len))
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
            || self.seq_len == 0
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
        self.q_tensor_len()?;
        self.kv_tensor_len()?;
        self.lse_len()?;
        Ok(())
    }
}

/// Deterministic scalar GQA/MQA oracle using online softmax.
///
/// K/V are indexed at their physical KV-head cardinality. No expanded K/V
/// tensor is created, including for MQA (`kv_heads == 1`).
pub fn forward_reference_grouped(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: GroupedAttentionShape,
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
    let head_stride = shape.seq_len * shape.head_dim;

    for batch in 0..shape.batch {
        for q_head in 0..shape.q_heads {
            let kv_head = q_head / group_size;
            let q_bh = batch * shape.q_heads + q_head;
            let kv_bh = batch * shape.kv_heads + kv_head;
            let q_head_base = q_bh * head_stride;
            let kv_head_base = kv_bh * head_stride;
            let lse_base = q_bh * shape.seq_len;

            for query_pos in 0..shape.seq_len {
                let q_base = q_head_base + query_pos * shape.head_dim;
                let out_base = q_base;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;

                for key_pos in 0..shape.seq_len {
                    if config.causal && key_pos > query_pos {
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

    #[test]
    fn grouping_contract_accepts_mha_gqa_and_mqa() {
        for (q_heads, kv_heads, group_size) in [(8, 8, 1), (8, 2, 4), (8, 1, 8)] {
            let shape = GroupedAttentionShape {
                batch: 2,
                q_heads,
                kv_heads,
                seq_len: 3,
                head_dim: 4,
            };
            assert_eq!(shape.group_size().unwrap(), group_size);
        }
    }

    #[test]
    fn rejects_non_divisible_head_groups() {
        let shape = GroupedAttentionShape {
            batch: 1,
            q_heads: 6,
            kv_heads: 4,
            seq_len: 2,
            head_dim: 8,
        };
        assert_eq!(
            shape.group_size().unwrap_err(),
            FlatAttentionError::InvalidHeadGrouping {
                q_heads: 6,
                kv_heads: 4
            }
        );
    }

    #[test]
    fn grouped_lengths_preserve_physical_kv_cardinality() {
        let shape = GroupedAttentionShape {
            batch: 2,
            q_heads: 8,
            kv_heads: 2,
            seq_len: 16,
            head_dim: 64,
        };
        assert_eq!(shape.q_tensor_len().unwrap(), 2 * 8 * 16 * 64);
        assert_eq!(shape.kv_tensor_len().unwrap(), 2 * 2 * 16 * 64);
        assert_eq!(shape.lse_len().unwrap(), 2 * 8 * 16);
    }
}
