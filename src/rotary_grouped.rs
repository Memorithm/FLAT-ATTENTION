//! FLAT-R1 fused head-local RoPE + GQA/MQA reference contract.
//!
//! This oracle intentionally does not materialize rotated Q/K tensors. RoPE is
//! evaluated pairwise inside each Q·K dot product, matching the fused GPU
//! design where rotation happens after staging Q/K into workgroup memory.

use super::{
    validate_input, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
    GroupedAttentionShape,
};

/// Head-local interleaved rotary embedding parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotaryEmbeddingConfig {
    /// RoPE base frequency (`theta` in SciRust's GQA block).
    pub theta: f32,
    /// Absolute position added to every token index in this attention call.
    pub position_offset: usize,
}

impl RotaryEmbeddingConfig {
    pub fn validate(self, head_dim: usize, seq_len: usize) -> Result<(), FlatAttentionError> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(FlatAttentionError::InvalidRotaryHeadDim { head_dim });
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(FlatAttentionError::InvalidRotaryTheta(self.theta));
        }
        self.position_offset
            .checked_add(seq_len.saturating_sub(1))
            .ok_or(FlatAttentionError::PositionOverflow)?;
        Ok(())
    }
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

/// Deterministic scalar oracle for fused head-local RoPE + GQA/MQA attention.
///
/// Layouts are the native grouped layouts from [`GroupedAttentionShape`]. Q and
/// K are raw projection outputs. V is never rotated. No rotated Q/K tensor and
/// no N×N score/probability matrix is allocated.
pub fn forward_reference_grouped_rope(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    rotary: RotaryEmbeddingConfig,
) -> Result<FlatAttentionOutput, FlatAttentionError> {
    shape.validate()?;
    rotary.validate(shape.head_dim, shape.seq_len)?;
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
                let query_position = rotary
                    .position_offset
                    .checked_add(query_pos)
                    .ok_or(FlatAttentionError::PositionOverflow)?;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;

                for key_pos in 0..shape.seq_len {
                    if config.causal && key_pos > query_pos {
                        break;
                    }
                    let kv_base = kv_head_base + key_pos * shape.head_dim;
                    let key_position = rotary
                        .position_offset
                        .checked_add(key_pos)
                        .ok_or(FlatAttentionError::PositionOverflow)?;
                    let mut dot = 0.0f32;
                    for pair in 0..shape.head_dim / 2 {
                        let dim = 2 * pair;
                        let (qe, qo) = rotated_pair(
                            q[q_base + dim],
                            q[q_base + dim + 1],
                            pair,
                            shape.head_dim,
                            query_position,
                            rotary.theta,
                        );
                        let (ke, ko) = rotated_pair(
                            k[kv_base + dim],
                            k[kv_base + dim + 1],
                            pair,
                            shape.head_dim,
                            key_position,
                            rotary.theta,
                        );
                        dot += qe * ke + qo * ko;
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
                        output[q_base + dim] = output[q_base + dim] * alpha
                            + probability_numerator * v[kv_base + dim];
                    }
                    running_sum = running_sum * alpha + probability_numerator;
                    running_max = new_max;
                }

                let inv_sum = running_sum.recip();
                for dim in 0..shape.head_dim {
                    output[q_base + dim] *= inv_sum;
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
    fn rejects_odd_head_dim() {
        let error = RotaryEmbeddingConfig {
            theta: 10_000.0,
            position_offset: 0,
        }
        .validate(63, 4)
        .unwrap_err();
        assert_eq!(
            error,
            FlatAttentionError::InvalidRotaryHeadDim { head_dim: 63 }
        );
    }

    #[test]
    fn rejects_invalid_theta() {
        for theta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                RotaryEmbeddingConfig {
                    theta,
                    position_offset: 0
                }
                .validate(64, 4),
                Err(FlatAttentionError::InvalidRotaryTheta(_))
            ));
        }
    }
}
