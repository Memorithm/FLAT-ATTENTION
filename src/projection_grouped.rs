//! Projection-layout GQA/MQA oracle for direct SciRust integration.
//!
//! Unlike the canonical head-major M10/R1 layout, projection outputs are
//! sequence-major matrices:
//!
//! - Q:   `[batch * seq_len, q_heads * head_dim]`
//! - K/V: `[batch * seq_len, kv_heads * head_dim]`
//! - O:   `[batch * seq_len, q_heads * head_dim]`
//!
//! This is the layout produced by SciRust's resident Q/K/V GEMMs. Reading it
//! directly eliminates per-head column slicing and output place/add stages.

use super::{
    validate_input, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
    GroupedAttentionShape, RotaryEmbeddingConfig,
};

#[inline]
fn projection_index(
    batch: usize,
    position: usize,
    head: usize,
    dim: usize,
    heads: usize,
    shape: GroupedAttentionShape,
) -> Result<usize, FlatAttentionError> {
    let width = heads
        .checked_mul(shape.head_dim)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    batch
        .checked_mul(shape.seq_len)
        .and_then(|row| row.checked_add(position))
        .and_then(|row| row.checked_mul(width))
        .and_then(|base| {
            head.checked_mul(shape.head_dim)
                .and_then(|h| base.checked_add(h))
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

/// Fused RoPE + GQA/MQA scalar oracle over sequence-major projection buffers.
///
/// The output is immediately compatible with a row-major output-projection GEMM
/// of shape `(batch * seq_len) × (q_heads * head_dim)`.
pub fn forward_reference_projection_grouped_rope(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    rotary: RotaryEmbeddingConfig,
) -> Result<FlatAttentionOutput, FlatAttentionError> {
    shape.validate()?;
    rotary.validate(shape.head_dim, shape.seq_len)?;
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
            let lse_base = (batch * shape.q_heads + q_head) * shape.seq_len;

            for query_pos in 0..shape.seq_len {
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
                    let key_position = rotary
                        .position_offset
                        .checked_add(key_pos)
                        .ok_or(FlatAttentionError::PositionOverflow)?;
                    let mut dot = 0.0f32;
                    for pair in 0..shape.head_dim / 2 {
                        let dim = 2 * pair;
                        let qe_idx =
                            projection_index(batch, query_pos, q_head, dim, shape.q_heads, shape)?;
                        let ke_idx =
                            projection_index(batch, key_pos, kv_head, dim, shape.kv_heads, shape)?;
                        let (qe, qo) = rotated_pair(
                            q[qe_idx],
                            q[qe_idx + 1],
                            pair,
                            shape.head_dim,
                            query_position,
                            rotary.theta,
                        );
                        let (ke, ko) = rotated_pair(
                            k[ke_idx],
                            k[ke_idx + 1],
                            pair,
                            shape.head_dim,
                            key_position,
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
                        let out_idx =
                            projection_index(batch, query_pos, q_head, dim, shape.q_heads, shape)?;
                        let v_idx =
                            projection_index(batch, key_pos, kv_head, dim, shape.kv_heads, shape)?;
                        output[out_idx] =
                            output[out_idx] * alpha + probability_numerator * v[v_idx];
                    }
                    running_sum = running_sum * alpha + probability_numerator;
                    running_max = new_max;
                }

                let inv_sum = running_sum.recip();
                for dim in 0..shape.head_dim {
                    let out_idx =
                        projection_index(batch, query_pos, q_head, dim, shape.q_heads, shape)?;
                    output[out_idx] *= inv_sum;
                }
                lse[lse_base + query_pos] = running_max + running_sum.ln();
            }
        }
    }

    Ok(FlatAttentionOutput { output, lse })
}
