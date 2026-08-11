//! M16 deterministic chunked-prefill contract for projection-layout GQA/MQA.
//!
//! This reference path partitions query rows into caller-selected chunks while
//! keeping K/V at native grouped-query cardinality. Each chunk is evaluated by
//! the already-qualified M11 asymmetric projection oracle with absolute causal
//! and RoPE positions preserved. Chunk boundaries therefore do not change the
//! per-query K/V visitation order or online-softmax arithmetic.

use core::fmt;

use super::{
    forward_reference_projection_grouped_rope_asymmetric, validate_input,
    AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig, FlatAttentionConfig,
    FlatAttentionError, FlatAttentionOutput, GroupedAttentionShape, RotaryEmbeddingConfig,
};

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkedProjectionPrefillError {
    Core(FlatAttentionError),
    ZeroQueryChunkSize,
}

impl fmt::Display for ChunkedProjectionPrefillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::ZeroQueryChunkSize => {
                write!(f, "chunked projection prefill requires a non-zero query chunk size")
            }
        }
    }
}

impl std::error::Error for ChunkedProjectionPrefillError {}

impl From<FlatAttentionError> for ChunkedProjectionPrefillError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

/// Deterministic projection-layout chunked-prefill oracle.
///
/// Logical storage remains exactly the SciRust projection layout:
///
/// - Q: `[batch, seq_len, q_heads * head_dim]`
/// - K/V: `[batch, seq_len, kv_heads * head_dim]`
/// - O: `[batch, seq_len, q_heads * head_dim]`
///
/// Only Q/O/LSE are partitioned into chunks. K/V are never expanded to query
/// head cardinality and no score/probability matrix is materialized. The
/// function makes no performance claim; it defines the correctness contract for
/// a later resident GPU chunked-prefill implementation.
pub fn forward_reference_projection_grouped_rope_chunked_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    rotary: RotaryEmbeddingConfig,
    query_chunk_size: usize,
) -> Result<FlatAttentionOutput, ChunkedProjectionPrefillError> {
    if query_chunk_size == 0 {
        return Err(ChunkedProjectionPrefillError::ZeroQueryChunkSize);
    }
    shape.validate()?;
    rotary.validate(shape.head_dim, shape.seq_len)?;
    let q_len = shape.q_tensor_len()?;
    let kv_len = shape.kv_tensor_len()?;
    validate_input("Q", q, q_len)?;
    validate_input("K", k, kv_len)?;
    validate_input("V", v, kv_len)?;

    let q_width = shape
        .q_heads
        .checked_mul(shape.head_dim)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let mut output = vec![0.0f32; q_len];
    let mut lse = vec![0.0f32; shape.lse_len()?];

    let mut query_start = 0usize;
    while query_start < shape.seq_len {
        let chunk_len = query_chunk_size.min(shape.seq_len - query_start);
        let chunk_q_len = shape
            .batch
            .checked_mul(chunk_len)
            .and_then(|rows| rows.checked_mul(q_width))
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let mut chunk_q = Vec::with_capacity(chunk_q_len);
        for batch in 0..shape.batch {
            let row_start = batch
                .checked_mul(shape.seq_len)
                .and_then(|row| row.checked_add(query_start))
                .and_then(|row| row.checked_mul(q_width))
                .ok_or(FlatAttentionError::ShapeOverflow)?;
            let row_end = row_start
                .checked_add(
                    chunk_len
                        .checked_mul(q_width)
                        .ok_or(FlatAttentionError::ShapeOverflow)?,
                )
                .ok_or(FlatAttentionError::ShapeOverflow)?;
            chunk_q.extend_from_slice(&q[row_start..row_end]);
        }

        let chunk_shape = AsymmetricGroupedAttentionShape {
            batch: shape.batch,
            q_heads: shape.q_heads,
            kv_heads: shape.kv_heads,
            query_len: chunk_len,
            kv_len: shape.seq_len,
            head_dim: shape.head_dim,
            query_position_offset: query_start,
        };
        let chunk_rotary = AsymmetricRotaryEmbeddingConfig {
            theta: rotary.theta,
            query_position_offset: rotary
                .position_offset
                .checked_add(query_start)
                .ok_or(FlatAttentionError::PositionOverflow)?,
            kv_position_offset: rotary.position_offset,
        };
        let chunk = forward_reference_projection_grouped_rope_asymmetric(
            &chunk_q,
            k,
            v,
            chunk_shape,
            config,
            chunk_rotary,
        )?;

        for batch in 0..shape.batch {
            let chunk_row_base = batch
                .checked_mul(chunk_len)
                .and_then(|row| row.checked_mul(q_width))
                .ok_or(FlatAttentionError::ShapeOverflow)?;
            let full_row_base = batch
                .checked_mul(shape.seq_len)
                .and_then(|row| row.checked_add(query_start))
                .and_then(|row| row.checked_mul(q_width))
                .ok_or(FlatAttentionError::ShapeOverflow)?;
            let elements = chunk_len
                .checked_mul(q_width)
                .ok_or(FlatAttentionError::ShapeOverflow)?;
            output[full_row_base..full_row_base + elements]
                .copy_from_slice(&chunk.output[chunk_row_base..chunk_row_base + elements]);

            for q_head in 0..shape.q_heads {
                let chunk_lse_base = (batch * shape.q_heads + q_head) * chunk_len;
                let full_lse_base =
                    (batch * shape.q_heads + q_head) * shape.seq_len + query_start;
                lse[full_lse_base..full_lse_base + chunk_len]
                    .copy_from_slice(&chunk.lse[chunk_lse_base..chunk_lse_base + chunk_len]);
            }
        }

        query_start += chunk_len;
    }

    Ok(FlatAttentionOutput { output, lse })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_reference_projection_grouped_rope;

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.021 + phase;
                x.sin() * 1.125 + (x * 0.43).cos() * 0.3125
            })
            .collect()
    }

    fn assert_bit_exact(shape: GroupedAttentionShape, causal: bool, chunk_size: usize) {
        let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
        let k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
        let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let rotary = RotaryEmbeddingConfig {
            theta: 10_000.0,
            position_offset: 7,
        };
        let contiguous =
            forward_reference_projection_grouped_rope(&q, &k, &v, shape, config, rotary).unwrap();
        let chunked = forward_reference_projection_grouped_rope_chunked_prefill(
            &q,
            &k,
            &v,
            shape,
            config,
            rotary,
            chunk_size,
        )
        .unwrap();
        assert_eq!(chunked, contiguous);
    }

    #[test]
    fn chunk_boundaries_are_bit_exact_for_gqa_and_mqa() {
        for causal in [false, true] {
            for chunk_size in [1usize, 2, 3, 4, 8, 16] {
                assert_bit_exact(
                    GroupedAttentionShape {
                        batch: 2,
                        q_heads: 4,
                        kv_heads: 2,
                        seq_len: 9,
                        head_dim: 8,
                    },
                    causal,
                    chunk_size,
                );
                assert_bit_exact(
                    GroupedAttentionShape {
                        batch: 2,
                        q_heads: 4,
                        kv_heads: 1,
                        seq_len: 7,
                        head_dim: 8,
                    },
                    causal,
                    chunk_size,
                );
            }
        }
    }

    #[test]
    fn oversized_chunk_is_bit_exact_to_contiguous_prefill() {
        assert_bit_exact(
            GroupedAttentionShape {
                batch: 1,
                q_heads: 6,
                kv_heads: 2,
                seq_len: 5,
                head_dim: 16,
            },
            true,
            64,
        );
    }

    #[test]
    fn zero_query_chunk_size_is_rejected() {
        let shape = GroupedAttentionShape {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            seq_len: 1,
            head_dim: 2,
        };
        assert_eq!(
            forward_reference_projection_grouped_rope_chunked_prefill(
                &[1.0, 2.0],
                &[3.0, 4.0],
                &[5.0, 6.0],
                shape,
                FlatAttentionConfig::default(),
                RotaryEmbeddingConfig {
                    theta: 10_000.0,
                    position_offset: 0,
                },
                0,
            ),
            Err(ChunkedProjectionPrefillError::ZeroQueryChunkSize)
        );
    }
}
