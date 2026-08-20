//! Elastic Positional Geometry (EPG) scalar reference contract.
//!
//! This module is intentionally a correctness oracle, not a claim that the
//! portable GPU kernels already implement EPG.  It establishes a versionable
//! geometry contract beside the existing RoPE oracle so future WGSL/CUDA paths
//! can be qualified against deterministic scalar behaviour.
//!
//! The first generation contains two `SO(4)` controls:
//!
//! - `Isoclinic`: both orthogonal planes in each four-channel block share one
//!   angular frequency;
//! - `Biplanar`: the two planes use the two consecutive frequencies that
//!   ordinary RoPE would assign to the same four channels.  This is an
//!   intentional equivalence control: with `so4_dims == head_dim` it must agree
//!   numerically with standard RoPE up to floating-point evaluation order.
//!
//! More expressive EPG geometries (for example learned/fixed basis changes or
//! structural coordinates) can be added as new versioned variants without
//! changing the attention contract.

use super::{
    validate_input, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
    GroupedAttentionShape,
};

/// Stable identifier for the scalar EPG contract implemented by this module.
pub const EPG_CONTRACT_VERSION: u32 = 1;

/// Four-dimensional rotation family used by the EPG tail of a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum So4Geometry {
    /// Left-isoclinic canonical control: both 2D planes share one frequency.
    Isoclinic,
    /// Canonical double rotation using the two consecutive RoPE frequencies.
    /// This is deliberately equivalent to ordinary RoPE on the same channels.
    Biplanar,
}

/// Head-local hybrid EPG configuration.
///
/// The first `head_dim - so4_dims` channels retain ordinary interleaved RoPE.
/// The final `so4_dims` channels are grouped into four-channel `SO(4)` blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpgEmbeddingConfig {
    /// RoPE-compatible base frequency.
    pub theta: f32,
    /// Absolute position added to every token index in this attention call.
    pub position_offset: usize,
    /// Number of trailing head channels assigned to four-dimensional blocks.
    /// Must be a multiple of four and no larger than `head_dim`.
    pub so4_dims: usize,
    /// Four-dimensional geometry used by the trailing blocks.
    pub so4_geometry: So4Geometry,
}

impl EpgEmbeddingConfig {
    /// Validate the geometry against one attention head.
    pub fn validate(self, head_dim: usize, seq_len: usize) -> Result<(), FlatAttentionError> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(FlatAttentionError::InvalidRotaryHeadDim { head_dim });
        }
        if self.so4_dims > head_dim || !self.so4_dims.is_multiple_of(4) {
            return Err(FlatAttentionError::InvalidEpgSo4Dims {
                head_dim,
                so4_dims: self.so4_dims,
            });
        }
        let so2_dims = head_dim - self.so4_dims;
        if !so2_dims.is_multiple_of(2) {
            return Err(FlatAttentionError::InvalidEpgSo4Dims {
                head_dim,
                so4_dims: self.so4_dims,
            });
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(FlatAttentionError::InvalidRotaryTheta(self.theta));
        }
        self.position_offset
            .checked_add(seq_len.saturating_sub(1))
            .ok_or(FlatAttentionError::PositionOverflow)?;
        Ok(())
    }

    /// Number of leading channels that remain ordinary `SO(2)` RoPE.
    #[inline]
    pub const fn so2_dims(self, head_dim: usize) -> usize {
        head_dim - self.so4_dims
    }
}

#[inline]
fn frequency(theta: f32, pair: usize, head_dim: usize) -> f32 {
    let exponent = -2.0 * pair as f32 / head_dim as f32;
    theta.powf(exponent)
}

#[inline]
fn rotate_pair(even: f32, odd: f32, angle: f32) -> (f32, f32) {
    let (sin, cos) = angle.sin_cos();
    (even * cos - odd * sin, even * sin + odd * cos)
}

#[inline]
fn rotate_so4_block(
    block: [f32; 4],
    first_pair: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
    geometry: So4Geometry,
) -> [f32; 4] {
    let omega_01 = frequency(theta, first_pair, head_dim);
    let omega_23 = match geometry {
        So4Geometry::Isoclinic => omega_01,
        So4Geometry::Biplanar => frequency(theta, first_pair + 1, head_dim),
    };
    let (r0, r1) = rotate_pair(block[0], block[1], position as f32 * omega_01);
    let (r2, r3) = rotate_pair(block[2], block[3], position as f32 * omega_23);
    [r0, r1, r2, r3]
}

fn epg_dot(
    q: &[f32],
    k: &[f32],
    head_dim: usize,
    query_position: usize,
    key_position: usize,
    epg: EpgEmbeddingConfig,
) -> f32 {
    let so2_dims = epg.so2_dims(head_dim);
    let mut dot = 0.0f32;

    for pair in 0..so2_dims / 2 {
        let dim = 2 * pair;
        let omega = frequency(epg.theta, pair, head_dim);
        let (qe, qo) = rotate_pair(q[dim], q[dim + 1], query_position as f32 * omega);
        let (ke, ko) = rotate_pair(k[dim], k[dim + 1], key_position as f32 * omega);
        dot += qe * ke + qo * ko;
    }

    let first_so4_pair = so2_dims / 2;
    for block_index in 0..epg.so4_dims / 4 {
        let dim = so2_dims + 4 * block_index;
        let first_pair = first_so4_pair + 2 * block_index;
        let qr = rotate_so4_block(
            [q[dim], q[dim + 1], q[dim + 2], q[dim + 3]],
            first_pair,
            head_dim,
            query_position,
            epg.theta,
            epg.so4_geometry,
        );
        let kr = rotate_so4_block(
            [k[dim], k[dim + 1], k[dim + 2], k[dim + 3]],
            first_pair,
            head_dim,
            key_position,
            epg.theta,
            epg.so4_geometry,
        );
        dot += qr[0] * kr[0] + qr[1] * kr[1] + qr[2] * kr[2] + qr[3] * kr[3];
    }

    dot
}

/// Deterministic scalar oracle for hybrid EPG + native grouped attention.
///
/// Q and K are raw projection outputs and V is never rotated.  Rotations are
/// evaluated directly inside each Q·K dot product; no rotated Q/K tensor and no
/// N×N score/probability matrix is materialized.
pub fn forward_reference_grouped_epg(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    epg: EpgEmbeddingConfig,
) -> Result<FlatAttentionOutput, FlatAttentionError> {
    shape.validate()?;
    epg.validate(shape.head_dim, shape.seq_len)?;
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
                let query_position = epg
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
                    let key_position = epg
                        .position_offset
                        .checked_add(key_pos)
                        .ok_or(FlatAttentionError::PositionOverflow)?;
                    let dot = epg_dot(
                        &q[q_base..q_base + shape.head_dim],
                        &k[kv_base..kv_base + shape.head_dim],
                        shape.head_dim,
                        query_position,
                        key_position,
                        epg,
                    );

                    let score = dot * scale;
                    let new_max = running_max.max(score);
                    let alpha = if running_max.is_infinite() {
                        0.0
                    } else {
                        (running_max - new_max).exp()
                    };
                    let probability_numerator = (score - new_max).exp();

                    for dim in 0..shape.head_dim {
                        output[q_base + dim] =
                            output[q_base + dim] * alpha + probability_numerator * v[kv_base + dim];
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
    use crate::{forward_reference_grouped_rope, RotaryEmbeddingConfig};

    fn fixture(shape: GroupedAttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_len = shape.q_tensor_len().unwrap();
        let kv_len = shape.kv_tensor_len().unwrap();
        let q = (0..q_len).map(|i| ((i * 17 + 3) % 101) as f32 / 53.0 - 0.9).collect();
        let k = (0..kv_len).map(|i| ((i * 29 + 7) % 103) as f32 / 59.0 - 0.8).collect();
        let v = (0..kv_len).map(|i| ((i * 11 + 5) % 97) as f32 / 47.0 - 1.0).collect();
        (q, k, v)
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "index {i}: {x} != {y}");
        }
    }

    #[test]
    fn zero_so4_dims_matches_existing_rope_oracle() {
        let shape = GroupedAttentionShape { batch: 1, q_heads: 4, kv_heads: 2, seq_len: 5, head_dim: 8 };
        let (q, k, v) = fixture(shape);
        let cfg = FlatAttentionConfig { causal: true, softmax_scale: None };
        let theta = 10_000.0;
        let rope = forward_reference_grouped_rope(
            &q, &k, &v, shape, cfg,
            RotaryEmbeddingConfig { theta, position_offset: 3 },
        ).unwrap();
        let epg = forward_reference_grouped_epg(
            &q, &k, &v, shape, cfg,
            EpgEmbeddingConfig { theta, position_offset: 3, so4_dims: 0, so4_geometry: So4Geometry::Biplanar },
        ).unwrap();
        assert_close(&rope.output, &epg.output, 1e-6);
        assert_close(&rope.lse, &epg.lse, 1e-6);
    }

    #[test]
    fn biplanar_full_head_is_an_equivalence_control_for_rope() {
        let shape = GroupedAttentionShape { batch: 1, q_heads: 2, kv_heads: 1, seq_len: 6, head_dim: 8 };
        let (q, k, v) = fixture(shape);
        let cfg = FlatAttentionConfig { causal: false, softmax_scale: None };
        let theta = 10_000.0;
        let rope = forward_reference_grouped_rope(
            &q, &k, &v, shape, cfg,
            RotaryEmbeddingConfig { theta, position_offset: 11 },
        ).unwrap();
        let epg = forward_reference_grouped_epg(
            &q, &k, &v, shape, cfg,
            EpgEmbeddingConfig { theta, position_offset: 11, so4_dims: 8, so4_geometry: So4Geometry::Biplanar },
        ).unwrap();
        assert_close(&rope.output, &epg.output, 2e-6);
        assert_close(&rope.lse, &epg.lse, 2e-6);
    }

    #[test]
    fn rejects_non_multiple_of_four_so4_tail() {
        let error = EpgEmbeddingConfig {
            theta: 10_000.0,
            position_offset: 0,
            so4_dims: 6,
            so4_geometry: So4Geometry::Isoclinic,
        }.validate(8, 4).unwrap_err();
        assert_eq!(error, FlatAttentionError::InvalidEpgSo4Dims { head_dim: 8, so4_dims: 6 });
    }

    #[test]
    fn isoclinic_dot_depends_only_on_relative_offset() {
        let q = [0.3, -0.2, 0.8, 1.1];
        let k = [-0.7, 0.4, 0.2, -1.3];
        let epg = EpgEmbeddingConfig {
            theta: 10_000.0,
            position_offset: 0,
            so4_dims: 4,
            so4_geometry: So4Geometry::Isoclinic,
        };
        let a = epg_dot(&q, &k, 4, 23, 7, epg);
        let b = epg_dot(&q, &k, 4, 119, 103, epg);
        assert!((a - b).abs() < 2e-5, "{a} != {b}");
    }
}
