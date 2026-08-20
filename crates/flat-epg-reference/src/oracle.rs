use crate::{rotation::epg_dot, EpgEmbeddingConfig, EpgError};
use flat_attention::{FlatAttentionConfig, FlatAttentionOutput, GroupedAttentionShape};

fn validate_input(name: &'static str, data: &[f32], expected: usize) -> Result<(), EpgError> {
    if data.len() != expected {
        return Err(EpgError::LengthMismatch {
            tensor: name,
            actual: data.len(),
            expected,
        });
    }
    if let Some(index) = data.iter().position(|x| !x.is_finite()) {
        return Err(EpgError::NonFiniteInput {
            tensor: name,
            index,
        });
    }
    Ok(())
}

/// Deterministic scalar oracle for hybrid EPG + native GQA/MQA attention.
///
/// Q and K are raw projection outputs. V is never transformed. Rotations are
/// evaluated inside each dot product, so no rotated Q/K tensor and no N×N
/// score/probability matrix is materialized.
pub fn forward_reference_grouped_epg(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    epg: EpgEmbeddingConfig,
) -> Result<FlatAttentionOutput, EpgError> {
    let group_size = shape.group_size()?;
    epg.validate(shape.head_dim, shape.seq_len)?;
    let q_tensor_len = shape.q_tensor_len()?;
    let kv_tensor_len = shape.kv_tensor_len()?;
    let lse_len = shape.lse_len()?;
    validate_input("Q", q, q_tensor_len)?;
    validate_input("K", k, kv_tensor_len)?;
    validate_input("V", v, kv_tensor_len)?;
    let scale = config.resolved_scale(shape.head_dim)?;

    let mut output = vec![0.0f32; q_tensor_len];
    let mut lse = vec![0.0f32; lse_len];
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
                let query_position = epg.resolve_position(query_pos)?;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;

                for key_pos in 0..shape.seq_len {
                    if config.causal && key_pos > query_pos {
                        break;
                    }
                    let kv_base = kv_head_base + key_pos * shape.head_dim;
                    let key_position = epg.resolve_position(key_pos)?;
                    let dot = epg_dot(
                        &q[q_base..q_base + shape.head_dim],
                        &k[kv_base..kv_base + shape.head_dim],
                        shape.head_dim,
                        query_position,
                        key_position,
                        epg,
                    )?;
                    let score = dot * scale;
                    let new_max = running_max.max(score);
                    let alpha = if running_max.is_infinite() {
                        0.0
                    } else {
                        (running_max - new_max).exp()
                    };
                    let numerator = (score - new_max).exp();

                    for dim in 0..shape.head_dim {
                        output[q_base + dim] =
                            output[q_base + dim] * alpha + numerator * v[kv_base + dim];
                    }
                    running_sum = running_sum * alpha + numerator;
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
    use crate::{rotation::epg_dot, So4Geometry};
    use flat_attention::{forward_reference_grouped_rope, RotaryEmbeddingConfig};

    fn fixture(shape: GroupedAttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_len = shape.q_tensor_len().unwrap();
        let kv_len = shape.kv_tensor_len().unwrap();
        let q = (0..q_len)
            .map(|i| ((i * 17 + 3) % 101) as f32 / 53.0 - 0.9)
            .collect();
        let k = (0..kv_len)
            .map(|i| ((i * 29 + 7) % 103) as f32 / 59.0 - 0.8)
            .collect();
        let v = (0..kv_len)
            .map(|i| ((i * 11 + 5) % 97) as f32 / 47.0 - 1.0)
            .collect();
        (q, k, v)
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "index {i}: {x} != {y}");
        }
    }

    #[test]
    fn pure_so2_matches_rope() {
        let shape = GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 5,
            head_dim: 8,
        };
        let (q, k, v) = fixture(shape);
        let cfg = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };
        let theta = 10_000.0;
        let rope = forward_reference_grouped_rope(
            &q,
            &k,
            &v,
            shape,
            cfg,
            RotaryEmbeddingConfig {
                theta,
                position_offset: 3,
            },
        )
        .unwrap();
        let epg = forward_reference_grouped_epg(
            &q,
            &k,
            &v,
            shape,
            cfg,
            EpgEmbeddingConfig::so2(theta, 3).unwrap(),
        )
        .unwrap();
        assert_close(&rope.output, &epg.output, 1e-6);
        assert_close(&rope.lse, &epg.lse, 1e-6);
    }

    #[test]
    fn full_biplanar_head_is_rope_equivalence_control() {
        let shape = GroupedAttentionShape {
            batch: 1,
            q_heads: 2,
            kv_heads: 1,
            seq_len: 6,
            head_dim: 8,
        };
        let (q, k, v) = fixture(shape);
        let cfg = FlatAttentionConfig {
            causal: false,
            softmax_scale: None,
        };
        let theta = 10_000.0;
        let rope = forward_reference_grouped_rope(
            &q,
            &k,
            &v,
            shape,
            cfg,
            RotaryEmbeddingConfig {
                theta,
                position_offset: 11,
            },
        )
        .unwrap();
        let epg = forward_reference_grouped_epg(
            &q,
            &k,
            &v,
            shape,
            cfg,
            EpgEmbeddingConfig::hybrid_so4(theta, 11, 8, So4Geometry::Biplanar).unwrap(),
        )
        .unwrap();
        assert_close(&rope.output, &epg.output, 2e-6);
        assert_close(&rope.lse, &epg.lse, 2e-6);
    }

    #[test]
    fn isoclinic_dot_is_translation_relative() {
        let q = [0.3, -0.2, 0.8, 1.1];
        let k = [-0.7, 0.4, 0.2, -1.3];
        let epg = EpgEmbeddingConfig::hybrid_so4(10_000.0, 0, 4, So4Geometry::Isoclinic)
            .unwrap();
        let a = epg_dot(&q, &k, 4, 23, 7, epg).unwrap();
        let b = epg_dot(&q, &k, 4, 119, 103, epg).unwrap();
        assert!((a - b).abs() < 2e-5, "{a} != {b}");
    }
}
