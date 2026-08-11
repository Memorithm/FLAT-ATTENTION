//! M19 deterministic scalar backward oracle for native GQA/MQA.
//!
//! Gradients are recomputed from Q/K/V plus the qualified grouped forward
//! O/LSE contract. K/V retain their physical KV-head cardinality; no expanded
//! query-head copy and no `N x N` score/probability matrix is materialized.

use crate::{
    validate_input, FlatAttentionBackwardOutput, FlatAttentionConfig, FlatAttentionError,
    FlatAttentionOutput, GroupedAttentionShape,
};

/// Deterministic scalar dQ/dK/dV oracle for equal-length native GQA/MQA.
///
/// Query head `qh` accumulates dK/dV into physical KV head
/// `qh / (q_heads / kv_heads)`. This makes the reduction across query heads
/// explicit while preserving native MQA/GQA storage.
pub fn backward_reference_grouped(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    forward: &FlatAttentionOutput,
) -> Result<FlatAttentionBackwardOutput, FlatAttentionError> {
    shape.validate()?;
    let q_tensor_len = shape.q_tensor_len()?;
    let kv_tensor_len = shape.kv_tensor_len()?;
    let lse_len = shape.lse_len()?;
    validate_input("Q", q, q_tensor_len)?;
    validate_input("K", k, kv_tensor_len)?;
    validate_input("V", v, kv_tensor_len)?;
    validate_input("dO", d_out, q_tensor_len)?;
    validate_input("O", &forward.output, q_tensor_len)?;
    validate_input("LSE", &forward.lse, lse_len)?;

    let scale = config.resolved_scale(shape.head_dim)?;
    let group_size = shape.q_heads / shape.kv_heads;
    let head_stride = shape.seq_len * shape.head_dim;

    let mut dq = vec![0.0f32; q_tensor_len];
    let mut dk = vec![0.0f32; kv_tensor_len];
    let mut dv = vec![0.0f32; kv_tensor_len];

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
                let lse = forward.lse[lse_base + query_pos];

                let mut delta = 0.0f32;
                for dim in 0..shape.head_dim {
                    delta += d_out[q_base + dim] * forward.output[q_base + dim];
                }

                let key_limit = if config.causal {
                    query_pos + 1
                } else {
                    shape.seq_len
                };
                for key_pos in 0..key_limit {
                    let kv_base = kv_head_base + key_pos * shape.head_dim;
                    let mut dot_qk = 0.0f32;
                    let mut d_probability = 0.0f32;
                    for dim in 0..shape.head_dim {
                        dot_qk += q[q_base + dim] * k[kv_base + dim];
                        d_probability += d_out[q_base + dim] * v[kv_base + dim];
                    }

                    let score = dot_qk * scale;
                    let probability = (score - lse).exp();
                    let d_score = probability * (d_probability - delta);
                    let d_dot = d_score * scale;

                    for dim in 0..shape.head_dim {
                        dq[q_base + dim] += d_dot * k[kv_base + dim];
                        dk[kv_base + dim] += d_dot * q[q_base + dim];
                        dv[kv_base + dim] += probability * d_out[q_base + dim];
                    }
                }
            }
        }
    }

    Ok(FlatAttentionBackwardOutput { dq, dk, dv })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backward_reference, forward_reference_grouped, AttentionShape};

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.31 + phase;
                x.sin() * 0.65 + (x * 0.43).cos() * 0.23
            })
            .collect()
    }

    fn objective(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d_out: &[f32],
        shape: GroupedAttentionShape,
        config: FlatAttentionConfig,
    ) -> f32 {
        forward_reference_grouped(q, k, v, shape, config)
            .unwrap()
            .output
            .iter()
            .zip(d_out)
            .map(|(output, grad)| output * grad)
            .sum()
    }

    fn finite_difference(
        tensor: &mut [f32],
        index: usize,
        epsilon: f32,
        evaluate: impl Fn(&[f32]) -> f32,
    ) -> f32 {
        let original = tensor[index];
        tensor[index] = original + epsilon;
        let plus = evaluate(tensor);
        tensor[index] = original - epsilon;
        let minus = evaluate(tensor);
        tensor[index] = original;
        (plus - minus) / (2.0 * epsilon)
    }

    fn assert_gradient_close(name: &str, analytic: &[f32], numerical: &[f32]) {
        assert_eq!(analytic.len(), numerical.len());
        for (index, (&analytic, &numerical)) in analytic.iter().zip(numerical).enumerate() {
            let tolerance = 4.0e-3 + 2.0e-2 * numerical.abs();
            let error = (analytic - numerical).abs();
            assert!(
                error <= tolerance,
                "{name}[{index}]: analytic={analytic}, numerical={numerical}, abs_error={error}, tolerance={tolerance}"
            );
        }
    }

    fn finite_difference_case(kv_heads: usize, causal: bool) {
        let shape = GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads,
            seq_len: 3,
            head_dim: 2,
        };
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: Some(0.67),
        };
        let mut q = fixture(shape.q_tensor_len().unwrap(), 0.1);
        let mut k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
        let mut v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
        let d_out = fixture(shape.q_tensor_len().unwrap(), 2.1);
        let forward = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
        let analytic =
            backward_reference_grouped(&q, &k, &v, &d_out, shape, config, &forward).unwrap();
        let epsilon = 1.0e-3;

        let mut numerical_q = vec![0.0; q.len()];
        for (index, numerical) in numerical_q.iter_mut().enumerate() {
            *numerical = finite_difference(&mut q, index, epsilon, |candidate| {
                objective(candidate, &k, &v, &d_out, shape, config)
            });
        }
        let mut numerical_k = vec![0.0; k.len()];
        for (index, numerical) in numerical_k.iter_mut().enumerate() {
            *numerical = finite_difference(&mut k, index, epsilon, |candidate| {
                objective(&q, candidate, &v, &d_out, shape, config)
            });
        }
        let mut numerical_v = vec![0.0; v.len()];
        for (index, numerical) in numerical_v.iter_mut().enumerate() {
            *numerical = finite_difference(&mut v, index, epsilon, |candidate| {
                objective(&q, &k, candidate, &d_out, shape, config)
            });
        }

        assert_gradient_close("dQ", &analytic.dq, &numerical_q);
        assert_gradient_close("dK", &analytic.dk, &numerical_k);
        assert_gradient_close("dV", &analytic.dv, &numerical_v);
    }

    #[test]
    fn gqa_backward_matches_finite_differences() {
        finite_difference_case(2, true);
    }

    #[test]
    fn mqa_backward_matches_finite_differences() {
        finite_difference_case(1, false);
    }

    #[test]
    fn mha_grouped_backward_matches_canonical_oracle_bitwise() {
        let grouped = GroupedAttentionShape {
            batch: 2,
            q_heads: 2,
            kv_heads: 2,
            seq_len: 3,
            head_dim: 2,
        };
        let canonical = AttentionShape {
            batch: grouped.batch,
            heads: grouped.q_heads,
            seq_len: grouped.seq_len,
            head_dim: grouped.head_dim,
        };
        let config = FlatAttentionConfig {
            causal: true,
            softmax_scale: Some(0.61),
        };
        let q = fixture(grouped.q_tensor_len().unwrap(), 0.2);
        let k = fixture(grouped.kv_tensor_len().unwrap(), 0.9);
        let v = fixture(grouped.kv_tensor_len().unwrap(), 1.6);
        let d_out = fixture(grouped.q_tensor_len().unwrap(), 2.4);
        let forward = forward_reference_grouped(&q, &k, &v, grouped, config).unwrap();

        let actual =
            backward_reference_grouped(&q, &k, &v, &d_out, grouped, config, &forward).unwrap();
        let expected = backward_reference(&q, &k, &v, &d_out, canonical, config, &forward).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn preserves_native_kv_gradient_cardinality() {
        let shape = GroupedAttentionShape {
            batch: 2,
            q_heads: 8,
            kv_heads: 1,
            seq_len: 4,
            head_dim: 2,
        };
        let config = FlatAttentionConfig::default();
        let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
        let k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
        let v = fixture(shape.kv_tensor_len().unwrap(), 1.3);
        let d_out = fixture(shape.q_tensor_len().unwrap(), 1.9);
        let forward = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
        let gradients =
            backward_reference_grouped(&q, &k, &v, &d_out, shape, config, &forward).unwrap();
        assert_eq!(gradients.dq.len(), shape.q_tensor_len().unwrap());
        assert_eq!(gradients.dk.len(), shape.kv_tensor_len().unwrap());
        assert_eq!(gradients.dv.len(), shape.kv_tensor_len().unwrap());
    }
}
