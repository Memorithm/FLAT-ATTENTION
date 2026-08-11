//! M17 deterministic scalar backward oracle.
//!
//! Gradients are recomputed from Q/K/V plus the qualified forward O/LSE
//! contract. No `N x N` score or probability matrix is materialized.

use crate::{
    validate_input, AttentionShape, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
};

/// Gradients of one attention forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatAttentionBackwardOutput {
    /// Gradient with respect to Q, same shape as Q.
    pub dq: Vec<f32>,
    /// Gradient with respect to K, same shape as K.
    pub dk: Vec<f32>,
    /// Gradient with respect to V, same shape as V.
    pub dv: Vec<f32>,
}

/// Deterministic scalar dQ/dK/dV oracle for the canonical MHA contract.
///
/// `forward` must be the O/LSE result for the same Q/K/V/configuration.
/// Probabilities are reconstructed one score at a time as
/// `exp(score - LSE)`, so backward does not retain or allocate an attention
/// probability matrix.
pub fn backward_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
    forward: &FlatAttentionOutput,
) -> Result<FlatAttentionBackwardOutput, FlatAttentionError> {
    shape.validate()?;
    let tensor_len = shape.tensor_len()?;
    let lse_len = shape.lse_len()?;
    validate_input("Q", q, tensor_len)?;
    validate_input("K", k, tensor_len)?;
    validate_input("V", v, tensor_len)?;
    validate_input("dO", d_out, tensor_len)?;
    validate_input("O", &forward.output, tensor_len)?;
    validate_input("LSE", &forward.lse, lse_len)?;
    let scale = config.resolved_scale(shape.head_dim)?;

    let mut dq = vec![0.0f32; tensor_len];
    let mut dk = vec![0.0f32; tensor_len];
    let mut dv = vec![0.0f32; tensor_len];
    let head_stride = shape.seq_len * shape.head_dim;

    for batch in 0..shape.batch {
        for head in 0..shape.heads {
            let bh = batch * shape.heads + head;
            let head_base = bh * head_stride;
            let lse_base = bh * shape.seq_len;

            for query_pos in 0..shape.seq_len {
                let q_base = head_base + query_pos * shape.head_dim;
                let lse = forward.lse[lse_base + query_pos];

                // For softmax backward, sum_j dP_j * P_j equals dO dot O.
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
                    let kv_base = head_base + key_pos * shape.head_dim;
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
    use crate::forward_reference;

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.37 + phase;
                x.sin() * 0.7 + (x * 0.41).cos() * 0.2
            })
            .collect()
    }

    fn objective(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d_out: &[f32],
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> f32 {
        let forward = forward_reference(q, k, v, shape, config).unwrap();
        forward
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
            let tolerance = 3.0e-3 + 1.5e-2 * numerical.abs();
            let error = (analytic - numerical).abs();
            assert!(
                error <= tolerance,
                "{name}[{index}]: analytic={analytic}, numerical={numerical}, abs_error={error}, tolerance={tolerance}"
            );
        }
    }

    fn finite_difference_case(causal: bool) {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 3,
            head_dim: 2,
        };
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: Some(0.73),
        };
        let mut q = fixture(6, 0.1);
        let mut k = fixture(6, 0.8);
        let mut v = fixture(6, 1.4);
        let d_out = fixture(6, 2.2);
        let forward = forward_reference(&q, &k, &v, shape, config).unwrap();
        let analytic = backward_reference(&q, &k, &v, &d_out, shape, config, &forward).unwrap();
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
    fn non_causal_backward_matches_finite_differences() {
        finite_difference_case(false);
    }

    #[test]
    fn causal_backward_matches_finite_differences() {
        finite_difference_case(true);
    }

    #[test]
    fn rejects_mismatched_forward_contract() {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 2,
            head_dim: 2,
        };
        let q = vec![0.0; 4];
        let k = vec![0.0; 4];
        let v = vec![0.0; 4];
        let d_out = vec![0.0; 4];
        let malformed = FlatAttentionOutput {
            output: vec![0.0; 3],
            lse: vec![0.0; 2],
        };
        assert!(matches!(
            backward_reference(
                &q,
                &k,
                &v,
                &d_out,
                shape,
                FlatAttentionConfig::default(),
                &malformed,
            ),
            Err(FlatAttentionError::LengthMismatch { tensor: "O", .. })
        ));
    }
}
