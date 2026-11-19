//! Deterministic, research-only sparse Ising reweighting of an attention vector.
//!
//! This is a mean-field approximation, not exact Boltzmann sampling, training,
//! numerical Q/K/V attention, or a production routing policy. The model operates
//! on a *single already-normalized attention vector*, without an N x N matrix.
//!
//! A spin s_i in {-1,+1} represents inactive/active attention. Unary fields are
//! calibrated as h_i = 0.5 * log(p_i / (1 - p_i)); thus the uncoupled J=0 case
//! reproduces the input distribution. Sparse symmetric signed couplings model
//! cooperative or antagonistic co-attention between positions.
//!
//! Damped parallel updates approximate m_i = tanh(h_i + sum_j J_ij * m_j).
//! Resulting Bernoulli probabilities (1+m_i)/2 are normalized for use as
//! attention weights. Mean-field magnetizations are not exact Ising marginals.
//! No convergence or model-quality claim follows from this standalone oracle.

use core::fmt;

/// An undirected Ising edge, supplied in strictly increasing (left, right)
/// lexicographic order; left must be smaller than right.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsingCoupling {
    pub left: usize,
    pub right: usize,
    pub weight: f64,
}

/// Numerics for deterministic damped parallel mean-field iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsingMeanFieldConfig {
    /// Must be in (0, 1].
    pub damping: f64,
    /// Absolute maximum magnetization change for a converged iteration.
    pub tolerance: f64,
    /// Strict positive upper bound on mean-field iterations.
    pub max_iterations: usize,
}

impl Default for IsingMeanFieldConfig {
    fn default() -> Self {
        Self {
            damping: 0.5,
            tolerance: 1e-10,
            max_iterations: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IsingAttentionVector {
    /// Normalized, nonnegative weights whose sum is one.
    pub weights: Vec<f64>,
    /// Final approximate magnetizations in [-1, 1].
    pub magnetizations: Vec<f64>,
    /// Number of mean-field iterations actually executed.
    pub iterations: usize,
    /// Whether the requested tolerance was reached.
    pub converged: bool,
    /// Maximum coordinate change in the last iteration.
    pub max_delta: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IsingAttentionError {
    EmptyVector,
    InvalidProbability { index: usize },
    NotNormalized { sum: f64 },
    InvalidConfig,
    InvalidEdge { edge_index: usize },
    NonCanonicalEdges { edge_index: usize },
    NonFiniteField { index: usize },
    NoActiveMass,
}

impl fmt::Display for IsingAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for IsingAttentionError {}

/// Apply sparse Ising couplings to a normalized attention vector.
///
/// The input must be finite, in [0,1], and sum to one within 1e-10. Exactly
/// zero/one components are frozen at -1/+1: structural zeros can never become
/// attended positions, and edges touching structural zeros are ignored so a
/// masked (e.g. future) position cannot indirectly influence valid attention. For an empty or all-zero coupling graph, this is the
/// exact identity transformation (including signed-zero representation).
///
/// Couplings are undirected and ordered with left < right, no duplicates,
/// ascending lexicographically. They are dimensionless beta*J at unit
/// temperature. Complexity is O(n + edges) memory and
/// O(max_iterations*(n + edges)) work. Full attention score matrices, learning,
/// differentiation, and GPU execution are deliberately out of scope.
pub fn ising_reweight_attention(
    attention: &[f64],
    couplings: &[IsingCoupling],
    config: IsingMeanFieldConfig,
) -> Result<IsingAttentionVector, IsingAttentionError> {
    if attention.is_empty() {
        return Err(IsingAttentionError::EmptyVector);
    }
    if !(config.damping.is_finite()
        && config.damping > 0.0
        && config.damping <= 1.0
        && config.tolerance.is_finite()
        && config.tolerance > 0.0
        && config.max_iterations > 0)
    {
        return Err(IsingAttentionError::InvalidConfig);
    }
    for (index, &value) in attention.iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(IsingAttentionError::InvalidProbability { index });
        }
    }
    let input_sum: f64 = attention.iter().sum();
    if !input_sum.is_finite() || (input_sum - 1.0).abs() > 1e-10 {
        return Err(IsingAttentionError::NotNormalized { sum: input_sum });
    }
    let mut previous = None;
    for (edge_index, edge) in couplings.iter().enumerate() {
        if edge.left >= edge.right || edge.right >= attention.len() || !edge.weight.is_finite() {
            return Err(IsingAttentionError::InvalidEdge { edge_index });
        }
        let pair = (edge.left, edge.right);
        if previous.is_some_and(|prior| pair <= prior) {
            return Err(IsingAttentionError::NonCanonicalEdges { edge_index });
        }
        previous = Some(pair);
    }

    let mut magnetizations: Vec<f64> = attention.iter().map(|p| 2.0 * p - 1.0).collect();
    if couplings.iter().all(|edge| {
        edge.weight == 0.0 || attention[edge.left] == 0.0 || attention[edge.right] == 0.0
    }) {
        return Ok(IsingAttentionVector {
            weights: attention.to_vec(),
            magnetizations,
            iterations: 0,
            converged: true,
            max_delta: 0.0,
        });
    }

    let fields: Vec<f64> = attention
        .iter()
        .map(|&p| {
            if p == 0.0 || p == 1.0 {
                0.0 // Boundary magnetizations are pinned and never updated.
            } else {
                0.5 * (p.ln() - (-p).ln_1p())
            }
        })
        .collect();
    let mut neighbors = vec![Vec::<(usize, f64)>::new(); attention.len()];
    for edge in couplings {
        // Masked positions must not influence valid attention, even indirectly.
        if attention[edge.left] == 0.0 || attention[edge.right] == 0.0 {
            continue;
        }
        neighbors[edge.left].push((edge.right, edge.weight));
        neighbors[edge.right].push((edge.left, edge.weight));
    }

    let mut next = magnetizations.clone();
    let mut max_delta = 0.0;
    let mut converged = false;
    let mut iterations = 0;
    for sweep in 0..config.max_iterations {
        max_delta = 0.0;
        for index in 0..attention.len() {
            if attention[index] == 0.0 || attention[index] == 1.0 {
                continue;
            }
            let mut field = fields[index];
            for &(neighbor, weight) in &neighbors[index] {
                field += weight * magnetizations[neighbor];
            }
            if !field.is_finite() {
                return Err(IsingAttentionError::NonFiniteField { index });
            }
            let target = field.tanh();
            let updated = magnetizations[index] + config.damping * (target - magnetizations[index]);
            let delta = (updated - magnetizations[index]).abs();
            max_delta = max_delta.max(delta);
            next[index] = updated;
        }
        std::mem::swap(&mut magnetizations, &mut next);
        iterations = sweep + 1;
        if max_delta <= config.tolerance {
            converged = true;
            break;
        }
    }

    let mut weights: Vec<f64> = magnetizations
        .iter()
        .map(|&m| ((1.0 + m) * 0.5).clamp(0.0, 1.0))
        .collect();
    let mass: f64 = weights.iter().sum();
    if !mass.is_finite() || mass <= 0.0 {
        return Err(IsingAttentionError::NoActiveMass);
    }
    for weight in &mut weights {
        *weight /= mass;
    }
    Ok(IsingAttentionVector {
        weights,
        magnetizations,
        iterations,
        converged,
        max_delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(weight: f64) -> IsingCoupling {
        IsingCoupling {
            left: 0,
            right: 1,
            weight,
        }
    }

    #[test]
    fn no_couplings_preserves_attention_bit_exactly() {
        let p = [0.8, 0.15, 0.05];
        for edges in [&[][..], &[edge(0.0)][..]] {
            let out = ising_reweight_attention(&p, edges, IsingMeanFieldConfig::default()).unwrap();
            assert_eq!(out.weights, p);
            assert!(out.converged);
            assert_eq!(out.iterations, 0);
        }
    }

    #[test]
    fn signed_couplings_alter_attention_in_opposite_directions() {
        let p = [0.8, 0.15, 0.05];
        let config = IsingMeanFieldConfig::default();
        let cooperative = ising_reweight_attention(&p, &[edge(0.1)], config).unwrap();
        let antagonistic = ising_reweight_attention(&p, &[edge(-0.1)], config).unwrap();
        assert!(cooperative.converged);
        assert!(antagonistic.converged);
        assert!(cooperative.weights[1] > p[1]);
        assert!(antagonistic.weights[1] < p[1]);
        for result in [cooperative, antagonistic] {
            assert!((result.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(result.weights.iter().all(|p| (0.0..=1.0).contains(p)));
        }
    }

    #[test]
    fn structural_zeros_and_ones_remain_pinned() {
        let p = [0.0, 0.6, 0.4];
        let out =
            ising_reweight_attention(&p, &[edge(0.4)], IsingMeanFieldConfig::default()).unwrap();
        assert_eq!(out.weights, p);
        assert_eq!(out.weights[0], 0.0);
        assert_eq!(out.magnetizations[0], -1.0);
        let certain =
            ising_reweight_attention(&[1.0, 0.0], &[edge(-2.0)], IsingMeanFieldConfig::default())
                .unwrap();
        assert_eq!(certain.weights, [1.0, 0.0]);
    }

    #[test]
    fn repeated_runs_are_deterministic_and_iteration_is_bounded() {
        let p = [0.7, 0.2, 0.1];
        let config = IsingMeanFieldConfig {
            max_iterations: 2,
            ..Default::default()
        };
        let first = ising_reweight_attention(&p, &[edge(2.0)], config).unwrap();
        let second = ising_reweight_attention(&p, &[edge(2.0)], config).unwrap();
        assert_eq!(first, second);
        assert!(first.iterations <= 2);
    }

    #[test]
    fn malformed_vectors_edges_and_config_fail_closed() {
        let config = IsingMeanFieldConfig::default();
        assert_eq!(
            ising_reweight_attention(&[], &[], config),
            Err(IsingAttentionError::EmptyVector)
        );
        assert!(matches!(
            ising_reweight_attention(&[0.2, 0.2], &[], config),
            Err(IsingAttentionError::NotNormalized { .. })
        ));
        assert!(matches!(
            ising_reweight_attention(&[f64::NAN, 1.0], &[], config),
            Err(IsingAttentionError::InvalidProbability { .. })
        ));
        assert!(matches!(
            ising_reweight_attention(&[0.4, 0.6], &[edge(f64::INFINITY)], config),
            Err(IsingAttentionError::InvalidEdge { .. })
        ));
        assert!(matches!(
            ising_reweight_attention(&[0.4, 0.6], &[edge(0.1), edge(0.2)], config),
            Err(IsingAttentionError::NonCanonicalEdges { .. })
        ));
        assert!(matches!(
            ising_reweight_attention(
                &[0.4, 0.6],
                &[IsingCoupling {
                    left: 1,
                    right: 0,
                    weight: 1.0
                }],
                config
            ),
            Err(IsingAttentionError::InvalidEdge { .. })
        ));
        assert_eq!(
            ising_reweight_attention(
                &[0.4, 0.6],
                &[],
                IsingMeanFieldConfig {
                    damping: 0.0,
                    ..config
                }
            ),
            Err(IsingAttentionError::InvalidConfig)
        );
    }
}
