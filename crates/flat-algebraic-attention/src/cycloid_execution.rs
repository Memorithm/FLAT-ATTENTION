//! CYCLOID-FLAT-2: execute a cost-gated selection with FLAT's numerical oracle.
//!
//! Research-only, host-only end-to-end integration. Dense QK scores are
//! recomputed from the authoritative Q/K inputs to qualify each candidate.
//! Therefore this is a diagnostic execution path, not an inference speedup.
//! No stable API or default GPU dispatch is modified.

use core::fmt;

use flat_attention::api::research_structural_routing::{
    forward_reference_structural_dense, forward_reference_structural_sparse,
    StructuralAttentionOutput, StructuralCandidateSet, StructuralRoutingError,
};
use flat_attention::{AttentionShape, FlatAttentionConfig, FlatAttentionError};

use crate::cycloid_cost::{
    choose_costed_attention, CycloidAttentionCandidate, CycloidAttentionCost,
    CycloidAttentionDecision, CycloidAttentionError, CycloidAttentionPolicy,
};

/// Candidate sets and dense cost attached to a single original query row.
#[derive(Debug, Clone, Copy)]
pub struct CycloidExecutionRow<'a> {
    pub candidate_routes: &'a [CycloidAttentionCandidate<'a>],
    pub dense_cost: CycloidAttentionCost,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CycloidExecutionReport {
    pub dense: StructuralAttentionOutput,
    pub selected: StructuralAttentionOutput,
    pub decisions: Vec<CycloidAttentionDecision>,
    pub max_output_abs_error: f64,
    pub max_lse_abs_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CycloidExecutionError {
    Attention(FlatAttentionError),
    Structural(StructuralRoutingError),
    Cost(CycloidAttentionError),
    InvalidRowCount {
        expected: usize,
        actual: usize,
    },
    IneligibleCandidateKey {
        row: usize,
        candidate_id: u32,
        key_position: usize,
    },
}

impl From<FlatAttentionError> for CycloidExecutionError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Attention(value)
    }
}

impl From<StructuralRoutingError> for CycloidExecutionError {
    fn from(value: StructuralRoutingError) -> Self {
        Self::Structural(value)
    }
}

impl From<CycloidAttentionError> for CycloidExecutionError {
    fn from(value: CycloidAttentionError) -> Self {
        Self::Cost(value)
    }
}

impl fmt::Display for CycloidExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CycloidExecutionError {}

/// Recompute dense QK scores, choose qualifying routes, execute each selected
/// numerical row with the existing sparse oracle and compare against the
/// independent dense-all-keys oracle. Every score and Value is checked for
/// finiteness by the existing oracles. Causal eligibility is enforced before
/// candidate comparison, not merely during sparse numerical execution.
///
/// Caller-owned route costs are estimates; full Q/K/V read and dense oracle
/// costs are NOT included in those estimates. A speedup cannot be inferred.
/// This accepts MHA geometry only, matching the authoritative structural
/// routing oracle, and does not authorize any cross-layer cache reuse.
pub fn execute_cycloid_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
    rows: &[CycloidExecutionRow<'_>],
    policy: CycloidAttentionPolicy,
) -> Result<CycloidExecutionReport, CycloidExecutionError> {
    // Numerical oracle validates exact Q/K/V shape and finite inputs before any
    // route decision, and retains the independent dense correctness control.
    let all = StructuralCandidateSet::all(shape)?;
    let dense = forward_reference_structural_dense(q, k, v, shape, config, &all)?;
    let query_count = shape.lse_len()?;
    if rows.len() != query_count {
        return Err(CycloidExecutionError::InvalidRowCount {
            expected: query_count,
            actual: rows.len(),
        });
    }

    let scale = f64::from(config.resolved_scale(shape.head_dim)?);
    let head_stride = shape
        .seq_len
        .checked_mul(shape.head_dim)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let mut decisions = Vec::with_capacity(query_count);
    let mut selected_rows = Vec::with_capacity(query_count);

    for (row, routes) in rows.iter().enumerate() {
        let query_position = row % shape.seq_len;
        let eligible = if config.causal {
            query_position + 1
        } else {
            shape.seq_len
        };
        // Forbid candidates referencing masked/future or nonexistent keys.
        // Dense fallback still uses only causal-eligible positions.
        for candidate in routes.candidate_routes {
            for &key_position in candidate.original_key_indices {
                if key_position >= eligible {
                    return Err(CycloidExecutionError::IneligibleCandidateKey {
                        row,
                        candidate_id: candidate.candidate_id,
                        key_position,
                    });
                }
            }
        }

        let head_base = (row / shape.seq_len) * head_stride;
        let query_base = head_base + query_position * shape.head_dim;
        let query = &q[query_base..query_base + shape.head_dim];
        let values = &v[head_base..head_base + head_stride];
        let max_abs_value = values
            .iter()
            .map(|value| f64::from(value.abs()))
            .fold(0.0_f64, f64::max);

        let mut scores = Vec::with_capacity(eligible);
        for key_pos in 0..eligible {
            let key_start = head_base + key_pos * shape.head_dim;
            let key = &k[key_start..key_start + shape.head_dim];
            let dot = query
                .iter()
                .zip(key)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>();
            scores.push(dot * scale);
        }
        let decision = choose_costed_attention(
            &scores,
            routes.candidate_routes,
            routes.dense_cost,
            policy,
            max_abs_value,
        )?;
        selected_rows.push(decision.original_key_indices.clone());
        decisions.push(decision);
    }

    let selected_set = StructuralCandidateSet::from_rows(shape, selected_rows)?;
    let selected = forward_reference_structural_sparse(q, k, v, shape, config, &selected_set)?;

    let max_output_abs_error = selected
        .attention
        .output
        .iter()
        .zip(&dense.attention.output)
        .map(|(candidate, reference)| (f64::from(*candidate) - f64::from(*reference)).abs())
        .fold(0.0_f64, f64::max);
    let max_lse_abs_error = selected
        .attention
        .lse
        .iter()
        .zip(&dense.attention.lse)
        .map(|(candidate, reference)| (f64::from(*candidate) - f64::from(*reference)).abs())
        .fold(0.0_f64, f64::max);

    Ok(CycloidExecutionReport {
        dense,
        selected,
        decisions,
        max_output_abs_error,
        max_lse_abs_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> AttentionShape {
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 2,
            head_dim: 1,
        }
    }

    fn cost(route_ns: u64, read_ns: u64, numerical_ns: u64) -> CycloidAttentionCost {
        CycloidAttentionCost {
            route_ns,
            kv_read_ns: read_ns,
            numerical_ns,
            estimated_read_bytes: 8,
        }
    }

    fn policy() -> CycloidAttentionPolicy {
        CycloidAttentionPolicy {
            min_retained_softmax_mass: 0.5,
            max_output_abs_error_bound: 3.0,
            max_estimated_read_bytes: 128,
        }
    }

    #[test]
    fn chosen_sparse_keys_execute_numerical_attention() {
        let first = [CycloidAttentionCandidate {
            candidate_id: 10,
            original_key_indices: &[0],
            estimated_cost: cost(10, 10, 10),
        }];
        let second = [CycloidAttentionCandidate {
            candidate_id: 11,
            original_key_indices: &[1],
            estimated_cost: cost(10, 10, 10),
        }];
        let rows = [
            CycloidExecutionRow {
                candidate_routes: &first,
                dense_cost: cost(0, 70, 70),
            },
            CycloidExecutionRow {
                candidate_routes: &second,
                dense_cost: cost(0, 70, 70),
            },
        ];
        // Uniform scores: dense yields [2,2], sparse yields [1,3].
        let result = execute_cycloid_reference(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 3.0],
            shape(),
            FlatAttentionConfig::default(),
            &rows,
            policy(),
        )
        .unwrap();
        assert_eq!(result.dense.attention.output, vec![2.0, 2.0]);
        assert_eq!(result.selected.attention.output, vec![1.0, 3.0]);
        assert_eq!(result.dense.counters.executed_pairs, 4);
        assert_eq!(result.selected.counters.executed_pairs, 2);
        assert_eq!(result.decisions[0].selected_candidate_id, Some(10));
        assert_eq!(result.max_output_abs_error, 1.0);
    }

    #[test]
    fn rejecting_dominant_key_loss_executes_dense_fallback() {
        let omitted = [CycloidAttentionCandidate {
            candidate_id: 10,
            original_key_indices: &[1],
            estimated_cost: cost(1, 1, 1),
        }];
        let rows = [
            CycloidExecutionRow {
                candidate_routes: &omitted,
                dense_cost: cost(0, 70, 70),
            },
            CycloidExecutionRow {
                candidate_routes: &omitted,
                dense_cost: cost(0, 70, 70),
            },
        ];
        let mut limits = policy();
        limits.min_retained_softmax_mass = 0.99;
        let result = execute_cycloid_reference(
            &[1.0, 1.0],
            &[10.0, 0.0],
            &[2.0, 9.0],
            shape(),
            FlatAttentionConfig::default(),
            &rows,
            limits,
        )
        .unwrap();
        assert_eq!(result.decisions[0].selected_candidate_id, None);
        assert_eq!(result.selected.attention, result.dense.attention);
        assert_eq!(result.selected.counters.executed_pairs, 4);
    }

    #[test]
    fn causal_future_candidate_fails_before_numeric_execution() {
        let future = [CycloidAttentionCandidate {
            candidate_id: 7,
            original_key_indices: &[1],
            estimated_cost: cost(1, 1, 1),
        }];
        let rows = [
            CycloidExecutionRow {
                candidate_routes: &future,
                dense_cost: cost(0, 70, 70),
            },
            CycloidExecutionRow {
                candidate_routes: &[],
                dense_cost: cost(0, 70, 70),
            },
        ];
        let error = execute_cycloid_reference(
            &[1.0, 1.0],
            &[1.0, 2.0],
            &[2.0, 3.0],
            shape(),
            FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
            &rows,
            policy(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            CycloidExecutionError::IneligibleCandidateKey {
                row: 0,
                candidate_id: 7,
                key_position: 1,
            }
        );
    }

    #[test]
    fn malformed_row_count_is_rejected() {
        let error = execute_cycloid_reference(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 3.0],
            shape(),
            FlatAttentionConfig::default(),
            &[],
            policy(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            CycloidExecutionError::InvalidRowCount {
                expected: 2,
                actual: 0,
            }
        );
    }

    #[test]
    fn causal_dense_fallback_remains_causal() {
        let rows = [
            CycloidExecutionRow {
                candidate_routes: &[],
                dense_cost: cost(0, 70, 70),
            },
            CycloidExecutionRow {
                candidate_routes: &[],
                dense_cost: cost(0, 70, 70),
            },
        ];
        let result = execute_cycloid_reference(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[2.0, 4.0],
            shape(),
            FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
            &rows,
            policy(),
        )
        .unwrap();
        assert_eq!(result.selected.attention.output, vec![2.0, 3.0]);
        assert_eq!(result.selected.attention, result.dense.attention);
        assert_eq!(result.selected.counters.executed_pairs, 3);
    }
}
