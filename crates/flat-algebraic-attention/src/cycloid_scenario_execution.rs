//! CYCLOID-FLAT-4: execute the multi-regime cost decision through FLAT's
//! canonical numerical sparse oracle, with a separate dense control.
//!
//! This is an OFFLINE two-pass research adapter. It intentionally recomputes
//! dense Q/K scores and executes the existing dense/sparse reference multiple
//! times to ensure all original numerical identities remain unchanged.
//! Neither this adapter nor any externally declared scenario costs establish
//! inference acceleration, device memory traffic or model-level parity.

use core::fmt;

use flat_attention::{AttentionShape, FlatAttentionConfig, FlatAttentionError};

use crate::{
    cycloid_cost::{
        CycloidAttentionCandidate, CycloidAttentionPolicy,
    },
    cycloid_execution::{
        execute_cycloid_reference, CycloidExecutionError, CycloidExecutionReport,
        CycloidExecutionRow,
    },
    cycloid_scenarios::{
        choose_scenario_robust_attention, CycloidCostScenario,
        CycloidRobustScenarioDecision, CycloidScenarioError,
    },
};

/// One numerical row, with its original baseline cost and complete independent
/// cost scenarios. Candidate IDs and their original key positions never change.
#[derive(Debug, Clone, Copy)]
pub struct CycloidScenarioExecutionRow<'a> {
    pub baseline: CycloidExecutionRow<'a>,
    pub scenarios: &'a [CycloidCostScenario<'a>],
}

#[derive(Debug, Clone, PartialEq)]
pub struct CycloidScenarioExecutionReport {
    /// Full numerical execution under the single baseline cost regime.
    pub baseline: CycloidExecutionReport,
    /// Full numerical execution under multi-regime robust selected key sets.
    pub robust: CycloidExecutionReport,
    /// Multi-regime verdict and original key IDs for every query row.
    pub robust_decisions: Vec<CycloidRobustScenarioDecision>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CycloidScenarioExecutionError {
    Numerical(CycloidExecutionError),
    Scenario { row: usize, source: CycloidScenarioError },
    Attention(FlatAttentionError),
    InvalidRowCount { expected: usize, actual: usize },
    DecisionDrift { row: usize },
    DenseOracleDrift,
}

impl fmt::Display for CycloidScenarioExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CycloidScenarioExecutionError {}

impl From<CycloidExecutionError> for CycloidScenarioExecutionError {
    fn from(value: CycloidExecutionError) -> Self {
        Self::Numerical(value)
    }
}

impl From<FlatAttentionError> for CycloidScenarioExecutionError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Attention(value)
    }
}

/// Two independently executable numerical paths, not a performance benchmark.
///
/// First, run the original dense/sparse reference path to validate Q/K/V,
/// masks, causal eligibility and original candidate identities. Then recompute
/// exact all-eligible-key Q/K scores, apply multi-scenario fidelity/cost gates,
/// pass ONLY robust winners to the *existing* numerical sparse oracle, and
/// assert that the oracle selected the exact same original candidate/keys.
/// A robust fallback executes the full dense key set (causally restricted).
///
/// The second canonical adapter reruns dense scoring and oracle execution on
/// purpose. This function MUST NOT be called as a per-token runtime router.
pub fn execute_cycloid_scenario_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
    rows: &[CycloidScenarioExecutionRow<'_>],
    policy: CycloidAttentionPolicy,
) -> Result<CycloidScenarioExecutionReport, CycloidScenarioExecutionError> {
    let baseline_rows: Vec<_> = rows.iter().map(|row| row.baseline).collect();

    // Checks the full Q/K/V oracle and every key mask/causal constraint before
    // any scenario-dependent candidate filtering takes place.
    let baseline =
        execute_cycloid_reference(q, k, v, shape, config, &baseline_rows, policy)?;
    let query_count = shape.lse_len()?;
    if rows.len() != query_count {
        return Err(CycloidScenarioExecutionError::InvalidRowCount {
            expected: query_count,
            actual: rows.len(),
        });
    }

    let scale = f64::from(config.resolved_scale(shape.head_dim)?);
    let head_stride = shape
        .seq_len
        .checked_mul(shape.head_dim)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let mut robust_decisions = Vec::with_capacity(query_count);

    for (row, request) in rows.iter().enumerate() {
        let query_position = row % shape.seq_len;
        let eligible = if config.causal {
            query_position + 1
        } else {
            shape.seq_len
        };
        let head_base = (row / shape.seq_len) * head_stride;
        let query_base = head_base + query_position * shape.head_dim;
        let query = &q[query_base..query_base + shape.head_dim];
        let values = &v[head_base..head_base + head_stride];
        let max_abs_value = values
            .iter()
            .map(|value| f64::from(value.abs()))
            .fold(0.0_f64, f64::max);

        let mut scores = Vec::with_capacity(eligible);
        for key_position in 0..eligible {
            let key_base = head_base + key_position * shape.head_dim;
            let key = &k[key_base..key_base + shape.head_dim];
            let dot = query
                .iter()
                .zip(key)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>();
            scores.push(dot * scale);
        }

        let decision = choose_scenario_robust_attention(
            &scores,
            request.baseline.candidate_routes,
            request.baseline.dense_cost,
            policy,
            max_abs_value,
            request.scenarios,
        )
        .map_err(|source| CycloidScenarioExecutionError::Scenario { row, source })?;
        robust_decisions.push(decision);
    }

    // Use the original baseline-cost entries for the retained original
    // candidate. The scenario decision may never invent a K/V position.
    let retained: Vec<Vec<CycloidAttentionCandidate<'_>>> = rows
        .iter()
        .zip(&robust_decisions)
        .map(|(row, decision)| {
            row.baseline
                .candidate_routes
                .iter()
                .copied()
                .filter(|candidate| {
                    Some(candidate.candidate_id) == decision.attention.selected_candidate_id
                })
                .collect()
        })
        .collect();
    let robust_rows: Vec<_> = rows
        .iter()
        .zip(&retained)
        .map(|(row, chosen)| CycloidExecutionRow {
            candidate_routes: chosen,
            dense_cost: row.baseline.dense_cost,
        })
        .collect();

    let robust = execute_cycloid_reference(q, k, v, shape, config, &robust_rows, policy)?;

    if robust.dense != baseline.dense {
        return Err(CycloidScenarioExecutionError::DenseOracleDrift);
    }
    for (row, (numeric, decision)) in robust
        .decisions
        .iter()
        .zip(&robust_decisions)
        .enumerate()
    {
        if numeric.selected_candidate_id != decision.attention.selected_candidate_id
            || numeric.original_key_indices != decision.attention.original_key_indices
        {
            return Err(CycloidScenarioExecutionError::DecisionDrift { row });
        }
    }

    Ok(CycloidScenarioExecutionReport {
        baseline,
        robust,
        robust_decisions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cycloid_cost::CycloidAttentionCost,
        cycloid_scenarios::CycloidScenarioCandidateCost,
    };

    fn shape() -> AttentionShape {
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 2,
            head_dim: 1,
        }
    }

    fn cost(total: u64) -> CycloidAttentionCost {
        CycloidAttentionCost {
            route_ns: 0,
            kv_read_ns: total,
            numerical_ns: 0,
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
    fn different_robust_candidate_is_actually_executed_against_dense() {
        let candidates = [
            CycloidAttentionCandidate {
                candidate_id: 1,
                original_key_indices: &[0],
                estimated_cost: cost(20),
            },
            CycloidAttentionCandidate {
                candidate_id: 2,
                original_key_indices: &[1],
                estimated_cost: cost(40),
            },
        ];
        let first = [
            CycloidScenarioCandidateCost { candidate_id: 1, cost: cost(20) },
            CycloidScenarioCandidateCost { candidate_id: 2, cost: cost(40) },
        ];
        let second = [
            CycloidScenarioCandidateCost { candidate_id: 1, cost: cost(80) },
            CycloidScenarioCandidateCost { candidate_id: 2, cost: cost(30) },
        ];
        let scenarios = [
            CycloidCostScenario {
                scenario_id: 1,
                dense_cost: cost(100),
                candidate_costs: &first,
            },
            CycloidCostScenario {
                scenario_id: 2,
                dense_cost: cost(50),
                candidate_costs: &second,
            },
        ];
        let rows = [
            CycloidScenarioExecutionRow {
                baseline: CycloidExecutionRow {
                    candidate_routes: &candidates,
                    dense_cost: cost(100),
                },
                scenarios: &scenarios,
            };
            2
        ];
        let out = execute_cycloid_scenario_reference(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 3.0],
            shape(),
            FlatAttentionConfig::default(),
            &rows,
            policy(),
        )
        .unwrap();
        assert_eq!(out.baseline.selected.attention.output, vec![1.0, 1.0]);
        assert_eq!(out.robust.dense.attention.output, vec![2.0, 2.0]);
        assert_eq!(out.robust.selected.attention.output, vec![3.0, 3.0]);
        assert_eq!(out.robust.selected.counters.executed_pairs, 2);
        assert_eq!(out.robust_decisions[0].attention.selected_candidate_id, Some(2));
    }

    #[test]
    fn one_regime_regression_executes_full_dense_fallback() {
        let candidates = [CycloidAttentionCandidate {
            candidate_id: 1,
            original_key_indices: &[0],
            estimated_cost: cost(20),
        }];
        let disqualified = [CycloidScenarioCandidateCost {
            candidate_id: 1,
            cost: cost(100),
        }];
        let scenarios = [
            CycloidCostScenario { scenario_id: 1, dense_cost: cost(100), candidate_costs: &disqualified },
            CycloidCostScenario { scenario_id: 2, dense_cost: cost(100), candidate_costs: &disqualified },
        ];
        let rows = [
            CycloidScenarioExecutionRow {
                baseline: CycloidExecutionRow {
                    candidate_routes: &candidates,
                    dense_cost: cost(100),
                },
                scenarios: &scenarios,
            };
            2
        ];
        let out = execute_cycloid_scenario_reference(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 3.0],
            shape(),
            FlatAttentionConfig::default(),
            &rows,
            policy(),
        )
        .unwrap();
        assert_eq!(out.robust.decisions[0].selected_candidate_id, None);
        assert_eq!(out.robust.selected.attention, out.robust.dense.attention);
        assert_eq!(out.robust.selected.counters.executed_pairs, 4);
    }

    #[test]
    fn causal_future_candidate_is_rejected_even_when_cheaper() {
        let candidates = [CycloidAttentionCandidate {
            candidate_id: 7,
            original_key_indices: &[1],
            estimated_cost: cost(10),
        }];
        let cost_list = [CycloidScenarioCandidateCost {
            candidate_id: 7,
            cost: cost(5),
        }];
        let scenarios = [
            CycloidCostScenario { scenario_id: 1, dense_cost: cost(100), candidate_costs: &cost_list },
            CycloidCostScenario { scenario_id: 2, dense_cost: cost(100), candidate_costs: &cost_list },
        ];
        let rows = [
            CycloidScenarioExecutionRow {
                baseline: CycloidExecutionRow {
                    candidate_routes: &candidates,
                    dense_cost: cost(100),
                },
                scenarios: &scenarios,
            };
            2
        ];
        let err = execute_cycloid_scenario_reference(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 3.0],
            shape(),
            FlatAttentionConfig { causal: true, softmax_scale: None },
            &rows,
            policy(),
        ).unwrap_err();
        assert_eq!(
            err,
            CycloidScenarioExecutionError::Numerical(
                CycloidExecutionError::IneligibleCandidateKey {
                    row: 0,
                    candidate_id: 7,
                    key_position: 1
                }
            )
        );
    }

    #[test]
    fn malformed_scenario_identity_is_not_executed() {
        let candidates = [CycloidAttentionCandidate {
            candidate_id: 1,
            original_key_indices: &[0],
            estimated_cost: cost(20),
        }];
        let forged = [CycloidScenarioCandidateCost {
            candidate_id: 99,
            cost: cost(10),
        }];
        let scenarios = [
            CycloidCostScenario { scenario_id: 1, dense_cost: cost(100), candidate_costs: &forged },
            CycloidCostScenario { scenario_id: 2, dense_cost: cost(100), candidate_costs: &forged },
        ];
        let rows = [
            CycloidScenarioExecutionRow {
                baseline: CycloidExecutionRow {
                    candidate_routes: &candidates,
                    dense_cost: cost(100),
                },
                scenarios: &scenarios,
            };
            2
        ];
        let err = execute_cycloid_scenario_reference(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 3.0],
            shape(),
            FlatAttentionConfig::default(),
            &rows,
            policy(),
        ).unwrap_err();
        assert_eq!(
            err,
            CycloidScenarioExecutionError::Scenario {
                row: 0,
                source: CycloidScenarioError::UnknownCandidateId {
                    scenario_id: 1,
                    candidate_id: 99
                },
            }
        );
    }
}
