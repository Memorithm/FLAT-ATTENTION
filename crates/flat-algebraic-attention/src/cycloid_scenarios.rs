//! CYCLOID-FLAT-3: a sparse attention candidate must repay its cost
//! across multiple explicitly named access-cost scenarios.
//!
//! This is an OFFLINE diagnostic layered on the existing dense QK oracle and
//! softmax-mass/output-error gate. Scenario costs are caller-declared estimates,
//! NOT observed GPU memory transactions. Different read order, cache state,
//! or backend may change the rank of a candidate; never promote a winner
//! chosen solely from one favorable latency estimate.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use crate::cycloid_cost::{
    choose_costed_attention, CycloidAttentionCandidate, CycloidAttentionCost,
    CycloidAttentionDecision, CycloidAttentionError, CycloidAttentionPolicy,
    CycloidAttentionVerdict,
};

pub const MIN_CYCLOID_SCENARIOS: usize = 2;
pub const MAX_CYCLOID_SCENARIOS: usize = 8;

/// Exact key-set identity is carried by the authoritative candidate list.
/// Each scenario supplies costs by candidate_id, never positional index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CycloidScenarioCandidateCost {
    pub candidate_id: u32,
    pub cost: CycloidAttentionCost,
}

#[derive(Debug, Clone, Copy)]
pub struct CycloidCostScenario<'a> {
    pub scenario_id: u32,
    pub dense_cost: CycloidAttentionCost,
    pub candidate_costs: &'a [CycloidScenarioCandidateCost],
}

/// The baseline numerical decision and per-candidate fidelity evidence are
/// retained; scenario-robust selection may prefer a different candidate or
/// return the unchanged dense oracle key set.
#[derive(Debug, Clone, PartialEq)]
pub struct CycloidRobustScenarioDecision {
    pub attention: CycloidAttentionDecision,
    pub scenario_count: usize,
    /// IDs that passed the fidelity/baseline check but failed at least
    /// one independent scenario's strict cost/read-budget requirement.
    pub rejected_by_scenario: Vec<u32>,
    /// Worst candidate/dense cost ratio across the baseline and all scenarios.
    /// None when the dense fallback is selected. Fractions are exact integers
    /// for deterministic comparison, not measured speedup values.
    pub selected_worst_ratio: Option<(u64, u64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CycloidScenarioError {
    Baseline(CycloidAttentionError),
    InvalidScenarioCount,
    DuplicateScenarioId(u32),
    DuplicateCandidateId { scenario_id: u32, candidate_id: u32 },
    UnknownCandidateId { scenario_id: u32, candidate_id: u32 },
    MissingCandidateId { scenario_id: u32, candidate_id: u32 },
    InvalidDenseCost(u32),
    InvalidCandidateCost { scenario_id: u32, candidate_id: u32 },
    CostOverflow,
}

impl fmt::Display for CycloidScenarioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CycloidScenarioError {}

impl From<CycloidAttentionError> for CycloidScenarioError {
    fn from(value: CycloidAttentionError) -> Self {
        Self::Baseline(value)
    }
}

fn total_ns(cost: CycloidAttentionCost) -> Result<u64, CycloidScenarioError> {
    cost.route_ns
        .checked_add(cost.kv_read_ns)
        .and_then(|v| v.checked_add(cost.numerical_ns))
        .ok_or(CycloidScenarioError::CostOverflow)
}

/// Evaluate numerical fidelity once against original dense QK scores, then
/// reject any sparse candidate that fails even one independent cost scenario.
/// For remaining candidates choose the lowest *worst-case relative cost*
/// (candidate/dense), using integer cross-products to avoid floating-point
/// ranking drift. Break ties by baseline time, read bytes, then candidate ID.
///
/// Work remains O(number of scenarios × candidate count) plus the existing
/// dense-oracle scorer. This is not a sublinear inference-time router and
/// must not be run per-token. Every scenario is merely externally supplied
/// cost evidence, not a physically measured benchmark certificate.
pub fn choose_scenario_robust_attention(
    dense_scores: &[f64],
    candidates: &[CycloidAttentionCandidate<'_>],
    dense_cost: CycloidAttentionCost,
    policy: CycloidAttentionPolicy,
    max_abs_value: f64,
    scenarios: &[CycloidCostScenario<'_>],
) -> Result<CycloidRobustScenarioDecision, CycloidScenarioError> {
    let mut baseline =
        choose_costed_attention(dense_scores, candidates, dense_cost, policy, max_abs_value)?;
    if !(MIN_CYCLOID_SCENARIOS..=MAX_CYCLOID_SCENARIOS).contains(&scenarios.len()) {
        return Err(CycloidScenarioError::InvalidScenarioCount);
    }

    let ids = candidates
        .iter()
        .map(|candidate| candidate.candidate_id)
        .collect::<BTreeSet<_>>();
    let mut scenario_ids = BTreeSet::new();
    let mut qualified = Vec::with_capacity(scenarios.len());
    for scenario in scenarios {
        if !scenario_ids.insert(scenario.scenario_id) {
            return Err(CycloidScenarioError::DuplicateScenarioId(
                scenario.scenario_id,
            ));
        }
        let dense_ns = total_ns(scenario.dense_cost)?;
        if dense_ns == 0 {
            return Err(CycloidScenarioError::InvalidDenseCost(scenario.scenario_id));
        }
        let mut costs = BTreeMap::new();
        for entry in scenario.candidate_costs {
            if !ids.contains(&entry.candidate_id) {
                return Err(CycloidScenarioError::UnknownCandidateId {
                    scenario_id: scenario.scenario_id,
                    candidate_id: entry.candidate_id,
                });
            }
            let candidate_ns = total_ns(entry.cost)?;
            if candidate_ns == 0 {
                return Err(CycloidScenarioError::InvalidCandidateCost {
                    scenario_id: scenario.scenario_id,
                    candidate_id: entry.candidate_id,
                });
            }
            if costs.insert(entry.candidate_id, entry.cost).is_some() {
                return Err(CycloidScenarioError::DuplicateCandidateId {
                    scenario_id: scenario.scenario_id,
                    candidate_id: entry.candidate_id,
                });
            }
        }
        for id in &ids {
            if !costs.contains_key(id) {
                return Err(CycloidScenarioError::MissingCandidateId {
                    scenario_id: scenario.scenario_id,
                    candidate_id: *id,
                });
            }
        }
        qualified.push((dense_ns, costs));
    }

    // Worst ratio, baseline time, baseline read bytes, ID, original keys.
    let mut best: Option<(u64, u64, u64, u64, u32, Vec<usize>)> = None;
    let mut rejected = Vec::new();
    for evidence in &baseline.evidence {
        if evidence.verdict != CycloidAttentionVerdict::CostFidelityCandidate {
            continue;
        }
        let mut worst_num = evidence.total_estimated_ns;
        let mut worst_den = baseline.dense_estimated_ns;
        let mut robust = true;
        for (dense_ns, costs) in &qualified {
            let cost = costs[&evidence.candidate_id];
            let candidate_ns = total_ns(cost)?;
            if candidate_ns >= *dense_ns
                || cost.estimated_read_bytes > policy.max_estimated_read_bytes
            {
                robust = false;
            }
            let left = u128::from(candidate_ns) * u128::from(worst_den);
            let right = u128::from(worst_num) * u128::from(*dense_ns);
            if left > right {
                worst_num = candidate_ns;
                worst_den = *dense_ns;
            }
        }
        if !robust {
            rejected.push(evidence.candidate_id);
            continue;
        }
        let current = (
            worst_num,
            worst_den,
            evidence.total_estimated_ns,
            evidence.estimated_read_bytes,
            evidence.candidate_id,
        );
        let better = match &best {
            Some((num, den, time, bytes, id, _)) => {
                let lhs = u128::from(current.0) * u128::from(*den);
                let rhs = u128::from(*num) * u128::from(current.1);
                lhs < rhs
                    || (lhs == rhs && (current.2, current.3, current.4) < (*time, *bytes, *id))
            }
            None => true,
        };
        if better {
            let keys = candidates
                .iter()
                .find(|c| c.candidate_id == evidence.candidate_id)
                .expect("baseline has already validated every candidate ID")
                .original_key_indices
                .to_vec();
            best = Some((current.0, current.1, current.2, current.3, current.4, keys));
        }
    }
    rejected.sort_unstable();

    let selected_worst_ratio = match best {
        Some((num, den, time, bytes, id, keys)) => {
            baseline.selected_candidate_id = Some(id);
            baseline.original_key_indices = keys;
            baseline.selected_estimated_ns = time;
            baseline.selected_estimated_read_bytes = bytes;
            Some((num, den))
        }
        None => {
            baseline.selected_candidate_id = None;
            baseline.original_key_indices = (0..dense_scores.len()).collect();
            baseline.selected_estimated_ns = baseline.dense_estimated_ns;
            baseline.selected_estimated_read_bytes = dense_cost.estimated_read_bytes;
            None
        }
    };

    Ok(CycloidRobustScenarioDecision {
        attention: baseline,
        scenario_count: scenarios.len(),
        rejected_by_scenario: rejected,
        selected_worst_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(route: u64, read: u64, numerical: u64, bytes: u64) -> CycloidAttentionCost {
        CycloidAttentionCost {
            route_ns: route,
            kv_read_ns: read,
            numerical_ns: numerical,
            estimated_read_bytes: bytes,
        }
    }

    fn policy() -> CycloidAttentionPolicy {
        CycloidAttentionPolicy {
            min_retained_softmax_mass: 0.5,
            max_output_abs_error_bound: 1.0,
            max_estimated_read_bytes: 100,
        }
    }

    fn candidate(id: u32, key: &'static [usize], ns: u64) -> CycloidAttentionCandidate<'static> {
        CycloidAttentionCandidate {
            candidate_id: id,
            original_key_indices: key,
            estimated_cost: cost(1, 1, ns - 2, 10),
        }
    }

    #[test]
    fn less_favorable_baseline_candidate_wins_across_all_scenarios() {
        let candidates = [candidate(1, &[0], 20), candidate(2, &[1], 40)];
        let first = [
            CycloidScenarioCandidateCost {
                candidate_id: 2,
                cost: cost(1, 1, 38, 10),
            },
            CycloidScenarioCandidateCost {
                candidate_id: 1,
                cost: cost(1, 1, 18, 10),
            },
        ];
        let second = [
            CycloidScenarioCandidateCost {
                candidate_id: 1,
                cost: cost(1, 1, 78, 10),
            },
            CycloidScenarioCandidateCost {
                candidate_id: 2,
                cost: cost(1, 1, 28, 10),
            },
        ];
        let scenarios = [
            CycloidCostScenario {
                scenario_id: 101,
                dense_cost: cost(0, 60, 40, 80),
                candidate_costs: &first,
            },
            CycloidCostScenario {
                scenario_id: 102,
                dense_cost: cost(0, 20, 30, 80),
                candidate_costs: &second,
            },
        ];
        let result = choose_scenario_robust_attention(
            &[0.0, 0.0],
            &candidates,
            cost(0, 60, 40, 80),
            policy(),
            1.0,
            &scenarios,
        )
        .unwrap();
        assert_eq!(result.attention.selected_candidate_id, Some(2));
        assert_eq!(result.attention.original_key_indices, vec![1]);
        assert_eq!(result.rejected_by_scenario, vec![1]);
        assert_eq!(result.selected_worst_ratio, Some((30, 50)));
    }

    #[test]
    fn sparse_candidate_that_loses_one_scenario_falls_back_to_dense() {
        let candidates = [candidate(1, &[0], 20)];
        let rates = [CycloidScenarioCandidateCost {
            candidate_id: 1,
            cost: cost(0, 50, 51, 10),
        }];
        let scenarios = [
            CycloidCostScenario {
                scenario_id: 1,
                dense_cost: cost(0, 60, 40, 80),
                candidate_costs: &rates,
            },
            CycloidCostScenario {
                scenario_id: 2,
                dense_cost: cost(0, 60, 40, 80),
                candidate_costs: &rates,
            },
        ];
        let result = choose_scenario_robust_attention(
            &[0.0, 0.0],
            &candidates,
            cost(0, 60, 40, 80),
            policy(),
            1.0,
            &scenarios,
        )
        .unwrap();
        assert_eq!(result.attention.selected_candidate_id, None);
        assert_eq!(result.attention.original_key_indices, vec![0, 1]);
        assert_eq!(result.rejected_by_scenario, vec![1]);
    }

    #[test]
    fn malformed_scenario_coverage_and_duplicate_id_fail_closed() {
        let c = [candidate(1, &[0], 20)];
        let proper = [CycloidScenarioCandidateCost {
            candidate_id: 1,
            cost: cost(1, 1, 18, 10),
        }];
        let empty = [];
        let malformed = [
            CycloidCostScenario {
                scenario_id: 1,
                dense_cost: cost(0, 50, 50, 80),
                candidate_costs: &proper,
            },
            CycloidCostScenario {
                scenario_id: 2,
                dense_cost: cost(0, 50, 50, 80),
                candidate_costs: &empty,
            },
        ];
        assert_eq!(
            choose_scenario_robust_attention(
                &[0.0, 0.0],
                &c,
                cost(0, 50, 50, 80),
                policy(),
                1.0,
                &malformed
            ),
            Err(CycloidScenarioError::MissingCandidateId {
                scenario_id: 2,
                candidate_id: 1
            })
        );
        let duplicate = [
            CycloidCostScenario {
                scenario_id: 3,
                dense_cost: cost(0, 50, 50, 80),
                candidate_costs: &proper,
            },
            CycloidCostScenario {
                scenario_id: 3,
                dense_cost: cost(0, 50, 50, 80),
                candidate_costs: &proper,
            },
        ];
        assert_eq!(
            choose_scenario_robust_attention(
                &[0.0, 0.0],
                &c,
                cost(0, 50, 50, 80),
                policy(),
                1.0,
                &duplicate
            ),
            Err(CycloidScenarioError::DuplicateScenarioId(3))
        );
    }

    #[test]
    fn invalid_scenario_count_and_overflow_are_rejected() {
        let c = [candidate(1, &[0], 20)];
        assert_eq!(
            choose_scenario_robust_attention(
                &[0.0, 0.0],
                &c,
                cost(0, 50, 50, 80),
                policy(),
                1.0,
                &[]
            ),
            Err(CycloidScenarioError::InvalidScenarioCount)
        );
        let bad = [CycloidScenarioCandidateCost {
            candidate_id: 1,
            cost: cost(u64::MAX, 1, 1, 10),
        }];
        let scenarios = [
            CycloidCostScenario {
                scenario_id: 1,
                dense_cost: cost(0, 50, 50, 80),
                candidate_costs: &bad,
            },
            CycloidCostScenario {
                scenario_id: 2,
                dense_cost: cost(0, 50, 50, 80),
                candidate_costs: &bad,
            },
        ];
        assert_eq!(
            choose_scenario_robust_attention(
                &[0.0, 0.0],
                &c,
                cost(0, 50, 50, 80),
                policy(),
                1.0,
                &scenarios
            ),
            Err(CycloidScenarioError::CostOverflow)
        );
    }
}
