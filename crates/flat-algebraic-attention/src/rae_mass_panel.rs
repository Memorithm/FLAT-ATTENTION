//! RAE-4b: paired numerical quality panel over RAE-3's five page policies.
//!
//! The original causal dense scores are required for every query. This
//! diagnostic cannot establish sparse execution, physical bandwidth savings,
//! hardware speedups, model quality or authority to skip a K/V read.

use core::fmt;

use crate::{
    rae_numeric_mass::{audit_causal_page_mass, RaeNumericMassAudit, RaeNumericMassError},
    rae_policy_audit::{RaeAudit, RaePolicy, RAE_POLICIES, RAE_POLICY_COUNT},
};

#[derive(Debug, Clone, Copy)]
pub struct RaeNumericalQuery<'a> {
    pub tick: u64,
    /// Independently computed dense logits in original key coordinates.
    pub scores: &'a [f64],
    /// Authoritative causal, length and padding eligibility mask.
    pub eligible: &'a [bool],
    /// Original page identity of each key.
    pub key_pages: &'a [usize],
    /// Independently established upper bound for each relevant V component.
    pub max_abs_value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RaeMassPanelError {
    EmptyPolicyAudit,
    WrongQueryCount { expected: usize, actual: usize },
    TickMismatch { row: usize, policy_tick: u64, numerical_tick: u64 },
    WrongPolicyCount { row: usize, expected: usize, actual: usize },
    InvalidMassBudget,
    InvalidOutputBudget,
    Numerical {
        row: usize,
        policy: RaePolicy,
        source: RaeNumericMassError,
    },
}

impl fmt::Display for RaeMassPanelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RaeMassPanelError {}

#[derive(Debug, Clone, PartialEq)]
pub struct RaeMassPolicyPoint {
    pub policy: RaePolicy,
    pub numerical: RaeNumericMassAudit,
    pub within_budgets: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaeMassPanelRow {
    pub tick: u64,
    /// Exactly one point per policy, in RAE_POLICIES order.
    pub policies: Vec<RaeMassPolicyPoint>,
}

impl RaeMassPanelRow {
    #[must_use]
    pub fn for_policy(&self, policy: RaePolicy) -> Option<&RaeMassPolicyPoint> {
        self.policies.iter().find(|point| point.policy == policy)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaeMassPolicySummary {
    pub policy: RaePolicy,
    pub queries: usize,
    pub budget_failures: usize,
    pub zero_selected_queries: usize,
    pub total_selected_eligible_keys: u128,
    /// Mean retained mass on exactly the same sampled query sequence.
    pub mean_retained_mass: f64,
    pub worst_dropped_mass: f64,
    pub worst_case_output_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaeMassPanel {
    pub rows: Vec<RaeMassPanelRow>,
    pub summaries: Vec<RaeMassPolicySummary>,
}

impl RaeMassPanel {
    #[must_use]
    pub fn summary(&self, policy: RaePolicy) -> Option<&RaeMassPolicySummary> {
        self.summaries.iter().find(|summary| summary.policy == policy)
    }
}

/// Join EXACTLY the same tick identities and original page IDs between
/// RAE-3 Boolean page masks and RAE-4 dense numerical softmax reference.
///
/// No policy may treat a pass in this host oracle as proof that skipping
/// attention is safe on future/changed queries, or that the query/key/value
/// read cost has been reduced. Every numerical query must supply complete,
/// independent causal logits and a bound on its values.
pub fn audit_rae_numerical_panel(
    selection: &RaeAudit,
    numerical_queries: &[RaeNumericalQuery<'_>],
    max_dropped_mass: f64,
    max_output_error: f64,
) -> Result<RaeMassPanel, RaeMassPanelError> {
    if selection.rows.is_empty() {
        return Err(RaeMassPanelError::EmptyPolicyAudit);
    }
    if selection.rows.len() != numerical_queries.len() {
        return Err(RaeMassPanelError::WrongQueryCount {
            expected: selection.rows.len(),
            actual: numerical_queries.len(),
        });
    }
    if !max_dropped_mass.is_finite() || !(0.0..=1.0).contains(&max_dropped_mass) {
        return Err(RaeMassPanelError::InvalidMassBudget);
    }
    if !max_output_error.is_finite() || max_output_error < 0.0 {
        return Err(RaeMassPanelError::InvalidOutputBudget);
    }

    let mut rows = Vec::with_capacity(selection.rows.len());
    let mut summaries = RAE_POLICIES
        .map(|policy| RaeMassPolicySummary {
            policy,
            queries: 0,
            budget_failures: 0,
            zero_selected_queries: 0,
            total_selected_eligible_keys: 0,
            mean_retained_mass: 0.0,
            worst_dropped_mass: 0.0,
            worst_case_output_bound: 0.0,
        })
        .to_vec();

    for (row, (selection_row, query)) in selection
        .rows
        .iter()
        .zip(numerical_queries)
        .enumerate()
    {
        if selection_row.tick != query.tick {
            return Err(RaeMassPanelError::TickMismatch {
                row,
                policy_tick: selection_row.tick,
                numerical_tick: query.tick,
            });
        }
        if selection_row.policy_masks.len() != RAE_POLICY_COUNT {
            return Err(RaeMassPanelError::WrongPolicyCount {
                row,
                expected: RAE_POLICY_COUNT,
                actual: selection_row.policy_masks.len(),
            });
        }
        let mut points = Vec::with_capacity(RAE_POLICY_COUNT);
        for (index, policy) in RAE_POLICIES.iter().copied().enumerate() {
            let numerical = audit_causal_page_mass(
                query.scores,
                query.eligible,
                query.key_pages,
                selection.page_count,
                &selection_row.policy_masks[index],
                query.max_abs_value,
            )
            .map_err(|source| RaeMassPanelError::Numerical { row, policy, source })?;
            let within_budgets = numerical
                .within_oracle_budgets(max_dropped_mass, max_output_error)
                .map_err(|source| RaeMassPanelError::Numerical { row, policy, source })?;
            let summary = &mut summaries[index];
            summary.queries += 1;
            summary.budget_failures += usize::from(!within_budgets);
            summary.zero_selected_queries += usize::from(numerical.selected_keys == 0);
            summary.total_selected_eligible_keys += numerical.selected_keys as u128;
            summary.mean_retained_mass += numerical.softmax.retained_mass();
            summary.worst_dropped_mass = summary
                .worst_dropped_mass
                .max(numerical.softmax.dropped_mass());
            summary.worst_case_output_bound = summary
                .worst_case_output_bound
                .max(numerical.worst_case_output_error_bound);
            points.push(RaeMassPolicyPoint {
                policy,
                numerical,
                within_budgets,
            });
        }
        rows.push(RaeMassPanelRow {
            tick: query.tick,
            policies: points,
        });
    }
    for summary in &mut summaries {
        summary.mean_retained_mass /= summary.queries as f64;
    }
    Ok(RaeMassPanel { rows, summaries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rae_policy_audit::{audit_paired_page_policies, RaeStep};

    fn policy_trace() -> RaeAudit {
        let steps = [
            RaeStep {
                tick: 1,
                feature_state: &[0b01],
                required_pages: &[0b01],
                context_invalidated: false,
            },
            RaeStep {
                tick: 2,
                feature_state: &[0b01],
                required_pages: &[0b10],
                context_invalidated: true,
            },
        ];
        audit_paired_page_policies(2, 2, &[0, 1], 0, &[0], &steps).unwrap()
    }

    #[test]
    fn held_high_query_can_drop_almost_all_mass_despite_context_recheck() {
        let trace = policy_trace();
        let queries = [
            RaeNumericalQuery {
                tick: 1,
                scores: &[0.0, 0.0],
                eligible: &[true, true],
                key_pages: &[0, 1],
                max_abs_value: 2.0,
            },
            RaeNumericalQuery {
                tick: 2,
                scores: &[0.0, 100.0],
                eligible: &[true, true],
                key_pages: &[0, 1],
                max_abs_value: 2.0,
            },
        ];
        let panel = audit_rae_numerical_panel(&trace, &queries, 0.1, 0.4).unwrap();
        assert_eq!(panel.rows.len(), 2);
        let all = panel.summary(RaePolicy::AllPages).unwrap();
        assert_eq!(all.budget_failures, 0);
        assert_eq!(all.worst_dropped_mass, 0.0);
        let rising = panel.summary(RaePolicy::RisingOnly).unwrap();
        assert_eq!(rising.zero_selected_queries, 1);
        assert_eq!(rising.budget_failures, 2);
        let active = panel.summary(RaePolicy::ActiveState).unwrap();
        assert!(active.worst_dropped_mass > 0.999999);
        assert!(active.worst_case_output_bound > 3.99999);
        let hybrid = panel.summary(RaePolicy::RisingWithContextInvalidation).unwrap();
        assert!(hybrid.worst_dropped_mass > 0.999999);
        assert!(!panel.rows[1]
            .for_policy(RaePolicy::RisingWithContextInvalidation)
            .unwrap()
            .within_budgets);
    }

    #[test]
    fn future_key_must_not_contribute_to_policy_mass() {
        let trace = policy_trace();
        let queries = [
            RaeNumericalQuery {
                tick: 1,
                scores: &[1000.0, 0.0],
                eligible: &[false, true],
                key_pages: &[0, 1],
                max_abs_value: 1.0,
            },
            RaeNumericalQuery {
                tick: 2,
                scores: &[1000.0, 0.0],
                eligible: &[false, true],
                key_pages: &[0, 1],
                max_abs_value: 1.0,
            },
        ];
        let panel = audit_rae_numerical_panel(&trace, &queries, 0.0, 0.0).unwrap();
        assert_eq!(
            panel.rows[0]
                .for_policy(RaePolicy::AllPages)
                .unwrap()
                .numerical
                .softmax
                .retained_mass(),
            1.0
        );
        assert_eq!(
            panel.rows[0]
                .for_policy(RaePolicy::RisingOnly)
                .unwrap()
                .numerical
                .softmax
                .retained_mass(),
            0.0
        );
        assert_eq!(
            panel.rows[1]
                .for_policy(RaePolicy::AllPages)
                .unwrap()
                .numerical
                .softmax
                .retained_mass(),
            1.0
        );
    }

    #[test]
    fn rejects_mismatched_ticks_query_count_and_bad_thresholds() {
        let trace = policy_trace();
        let single = [RaeNumericalQuery {
            tick: 2,
            scores: &[0.0],
            eligible: &[true],
            key_pages: &[0],
            max_abs_value: 1.0,
        }];
        assert_eq!(
            audit_rae_numerical_panel(&trace, &single, 0.2, 1.0),
            Err(RaeMassPanelError::WrongQueryCount {
                expected: 2,
                actual: 1,
            })
        );
        let pair = [single[0], single[0]];
        assert_eq!(
            audit_rae_numerical_panel(&trace, &pair, 0.2, 1.0),
            Err(RaeMassPanelError::TickMismatch {
                row: 0,
                policy_tick: 1,
                numerical_tick: 2,
            })
        );
        assert_eq!(
            audit_rae_numerical_panel(&trace, &pair, 1.1, 1.0),
            Err(RaeMassPanelError::InvalidMassBudget)
        );
        assert_eq!(
            audit_rae_numerical_panel(&trace, &pair, 0.1, -1.0),
            Err(RaeMassPanelError::InvalidOutputBudget)
        );
    }
}
