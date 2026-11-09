//! CYCLOID-FLAT-1: select among independently generated sparse-key candidates.
//!
//! This is a bounded, deterministic OFFLINE evaluation/decision instrument.
//! Dense QK scores are supplied by an independent oracle, so this code is not
//! an inference prefilter and cannot establish a runtime speedup.
//! Existing MAA Boolean/F2 candidate generation and exact numerical FLAT
//! execution remain authoritative. The selector estimates whether a longer
//! routing path could repay its overhead while satisfying a mass/output bound.

use core::fmt;
use std::collections::BTreeSet;

use crate::softmax_mass::{output_error_bound, retained_softmax_mass, SoftmaxMassError};

pub const MAX_CYCLOID_FLAT_CANDIDATES: usize = 64;

/// Costs are externally supplied estimates, not measured GPU transactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CycloidAttentionCost {
    pub route_ns: u64,
    pub kv_read_ns: u64,
    pub numerical_ns: u64,
    pub estimated_read_bytes: u64,
}

impl CycloidAttentionCost {
    fn total_ns(self) -> Result<u64, CycloidAttentionError> {
        self.route_ns
            .checked_add(self.kv_read_ns)
            .and_then(|v| v.checked_add(self.numerical_ns))
            .ok_or(CycloidAttentionError::CostOverflow)
    }
}

/// Prequalified candidate key set (e.g., derived by MAA), with its independent
/// estimated execution cost. Key indices are original numerical K/V positions.
#[derive(Debug, Clone, Copy)]
pub struct CycloidAttentionCandidate<'a> {
    pub candidate_id: u32,
    pub original_key_indices: &'a [usize],
    pub estimated_cost: CycloidAttentionCost,
}

/// Research-only fidelity threshold, not a language-model quality guarantee.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CycloidAttentionPolicy {
    pub min_retained_softmax_mass: f64,
    pub max_output_abs_error_bound: f64,
    pub max_estimated_read_bytes: u64,
}

impl CycloidAttentionPolicy {
    fn validate(self) -> Result<Self, CycloidAttentionError> {
        if !self.min_retained_softmax_mass.is_finite()
            || !(0.0..=1.0).contains(&self.min_retained_softmax_mass)
            || !self.max_output_abs_error_bound.is_finite()
            || self.max_output_abs_error_bound < 0.0
        {
            return Err(CycloidAttentionError::InvalidPolicy);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycloidAttentionVerdict {
    CostFidelityCandidate,
    LowRetainedMass,
    ExcessiveOutputBound,
    ExcessiveReadBudget,
    NotFasterThanDense,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CycloidAttentionEvidence {
    pub candidate_id: u32,
    pub retained_mass: f64,
    pub max_output_error_bound: f64,
    pub total_estimated_ns: u64,
    pub estimated_read_bytes: u64,
    pub verdict: CycloidAttentionVerdict,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CycloidAttentionDecision {
    /// None means retain the authoritative dense baseline.
    pub selected_candidate_id: Option<u32>,
    /// Original key identity; dense fallback contains 0..scores.len().
    pub original_key_indices: Vec<usize>,
    pub dense_estimated_ns: u64,
    pub selected_estimated_ns: u64,
    pub selected_estimated_read_bytes: u64,
    pub evidence: Vec<CycloidAttentionEvidence>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CycloidAttentionError {
    Softmax(SoftmaxMassError),
    InvalidPolicy,
    TooManyCandidates,
    DuplicateCandidateId(u32),
    EmptyCandidate(u32),
    CostOverflow,
    ZeroDenseCost,
}

impl fmt::Display for CycloidAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CycloidAttentionError {}

impl From<SoftmaxMassError> for CycloidAttentionError {
    fn from(value: SoftmaxMassError) -> Self {
        Self::Softmax(value)
    }
}

/// Compare sparse candidates with the exact same dense-reference score vector.
/// Work is O(candidate_count * key_count), and SHOULD NOT run per decode token.
/// Candidate-set shape/causality/epoch qualification is the caller's obligation;
/// dense scores and the selected numerical K/V must share the same identity.
/// This host diagnostic neither selects unscored future positions nor changes
/// any stable FLAT attention API.
pub fn choose_costed_attention(
    dense_scores: &[f64],
    candidates: &[CycloidAttentionCandidate<'_>],
    dense_cost: CycloidAttentionCost,
    policy: CycloidAttentionPolicy,
    max_abs_value: f64,
) -> Result<CycloidAttentionDecision, CycloidAttentionError> {
    let policy = policy.validate()?;
    if candidates.len() > MAX_CYCLOID_FLAT_CANDIDATES {
        return Err(CycloidAttentionError::TooManyCandidates);
    }
    // Validates even unselected dense scores and the value magnitude bound.
    retained_softmax_mass(dense_scores, &[])?;
    output_error_bound(1.0, max_abs_value)?;
    let dense_total_ns = dense_cost.total_ns()?;
    if dense_total_ns == 0 {
        return Err(CycloidAttentionError::ZeroDenseCost);
    }

    let mut seen_ids = BTreeSet::new();
    let mut evidence = Vec::with_capacity(candidates.len());
    // Estimated total time, read bytes, candidate ID, original key positions.
    let mut best: Option<(u64, u64, u32, Vec<usize>)> = None;

    for candidate in candidates {
        if !seen_ids.insert(candidate.candidate_id) {
            return Err(CycloidAttentionError::DuplicateCandidateId(
                candidate.candidate_id,
            ));
        }
        if candidate.original_key_indices.is_empty() {
            return Err(CycloidAttentionError::EmptyCandidate(
                candidate.candidate_id,
            ));
        }
        let mass = retained_softmax_mass(dense_scores, candidate.original_key_indices)?;
        let retained_mass = mass.retained_mass();
        let error_bound = output_error_bound(retained_mass, max_abs_value)?;
        let total_ns = candidate.estimated_cost.total_ns()?;
        let read_bytes = candidate.estimated_cost.estimated_read_bytes;

        let verdict = if retained_mass < policy.min_retained_softmax_mass {
            CycloidAttentionVerdict::LowRetainedMass
        } else if error_bound > policy.max_output_abs_error_bound {
            CycloidAttentionVerdict::ExcessiveOutputBound
        } else if read_bytes > policy.max_estimated_read_bytes {
            CycloidAttentionVerdict::ExcessiveReadBudget
        } else if total_ns >= dense_total_ns {
            CycloidAttentionVerdict::NotFasterThanDense
        } else {
            CycloidAttentionVerdict::CostFidelityCandidate
        };

        evidence.push(CycloidAttentionEvidence {
            candidate_id: candidate.candidate_id,
            retained_mass,
            max_output_error_bound: error_bound,
            total_estimated_ns: total_ns,
            estimated_read_bytes: read_bytes,
            verdict,
        });
        if verdict == CycloidAttentionVerdict::CostFidelityCandidate {
            let key = (total_ns, read_bytes, candidate.candidate_id);
            let replace = match &best {
                Some((time, bytes, id, _)) => key < (*time, *bytes, *id),
                None => true,
            };
            if replace {
                best = Some((
                    total_ns,
                    read_bytes,
                    candidate.candidate_id,
                    candidate.original_key_indices.to_vec(),
                ));
            }
        }
    }
    let (selected_candidate_id, original_key_indices, selected_estimated_ns, selected_read_bytes) =
        match best {
            Some((time, bytes, id, positions)) => (Some(id), positions, time, bytes),
            None => (
                None,
                (0..dense_scores.len()).collect(),
                dense_total_ns,
                dense_cost.estimated_read_bytes,
            ),
        };

    Ok(CycloidAttentionDecision {
        selected_candidate_id,
        original_key_indices,
        dense_estimated_ns: dense_total_ns,
        selected_estimated_ns,
        selected_estimated_read_bytes: selected_read_bytes,
        evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(route: u64, read: u64, math: u64, bytes: u64) -> CycloidAttentionCost {
        CycloidAttentionCost {
            route_ns: route,
            kv_read_ns: read,
            numerical_ns: math,
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

    #[test]
    fn longer_routed_selection_can_beat_dense_under_fidelity_budget() {
        let candidate = CycloidAttentionCandidate {
            candidate_id: 3,
            original_key_indices: &[0],
            estimated_cost: cost(10, 10, 20, 20),
        };
        let answer = choose_costed_attention(
            &[0.0, 0.0],
            &[candidate],
            cost(0, 60, 40, 80),
            policy(),
            1.0,
        )
        .unwrap();
        assert_eq!(answer.selected_candidate_id, Some(3));
        assert_eq!(answer.original_key_indices, vec![0]);
        assert_eq!(answer.selected_estimated_ns, 40);
        assert_eq!(
            answer.evidence[0].verdict,
            CycloidAttentionVerdict::CostFidelityCandidate
        );
    }

    #[test]
    fn dropped_dominant_key_forces_dense_fallback() {
        let candidate = CycloidAttentionCandidate {
            candidate_id: 1,
            original_key_indices: &[1],
            estimated_cost: cost(1, 1, 1, 1),
        };
        let mut rules = policy();
        rules.min_retained_softmax_mass = 0.99;
        let answer =
            choose_costed_attention(&[10.0, 0.0], &[candidate], cost(0, 50, 50, 80), rules, 1.0)
                .unwrap();
        assert_eq!(answer.selected_candidate_id, None);
        assert_eq!(answer.original_key_indices, vec![0, 1]);
        assert_eq!(
            answer.evidence[0].verdict,
            CycloidAttentionVerdict::LowRetainedMass
        );
    }

    #[test]
    fn routing_overhead_and_read_budget_can_reject_candidate() {
        let candidates = [
            CycloidAttentionCandidate {
                candidate_id: 1,
                original_key_indices: &[0],
                estimated_cost: cost(100, 10, 10, 20),
            },
            CycloidAttentionCandidate {
                candidate_id: 2,
                original_key_indices: &[1],
                estimated_cost: cost(1, 1, 1, 101),
            },
        ];
        let answer =
            choose_costed_attention(&[0.0, 0.0], &candidates, cost(0, 50, 50, 80), policy(), 1.0)
                .unwrap();
        assert_eq!(answer.selected_candidate_id, None);
        assert_eq!(
            answer.evidence[0].verdict,
            CycloidAttentionVerdict::NotFasterThanDense
        );
        assert_eq!(
            answer.evidence[1].verdict,
            CycloidAttentionVerdict::ExcessiveReadBudget
        );
    }

    #[test]
    fn cost_ties_prefer_lower_read_bytes_then_id() {
        let candidates = [
            CycloidAttentionCandidate {
                candidate_id: 9,
                original_key_indices: &[1],
                estimated_cost: cost(10, 10, 10, 30),
            },
            CycloidAttentionCandidate {
                candidate_id: 2,
                original_key_indices: &[0],
                estimated_cost: cost(10, 10, 10, 20),
            },
            CycloidAttentionCandidate {
                candidate_id: 1,
                original_key_indices: &[1],
                estimated_cost: cost(10, 10, 10, 20),
            },
        ];
        let a =
            choose_costed_attention(&[0.0, 0.0], &candidates, cost(0, 50, 50, 80), policy(), 1.0)
                .unwrap();
        assert_eq!(a.selected_candidate_id, Some(1));
        let b =
            choose_costed_attention(&[0.0, 0.0], &candidates, cost(0, 50, 50, 80), policy(), 1.0)
                .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn untrusted_or_malformed_evidence_fails_closed() {
        let empty = CycloidAttentionCandidate {
            candidate_id: 1,
            original_key_indices: &[],
            estimated_cost: cost(1, 1, 1, 1),
        };
        assert_eq!(
            choose_costed_attention(&[0.0], &[empty], cost(0, 1, 1, 1), policy(), 1.0),
            Err(CycloidAttentionError::EmptyCandidate(1))
        );
        let duplicate = CycloidAttentionCandidate {
            original_key_indices: &[0],
            ..empty
        };
        assert_eq!(
            choose_costed_attention(
                &[0.0],
                &[duplicate, duplicate],
                cost(0, 1, 1, 1),
                policy(),
                1.0
            ),
            Err(CycloidAttentionError::DuplicateCandidateId(1))
        );
        assert_eq!(
            choose_costed_attention(
                &[0.0, f64::NAN],
                &[duplicate],
                cost(0, 1, 1, 1),
                policy(),
                1.0
            ),
            Err(CycloidAttentionError::Softmax(
                SoftmaxMassError::NonFiniteScore { index: 1 }
            ))
        );
    }
}
