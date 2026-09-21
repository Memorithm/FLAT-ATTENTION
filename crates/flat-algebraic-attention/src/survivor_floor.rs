use core::fmt;
use std::collections::BTreeSet;

use crate::qualification::{CandidateDisposition, CandidateQualificationDecision};

#[derive(Debug, Clone, Copy)]
pub struct SurvivorFloorCandidate<'a> {
    candidate_index: usize,
    decision: &'a CandidateQualificationDecision,
}

impl<'a> SurvivorFloorCandidate<'a> {
    #[must_use]
    pub const fn new(
        candidate_index: usize,
        decision: &'a CandidateQualificationDecision,
    ) -> Self {
        Self {
            candidate_index,
            decision,
        }
    }

    #[must_use]
    pub const fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    #[must_use]
    pub const fn decision(&self) -> &'a CandidateQualificationDecision {
        self.decision
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurvivorFloorResult {
    requested_floor: usize,
    target_floor: usize,
    boolean_survivors: usize,
    strict_survivors: usize,
    survivor_indices: Vec<usize>,
    rescued_indices: Vec<usize>,
}

impl SurvivorFloorResult {
    #[must_use]
    pub const fn requested_floor(&self) -> usize {
        self.requested_floor
    }

    #[must_use]
    pub const fn target_floor(&self) -> usize {
        self.target_floor
    }

    #[must_use]
    pub const fn boolean_survivors(&self) -> usize {
        self.boolean_survivors
    }

    #[must_use]
    pub const fn strict_survivors(&self) -> usize {
        self.strict_survivors
    }

    #[must_use]
    pub fn survivor_indices(&self) -> &[usize] {
        &self.survivor_indices
    }

    #[must_use]
    pub fn rescued_indices(&self) -> &[usize] {
        &self.rescued_indices
    }

    #[must_use]
    pub const fn rescued_count(&self) -> usize {
        self.rescued_indices.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurvivorFloorError {
    DuplicateCandidateIndex { candidate_index: usize },
    MissingBooleanDecision { candidate_index: usize },
    MaxPlusDecisionNotSupported { candidate_index: usize },
    InconsistentStrictAdmission { candidate_index: usize },
}

impl fmt::Display for SurvivorFloorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateCandidateIndex { candidate_index } => write!(
                formatter,
                "candidate index {candidate_index} occurs more than once in a survivor-floor batch"
            ),
            Self::MissingBooleanDecision { candidate_index } => write!(
                formatter,
                "candidate {candidate_index} has no Boolean verdict for survivor-floor rescue"
            ),
            Self::MaxPlusDecisionNotSupported { candidate_index } => write!(
                formatter,
                "candidate {candidate_index} carries Max-Plus readiness evidence; survivor-floor rescue is semantic-only"
            ),
            Self::InconsistentStrictAdmission { candidate_index } => write!(
                formatter,
                "candidate {candidate_index} was strictly admitted despite failing Boolean admission"
            ),
        }
    }
}

impl std::error::Error for SurvivorFloorError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RescueCandidate {
    candidate_index: usize,
    support: u8,
}

pub fn apply_survivor_floor(
    candidates: &[SurvivorFloorCandidate<'_>],
    minimum_survivors: usize,
) -> Result<SurvivorFloorResult, SurvivorFloorError> {
    let mut seen = BTreeSet::new();
    let mut strict = BTreeSet::new();
    let mut rescue_pool = Vec::new();
    let mut boolean_survivors = 0usize;

    for candidate in candidates {
        if !seen.insert(candidate.candidate_index) {
            return Err(SurvivorFloorError::DuplicateCandidateIndex {
                candidate_index: candidate.candidate_index,
            });
        }

        let decision = candidate.decision;
        let boolean = decision.boolean().ok_or(
            SurvivorFloorError::MissingBooleanDecision {
                candidate_index: candidate.candidate_index,
            },
        )?;

        if decision.max_plus().is_some() {
            return Err(SurvivorFloorError::MaxPlusDecisionNotSupported {
                candidate_index: candidate.candidate_index,
            });
        }

        if decision.disposition() == CandidateDisposition::Admit && !boolean.qualified() {
            return Err(SurvivorFloorError::InconsistentStrictAdmission {
                candidate_index: candidate.candidate_index,
            });
        }

        if !boolean.qualified() {
            continue;
        }

        boolean_survivors += 1;

        if decision.disposition() == CandidateDisposition::Admit {
            strict.insert(candidate.candidate_index);
            continue;
        }

        let support = u8::from(decision.f2().is_some_and(|value| value.qualified()))
            + u8::from(
                decision
                    .zhegalkin()
                    .is_some_and(|value| value.qualified()),
            );
        rescue_pool.push(RescueCandidate {
            candidate_index: candidate.candidate_index,
            support,
        });
    }

    let strict_survivors = strict.len();
    let target_floor = minimum_survivors.min(boolean_survivors);
    let needed = target_floor.saturating_sub(strict_survivors);

    rescue_pool.sort_by(|left, right| {
        right
            .support
            .cmp(&left.support)
            .then_with(|| left.candidate_index.cmp(&right.candidate_index))
    });

    let rescued = rescue_pool
        .iter()
        .take(needed)
        .map(|candidate| candidate.candidate_index)
        .collect::<BTreeSet<_>>();

    let mut survivor_indices = Vec::with_capacity(strict_survivors + rescued.len());
    let mut rescued_indices = Vec::with_capacity(rescued.len());
    for candidate in candidates {
        if strict.contains(&candidate.candidate_index) || rescued.contains(&candidate.candidate_index)
        {
            survivor_indices.push(candidate.candidate_index);
            if rescued.contains(&candidate.candidate_index) {
                rescued_indices.push(candidate.candidate_index);
            }
        }
    }

    Ok(SurvivorFloorResult {
        requested_floor: minimum_survivors,
        target_floor,
        boolean_survivors,
        strict_survivors,
        survivor_indices,
        rescued_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cooperation::{
        route_attention_needs, AttentionAlgebraNeeds, M13bBooleanRoutingEvidence,
    };
    use crate::f2::{F2AffinePredicate, F2Vector};
    use crate::qualification::{
        qualify_candidate, CandidateQualificationInputs, F2CandidateEvaluation,
        MaxPlusCandidateEvaluation, RecompositionPolicy, ZhegalkinCandidateEvaluation,
    };
    use crate::max_plus::{MaxPlusSchedule, MaxPlusValue};
    use crate::zhegalkin::ZhegalkinPolynomial;

    fn route(mask: u64, blocks: usize) -> crate::cooperation::AlgebraicRoute {
        route_attention_needs(
            AttentionAlgebraNeeds {
                eligibility_logic: true,
                parity_or_binary_linear: true,
                nonlinear_boolean_interaction: true,
                ..AttentionAlgebraNeeds::default()
            },
            Some(
                M13bBooleanRoutingEvidence::new(
                    1,
                    blocks,
                    vec![mask],
                    1,
                    blocks,
                    vec![(1u64 << blocks) - 1],
                    vec![mask],
                    None,
                    None,
                )
                .unwrap(),
            ),
        )
        .unwrap()
    }

    fn decisions(
        mask: u64,
        features: &[[bool; 3]],
    ) -> Vec<crate::qualification::CandidateQualificationDecision> {
        let route = route(mask, features.len());
        let f2 =
            F2AffinePredicate::new(F2Vector::from_bools(&[true, false, false]).unwrap(), false);
        let zhegalkin =
            ZhegalkinPolynomial::from_variable_sets(3, vec![vec![1, 2]]).unwrap();

        features
            .iter()
            .enumerate()
            .map(|(candidate_index, bits)| {
                let input = F2Vector::from_bools(bits).unwrap();
                qualify_candidate(
                    &route,
                    CandidateQualificationInputs {
                        boolean_block: Some(candidate_index),
                        f2: Some(F2CandidateEvaluation {
                            predicate: &f2,
                            input: &input,
                        }),
                        zhegalkin: Some(ZhegalkinCandidateEvaluation {
                            polynomial: &zhegalkin,
                            input: &input,
                        }),
                        max_plus: None,
                    },
                    RecompositionPolicy::AllSelectedMustQualify,
                )
                .unwrap()
            })
            .collect()
    }

    fn frames<'a>(
        decisions: &'a [crate::qualification::CandidateQualificationDecision],
    ) -> Vec<SurvivorFloorCandidate<'a>> {
        decisions
            .iter()
            .enumerate()
            .map(|(candidate_index, decision)| SurvivorFloorCandidate::new(candidate_index, decision))
            .collect()
    }

    #[test]
    fn floor_zero_reproduces_strict_survivors() {
        let decisions = decisions(
            0b1111,
            &[
                [true, true, true],
                [true, true, false],
                [false, true, true],
                [true, true, true],
            ],
        );
        let result = apply_survivor_floor(&frames(&decisions), 0).unwrap();

        assert_eq!(result.survivor_indices(), &[0, 3]);
        assert_eq!(result.strict_survivors(), 2);
        assert!(result.rescued_indices().is_empty());
    }

    #[test]
    fn floor_rescues_by_auxiliary_support_then_candidate_id() {
        let decisions = decisions(
            0b1111,
            &[
                [false, false, false],
                [true, true, false],
                [false, true, true],
                [false, false, false],
            ],
        );
        let result = apply_survivor_floor(&frames(&decisions), 2).unwrap();

        assert_eq!(result.strict_survivors(), 0);
        assert_eq!(result.survivor_indices(), &[1, 2]);
        assert_eq!(result.rescued_indices(), &[1, 2]);
    }

    #[test]
    fn rescue_never_resurrects_boolean_rejection() {
        let decisions = decisions(
            0b0101,
            &[
                [true, true, true],
                [true, true, true],
                [false, true, false],
                [true, true, true],
            ],
        );
        let result = apply_survivor_floor(&frames(&decisions), 4).unwrap();

        assert_eq!(result.boolean_survivors(), 2);
        assert_eq!(result.target_floor(), 2);
        assert!(result.survivor_indices().iter().all(|id| [0, 2].contains(id)));
    }

    #[test]
    fn floor_at_boolean_count_reproduces_boolean_survivor_set() {
        let decisions = decisions(
            0b1101,
            &[
                [false, false, false],
                [true, true, true],
                [true, false, false],
                [false, true, false],
            ],
        );
        let result = apply_survivor_floor(&frames(&decisions), usize::MAX).unwrap();

        assert_eq!(result.boolean_survivors(), 3);
        assert_eq!(result.survivor_indices(), &[0, 2, 3]);
    }

    #[test]
    fn strict_survivors_are_never_removed_when_floor_is_smaller() {
        let decisions = decisions(
            0b1111,
            &[
                [true, true, true],
                [true, true, true],
                [true, true, true],
                [false, false, false],
            ],
        );
        let result = apply_survivor_floor(&frames(&decisions), 1).unwrap();

        assert_eq!(result.strict_survivors(), 3);
        assert_eq!(result.survivor_indices(), &[0, 1, 2]);
        assert_eq!(result.target_floor(), 1);
    }

    #[test]
    fn final_output_order_is_input_order_after_ranked_rescue() {
        let decisions = decisions(
            0b1111,
            &[
                [false, false, false],
                [false, true, false],
                [false, true, false],
                [false, false, false],
            ],
        );
        let custom = [
            SurvivorFloorCandidate::new(3, &decisions[3]),
            SurvivorFloorCandidate::new(2, &decisions[2]),
            SurvivorFloorCandidate::new(1, &decisions[1]),
            SurvivorFloorCandidate::new(0, &decisions[0]),
        ];
        let result = apply_survivor_floor(&custom, 2).unwrap();

        assert_eq!(result.survivor_indices(), &[2, 1]);
        assert_eq!(result.rescued_indices(), &[2, 1]);
    }

    #[test]
    fn duplicate_candidate_ids_fail_closed() {
        let decisions = decisions(0b1, &[[true, true, true]]);
        let duplicate = [
            SurvivorFloorCandidate::new(7, &decisions[0]),
            SurvivorFloorCandidate::new(7, &decisions[0]),
        ];
        assert_eq!(
            apply_survivor_floor(&duplicate, 1),
            Err(SurvivorFloorError::DuplicateCandidateIndex { candidate_index: 7 })
        );
    }

    #[test]
    fn missing_boolean_verdict_fails_closed() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                parity_or_binary_linear: true,
                ..AttentionAlgebraNeeds::default()
            },
            None,
        )
        .unwrap();
        let input = F2Vector::from_bools(&[true]).unwrap();
        let predicate = F2AffinePredicate::new(input.clone(), false);
        let decision = qualify_candidate(
            &route,
            CandidateQualificationInputs {
                f2: Some(F2CandidateEvaluation {
                    predicate: &predicate,
                    input: &input,
                }),
                ..CandidateQualificationInputs::default()
            },
            RecompositionPolicy::AllSelectedMustQualify,
        )
        .unwrap();
        let candidate = [SurvivorFloorCandidate::new(0, &decision)];

        assert_eq!(
            apply_survivor_floor(&candidate, 1),
            Err(SurvivorFloorError::MissingBooleanDecision { candidate_index: 0 })
        );
    }

    #[test]
    fn max_plus_readiness_is_not_reinterpreted_as_rescue_evidence() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                eligibility_logic: true,
                precedence_or_critical_path: true,
                ..AttentionAlgebraNeeds::default()
            },
            Some(
                M13bBooleanRoutingEvidence::new(
                    1,
                    1,
                    vec![1],
                    1,
                    1,
                    vec![1],
                    vec![1],
                    None,
                    None,
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let schedule = MaxPlusSchedule::new(1, vec![]).unwrap();
        let initial = [MaxPlusValue::Finite(0)];
        let decision = qualify_candidate(
            &route,
            CandidateQualificationInputs {
                boolean_block: Some(0),
                max_plus: Some(MaxPlusCandidateEvaluation {
                    schedule: &schedule,
                    initial: &initial,
                    node: 0,
                    not_after: 0,
                }),
                ..CandidateQualificationInputs::default()
            },
            RecompositionPolicy::AllSelectedMustQualify,
        )
        .unwrap();
        let candidate = [SurvivorFloorCandidate::new(0, &decision)];

        assert_eq!(
            apply_survivor_floor(&candidate, 1),
            Err(SurvivorFloorError::MaxPlusDecisionNotSupported { candidate_index: 0 })
        );
    }
}
