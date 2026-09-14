use core::fmt;
use std::collections::BTreeSet;

use crate::cooperation::{AlgebraDomain, AlgebraicRoute};
use crate::qualification::{
    qualify_candidate, CandidateDisposition, CandidateQualificationInputs, QualificationError,
    RecompositionPolicy,
};

#[derive(Debug, Clone, Copy)]
pub struct CandidateFrame<'a> {
    candidate_index: usize,
    inputs: CandidateQualificationInputs<'a>,
}

impl<'a> CandidateFrame<'a> {
    #[must_use]
    pub const fn new(candidate_index: usize, inputs: CandidateQualificationInputs<'a>) -> Self {
        Self {
            candidate_index,
            inputs,
        }
    }

    #[must_use]
    pub const fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    #[must_use]
    pub const fn inputs(&self) -> CandidateQualificationInputs<'a> {
        self.inputs
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateRejection {
    candidate_index: usize,
    domains: Vec<AlgebraDomain>,
}

impl CandidateRejection {
    #[must_use]
    pub const fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    #[must_use]
    pub fn domains(&self) -> &[AlgebraDomain] {
        &self.domains
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DomainRejectionCounts {
    boolean: usize,
    f2: usize,
    zhegalkin: usize,
    max_plus: usize,
}

impl DomainRejectionCounts {
    #[must_use]
    pub const fn boolean(&self) -> usize {
        self.boolean
    }

    #[must_use]
    pub const fn f2(&self) -> usize {
        self.f2
    }

    #[must_use]
    pub const fn zhegalkin(&self) -> usize {
        self.zhegalkin
    }

    #[must_use]
    pub const fn max_plus(&self) -> usize {
        self.max_plus
    }

    #[must_use]
    pub const fn for_domain(&self, domain: AlgebraDomain) -> usize {
        match domain {
            AlgebraDomain::Boolean => self.boolean,
            AlgebraDomain::F2 => self.f2,
            AlgebraDomain::Zhegalkin => self.zhegalkin,
            AlgebraDomain::MaxPlus => self.max_plus,
        }
    }

    fn record(&mut self, domain: AlgebraDomain) {
        match domain {
            AlgebraDomain::Boolean => self.boolean += 1,
            AlgebraDomain::F2 => self.f2 += 1,
            AlgebraDomain::Zhegalkin => self.zhegalkin += 1,
            AlgebraDomain::MaxPlus => self.max_plus += 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurvivorSet {
    evaluated: usize,
    survivor_indices: Vec<usize>,
    rejections: Vec<CandidateRejection>,
    rejection_counts: DomainRejectionCounts,
}

impl SurvivorSet {
    #[must_use]
    pub const fn evaluated(&self) -> usize {
        self.evaluated
    }

    #[must_use]
    pub fn survivor_indices(&self) -> &[usize] {
        &self.survivor_indices
    }

    #[must_use]
    pub const fn survivor_count(&self) -> usize {
        self.survivor_indices.len()
    }

    #[must_use]
    pub const fn rejected_count(&self) -> usize {
        self.rejections.len()
    }

    #[must_use]
    pub fn rejections(&self) -> &[CandidateRejection] {
        &self.rejections
    }

    #[must_use]
    pub const fn rejection_counts(&self) -> DomainRejectionCounts {
        self.rejection_counts
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurvivorSetError {
    DuplicateCandidateIndex { candidate_index: usize },
    CandidateQualification {
        candidate_index: usize,
        source: QualificationError,
    },
}

impl fmt::Display for SurvivorSetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateCandidateIndex { candidate_index } => write!(
                formatter,
                "candidate index {candidate_index} occurs more than once in a survivor-set batch"
            ),
            Self::CandidateQualification {
                candidate_index,
                source,
            } => write!(
                formatter,
                "candidate {candidate_index} qualification failed: {source}"
            ),
        }
    }
}

impl std::error::Error for SurvivorSetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DuplicateCandidateIndex { .. } => None,
            Self::CandidateQualification { source, .. } => Some(source),
        }
    }
}

pub fn qualify_survivor_set(
    route: &AlgebraicRoute,
    candidates: &[CandidateFrame<'_>],
    policy: RecompositionPolicy,
) -> Result<SurvivorSet, SurvivorSetError> {
    let mut seen = BTreeSet::new();
    let mut survivor_indices = Vec::with_capacity(candidates.len());
    let mut rejections = Vec::new();
    let mut rejection_counts = DomainRejectionCounts::default();

    for candidate in candidates {
        if !seen.insert(candidate.candidate_index) {
            return Err(SurvivorSetError::DuplicateCandidateIndex {
                candidate_index: candidate.candidate_index,
            });
        }

        let decision = qualify_candidate(route, candidate.inputs, policy).map_err(|source| {
            SurvivorSetError::CandidateQualification {
                candidate_index: candidate.candidate_index,
                source,
            }
        })?;

        match decision.disposition() {
            CandidateDisposition::Admit => survivor_indices.push(candidate.candidate_index),
            CandidateDisposition::Reject => {
                let domains = rejected_domains(&decision);
                debug_assert!(!domains.is_empty());
                for domain in &domains {
                    rejection_counts.record(*domain);
                }
                rejections.push(CandidateRejection {
                    candidate_index: candidate.candidate_index,
                    domains,
                });
            }
        }
    }

    Ok(SurvivorSet {
        evaluated: candidates.len(),
        survivor_indices,
        rejections,
        rejection_counts,
    })
}

fn rejected_domains(
    decision: &crate::qualification::CandidateQualificationDecision,
) -> Vec<AlgebraDomain> {
    let mut domains = Vec::with_capacity(4);
    if decision.boolean().is_some_and(|value| !value.qualified()) {
        domains.push(AlgebraDomain::Boolean);
    }
    if decision.f2().is_some_and(|value| !value.qualified()) {
        domains.push(AlgebraDomain::F2);
    }
    if decision
        .zhegalkin()
        .is_some_and(|value| !value.qualified())
    {
        domains.push(AlgebraDomain::Zhegalkin);
    }
    if decision
        .max_plus()
        .is_some_and(|value| !value.qualified())
    {
        domains.push(AlgebraDomain::MaxPlus);
    }
    domains
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cooperation::{
        route_attention_needs, AttentionAlgebraNeeds, M13bBooleanRoutingEvidence,
    };
    use crate::f2::{F2AffinePredicate, F2Vector};
    use crate::max_plus::{MaxPlusEdge, MaxPlusSchedule, MaxPlusValue};
    use crate::qualification::{
        F2CandidateEvaluation, MaxPlusCandidateEvaluation, ZhegalkinCandidateEvaluation,
    };
    use crate::zhegalkin::ZhegalkinPolynomial;

    fn boolean_evidence() -> M13bBooleanRoutingEvidence {
        M13bBooleanRoutingEvidence::new(
            1,
            4,
            vec![0b1101],
            1,
            4,
            vec![0b1011],
            vec![0b1001],
            Some(1),
            Some(7),
        )
        .unwrap()
    }

    fn route() -> AlgebraicRoute {
        route_attention_needs(
            AttentionAlgebraNeeds {
                eligibility_logic: true,
                parity_or_binary_linear: true,
                nonlinear_boolean_interaction: true,
                precedence_or_critical_path: true,
            },
            Some(boolean_evidence()),
        )
        .unwrap()
    }

    #[test]
    fn batch_produces_stable_survivors_and_domain_attribution() {
        let route = route();
        let f2_predicate =
            F2AffinePredicate::new(F2Vector::from_bools(&[true, false]).unwrap(), false);
        let f2_true = F2Vector::from_bools(&[true, false]).unwrap();
        let f2_false = F2Vector::from_bools(&[false, false]).unwrap();
        let zhegalkin = ZhegalkinPolynomial::from_variable_sets(2, vec![vec![0, 1]]).unwrap();
        let z_true = F2Vector::from_bools(&[true, true]).unwrap();
        let z_false = F2Vector::from_bools(&[true, false]).unwrap();
        let schedule = MaxPlusSchedule::new(2, vec![MaxPlusEdge::new(0, 1, 3)]).unwrap();
        let initial = [MaxPlusValue::Finite(0), MaxPlusValue::ZERO];

        let candidates = [
            CandidateFrame::new(
                10,
                CandidateQualificationInputs {
                    boolean_block: Some(0),
                    f2: Some(F2CandidateEvaluation {
                        predicate: &f2_predicate,
                        input: &f2_true,
                    }),
                    zhegalkin: Some(ZhegalkinCandidateEvaluation {
                        polynomial: &zhegalkin,
                        input: &z_true,
                    }),
                    max_plus: Some(MaxPlusCandidateEvaluation {
                        schedule: &schedule,
                        initial: &initial,
                        node: 1,
                        not_after: 3,
                    }),
                },
            ),
            CandidateFrame::new(
                20,
                CandidateQualificationInputs {
                    boolean_block: Some(1),
                    f2: Some(F2CandidateEvaluation {
                        predicate: &f2_predicate,
                        input: &f2_false,
                    }),
                    zhegalkin: Some(ZhegalkinCandidateEvaluation {
                        polynomial: &zhegalkin,
                        input: &z_false,
                    }),
                    max_plus: Some(MaxPlusCandidateEvaluation {
                        schedule: &schedule,
                        initial: &initial,
                        node: 1,
                        not_after: 2,
                    }),
                },
            ),
        ];

        let result = qualify_survivor_set(
            &route,
            &candidates,
            RecompositionPolicy::AllSelectedMustQualify,
        )
        .unwrap();

        assert_eq!(result.evaluated(), 2);
        assert_eq!(result.survivor_indices(), &[10]);
        assert_eq!(result.survivor_count(), 1);
        assert_eq!(result.rejected_count(), 1);
        assert_eq!(result.rejections()[0].candidate_index(), 20);
        assert_eq!(
            result.rejections()[0].domains(),
            &[
                AlgebraDomain::Boolean,
                AlgebraDomain::F2,
                AlgebraDomain::Zhegalkin,
                AlgebraDomain::MaxPlus,
            ]
        );
        let counts = result.rejection_counts();
        for domain in [
            AlgebraDomain::Boolean,
            AlgebraDomain::F2,
            AlgebraDomain::Zhegalkin,
            AlgebraDomain::MaxPlus,
        ] {
            assert_eq!(counts.for_domain(domain), 1);
        }
    }

    #[test]
    fn survivor_order_follows_input_order_without_sorting_candidate_ids() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                parity_or_binary_linear: true,
                ..AttentionAlgebraNeeds::default()
            },
            None,
        )
        .unwrap();
        let predicate = F2AffinePredicate::new(F2Vector::from_bools(&[true]).unwrap(), false);
        let input = F2Vector::from_bools(&[true]).unwrap();
        let evaluation = F2CandidateEvaluation {
            predicate: &predicate,
            input: &input,
        };
        let candidates = [
            CandidateFrame::new(
                9,
                CandidateQualificationInputs {
                    f2: Some(evaluation),
                    ..CandidateQualificationInputs::default()
                },
            ),
            CandidateFrame::new(
                2,
                CandidateQualificationInputs {
                    f2: Some(evaluation),
                    ..CandidateQualificationInputs::default()
                },
            ),
        ];

        let result = qualify_survivor_set(
            &route,
            &candidates,
            RecompositionPolicy::AllSelectedMustQualify,
        )
        .unwrap();
        assert_eq!(result.survivor_indices(), &[9, 2]);
    }

    #[test]
    fn duplicate_candidate_ids_fail_closed() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                parity_or_binary_linear: true,
                ..AttentionAlgebraNeeds::default()
            },
            None,
        )
        .unwrap();
        let predicate = F2AffinePredicate::new(F2Vector::from_bools(&[true]).unwrap(), false);
        let input = F2Vector::from_bools(&[true]).unwrap();
        let evaluation = F2CandidateEvaluation {
            predicate: &predicate,
            input: &input,
        };
        let frame = CandidateFrame::new(
            7,
            CandidateQualificationInputs {
                f2: Some(evaluation),
                ..CandidateQualificationInputs::default()
            },
        );

        assert_eq!(
            qualify_survivor_set(
                &route,
                &[frame, frame],
                RecompositionPolicy::AllSelectedMustQualify,
            ),
            Err(SurvivorSetError::DuplicateCandidateIndex { candidate_index: 7 })
        );
    }

    #[test]
    fn candidate_errors_are_contextualized_with_candidate_id() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                parity_or_binary_linear: true,
                ..AttentionAlgebraNeeds::default()
            },
            None,
        )
        .unwrap();
        let candidates = [CandidateFrame::new(42, CandidateQualificationInputs::default())];

        assert_eq!(
            qualify_survivor_set(
                &route,
                &candidates,
                RecompositionPolicy::AllSelectedMustQualify,
            ),
            Err(SurvivorSetError::CandidateQualification {
                candidate_index: 42,
                source: QualificationError::MissingDomainEvaluation {
                    domain: AlgebraDomain::F2,
                },
            })
        );
    }

    #[test]
    fn empty_batch_is_an_exact_empty_survivor_set() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                parity_or_binary_linear: true,
                ..AttentionAlgebraNeeds::default()
            },
            None,
        )
        .unwrap();

        let result = qualify_survivor_set(
            &route,
            &[],
            RecompositionPolicy::AllSelectedMustQualify,
        )
        .unwrap();
        assert_eq!(result.evaluated(), 0);
        assert!(result.survivor_indices().is_empty());
        assert!(result.rejections().is_empty());
    }
}
