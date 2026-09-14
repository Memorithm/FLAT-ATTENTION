use core::fmt;

use crate::cooperation::{AlgebraDomain, AlgebraicRoute, ConversionQuality};
use crate::f2::{F2AffinePredicate, F2Error, F2Vector};
use crate::max_plus::{MaxPlusError, MaxPlusSchedule, MaxPlusValue};
use crate::zhegalkin::{ZhegalkinError, ZhegalkinPolynomial};

/// Explicit system-level policy used to recompose independent algebraic verdicts.
///
/// This is not an algebraic identity. It is a research policy layered above the
/// Boolean, F2, Zhegalkin and Max-Plus engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecompositionPolicy {
    AllSelectedMustQualify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateDisposition {
    Admit,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanCandidateDecision {
    block_index: usize,
    qualified: bool,
}

impl BooleanCandidateDecision {
    #[must_use]
    pub const fn block_index(&self) -> usize {
        self.block_index
    }

    #[must_use]
    pub const fn qualified(&self) -> bool {
        self.qualified
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F2CandidateDecision {
    qualified: bool,
}

impl F2CandidateDecision {
    #[must_use]
    pub const fn qualified(&self) -> bool {
        self.qualified
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZhegalkinCandidateDecision {
    qualified: bool,
}

impl ZhegalkinCandidateDecision {
    #[must_use]
    pub const fn qualified(&self) -> bool {
        self.qualified
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxPlusCandidateDecision {
    node: usize,
    feasible_time: MaxPlusValue,
    not_after: i64,
    qualified: bool,
}

impl MaxPlusCandidateDecision {
    #[must_use]
    pub const fn node(&self) -> usize {
        self.node
    }

    #[must_use]
    pub const fn feasible_time(&self) -> MaxPlusValue {
        self.feasible_time
    }

    #[must_use]
    pub const fn not_after(&self) -> i64 {
        self.not_after
    }

    #[must_use]
    pub const fn qualified(&self) -> bool {
        self.qualified
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateQualificationDecision {
    policy: RecompositionPolicy,
    disposition: CandidateDisposition,
    boolean: Option<BooleanCandidateDecision>,
    f2: Option<F2CandidateDecision>,
    zhegalkin: Option<ZhegalkinCandidateDecision>,
    max_plus: Option<MaxPlusCandidateDecision>,
}

impl CandidateQualificationDecision {
    #[must_use]
    pub const fn policy(&self) -> RecompositionPolicy {
        self.policy
    }

    #[must_use]
    pub const fn recomposition_quality(&self) -> ConversionQuality {
        ConversionQuality::PolicyDefined
    }

    #[must_use]
    pub const fn disposition(&self) -> CandidateDisposition {
        self.disposition
    }

    #[must_use]
    pub const fn boolean(&self) -> Option<BooleanCandidateDecision> {
        self.boolean
    }

    #[must_use]
    pub const fn f2(&self) -> Option<F2CandidateDecision> {
        self.f2
    }

    #[must_use]
    pub const fn zhegalkin(&self) -> Option<ZhegalkinCandidateDecision> {
        self.zhegalkin
    }

    #[must_use]
    pub const fn max_plus(&self) -> Option<MaxPlusCandidateDecision> {
        self.max_plus
    }
}

#[derive(Debug, Clone, Copy)]
pub struct F2CandidateEvaluation<'a> {
    pub predicate: &'a F2AffinePredicate,
    pub input: &'a F2Vector,
}

#[derive(Debug, Clone, Copy)]
pub struct ZhegalkinCandidateEvaluation<'a> {
    pub polynomial: &'a ZhegalkinPolynomial,
    pub input: &'a F2Vector,
}

#[derive(Debug, Clone, Copy)]
pub struct MaxPlusCandidateEvaluation<'a> {
    pub schedule: &'a MaxPlusSchedule,
    pub initial: &'a [MaxPlusValue],
    pub node: usize,
    pub not_after: i64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CandidateQualificationInputs<'a> {
    pub boolean_block: Option<usize>,
    pub f2: Option<F2CandidateEvaluation<'a>>,
    pub zhegalkin: Option<ZhegalkinCandidateEvaluation<'a>>,
    pub max_plus: Option<MaxPlusCandidateEvaluation<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum QualificationError {
    MissingDomainEvaluation { domain: AlgebraDomain },
    UnexpectedDomainEvaluation { domain: AlgebraDomain },
    BooleanBlockOutOfBounds { block: usize, blocks: usize },
    MaxPlusNodeOutOfBounds { node: usize, node_count: usize },
    F2(F2Error),
    Zhegalkin(ZhegalkinError),
    MaxPlus(MaxPlusError),
}

impl fmt::Display for QualificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDomainEvaluation { domain } => {
                write!(formatter, "missing candidate evaluation for {domain:?}")
            }
            Self::UnexpectedDomainEvaluation { domain } => {
                write!(formatter, "unexpected candidate evaluation for {domain:?}")
            }
            Self::BooleanBlockOutOfBounds { block, blocks } => write!(
                formatter,
                "Boolean candidate block {block} is outside 0..{blocks}"
            ),
            Self::MaxPlusNodeOutOfBounds { node, node_count } => write!(
                formatter,
                "Max-Plus candidate node {node} is outside 0..{node_count}"
            ),
            Self::F2(error) => write!(formatter, "F2 candidate evaluation failed: {error}"),
            Self::Zhegalkin(error) => {
                write!(formatter, "Zhegalkin candidate evaluation failed: {error}")
            }
            Self::MaxPlus(error) => {
                write!(formatter, "Max-Plus candidate evaluation failed: {error}")
            }
        }
    }
}

impl std::error::Error for QualificationError {}

impl From<F2Error> for QualificationError {
    fn from(error: F2Error) -> Self {
        Self::F2(error)
    }
}

impl From<ZhegalkinError> for QualificationError {
    fn from(error: ZhegalkinError) -> Self {
        Self::Zhegalkin(error)
    }
}

impl From<MaxPlusError> for QualificationError {
    fn from(error: MaxPlusError) -> Self {
        Self::MaxPlus(error)
    }
}

pub fn qualify_candidate(
    route: &AlgebraicRoute,
    inputs: CandidateQualificationInputs<'_>,
    policy: RecompositionPolicy,
) -> Result<CandidateQualificationDecision, QualificationError> {
    require_evaluation(
        route,
        AlgebraDomain::Boolean,
        inputs.boolean_block.is_some(),
    )?;
    require_evaluation(route, AlgebraDomain::F2, inputs.f2.is_some())?;
    require_evaluation(
        route,
        AlgebraDomain::Zhegalkin,
        inputs.zhegalkin.is_some(),
    )?;
    require_evaluation(route, AlgebraDomain::MaxPlus, inputs.max_plus.is_some())?;

    let boolean = if let Some(block_index) = inputs.boolean_block {
        let evidence = route
            .boolean_evidence()
            .expect("validated Boolean routes always carry M13B evidence");
        if block_index >= evidence.blocks() {
            return Err(QualificationError::BooleanBlockOutOfBounds {
                block: block_index,
                blocks: evidence.blocks(),
            });
        }
        let word = evidence.mask_words()[block_index / u64::BITS as usize];
        let bit = 1u64 << (block_index % u64::BITS as usize);
        Some(BooleanCandidateDecision {
            block_index,
            qualified: word & bit != 0,
        })
    } else {
        None
    };

    let f2 = inputs
        .f2
        .map(|evaluation| {
            Ok(F2CandidateDecision {
                qualified: evaluation.predicate.evaluate(evaluation.input)?,
            })
        })
        .transpose()?;

    let zhegalkin = inputs
        .zhegalkin
        .map(|evaluation| {
            Ok(ZhegalkinCandidateDecision {
                qualified: evaluation.polynomial.evaluate(evaluation.input)?,
            })
        })
        .transpose()?;

    let max_plus = inputs
        .max_plus
        .map(|evaluation| evaluate_max_plus(evaluation))
        .transpose()?;

    let all_qualified = boolean.map_or(true, |decision| decision.qualified())
        && f2.map_or(true, |decision| decision.qualified())
        && zhegalkin.map_or(true, |decision| decision.qualified())
        && max_plus.map_or(true, |decision| decision.qualified());

    let disposition = match policy {
        RecompositionPolicy::AllSelectedMustQualify if all_qualified => CandidateDisposition::Admit,
        RecompositionPolicy::AllSelectedMustQualify => CandidateDisposition::Reject,
    };

    Ok(CandidateQualificationDecision {
        policy,
        disposition,
        boolean,
        f2,
        zhegalkin,
        max_plus,
    })
}

fn require_evaluation(
    route: &AlgebraicRoute,
    domain: AlgebraDomain,
    supplied: bool,
) -> Result<(), QualificationError> {
    match (route.contains(domain), supplied) {
        (true, false) => Err(QualificationError::MissingDomainEvaluation { domain }),
        (false, true) => Err(QualificationError::UnexpectedDomainEvaluation { domain }),
        _ => Ok(()),
    }
}

fn evaluate_max_plus(
    evaluation: MaxPlusCandidateEvaluation<'_>,
) -> Result<MaxPlusCandidateDecision, QualificationError> {
    if evaluation.node >= evaluation.schedule.node_count() {
        return Err(QualificationError::MaxPlusNodeOutOfBounds {
            node: evaluation.node,
            node_count: evaluation.schedule.node_count(),
        });
    }

    let times = evaluation
        .schedule
        .earliest_feasible_times(evaluation.initial)?;
    let feasible_time = times[evaluation.node];
    let qualified = match feasible_time {
        MaxPlusValue::NegInfinity => false,
        MaxPlusValue::Finite(value) => value <= evaluation.not_after,
    };

    Ok(MaxPlusCandidateDecision {
        node: evaluation.node,
        feasible_time,
        not_after: evaluation.not_after,
        qualified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cooperation::{
        route_attention_needs, AttentionAlgebraNeeds, M13bBooleanRoutingEvidence,
    };
    use crate::max_plus::MaxPlusEdge;

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

    fn all_domain_route() -> AlgebraicRoute {
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
    fn all_four_domains_can_cooperate_without_losing_individual_verdicts() {
        let route = all_domain_route();
        let f2_input = F2Vector::from_bools(&[true, false]).unwrap();
        let f2_predicate = F2AffinePredicate::new(
            F2Vector::from_bools(&[true, false]).unwrap(),
            false,
        );
        let zhegalkin_input = F2Vector::from_bools(&[true, true]).unwrap();
        let zhegalkin =
            ZhegalkinPolynomial::from_variable_sets(2, vec![vec![0, 1]]).unwrap();
        let schedule = MaxPlusSchedule::new(2, vec![MaxPlusEdge::new(0, 1, 3)]).unwrap();
        let initial = [MaxPlusValue::Finite(0), MaxPlusValue::ZERO];

        let decision = qualify_candidate(
            &route,
            CandidateQualificationInputs {
                boolean_block: Some(0),
                f2: Some(F2CandidateEvaluation {
                    predicate: &f2_predicate,
                    input: &f2_input,
                }),
                zhegalkin: Some(ZhegalkinCandidateEvaluation {
                    polynomial: &zhegalkin,
                    input: &zhegalkin_input,
                }),
                max_plus: Some(MaxPlusCandidateEvaluation {
                    schedule: &schedule,
                    initial: &initial,
                    node: 1,
                    not_after: 3,
                }),
            },
            RecompositionPolicy::AllSelectedMustQualify,
        )
        .unwrap();

        assert_eq!(decision.disposition(), CandidateDisposition::Admit);
        assert_eq!(decision.recomposition_quality(), ConversionQuality::PolicyDefined);
        assert!(decision.boolean().unwrap().qualified());
        assert!(decision.f2().unwrap().qualified());
        assert!(decision.zhegalkin().unwrap().qualified());
        assert!(decision.max_plus().unwrap().qualified());
        assert_eq!(
            decision.max_plus().unwrap().feasible_time(),
            MaxPlusValue::Finite(3)
        );
    }

    #[test]
    fn one_domain_rejection_rejects_without_erasing_other_evidence() {
        let route = all_domain_route();
        let f2_input = F2Vector::from_bools(&[false, false]).unwrap();
        let f2_predicate = F2AffinePredicate::new(
            F2Vector::from_bools(&[true, false]).unwrap(),
            false,
        );
        let zhegalkin_input = F2Vector::from_bools(&[true, true]).unwrap();
        let zhegalkin =
            ZhegalkinPolynomial::from_variable_sets(2, vec![vec![0, 1]]).unwrap();
        let schedule = MaxPlusSchedule::new(2, vec![MaxPlusEdge::new(0, 1, 3)]).unwrap();
        let initial = [MaxPlusValue::Finite(0), MaxPlusValue::ZERO];

        let decision = qualify_candidate(
            &route,
            CandidateQualificationInputs {
                boolean_block: Some(0),
                f2: Some(F2CandidateEvaluation {
                    predicate: &f2_predicate,
                    input: &f2_input,
                }),
                zhegalkin: Some(ZhegalkinCandidateEvaluation {
                    polynomial: &zhegalkin,
                    input: &zhegalkin_input,
                }),
                max_plus: Some(MaxPlusCandidateEvaluation {
                    schedule: &schedule,
                    initial: &initial,
                    node: 1,
                    not_after: 3,
                }),
            },
            RecompositionPolicy::AllSelectedMustQualify,
        )
        .unwrap();

        assert_eq!(decision.disposition(), CandidateDisposition::Reject);
        assert!(decision.boolean().unwrap().qualified());
        assert!(!decision.f2().unwrap().qualified());
        assert!(decision.zhegalkin().unwrap().qualified());
        assert!(decision.max_plus().unwrap().qualified());
    }

    #[test]
    fn route_and_supplied_evaluations_must_match_exactly() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                parity_or_binary_linear: true,
                ..AttentionAlgebraNeeds::default()
            },
            None,
        )
        .unwrap();

        assert_eq!(
            qualify_candidate(
                &route,
                CandidateQualificationInputs::default(),
                RecompositionPolicy::AllSelectedMustQualify,
            ),
            Err(QualificationError::MissingDomainEvaluation {
                domain: AlgebraDomain::F2,
            })
        );
    }

    #[test]
    fn boolean_block_bounds_fail_closed() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                eligibility_logic: true,
                ..AttentionAlgebraNeeds::default()
            },
            Some(boolean_evidence()),
        )
        .unwrap();

        assert_eq!(
            qualify_candidate(
                &route,
                CandidateQualificationInputs {
                    boolean_block: Some(4),
                    ..CandidateQualificationInputs::default()
                },
                RecompositionPolicy::AllSelectedMustQualify,
            ),
            Err(QualificationError::BooleanBlockOutOfBounds {
                block: 4,
                blocks: 4,
            })
        );
    }

    #[test]
    fn unreachable_or_late_max_plus_candidate_is_a_negative_verdict() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                precedence_or_critical_path: true,
                ..AttentionAlgebraNeeds::default()
            },
            None,
        )
        .unwrap();
        let schedule = MaxPlusSchedule::new(2, vec![MaxPlusEdge::new(0, 1, 5)]).unwrap();
        let unreachable = [MaxPlusValue::ZERO, MaxPlusValue::ZERO];
        let late = [MaxPlusValue::Finite(0), MaxPlusValue::ZERO];

        for initial in [&unreachable[..], &late[..]] {
            let decision = qualify_candidate(
                &route,
                CandidateQualificationInputs {
                    max_plus: Some(MaxPlusCandidateEvaluation {
                        schedule: &schedule,
                        initial,
                        node: 1,
                        not_after: 4,
                    }),
                    ..CandidateQualificationInputs::default()
                },
                RecompositionPolicy::AllSelectedMustQualify,
            )
            .unwrap();
            assert_eq!(decision.disposition(), CandidateDisposition::Reject);
            assert!(!decision.max_plus().unwrap().qualified());
        }
    }

    #[test]
    fn max_plus_node_bounds_fail_closed() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                precedence_or_critical_path: true,
                ..AttentionAlgebraNeeds::default()
            },
            None,
        )
        .unwrap();
        let schedule = MaxPlusSchedule::new(1, vec![]).unwrap();
        let initial = [MaxPlusValue::Finite(0)];

        assert_eq!(
            qualify_candidate(
                &route,
                CandidateQualificationInputs {
                    max_plus: Some(MaxPlusCandidateEvaluation {
                        schedule: &schedule,
                        initial: &initial,
                        node: 1,
                        not_after: 0,
                    }),
                    ..CandidateQualificationInputs::default()
                },
                RecompositionPolicy::AllSelectedMustQualify,
            ),
            Err(QualificationError::MaxPlusNodeOutOfBounds {
                node: 1,
                node_count: 1,
            })
        );
    }
}
