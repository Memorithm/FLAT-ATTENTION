use core::fmt;
use std::collections::BTreeSet;

use crate::cooperation::{AlgebraDomain, AlgebraicRoute};
use crate::evidence::{
    ArmEvidence, EvidenceArm, EvidenceError, LatencyObservation, MatchedAttentionEvidence,
    MatchedWorkloadIdentity,
};
use crate::qualification::RecompositionPolicy;
use crate::survivor_set::SurvivorSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchedLatencySet {
    pub dense: LatencyObservation,
    pub boolean_only: LatencyObservation,
    pub multi_algebra: LatencyObservation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatchedExperimentError {
    RouteMissingBoolean,
    BooleanGeometryMismatch {
        expected: usize,
        actual: usize,
    },
    DuplicateReferenceRelevant {
        candidate_index: usize,
    },
    ReferenceRelevantOutOfBounds {
        candidate_index: usize,
        candidate_count: usize,
    },
    SurvivorCandidateOutOfBounds {
        candidate_index: usize,
        candidate_count: usize,
    },
    RejectedCandidateOutOfBounds {
        candidate_index: usize,
        candidate_count: usize,
    },
    CandidateGeometryIncomplete {
        expected: usize,
        actual: usize,
    },
    Evidence(EvidenceError),
}

impl fmt::Display for MatchedExperimentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MatchedExperimentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evidence(source) => Some(source),
            _ => None,
        }
    }
}

impl From<EvidenceError> for MatchedExperimentError {
    fn from(error: EvidenceError) -> Self {
        Self::Evidence(error)
    }
}

/// Derive a matched dense / Boolean-only / multi-algebra evidence record from
/// canonical host oracles.
///
/// `reference_relevant_indices` is the dense-reference relevance set for the
/// matched workload. Boolean-only counts are derived from the M13B mask carried
/// by `route`; multi-algebra counts are derived from `survivor_set`. No timing is
/// inferred from logical work reduction. Optional timing observations are merely
/// forwarded to the evidence schema, which enforces matched sample counts.
pub fn derive_matched_host_evidence(
    identity: MatchedWorkloadIdentity,
    reference_relevant_indices: &[usize],
    route: &AlgebraicRoute,
    policy: RecompositionPolicy,
    survivor_set: &SurvivorSet,
    latency: Option<MatchedLatencySet>,
) -> Result<MatchedAttentionEvidence, MatchedExperimentError> {
    if !route.contains(AlgebraDomain::Boolean) {
        return Err(MatchedExperimentError::RouteMissingBoolean);
    }

    let boolean = route
        .boolean_evidence()
        .ok_or(MatchedExperimentError::RouteMissingBoolean)?;
    let candidate_count = identity.candidate_count();
    if boolean.blocks() != candidate_count {
        return Err(MatchedExperimentError::BooleanGeometryMismatch {
            expected: candidate_count,
            actual: boolean.blocks(),
        });
    }

    let relevant = validate_reference_relevant(reference_relevant_indices, candidate_count)?;
    validate_survivor_geometry(survivor_set, candidate_count)?;

    let boolean_survivor_count = (0..candidate_count)
        .filter(|&candidate_index| boolean_admits(boolean.mask_words(), candidate_index))
        .count();
    let boolean_retained_relevant = relevant
        .iter()
        .copied()
        .filter(|&candidate_index| boolean_admits(boolean.mask_words(), candidate_index))
        .count();
    let multi_survivors = survivor_set
        .survivor_indices()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let multi_retained_relevant = relevant.intersection(&multi_survivors).count();

    let (dense_latency, boolean_latency, multi_latency) = match latency {
        Some(latency) => (
            Some(latency.dense),
            Some(latency.boolean_only),
            Some(latency.multi_algebra),
        ),
        None => (None, None, None),
    };

    let reference_relevant = relevant.len();
    let dense = ArmEvidence::new(
        EvidenceArm::DenseReference,
        candidate_count,
        candidate_count,
        candidate_count,
        reference_relevant,
        reference_relevant,
        dense_latency,
    )?;
    let boolean_only = ArmEvidence::new(
        EvidenceArm::BooleanOnlyControl,
        candidate_count,
        boolean_survivor_count,
        boolean_survivor_count,
        reference_relevant,
        boolean_retained_relevant,
        boolean_latency,
    )?;
    let multi_algebra = ArmEvidence::new(
        EvidenceArm::MultiAlgebraCandidate,
        candidate_count,
        survivor_set.survivor_count(),
        survivor_set.survivor_count(),
        reference_relevant,
        multi_retained_relevant,
        multi_latency,
    )?;

    MatchedAttentionEvidence::new(
        identity,
        dense,
        boolean_only,
        multi_algebra,
        route,
        policy,
        survivor_set,
    )
    .map_err(Into::into)
}

fn validate_reference_relevant(
    indices: &[usize],
    candidate_count: usize,
) -> Result<BTreeSet<usize>, MatchedExperimentError> {
    let mut relevant = BTreeSet::new();
    for &candidate_index in indices {
        if candidate_index >= candidate_count {
            return Err(MatchedExperimentError::ReferenceRelevantOutOfBounds {
                candidate_index,
                candidate_count,
            });
        }
        if !relevant.insert(candidate_index) {
            return Err(MatchedExperimentError::DuplicateReferenceRelevant { candidate_index });
        }
    }
    Ok(relevant)
}

fn validate_survivor_geometry(
    survivor_set: &SurvivorSet,
    candidate_count: usize,
) -> Result<(), MatchedExperimentError> {
    let mut seen = BTreeSet::new();
    for &candidate_index in survivor_set.survivor_indices() {
        if candidate_index >= candidate_count {
            return Err(MatchedExperimentError::SurvivorCandidateOutOfBounds {
                candidate_index,
                candidate_count,
            });
        }
        seen.insert(candidate_index);
    }
    for rejection in survivor_set.rejections() {
        let candidate_index = rejection.candidate_index();
        if candidate_index >= candidate_count {
            return Err(MatchedExperimentError::RejectedCandidateOutOfBounds {
                candidate_index,
                candidate_count,
            });
        }
        seen.insert(candidate_index);
    }
    if seen.len() != candidate_count {
        return Err(MatchedExperimentError::CandidateGeometryIncomplete {
            expected: candidate_count,
            actual: seen.len(),
        });
    }
    Ok(())
}

fn boolean_admits(mask_words: &[u64], candidate_index: usize) -> bool {
    let word = mask_words[candidate_index / u64::BITS as usize];
    let bit = 1u64 << (candidate_index % u64::BITS as usize);
    word & bit != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cooperation::{
        route_attention_needs, AttentionAlgebraNeeds, M13bBooleanRoutingEvidence,
    };
    use crate::f2::{F2AffinePredicate, F2Vector};
    use crate::qualification::{CandidateQualificationInputs, F2CandidateEvaluation};
    use crate::survivor_set::{qualify_survivor_set, CandidateFrame};

    fn route() -> AlgebraicRoute {
        let boolean = M13bBooleanRoutingEvidence::new(
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
        .unwrap();
        route_attention_needs(
            AttentionAlgebraNeeds {
                eligibility_logic: true,
                parity_or_binary_linear: true,
                ..AttentionAlgebraNeeds::default()
            },
            Some(boolean),
        )
        .unwrap()
    }

    fn survivor_set(route: &AlgebraicRoute, final_id: usize) -> SurvivorSet {
        let predicate = F2AffinePredicate::new(F2Vector::from_bools(&[true]).unwrap(), false);
        let pass = F2Vector::from_bools(&[true]).unwrap();
        let fail = F2Vector::from_bools(&[false]).unwrap();
        let pass_eval = F2CandidateEvaluation {
            predicate: &predicate,
            input: &pass,
        };
        let fail_eval = F2CandidateEvaluation {
            predicate: &predicate,
            input: &fail,
        };
        let candidates = [
            CandidateFrame::new(
                0,
                CandidateQualificationInputs {
                    boolean_block: Some(0),
                    f2: Some(pass_eval),
                    ..CandidateQualificationInputs::default()
                },
            ),
            CandidateFrame::new(
                1,
                CandidateQualificationInputs {
                    boolean_block: Some(1),
                    f2: Some(pass_eval),
                    ..CandidateQualificationInputs::default()
                },
            ),
            CandidateFrame::new(
                2,
                CandidateQualificationInputs {
                    boolean_block: Some(2),
                    f2: Some(fail_eval),
                    ..CandidateQualificationInputs::default()
                },
            ),
            CandidateFrame::new(
                final_id,
                CandidateQualificationInputs {
                    boolean_block: Some(3),
                    f2: Some(pass_eval),
                    ..CandidateQualificationInputs::default()
                },
            ),
        ];
        qualify_survivor_set(
            route,
            &candidates,
            RecompositionPolicy::AllSelectedMustQualify,
        )
        .unwrap()
    }

    fn identity(candidate_count: usize) -> MatchedWorkloadIdentity {
        MatchedWorkloadIdentity::new(
            "maa-matched-host",
            "canonical-mask",
            "sha256:canonical",
            candidate_count,
        )
        .unwrap()
    }

    #[test]
    fn derives_all_three_arms_from_dense_labels_and_canonical_oracles() {
        let route = route();
        let survivors = survivor_set(&route, 3);
        let evidence = derive_matched_host_evidence(
            identity(4),
            &[0, 3],
            &route,
            RecompositionPolicy::AllSelectedMustQualify,
            &survivors,
            None,
        )
        .unwrap();

        assert_eq!(evidence.dense().survivor_count(), 4);
        assert_eq!(evidence.boolean_only().survivor_count(), 3);
        assert_eq!(evidence.multi_algebra().survivor_count(), 2);
        assert_eq!(evidence.dense().retained_relevant(), 2);
        assert_eq!(evidence.boolean_only().retained_relevant(), 2);
        assert_eq!(evidence.multi_algebra().retained_relevant(), 2);
        assert_eq!(evidence.additional_score_work_avoided_vs_boolean(), 1);
        assert_eq!(evidence.rejection_counts().boolean(), 1);
        assert_eq!(evidence.rejection_counts().f2(), 1);
        assert!(!evidence.has_matched_latency());
    }

    #[test]
    fn duplicate_dense_reference_labels_fail_closed() {
        let route = route();
        let survivors = survivor_set(&route, 3);
        assert_eq!(
            derive_matched_host_evidence(
                identity(4),
                &[0, 0],
                &route,
                RecompositionPolicy::AllSelectedMustQualify,
                &survivors,
                None,
            ),
            Err(MatchedExperimentError::DuplicateReferenceRelevant { candidate_index: 0 })
        );
    }

    #[test]
    fn dense_reference_label_outside_geometry_fails_closed() {
        let route = route();
        let survivors = survivor_set(&route, 3);
        assert_eq!(
            derive_matched_host_evidence(
                identity(4),
                &[4],
                &route,
                RecompositionPolicy::AllSelectedMustQualify,
                &survivors,
                None,
            ),
            Err(MatchedExperimentError::ReferenceRelevantOutOfBounds {
                candidate_index: 4,
                candidate_count: 4,
            })
        );
    }

    #[test]
    fn survivor_ids_must_share_the_boolean_candidate_geometry() {
        let route = route();
        let survivors = survivor_set(&route, 4);
        assert_eq!(
            derive_matched_host_evidence(
                identity(4),
                &[0, 3],
                &route,
                RecompositionPolicy::AllSelectedMustQualify,
                &survivors,
                None,
            ),
            Err(MatchedExperimentError::SurvivorCandidateOutOfBounds {
                candidate_index: 4,
                candidate_count: 4,
            })
        );
    }

    #[test]
    fn boolean_mask_geometry_must_match_the_workload() {
        let route = route();
        let survivors = survivor_set(&route, 3);
        assert_eq!(
            derive_matched_host_evidence(
                identity(5),
                &[0, 3],
                &route,
                RecompositionPolicy::AllSelectedMustQualify,
                &survivors,
                None,
            ),
            Err(MatchedExperimentError::BooleanGeometryMismatch {
                expected: 5,
                actual: 4,
            })
        );
    }

    #[test]
    fn matched_latency_is_forwarded_without_being_inferred() {
        let route = route();
        let survivors = survivor_set(&route, 3);
        let latency = MatchedLatencySet {
            dense: LatencyObservation::new(10, 1_200, 100, 140).unwrap(),
            boolean_only: LatencyObservation::new(10, 1_100, 90, 130).unwrap(),
            multi_algebra: LatencyObservation::new(10, 1_000, 80, 120).unwrap(),
        };
        let evidence = derive_matched_host_evidence(
            identity(4),
            &[0, 3],
            &route,
            RecompositionPolicy::AllSelectedMustQualify,
            &survivors,
            Some(latency),
        )
        .unwrap();

        assert!(evidence.has_matched_latency());
        assert_eq!(evidence.multi_algebra().latency().unwrap().mean_ns(), 100);
    }
}
