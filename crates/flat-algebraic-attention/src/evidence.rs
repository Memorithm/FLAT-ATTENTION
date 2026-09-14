use core::fmt;

use crate::cooperation::{AlgebraDomain, AlgebraicRoute};
use crate::qualification::RecompositionPolicy;
use crate::survivor_set::{DomainRejectionCounts, SurvivorSet};

pub const MAA_HOST_EVIDENCE_SCHEMA_VERSION: u32 = 1;
const PPM: u128 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceArm {
    DenseReference,
    BooleanOnlyControl,
    MultiAlgebraCandidate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedWorkloadIdentity {
    suite: String,
    case: String,
    input_digest: String,
    candidate_count: usize,
}

impl MatchedWorkloadIdentity {
    pub fn new(
        suite: impl Into<String>,
        case: impl Into<String>,
        input_digest: impl Into<String>,
        candidate_count: usize,
    ) -> Result<Self, EvidenceError> {
        let suite = suite.into();
        let case = case.into();
        let input_digest = input_digest.into();
        require_text("suite", &suite)?;
        require_text("case", &case)?;
        require_text("input_digest", &input_digest)?;
        if candidate_count == 0 {
            return Err(EvidenceError::ZeroCandidateCount);
        }
        Ok(Self {
            suite,
            case,
            input_digest,
            candidate_count,
        })
    }

    #[must_use]
    pub fn suite(&self) -> &str {
        &self.suite
    }

    #[must_use]
    pub fn case(&self) -> &str {
        &self.case
    }

    #[must_use]
    pub fn input_digest(&self) -> &str {
        &self.input_digest
    }

    #[must_use]
    pub const fn candidate_count(&self) -> usize {
        self.candidate_count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyObservation {
    sample_count: u64,
    total_ns: u128,
    min_ns: u64,
    max_ns: u64,
}

impl LatencyObservation {
    pub fn new(
        sample_count: u64,
        total_ns: u128,
        min_ns: u64,
        max_ns: u64,
    ) -> Result<Self, EvidenceError> {
        if sample_count == 0 {
            return Err(EvidenceError::ZeroLatencySamples);
        }
        if min_ns > max_ns {
            return Err(EvidenceError::InvalidLatencyRange { min_ns, max_ns });
        }
        let samples = u128::from(sample_count);
        let lower = u128::from(min_ns) * samples;
        let upper = u128::from(max_ns) * samples;
        if !(lower..=upper).contains(&total_ns) {
            return Err(EvidenceError::InvalidLatencyTotal {
                sample_count,
                total_ns,
                min_ns,
                max_ns,
            });
        }
        Ok(Self {
            sample_count,
            total_ns,
            min_ns,
            max_ns,
        })
    }

    #[must_use]
    pub const fn sample_count(&self) -> u64 {
        self.sample_count
    }

    #[must_use]
    pub const fn total_ns(&self) -> u128 {
        self.total_ns
    }

    #[must_use]
    pub const fn min_ns(&self) -> u64 {
        self.min_ns
    }

    #[must_use]
    pub const fn max_ns(&self) -> u64 {
        self.max_ns
    }

    #[must_use]
    pub fn mean_ns(&self) -> u128 {
        self.total_ns / u128::from(self.sample_count)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmEvidence {
    arm: EvidenceArm,
    candidate_count: usize,
    survivor_count: usize,
    exact_score_evaluations: usize,
    reference_relevant: usize,
    retained_relevant: usize,
    latency: Option<LatencyObservation>,
}

impl ArmEvidence {
    pub fn new(
        arm: EvidenceArm,
        candidate_count: usize,
        survivor_count: usize,
        exact_score_evaluations: usize,
        reference_relevant: usize,
        retained_relevant: usize,
        latency: Option<LatencyObservation>,
    ) -> Result<Self, EvidenceError> {
        if candidate_count == 0 {
            return Err(EvidenceError::ZeroCandidateCount);
        }
        if survivor_count > candidate_count {
            return Err(EvidenceError::TooManySurvivors {
                arm,
                survivor_count,
                candidate_count,
            });
        }
        if exact_score_evaluations != survivor_count {
            return Err(EvidenceError::ScoreSurvivorMismatch {
                arm,
                exact_score_evaluations,
                survivor_count,
            });
        }
        if reference_relevant > candidate_count {
            return Err(EvidenceError::TooManyReferenceRelevant {
                arm,
                reference_relevant,
                candidate_count,
            });
        }
        if retained_relevant > reference_relevant {
            return Err(EvidenceError::TooManyRetainedRelevant {
                arm,
                retained_relevant,
                reference_relevant,
            });
        }
        if arm == EvidenceArm::DenseReference {
            if survivor_count != candidate_count {
                return Err(EvidenceError::DenseDidNotScoreAll {
                    candidate_count,
                    survivor_count,
                });
            }
            if retained_relevant != reference_relevant {
                return Err(EvidenceError::DenseLostReferenceRelevant {
                    reference_relevant,
                    retained_relevant,
                });
            }
        }
        Ok(Self {
            arm,
            candidate_count,
            survivor_count,
            exact_score_evaluations,
            reference_relevant,
            retained_relevant,
            latency,
        })
    }

    #[must_use]
    pub const fn arm(&self) -> EvidenceArm {
        self.arm
    }

    #[must_use]
    pub const fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    #[must_use]
    pub const fn survivor_count(&self) -> usize {
        self.survivor_count
    }

    #[must_use]
    pub const fn exact_score_evaluations(&self) -> usize {
        self.exact_score_evaluations
    }

    #[must_use]
    pub const fn reference_relevant(&self) -> usize {
        self.reference_relevant
    }

    #[must_use]
    pub const fn retained_relevant(&self) -> usize {
        self.retained_relevant
    }

    #[must_use]
    pub const fn latency(&self) -> Option<LatencyObservation> {
        self.latency
    }

    #[must_use]
    pub const fn score_work_avoided(&self) -> usize {
        self.candidate_count - self.exact_score_evaluations
    }

    #[must_use]
    pub fn score_reduction_ppm(&self) -> u32 {
        ratio_ppm(self.score_work_avoided(), self.candidate_count)
            .expect("candidate count is validated as non-zero")
    }

    #[must_use]
    pub fn relevant_coverage_ppm(&self) -> Option<u32> {
        ratio_ppm(self.retained_relevant, self.reference_relevant)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedAttentionEvidence {
    schema_version: u32,
    identity: MatchedWorkloadIdentity,
    dense: ArmEvidence,
    boolean_only: ArmEvidence,
    multi_algebra: ArmEvidence,
    route_domains: Vec<AlgebraDomain>,
    policy: RecompositionPolicy,
    rejection_counts: DomainRejectionCounts,
}

impl MatchedAttentionEvidence {
    pub fn new(
        identity: MatchedWorkloadIdentity,
        dense: ArmEvidence,
        boolean_only: ArmEvidence,
        multi_algebra: ArmEvidence,
        route: &AlgebraicRoute,
        policy: RecompositionPolicy,
        survivor_set: &SurvivorSet,
    ) -> Result<Self, EvidenceError> {
        require_arm(EvidenceArm::DenseReference, dense.arm())?;
        require_arm(EvidenceArm::BooleanOnlyControl, boolean_only.arm())?;
        require_arm(EvidenceArm::MultiAlgebraCandidate, multi_algebra.arm())?;

        for arm in [&dense, &boolean_only, &multi_algebra] {
            if arm.candidate_count() != identity.candidate_count() {
                return Err(EvidenceError::CandidateCountMismatch {
                    arm: arm.arm(),
                    expected: identity.candidate_count(),
                    actual: arm.candidate_count(),
                });
            }
        }

        if dense.reference_relevant() != boolean_only.reference_relevant()
            || dense.reference_relevant() != multi_algebra.reference_relevant()
        {
            return Err(EvidenceError::ReferenceRelevantMismatch {
                dense: dense.reference_relevant(),
                boolean_only: boolean_only.reference_relevant(),
                multi_algebra: multi_algebra.reference_relevant(),
            });
        }
        if !route.contains(AlgebraDomain::Boolean) {
            return Err(EvidenceError::RouteMissingBoolean);
        }
        if route.domains().len() < 2 {
            return Err(EvidenceError::RouteMissingAdditionalDomain);
        }
        if multi_algebra.survivor_count() > boolean_only.survivor_count() {
            return Err(EvidenceError::MultiResurrectedBooleanRejection {
                boolean_only: boolean_only.survivor_count(),
                multi_algebra: multi_algebra.survivor_count(),
            });
        }
        if multi_algebra.retained_relevant() > boolean_only.retained_relevant() {
            return Err(EvidenceError::MultiResurrectedRelevantCandidate {
                boolean_only: boolean_only.retained_relevant(),
                multi_algebra: multi_algebra.retained_relevant(),
            });
        }
        if survivor_set.evaluated() != identity.candidate_count() {
            return Err(EvidenceError::OracleEvaluatedMismatch {
                expected: identity.candidate_count(),
                actual: survivor_set.evaluated(),
            });
        }
        if survivor_set.survivor_count() != multi_algebra.survivor_count() {
            return Err(EvidenceError::OracleSurvivorMismatch {
                evidence: multi_algebra.survivor_count(),
                oracle: survivor_set.survivor_count(),
            });
        }

        validate_latency_triplet(&dense, &boolean_only, &multi_algebra)?;

        Ok(Self {
            schema_version: MAA_HOST_EVIDENCE_SCHEMA_VERSION,
            identity,
            dense,
            boolean_only,
            multi_algebra,
            route_domains: route.domains().to_vec(),
            policy,
            rejection_counts: survivor_set.rejection_counts(),
        })
    }

    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[must_use]
    pub const fn identity(&self) -> &MatchedWorkloadIdentity {
        &self.identity
    }

    #[must_use]
    pub const fn dense(&self) -> &ArmEvidence {
        &self.dense
    }

    #[must_use]
    pub const fn boolean_only(&self) -> &ArmEvidence {
        &self.boolean_only
    }

    #[must_use]
    pub const fn multi_algebra(&self) -> &ArmEvidence {
        &self.multi_algebra
    }

    #[must_use]
    pub fn route_domains(&self) -> &[AlgebraDomain] {
        &self.route_domains
    }

    #[must_use]
    pub const fn policy(&self) -> RecompositionPolicy {
        self.policy
    }

    #[must_use]
    pub const fn rejection_counts(&self) -> DomainRejectionCounts {
        self.rejection_counts
    }

    #[must_use]
    pub const fn additional_score_work_avoided_vs_boolean(&self) -> usize {
        self.boolean_only.exact_score_evaluations - self.multi_algebra.exact_score_evaluations
    }

    #[must_use]
    pub fn has_matched_latency(&self) -> bool {
        self.dense.latency().is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvidenceError {
    EmptyIdentityField {
        field: &'static str,
    },
    ZeroCandidateCount,
    ZeroLatencySamples,
    InvalidLatencyRange {
        min_ns: u64,
        max_ns: u64,
    },
    InvalidLatencyTotal {
        sample_count: u64,
        total_ns: u128,
        min_ns: u64,
        max_ns: u64,
    },
    TooManySurvivors {
        arm: EvidenceArm,
        survivor_count: usize,
        candidate_count: usize,
    },
    ScoreSurvivorMismatch {
        arm: EvidenceArm,
        exact_score_evaluations: usize,
        survivor_count: usize,
    },
    TooManyReferenceRelevant {
        arm: EvidenceArm,
        reference_relevant: usize,
        candidate_count: usize,
    },
    TooManyRetainedRelevant {
        arm: EvidenceArm,
        retained_relevant: usize,
        reference_relevant: usize,
    },
    DenseDidNotScoreAll {
        candidate_count: usize,
        survivor_count: usize,
    },
    DenseLostReferenceRelevant {
        reference_relevant: usize,
        retained_relevant: usize,
    },
    ArmKindMismatch {
        expected: EvidenceArm,
        actual: EvidenceArm,
    },
    CandidateCountMismatch {
        arm: EvidenceArm,
        expected: usize,
        actual: usize,
    },
    ReferenceRelevantMismatch {
        dense: usize,
        boolean_only: usize,
        multi_algebra: usize,
    },
    RouteMissingBoolean,
    RouteMissingAdditionalDomain,
    MultiResurrectedBooleanRejection {
        boolean_only: usize,
        multi_algebra: usize,
    },
    MultiResurrectedRelevantCandidate {
        boolean_only: usize,
        multi_algebra: usize,
    },
    OracleEvaluatedMismatch {
        expected: usize,
        actual: usize,
    },
    OracleSurvivorMismatch {
        evidence: usize,
        oracle: usize,
    },
    IncompleteLatencyTriplet,
    LatencySampleCountMismatch {
        dense: u64,
        boolean_only: u64,
        multi_algebra: u64,
    },
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for EvidenceError {}

fn require_text(field: &'static str, value: &str) -> Result<(), EvidenceError> {
    if value.trim().is_empty() {
        Err(EvidenceError::EmptyIdentityField { field })
    } else {
        Ok(())
    }
}

fn require_arm(expected: EvidenceArm, actual: EvidenceArm) -> Result<(), EvidenceError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EvidenceError::ArmKindMismatch { expected, actual })
    }
}

fn validate_latency_triplet(
    dense: &ArmEvidence,
    boolean_only: &ArmEvidence,
    multi_algebra: &ArmEvidence,
) -> Result<(), EvidenceError> {
    match (
        dense.latency(),
        boolean_only.latency(),
        multi_algebra.latency(),
    ) {
        (None, None, None) => Ok(()),
        (Some(dense), Some(boolean_only), Some(multi_algebra)) => {
            if dense.sample_count() != boolean_only.sample_count()
                || dense.sample_count() != multi_algebra.sample_count()
            {
                return Err(EvidenceError::LatencySampleCountMismatch {
                    dense: dense.sample_count(),
                    boolean_only: boolean_only.sample_count(),
                    multi_algebra: multi_algebra.sample_count(),
                });
            }
            Ok(())
        }
        _ => Err(EvidenceError::IncompleteLatencyTriplet),
    }
}

fn ratio_ppm(numerator: usize, denominator: usize) -> Option<u32> {
    if denominator == 0 {
        return None;
    }
    let scaled = (numerator as u128 * PPM) / denominator as u128;
    Some(scaled as u32)
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

    fn fixture() -> (AlgebraicRoute, SurvivorSet) {
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
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                eligibility_logic: true,
                parity_or_binary_linear: true,
                ..AttentionAlgebraNeeds::default()
            },
            Some(boolean),
        )
        .unwrap();
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
                3,
                CandidateQualificationInputs {
                    boolean_block: Some(3),
                    f2: Some(pass_eval),
                    ..CandidateQualificationInputs::default()
                },
            ),
        ];
        let survivors = qualify_survivor_set(
            &route,
            &candidates,
            RecompositionPolicy::AllSelectedMustQualify,
        )
        .unwrap();
        (route, survivors)
    }

    fn identity() -> MatchedWorkloadIdentity {
        MatchedWorkloadIdentity::new("maa-host", "four-candidates", "sha256:test", 4).unwrap()
    }

    fn latency(total_ns: u128) -> LatencyObservation {
        LatencyObservation::new(10, total_ns, 90, 200).unwrap()
    }

    fn dense(with_latency: bool) -> ArmEvidence {
        ArmEvidence::new(
            EvidenceArm::DenseReference,
            4,
            4,
            4,
            2,
            2,
            with_latency.then(|| latency(1_200)),
        )
        .unwrap()
    }

    fn boolean_only(with_latency: bool) -> ArmEvidence {
        ArmEvidence::new(
            EvidenceArm::BooleanOnlyControl,
            4,
            3,
            3,
            2,
            2,
            with_latency.then(|| latency(1_100)),
        )
        .unwrap()
    }

    fn multi(with_latency: bool) -> ArmEvidence {
        ArmEvidence::new(
            EvidenceArm::MultiAlgebraCandidate,
            4,
            2,
            2,
            2,
            2,
            with_latency.then(|| latency(1_000)),
        )
        .unwrap()
    }

    #[test]
    fn matched_triplet_records_score_reduction_and_rejections() {
        let (route, survivors) = fixture();
        let evidence = MatchedAttentionEvidence::new(
            identity(),
            dense(true),
            boolean_only(true),
            multi(true),
            &route,
            RecompositionPolicy::AllSelectedMustQualify,
            &survivors,
        )
        .unwrap();

        assert_eq!(evidence.schema_version(), 1);
        assert_eq!(
            evidence.route_domains(),
            &[AlgebraDomain::Boolean, AlgebraDomain::F2]
        );
        assert_eq!(evidence.boolean_only().score_work_avoided(), 1);
        assert_eq!(evidence.multi_algebra().score_work_avoided(), 2);
        assert_eq!(evidence.additional_score_work_avoided_vs_boolean(), 1);
        assert_eq!(evidence.multi_algebra().score_reduction_ppm(), 500_000);
        assert_eq!(
            evidence.multi_algebra().relevant_coverage_ppm(),
            Some(1_000_000)
        );
        assert_eq!(evidence.rejection_counts().boolean(), 1);
        assert_eq!(evidence.rejection_counts().f2(), 1);
        assert!(evidence.has_matched_latency());
        assert_eq!(evidence.multi_algebra().latency().unwrap().mean_ns(), 100);
    }

    #[test]
    fn nested_candidate_cannot_resurrect_boolean_rejections() {
        let (route, survivors) = fixture();
        let resurrected = ArmEvidence::new(
            EvidenceArm::MultiAlgebraCandidate,
            4,
            4,
            4,
            2,
            2,
            Some(latency(1_000)),
        )
        .unwrap();
        assert_eq!(
            MatchedAttentionEvidence::new(
                identity(),
                dense(true),
                boolean_only(true),
                resurrected,
                &route,
                RecompositionPolicy::AllSelectedMustQualify,
                &survivors,
            ),
            Err(EvidenceError::MultiResurrectedBooleanRejection {
                boolean_only: 3,
                multi_algebra: 4,
            })
        );
    }

    #[test]
    fn recorded_survivors_must_match_the_host_oracle() {
        let (route, survivors) = fixture();
        let wrong = ArmEvidence::new(
            EvidenceArm::MultiAlgebraCandidate,
            4,
            1,
            1,
            2,
            1,
            Some(latency(1_000)),
        )
        .unwrap();
        assert_eq!(
            MatchedAttentionEvidence::new(
                identity(),
                dense(true),
                boolean_only(true),
                wrong,
                &route,
                RecompositionPolicy::AllSelectedMustQualify,
                &survivors,
            ),
            Err(EvidenceError::OracleSurvivorMismatch {
                evidence: 1,
                oracle: 2,
            })
        );
    }

    #[test]
    fn latency_must_be_all_present_or_all_absent() {
        let (route, survivors) = fixture();
        assert_eq!(
            MatchedAttentionEvidence::new(
                identity(),
                dense(true),
                boolean_only(true),
                multi(false),
                &route,
                RecompositionPolicy::AllSelectedMustQualify,
                &survivors,
            ),
            Err(EvidenceError::IncompleteLatencyTriplet)
        );

        let logical_only = MatchedAttentionEvidence::new(
            identity(),
            dense(false),
            boolean_only(false),
            multi(false),
            &route,
            RecompositionPolicy::AllSelectedMustQualify,
            &survivors,
        )
        .unwrap();
        assert!(!logical_only.has_matched_latency());
    }

    #[test]
    fn latency_summary_rejects_an_impossible_total() {
        assert_eq!(
            LatencyObservation::new(10, 500, 90, 200),
            Err(EvidenceError::InvalidLatencyTotal {
                sample_count: 10,
                total_ns: 500,
                min_ns: 90,
                max_ns: 200,
            })
        );
    }
}
