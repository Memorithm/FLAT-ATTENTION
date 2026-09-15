use core::fmt;

use crate::cooperation::{AlgebraDomain, AlgebraicRoute};
use crate::qualification::RecompositionPolicy;

pub const MAA_HOLDOUT_POLICY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    pub fn parse(value: &str) -> Result<Self, HoldoutError> {
        if value.len() != 64 {
            return Err(HoldoutError::InvalidSha256Length {
                actual: value.len(),
            });
        }

        let bytes = value.as_bytes();
        let mut output = [0u8; 32];
        for (index, pair) in bytes.chunks_exact(2).enumerate() {
            let high = decode_hex(pair[0], index * 2)?;
            let low = decode_hex(pair[1], index * 2 + 1)?;
            output[index] = (high << 4) | low;
        }
        Ok(Self(output))
    }

    #[must_use]
    pub fn to_hex(self) -> String {
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            use core::fmt::Write as _;
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

fn decode_hex(byte: u8, index: usize) -> Result<u8, HoldoutError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(HoldoutError::InvalidSha256Character { index, byte }),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenPolicyManifest {
    schema_version: u32,
    policy_id: String,
    source_revision: String,
    route_domains: Vec<AlgebraDomain>,
    recomposition_policy: RecompositionPolicy,
    predicate_digest: Sha256Digest,
    feature_schema_digest: Sha256Digest,
    tuning_dataset_digest: Sha256Digest,
    confirmatory_dataset_digest: Sha256Digest,
}

impl FrozenPolicyManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        policy_id: impl Into<String>,
        source_revision: impl Into<String>,
        route: &AlgebraicRoute,
        recomposition_policy: RecompositionPolicy,
        predicate_digest: Sha256Digest,
        feature_schema_digest: Sha256Digest,
        tuning_dataset_digest: Sha256Digest,
        confirmatory_dataset_digest: Sha256Digest,
    ) -> Result<Self, HoldoutError> {
        let policy_id = policy_id.into();
        let source_revision = source_revision.into();
        require_text("policy_id", &policy_id)?;
        require_text("source_revision", &source_revision)?;
        if tuning_dataset_digest == confirmatory_dataset_digest {
            return Err(HoldoutError::HoldoutDatasetsOverlap);
        }

        Ok(Self {
            schema_version: MAA_HOLDOUT_POLICY_SCHEMA_VERSION,
            policy_id,
            source_revision,
            route_domains: route.domains().to_vec(),
            recomposition_policy,
            predicate_digest,
            feature_schema_digest,
            tuning_dataset_digest,
            confirmatory_dataset_digest,
        })
    }

    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[must_use]
    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    #[must_use]
    pub fn source_revision(&self) -> &str {
        &self.source_revision
    }

    #[must_use]
    pub fn route_domains(&self) -> &[AlgebraDomain] {
        &self.route_domains
    }

    #[must_use]
    pub const fn recomposition_policy(&self) -> RecompositionPolicy {
        self.recomposition_policy
    }

    #[must_use]
    pub const fn predicate_digest(&self) -> Sha256Digest {
        self.predicate_digest
    }

    #[must_use]
    pub const fn feature_schema_digest(&self) -> Sha256Digest {
        self.feature_schema_digest
    }

    #[must_use]
    pub const fn tuning_dataset_digest(&self) -> Sha256Digest {
        self.tuning_dataset_digest
    }

    #[must_use]
    pub const fn confirmatory_dataset_digest(&self) -> Sha256Digest {
        self.confirmatory_dataset_digest
    }

    pub fn bind_confirmatory<'a>(
        &'a self,
        context: &ConfirmatoryContext<'_>,
    ) -> Result<ConfirmatoryBinding<'a>, HoldoutError> {
        if context.policy_id != self.policy_id {
            return Err(HoldoutError::PolicyIdMismatch);
        }
        if context.source_revision != self.source_revision {
            return Err(HoldoutError::SourceRevisionMismatch);
        }
        if context.route.domains() != self.route_domains {
            return Err(HoldoutError::RouteMismatch {
                expected: self.route_domains.clone(),
                actual: context.route.domains().to_vec(),
            });
        }
        if context.recomposition_policy != self.recomposition_policy {
            return Err(HoldoutError::RecompositionPolicyMismatch {
                expected: self.recomposition_policy,
                actual: context.recomposition_policy,
            });
        }
        if context.predicate_digest != self.predicate_digest {
            return Err(HoldoutError::PredicateDigestMismatch);
        }
        if context.feature_schema_digest != self.feature_schema_digest {
            return Err(HoldoutError::FeatureSchemaDigestMismatch);
        }
        if context.dataset_digest == self.tuning_dataset_digest {
            return Err(HoldoutError::TuningDatasetUsedAsConfirmatory);
        }
        if context.dataset_digest != self.confirmatory_dataset_digest {
            return Err(HoldoutError::ConfirmatoryDatasetDigestMismatch);
        }

        Ok(ConfirmatoryBinding { manifest: self })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConfirmatoryContext<'a> {
    policy_id: &'a str,
    source_revision: &'a str,
    route: &'a AlgebraicRoute,
    recomposition_policy: RecompositionPolicy,
    predicate_digest: Sha256Digest,
    feature_schema_digest: Sha256Digest,
    dataset_digest: Sha256Digest,
}

impl<'a> ConfirmatoryContext<'a> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        policy_id: &'a str,
        source_revision: &'a str,
        route: &'a AlgebraicRoute,
        recomposition_policy: RecompositionPolicy,
        predicate_digest: Sha256Digest,
        feature_schema_digest: Sha256Digest,
        dataset_digest: Sha256Digest,
    ) -> Self {
        Self {
            policy_id,
            source_revision,
            route,
            recomposition_policy,
            predicate_digest,
            feature_schema_digest,
            dataset_digest,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConfirmatoryBinding<'a> {
    manifest: &'a FrozenPolicyManifest,
}

impl<'a> ConfirmatoryBinding<'a> {
    #[must_use]
    pub const fn manifest(&self) -> &'a FrozenPolicyManifest {
        self.manifest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HoldoutError {
    EmptyTextField {
        field: &'static str,
    },
    InvalidSha256Length {
        actual: usize,
    },
    InvalidSha256Character {
        index: usize,
        byte: u8,
    },
    HoldoutDatasetsOverlap,
    PolicyIdMismatch,
    SourceRevisionMismatch,
    RouteMismatch {
        expected: Vec<AlgebraDomain>,
        actual: Vec<AlgebraDomain>,
    },
    RecompositionPolicyMismatch {
        expected: RecompositionPolicy,
        actual: RecompositionPolicy,
    },
    PredicateDigestMismatch,
    FeatureSchemaDigestMismatch,
    TuningDatasetUsedAsConfirmatory,
    ConfirmatoryDatasetDigestMismatch,
}

impl fmt::Display for HoldoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HoldoutError {}

fn require_text(field: &'static str, value: &str) -> Result<(), HoldoutError> {
    if value.trim().is_empty() {
        Err(HoldoutError::EmptyTextField { field })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cooperation::{
        route_attention_needs, AttentionAlgebraNeeds, M13bBooleanRoutingEvidence,
    };

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn digest(value: &str) -> Sha256Digest {
        Sha256Digest::parse(value).unwrap()
    }

    fn boolean_evidence() -> M13bBooleanRoutingEvidence {
        M13bBooleanRoutingEvidence::new(
            1,
            4,
            vec![0b1111],
            1,
            4,
            vec![0b1011],
            vec![0b1101],
            None,
            None,
        )
        .unwrap()
    }

    fn all_domains_route() -> AlgebraicRoute {
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

    fn boolean_f2_route() -> AlgebraicRoute {
        route_attention_needs(
            AttentionAlgebraNeeds {
                eligibility_logic: true,
                parity_or_binary_linear: true,
                ..AttentionAlgebraNeeds::default()
            },
            Some(boolean_evidence()),
        )
        .unwrap()
    }

    fn manifest(route: &AlgebraicRoute) -> FrozenPolicyManifest {
        FrozenPolicyManifest::new(
            "maa-policy-v1",
            "commit:0123456789abcdef",
            route,
            RecompositionPolicy::AllSelectedMustQualify,
            digest(A),
            digest(B),
            digest(C),
            digest(D),
        )
        .unwrap()
    }

    fn matching_context<'a>(route: &'a AlgebraicRoute) -> ConfirmatoryContext<'a> {
        ConfirmatoryContext::new(
            "maa-policy-v1",
            "commit:0123456789abcdef",
            route,
            RecompositionPolicy::AllSelectedMustQualify,
            digest(A),
            digest(B),
            digest(D),
        )
    }

    #[test]
    fn digest_accepts_mixed_case_and_normalizes_to_lowercase() {
        let value = Sha256Digest::parse(
            "AaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAa",
        )
        .unwrap();
        assert_eq!(value.to_hex(), A);
        assert_eq!(value.to_string(), A);
    }

    #[test]
    fn malformed_digest_fails_closed() {
        assert_eq!(
            Sha256Digest::parse("abcd"),
            Err(HoldoutError::InvalidSha256Length { actual: 4 })
        );
        let invalid = format!("{}g", &A[..63]);
        assert_eq!(
            Sha256Digest::parse(&invalid),
            Err(HoldoutError::InvalidSha256Character {
                index: 63,
                byte: b'g',
            })
        );
    }

    #[test]
    fn blank_identity_and_overlapping_datasets_fail_closed() {
        let route = boolean_f2_route();
        assert_eq!(
            FrozenPolicyManifest::new(
                "   ",
                "commit:1",
                &route,
                RecompositionPolicy::AllSelectedMustQualify,
                digest(A),
                digest(B),
                digest(C),
                digest(D),
            ),
            Err(HoldoutError::EmptyTextField { field: "policy_id" })
        );
        assert_eq!(
            FrozenPolicyManifest::new(
                "policy",
                "\t",
                &route,
                RecompositionPolicy::AllSelectedMustQualify,
                digest(A),
                digest(B),
                digest(C),
                digest(D),
            ),
            Err(HoldoutError::EmptyTextField {
                field: "source_revision",
            })
        );
        assert_eq!(
            FrozenPolicyManifest::new(
                "policy",
                "commit:1",
                &route,
                RecompositionPolicy::AllSelectedMustQualify,
                digest(A),
                digest(B),
                digest(C),
                digest(C),
            ),
            Err(HoldoutError::HoldoutDatasetsOverlap)
        );
    }

    #[test]
    fn exact_all_domain_confirmatory_context_binds_successfully() {
        let route = all_domains_route();
        let frozen = manifest(&route);
        let before = frozen.clone();
        let context = matching_context(&route);
        let binding = frozen.bind_confirmatory(&context).unwrap();

        assert_eq!(binding.manifest(), &frozen);
        assert_eq!(frozen, before);
        assert_eq!(frozen.schema_version(), MAA_HOLDOUT_POLICY_SCHEMA_VERSION);
        assert_eq!(
            frozen.route_domains(),
            &[
                AlgebraDomain::Boolean,
                AlgebraDomain::F2,
                AlgebraDomain::Zhegalkin,
                AlgebraDomain::MaxPlus,
            ]
        );
    }

    #[test]
    fn route_identity_is_not_inferred_from_policy_label() {
        let all = all_domains_route();
        let smaller = boolean_f2_route();
        let frozen = manifest(&all);
        let context = matching_context(&smaller);

        assert_eq!(
            frozen.bind_confirmatory(&context),
            Err(HoldoutError::RouteMismatch {
                expected: vec![
                    AlgebraDomain::Boolean,
                    AlgebraDomain::F2,
                    AlgebraDomain::Zhegalkin,
                    AlgebraDomain::MaxPlus,
                ],
                actual: vec![AlgebraDomain::Boolean, AlgebraDomain::F2],
            })
        );
    }

    #[test]
    fn every_frozen_identity_mismatch_fails_closed() {
        let route = all_domains_route();
        let frozen = manifest(&route);

        let wrong_policy = ConfirmatoryContext::new(
            "other-policy",
            "commit:0123456789abcdef",
            &route,
            RecompositionPolicy::AllSelectedMustQualify,
            digest(A),
            digest(B),
            digest(D),
        );
        assert_eq!(
            frozen.bind_confirmatory(&wrong_policy),
            Err(HoldoutError::PolicyIdMismatch)
        );

        let wrong_revision = ConfirmatoryContext::new(
            "maa-policy-v1",
            "commit:other",
            &route,
            RecompositionPolicy::AllSelectedMustQualify,
            digest(A),
            digest(B),
            digest(D),
        );
        assert_eq!(
            frozen.bind_confirmatory(&wrong_revision),
            Err(HoldoutError::SourceRevisionMismatch)
        );

        let wrong_recomposition = ConfirmatoryContext::new(
            "maa-policy-v1",
            "commit:0123456789abcdef",
            &route,
            RecompositionPolicy::AnySelectedMayQualify,
            digest(A),
            digest(B),
            digest(D),
        );
        assert_eq!(
            frozen.bind_confirmatory(&wrong_recomposition),
            Err(HoldoutError::RecompositionPolicyMismatch {
                expected: RecompositionPolicy::AllSelectedMustQualify,
                actual: RecompositionPolicy::AnySelectedMayQualify,
            })
        );

        let wrong_predicate = ConfirmatoryContext::new(
            "maa-policy-v1",
            "commit:0123456789abcdef",
            &route,
            RecompositionPolicy::AllSelectedMustQualify,
            digest(E),
            digest(B),
            digest(D),
        );
        assert_eq!(
            frozen.bind_confirmatory(&wrong_predicate),
            Err(HoldoutError::PredicateDigestMismatch)
        );

        let wrong_features = ConfirmatoryContext::new(
            "maa-policy-v1",
            "commit:0123456789abcdef",
            &route,
            RecompositionPolicy::AllSelectedMustQualify,
            digest(A),
            digest(E),
            digest(D),
        );
        assert_eq!(
            frozen.bind_confirmatory(&wrong_features),
            Err(HoldoutError::FeatureSchemaDigestMismatch)
        );

        let wrong_holdout = ConfirmatoryContext::new(
            "maa-policy-v1",
            "commit:0123456789abcdef",
            &route,
            RecompositionPolicy::AllSelectedMustQualify,
            digest(A),
            digest(B),
            digest(F),
        );
        assert_eq!(
            frozen.bind_confirmatory(&wrong_holdout),
            Err(HoldoutError::ConfirmatoryDatasetDigestMismatch)
        );
    }

    #[test]
    fn tuning_dataset_cannot_be_relabelled_as_confirmatory() {
        let route = all_domains_route();
        let frozen = manifest(&route);
        let context = ConfirmatoryContext::new(
            "maa-policy-v1",
            "commit:0123456789abcdef",
            &route,
            RecompositionPolicy::AllSelectedMustQualify,
            digest(A),
            digest(B),
            digest(C),
        );

        assert_eq!(
            frozen.bind_confirmatory(&context),
            Err(HoldoutError::TuningDatasetUsedAsConfirmatory)
        );
    }
}
