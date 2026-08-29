//! Deterministic semantic selection without execution routing.
//!
//! This crate consumes the stable rule identities introduced by `flat-semantic`
//! and cataloged by `flat-semantic-registry`. Its policy answers only:
//!
//! > Which registered semantic rule did the caller explicitly request?
//!
//! It does **not** choose a kernel, backend, device, implementation candidate,
//! fallback semantic, or research winner. Those decisions belong to separate
//! contracts and evidence gates.

#![forbid(unsafe_code)]

use std::fmt::{Display, Formatter};

use flat_semantic::v1::SemanticId;
use flat_semantic_registry::{SemanticRegistry, SemanticRegistryError};

/// Version of the deterministic semantic-selection contract.
pub const SELECTION_CONTRACT_VERSION: u16 = 1;

/// Caller-owned request for one exact semantic rule identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticSelectionRequest {
    semantic: SemanticId,
}

impl SemanticSelectionRequest {
    /// Construct an exact semantic request from an already-validated identity.
    #[must_use]
    pub const fn new(semantic: SemanticId) -> Self {
        Self { semantic }
    }

    /// Exact semantic identity requested by the caller.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }
}

/// Deterministic exact-match policy.
///
/// This policy performs no ranking and has no fallback path. If the requested
/// identity is not registered exactly, selection fails.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ExactSemanticSelectionPolicy;

impl ExactSemanticSelectionPolicy {
    /// Resolve exactly one registered semantic rule identity.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticSelectionError::UnregisteredSemantic`] when the exact
    /// requested identity is absent. A different revision or same-family rule
    /// is never substituted.
    pub fn select(
        self,
        registry: &SemanticRegistry,
        request: &SemanticSelectionRequest,
    ) -> Result<SemanticSelectionDecision, SemanticSelectionError> {
        let selected = registry.get(request.semantic()).ok_or_else(|| {
            SemanticSelectionError::UnregisteredSemantic {
                requested: request.semantic().clone(),
            }
        })?;

        let single = SemanticRegistry::new([selected.clone()]).map_err(|error| {
            SemanticSelectionError::RegistryInvariant {
                reason: error.to_string(),
            }
        })?;

        Ok(SemanticSelectionDecision {
            semantic: selected.clone(),
            registry_fingerprint: registry.stable_fingerprint(),
            semantic_identity_fingerprint: single.stable_fingerprint(),
        })
    }
}

/// Immutable result of one exact semantic-selection decision.
///
/// Provenance includes both the selected rule identity and the fingerprint of
/// the registry that admitted it. No execution/kernel identity is present.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticSelectionDecision {
    semantic: SemanticId,
    registry_fingerprint: u64,
    semantic_identity_fingerprint: u64,
}

impl SemanticSelectionDecision {
    /// Selected stable semantic rule identity.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }

    /// Fingerprint of the complete registry used for this decision.
    #[must_use]
    pub const fn registry_fingerprint(&self) -> u64 {
        self.registry_fingerprint
    }

    /// Stable fingerprint of the selected identity using the registry's own
    /// canonical single-entry encoding.
    #[must_use]
    pub const fn semantic_identity_fingerprint(&self) -> u64 {
        self.semantic_identity_fingerprint
    }

    /// Validate this decision against a current registry without reselecting or
    /// substituting another semantic.
    ///
    /// # Errors
    ///
    /// Fails closed when the exact semantic is no longer registered, when the
    /// selected identity fingerprint no longer matches, or when the complete
    /// registry fingerprint differs from the registry that authorized this
    /// decision.
    pub fn validate_against_registry(
        &self,
        registry: &SemanticRegistry,
    ) -> Result<(), SemanticSelectionValidationError> {
        let current = ExactSemanticSelectionPolicy
            .select(
                registry,
                &SemanticSelectionRequest::new(self.semantic.clone()),
            )
            .map_err(SemanticSelectionValidationError::Selection)?;

        if current.semantic_identity_fingerprint != self.semantic_identity_fingerprint {
            return Err(
                SemanticSelectionValidationError::SemanticIdentityFingerprintMismatch {
                    decision: self.semantic_identity_fingerprint,
                    current: current.semantic_identity_fingerprint,
                },
            );
        }
        if current.registry_fingerprint != self.registry_fingerprint {
            return Err(SemanticSelectionValidationError::RegistryFingerprintMismatch {
                decision: self.registry_fingerprint,
                current: current.registry_fingerprint,
            });
        }
        Ok(())
    }

    /// Canonical selection record.
    ///
    /// The record deliberately excludes kernel, backend, device, benchmark,
    /// fallback, and external-evidence fields.
    #[must_use]
    pub fn canonical_record(&self) -> String {
        format!(
            "flat-semantic-selection-v{SELECTION_CONTRACT_VERSION};policy=exact-registered;registry={:016x};semantic={:016x}",
            self.registry_fingerprint, self.semantic_identity_fingerprint
        )
    }

    /// Stable FNV-1a-64 fingerprint of [`Self::canonical_record`].
    ///
    /// This is a deterministic provenance/cache-key aid, not a cryptographic
    /// authenticity primitive.
    #[must_use]
    pub fn stable_fingerprint(&self) -> u64 {
        fnv1a64(self.canonical_record().as_bytes())
    }
}

/// Fail-closed validation errors for a previously produced selection decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticSelectionValidationError {
    /// Exact revalidation of the selected semantic failed.
    Selection(SemanticSelectionError),
    /// The selected identity no longer has the same canonical fingerprint.
    SemanticIdentityFingerprintMismatch {
        /// Fingerprint recorded when the decision was produced.
        decision: u64,
        /// Fingerprint computed from the current registry entry.
        current: u64,
    },
    /// The complete registry changed after the decision was produced.
    RegistryFingerprintMismatch {
        /// Registry fingerprint recorded by the decision.
        decision: u64,
        /// Fingerprint of the current registry.
        current: u64,
    },
}

impl Display for SemanticSelectionValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Selection(error) => write!(formatter, "semantic selection is no longer valid: {error}"),
            Self::SemanticIdentityFingerprintMismatch { decision, current } => write!(
                formatter,
                "semantic identity fingerprint changed: decision={decision:016x} current={current:016x}"
            ),
            Self::RegistryFingerprintMismatch { decision, current } => write!(
                formatter,
                "semantic registry fingerprint changed: decision={decision:016x} current={current:016x}"
            ),
        }
    }
}

impl std::error::Error for SemanticSelectionValidationError {}

/// Fail-closed semantic-selection errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticSelectionError {
    /// The exact requested rule identity is absent from the registry.
    UnregisteredSemantic {
        /// Identity that was requested without an exact registry match.
        requested: SemanticId,
    },
    /// A post-lookup registry invariant unexpectedly failed while deriving the
    /// canonical selected-identity fingerprint.
    RegistryInvariant {
        /// Deterministic human-readable reason from the registry contract.
        reason: String,
    },
}

impl Display for SemanticSelectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnregisteredSemantic { requested } => write!(
                formatter,
                "semantic {} revision {} is not registered exactly",
                requested.name(),
                requested.revision()
            ),
            Self::RegistryInvariant { reason } => {
                write!(formatter, "semantic registry invariant failed: {reason}")
            }
        }
    }
}

impl std::error::Error for SemanticSelectionError {}

impl From<SemanticRegistryError> for SemanticSelectionError {
    fn from(error: SemanticRegistryError) -> Self {
        Self::RegistryInvariant {
            reason: error.to_string(),
        }
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_semantic::v1::{SemanticFamily, StandardSoftmaxSemantic};

    fn semantic(family: SemanticFamily, name: &str, revision: u32) -> SemanticId {
        SemanticId::new(family, name, revision).unwrap()
    }

    #[test]
    fn exact_registered_selection_is_deterministic() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let toeplitz = semantic(SemanticFamily::ToeplitzStructured, "toeplitz-research", 2);
        let registry = SemanticRegistry::new([toeplitz, standard.clone()]).unwrap();
        let request = SemanticSelectionRequest::new(standard.clone());
        let policy = ExactSemanticSelectionPolicy;

        let first = policy.select(&registry, &request).unwrap();
        let second = policy.select(&registry, &request).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.semantic(), &standard);
        assert_eq!(first.registry_fingerprint(), registry.stable_fingerprint());
        assert_eq!(first.stable_fingerprint(), second.stable_fingerprint());
        first.validate_against_registry(&registry).unwrap();
    }

    #[test]
    fn missing_revision_does_not_fallback() {
        let registered = semantic(SemanticFamily::Experimental, "candidate", 1);
        let requested = semantic(SemanticFamily::Experimental, "candidate", 2);
        let registry = SemanticRegistry::new([registered]).unwrap();
        let request = SemanticSelectionRequest::new(requested.clone());

        assert_eq!(
            ExactSemanticSelectionPolicy
                .select(&registry, &request)
                .unwrap_err(),
            SemanticSelectionError::UnregisteredSemantic { requested }
        );
    }

    #[test]
    fn same_family_does_not_create_implicit_fallback() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let requested = semantic(SemanticFamily::StandardSoftmax, "alternate-softmax", 1);
        let registry = SemanticRegistry::new([standard]).unwrap();
        let request = SemanticSelectionRequest::new(requested.clone());

        assert!(matches!(
            ExactSemanticSelectionPolicy.select(&registry, &request),
            Err(SemanticSelectionError::UnregisteredSemantic { requested: value }) if value == requested
        ));
    }

    #[test]
    fn registry_provenance_changes_without_changing_selected_rule() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let extra = semantic(SemanticFamily::RecurrentMemory, "delta-memory", 1);
        let narrow = SemanticRegistry::new([standard.clone()]).unwrap();
        let broad = SemanticRegistry::new([extra, standard.clone()]).unwrap();
        let request = SemanticSelectionRequest::new(standard.clone());

        let narrow_decision = ExactSemanticSelectionPolicy
            .select(&narrow, &request)
            .unwrap();
        let broad_decision = ExactSemanticSelectionPolicy
            .select(&broad, &request)
            .unwrap();

        assert_eq!(narrow_decision.semantic(), &standard);
        assert_eq!(broad_decision.semantic(), &standard);
        assert_eq!(
            narrow_decision.semantic_identity_fingerprint(),
            broad_decision.semantic_identity_fingerprint()
        );
        assert_ne!(
            narrow_decision.registry_fingerprint(),
            broad_decision.registry_fingerprint()
        );
        assert_ne!(
            narrow_decision.stable_fingerprint(),
            broad_decision.stable_fingerprint()
        );
        assert_eq!(
            narrow_decision.validate_against_registry(&broad),
            Err(SemanticSelectionValidationError::RegistryFingerprintMismatch {
                decision: narrow.stable_fingerprint(),
                current: broad.stable_fingerprint(),
            })
        );
    }

    #[test]
    fn removed_exact_semantic_fails_revalidation_without_fallback() {
        let selected = semantic(SemanticFamily::Experimental, "candidate", 1);
        let replacement = semantic(SemanticFamily::Experimental, "candidate", 2);
        let original = SemanticRegistry::new([selected.clone()]).unwrap();
        let current = SemanticRegistry::new([replacement]).unwrap();
        let decision = ExactSemanticSelectionPolicy
            .select(
                &original,
                &SemanticSelectionRequest::new(selected.clone()),
            )
            .unwrap();

        assert_eq!(
            decision.validate_against_registry(&current),
            Err(SemanticSelectionValidationError::Selection(
                SemanticSelectionError::UnregisteredSemantic {
                    requested: selected,
                },
            ))
        );
    }

    #[test]
    fn selection_is_rule_level_not_instance_or_kernel_level() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum TestKernel {
            Portable,
            Optimized,
        }

        let bidirectional = StandardSoftmaxSemantic::new(false, 0.5).unwrap();
        let causal = StandardSoftmaxSemantic::new(true, 0.25).unwrap();
        let rule = bidirectional.descriptor().id().clone();
        assert_eq!(rule, causal.descriptor().id().clone());
        assert_ne!(
            bidirectional.stable_fingerprint(),
            causal.stable_fingerprint()
        );

        let registry = SemanticRegistry::new([rule.clone()]).unwrap();
        let decision = ExactSemanticSelectionPolicy
            .select(&registry, &SemanticSelectionRequest::new(rule))
            .unwrap();
        let execution_plans = [
            (decision.stable_fingerprint(), TestKernel::Portable),
            (decision.stable_fingerprint(), TestKernel::Optimized),
        ];

        assert_eq!(execution_plans[0].0, execution_plans[1].0);
        assert_ne!(execution_plans[0].1, execution_plans[1].1);
        let record = decision.canonical_record();
        assert!(!record.contains("Portable"));
        assert!(!record.contains("Optimized"));
        assert!(!record.contains("device"));
    }
}
