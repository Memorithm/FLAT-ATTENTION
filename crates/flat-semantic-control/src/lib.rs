//! Backend-neutral semantic request/state and selection-policy contracts.
//!
//! This crate is a control-plane layer above `flat-semantic` and
//! `flat-semantic-registry`. It deliberately does not depend on FLAT kernel,
//! device, WGPU, autotuning, or benchmark types.
//!
//! The payload and carried-state types are generic. Future recurrent, Toeplitz,
//! Green-kernel, or hybrid semantics are therefore not forced through a Q/K/V
//! request shape merely because StandardSoftmax uses one today.

#![forbid(unsafe_code)]

use std::fmt::{Display, Formatter, Write};

use flat_semantic::v1::{SemanticDescriptor, SemanticId, StateSemantics};
use flat_semantic_registry::SemanticRegistry;

/// Version of this semantic control-plane contract.
pub const SEMANTIC_CONTROL_VERSION: u16 = 1;

/// Runtime semantic state attached to one typed request.
///
/// `Stateless` is explicit instead of being encoded as an empty user payload.
/// The descriptor and state value must agree before construction succeeds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticState<S> {
    /// The semantic carries no state between declared invocations.
    Stateless,
    /// A semantic-specific carried state supplied by the caller.
    Recurrent(S),
}

impl<S> SemanticState<S> {
    /// State semantics represented by this value.
    #[must_use]
    pub const fn semantics(&self) -> StateSemantics {
        match self {
            Self::Stateless => StateSemantics::Stateless,
            Self::Recurrent(_) => StateSemantics::Recurrent,
        }
    }
}

/// Typed semantic invocation whose payload and state remain semantic-specific.
///
/// This is an admission/control object, not an execution request. It contains
/// no kernel, device, backend, or buffer handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticRequest<P, S> {
    descriptor: SemanticDescriptor,
    payload: P,
    state: SemanticState<S>,
}

impl<P, S> SemanticRequest<P, S> {
    /// Construct a request after checking the declared state contract.
    ///
    /// # Errors
    ///
    /// Fails closed when a stateless descriptor receives recurrent state or a
    /// recurrent descriptor receives no carried state.
    pub fn new(
        descriptor: SemanticDescriptor,
        payload: P,
        state: SemanticState<S>,
    ) -> Result<Self, SemanticControlError> {
        let actual = state.semantics();
        if descriptor.state() != actual {
            return Err(SemanticControlError::StateContractMismatch {
                expected: descriptor.state(),
                actual,
            });
        }
        Ok(Self {
            descriptor,
            payload,
            state,
        })
    }

    /// Stable semantic rule identity requested by the caller.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        self.descriptor.id()
    }

    /// Full semantic-instance descriptor.
    #[must_use]
    pub const fn descriptor(&self) -> &SemanticDescriptor {
        &self.descriptor
    }

    /// Semantic-specific payload, with no imposed Q/K/V interpretation.
    #[must_use]
    pub const fn payload(&self) -> &P {
        &self.payload
    }

    /// Semantic-specific state value.
    #[must_use]
    pub const fn state(&self) -> &SemanticState<S> {
        &self.state
    }

    /// Consume the request without copying payload or state.
    #[must_use]
    pub fn into_parts(self) -> (SemanticDescriptor, P, SemanticState<S>) {
        (self.descriptor, self.payload, self.state)
    }
}

/// Explicit caller-owned semantic preference order.
///
/// The policy never invents a fallback. Only identities listed here may be
/// selected, and only when the registry contains the exact identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticSelectionPolicy {
    preferences: Vec<SemanticId>,
}

impl SemanticSelectionPolicy {
    /// Construct an ordered preference policy.
    ///
    /// # Errors
    ///
    /// Empty policies and duplicate exact identities are rejected because both
    /// would make provenance or fallback intent ambiguous.
    pub fn new(
        preferences: impl IntoIterator<Item = SemanticId>,
    ) -> Result<Self, SemanticControlError> {
        let preferences = preferences.into_iter().collect::<Vec<_>>();
        if preferences.is_empty() {
            return Err(SemanticControlError::EmptySelectionPolicy);
        }
        for (index, semantic) in preferences.iter().enumerate() {
            if preferences[..index].contains(semantic) {
                return Err(SemanticControlError::DuplicatePreference {
                    semantic: semantic.clone(),
                });
            }
        }
        Ok(Self { preferences })
    }

    /// Caller-declared ordered semantic identities.
    #[must_use]
    pub fn preferences(&self) -> &[SemanticId] {
        &self.preferences
    }

    /// Select the first caller-approved semantic present in the registry.
    ///
    /// This function does not instantiate a semantic implementation and does
    /// not select a kernel/backend. The returned decision is semantic-only.
    #[must_use]
    pub fn select(&self, registry: &SemanticRegistry) -> SemanticSelectionDecision {
        self.preferences
            .iter()
            .enumerate()
            .find(|(_, semantic)| registry.contains(semantic))
            .map_or(SemanticSelectionDecision::NoRegisteredPreference, |(rank, semantic)| {
                SemanticSelectionDecision::Selected {
                    semantic: semantic.clone(),
                    preference_rank: rank,
                }
            })
    }

    /// Stable caller-intent record excluding execution information.
    #[must_use]
    pub fn canonical_record(&self) -> String {
        let mut record = format!(
            "flat-semantic-control-v{SEMANTIC_CONTROL_VERSION};preferences={}\n",
            self.preferences.len()
        );
        for (rank, semantic) in self.preferences.iter().enumerate() {
            let _ = writeln!(
                record,
                "rank={rank};name={};revision={}",
                semantic.name(),
                semantic.revision()
            );
        }
        record
    }

    /// Stable FNV-1a-64 fingerprint of [`Self::canonical_record`].
    ///
    /// This is a deterministic provenance aid, not a cryptographic digest.
    #[must_use]
    pub fn stable_fingerprint(&self) -> u64 {
        fnv1a64(self.canonical_record().as_bytes())
    }
}

/// Result of semantic-only selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticSelectionDecision {
    /// The first caller-approved registered semantic was selected.
    Selected {
        /// Exact stable semantic rule identity.
        semantic: SemanticId,
        /// Zero-based rank in the caller-owned preference list.
        preference_rank: usize,
    },
    /// None of the explicitly allowed semantics were registered.
    NoRegisteredPreference,
}

impl SemanticSelectionDecision {
    /// Selected semantic identity, when selection succeeded.
    #[must_use]
    pub const fn semantic(&self) -> Option<&SemanticId> {
        match self {
            Self::Selected { semantic, .. } => Some(semantic),
            Self::NoRegisteredPreference => None,
        }
    }
}

/// Typed control-plane validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticControlError {
    /// Request state does not match the semantic descriptor.
    StateContractMismatch {
        /// State kind declared by the semantic descriptor.
        expected: StateSemantics,
        /// State kind supplied by the request.
        actual: StateSemantics,
    },
    /// No semantic was supplied to the explicit preference policy.
    EmptySelectionPolicy,
    /// One exact semantic identity appears more than once in the policy.
    DuplicatePreference {
        /// Duplicated identity.
        semantic: SemanticId,
    },
}

impl Display for SemanticControlError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateContractMismatch { expected, actual } => {
                write!(formatter, "semantic state mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::EmptySelectionPolicy => {
                formatter.write_str("semantic selection policy must not be empty")
            }
            Self::DuplicatePreference { semantic } => write!(
                formatter,
                "semantic {} revision {} appears more than once in the selection policy",
                semantic.name(),
                semantic.revision()
            ),
        }
    }
}

impl std::error::Error for SemanticControlError {}

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
    use flat_semantic::v1::{
        MaskSemantics, SavedStateContract, SemanticFamily, StandardSoftmaxSemantic,
        WeightSemantics,
    };

    fn semantic(family: SemanticFamily, name: &str, revision: u32) -> SemanticId {
        SemanticId::new(family, name, revision).unwrap()
    }

    fn recurrent_descriptor() -> SemanticDescriptor {
        SemanticDescriptor::new(
            semantic(SemanticFamily::RecurrentMemory, "delta-memory", 1),
            MaskSemantics::Causal,
            StateSemantics::Recurrent,
            WeightSemantics::StateDependent,
            SavedStateContract::None,
        )
    }

    #[test]
    fn request_payload_is_not_forced_into_qkv_shape() {
        #[derive(Debug, Clone, PartialEq, Eq)]
        struct RecurrentPayload {
            token: u32,
            gate: u16,
        }

        let request = SemanticRequest::new(
            recurrent_descriptor(),
            RecurrentPayload { token: 7, gate: 3 },
            SemanticState::Recurrent(vec![11_u8, 12, 13]),
        )
        .unwrap();

        assert_eq!(request.payload().token, 7);
        assert_eq!(request.payload().gate, 3);
        assert_eq!(request.semantic().name(), "delta-memory");
        assert!(matches!(request.state(), SemanticState::Recurrent(state) if state == &[11, 12, 13]));
    }

    #[test]
    fn request_state_contract_fails_closed() {
        let standard = StandardSoftmaxSemantic::new(false, 0.5).unwrap().descriptor();
        let error = SemanticRequest::new(standard, (), SemanticState::Recurrent(5_u32)).unwrap_err();
        assert_eq!(
            error,
            SemanticControlError::StateContractMismatch {
                expected: StateSemantics::Stateless,
                actual: StateSemantics::Recurrent,
            }
        );

        let error = SemanticRequest::<(), u32>::new(
            recurrent_descriptor(),
            (),
            SemanticState::Stateless,
        )
        .unwrap_err();
        assert_eq!(
            error,
            SemanticControlError::StateContractMismatch {
                expected: StateSemantics::Recurrent,
                actual: StateSemantics::Stateless,
            }
        );
    }

    #[test]
    fn caller_order_controls_semantic_selection_without_implicit_fallback() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let toeplitz = semantic(SemanticFamily::ToeplitzStructured, "toeplitz-research", 1);
        let recurrent = semantic(SemanticFamily::RecurrentMemory, "delta-memory", 1);
        let registry = SemanticRegistry::new([standard.clone(), recurrent.clone()]).unwrap();

        let prefer_toeplitz =
            SemanticSelectionPolicy::new([toeplitz, recurrent.clone(), standard.clone()]).unwrap();
        assert_eq!(
            prefer_toeplitz.select(&registry),
            SemanticSelectionDecision::Selected {
                semantic: recurrent,
                preference_rank: 1,
            }
        );

        let unavailable = SemanticSelectionPolicy::new([semantic(
            SemanticFamily::ProlateConcentration,
            "prolate-research",
            1,
        )])
        .unwrap();
        assert_eq!(
            unavailable.select(&registry),
            SemanticSelectionDecision::NoRegisteredPreference
        );
    }

    #[test]
    fn selection_identity_contains_no_execution_choice() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum TestKernel {
            Portable,
            Optimized,
        }

        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let registry = SemanticRegistry::new([standard.clone()]).unwrap();
        let policy = SemanticSelectionPolicy::new([standard]).unwrap();
        let decision = policy.select(&registry);
        let fingerprint = policy.stable_fingerprint();

        let execution_bindings = [
            (decision.clone(), fingerprint, TestKernel::Portable),
            (decision.clone(), fingerprint, TestKernel::Optimized),
        ];
        assert_eq!(execution_bindings[0].0, execution_bindings[1].0);
        assert_eq!(execution_bindings[0].1, execution_bindings[1].1);
        assert_ne!(execution_bindings[0].2, execution_bindings[1].2);
        let record = policy.canonical_record();
        assert!(!record.contains("Portable"));
        assert!(!record.contains("Optimized"));
    }

    #[test]
    fn invalid_preference_policies_fail_closed() {
        assert_eq!(
            SemanticSelectionPolicy::new(Vec::<SemanticId>::new()).unwrap_err(),
            SemanticControlError::EmptySelectionPolicy
        );
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        assert_eq!(
            SemanticSelectionPolicy::new([standard.clone(), standard.clone()]).unwrap_err(),
            SemanticControlError::DuplicatePreference { semantic: standard }
        );
    }
}
