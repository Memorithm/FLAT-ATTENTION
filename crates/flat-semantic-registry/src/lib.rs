//! Deterministic registry of FLAT semantic rule identities.
//!
//! The registry is deliberately narrower than an execution dispatcher. It
//! catalogs stable [`SemanticId`] values only. Instance parameters such as
//! causal visibility or score scale remain in concrete semantic values, while
//! kernel/device selection remains outside this crate entirely.
//!
//! This separation prevents three concepts from being collapsed into one key:
//! semantic rule identity, semantic instance configuration, and execution
//! strategy. Registration therefore makes no claim that a rule has a GPU
//! implementation, production route, benchmark advantage, or positive research
//! result.

#![forbid(unsafe_code)]

use std::fmt::{Display, Formatter, Write};

use flat_semantic::v1::{SemanticFamily, SemanticId};

/// Version of the deterministic registry serialization contract.
pub const REGISTRY_CONTRACT_VERSION: u16 = 1;

/// Deterministic catalog of stable semantic rule identities.
///
/// Entries are canonicalized by family, stable slug, then semantic revision.
/// Construction rejects duplicate identities instead of silently deduplicating
/// them, so independently assembled registries fail closed on ambiguous input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticRegistry {
    entries: Vec<SemanticId>,
}

impl SemanticRegistry {
    /// Build a canonical registry from semantic rule identities.
    ///
    /// Input order does not affect [`Self::canonical_record`] or
    /// [`Self::stable_fingerprint`].
    ///
    /// # Errors
    ///
    /// Returns an error when the linked semantic-contract version contains a
    /// family unknown to this registry contract, or when the same semantic
    /// identity is supplied more than once.
    pub fn new(
        entries: impl IntoIterator<Item = SemanticId>,
    ) -> Result<Self, SemanticRegistryError> {
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        for semantic in &entries {
            if family_rank(semantic.family()) == u8::MAX {
                return Err(SemanticRegistryError::UnsupportedFamily {
                    semantic: semantic.clone(),
                });
            }
        }

        entries.sort_by(|left, right| {
            family_rank(left.family())
                .cmp(&family_rank(right.family()))
                .then_with(|| left.name().cmp(right.name()))
                .then_with(|| left.revision().cmp(&right.revision()))
        });

        if let Some(duplicate) = entries
            .windows(2)
            .find(|pair| pair[0] == pair[1])
            .map(|pair| pair[0].clone())
        {
            return Err(SemanticRegistryError::DuplicateSemantic {
                semantic: duplicate,
            });
        }

        Ok(Self { entries })
    }

    /// Number of registered semantic rule identities.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry contains no semantic rule identities.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Canonically ordered semantic identities.
    #[must_use]
    pub fn entries(&self) -> &[SemanticId] {
        &self.entries
    }

    /// Resolve an exact semantic rule identity.
    ///
    /// This lookup returns metadata only; it does not select or instantiate an
    /// execution kernel.
    #[must_use]
    pub fn get(&self, semantic: &SemanticId) -> Option<&SemanticId> {
        self.entries.iter().find(|candidate| *candidate == semantic)
    }

    /// Whether an exact semantic rule identity is cataloged.
    #[must_use]
    pub fn contains(&self, semantic: &SemanticId) -> bool {
        self.get(semantic).is_some()
    }

    /// Stable, order-independent textual representation of this registry.
    ///
    /// The record intentionally contains no kernel, backend, device, benchmark,
    /// or external-evidence field.
    #[must_use]
    pub fn canonical_record(&self) -> String {
        let mut record = format!(
            "flat-semantic-registry-v{REGISTRY_CONTRACT_VERSION};count={}\n",
            self.entries.len()
        );
        for semantic in &self.entries {
            let _ = writeln!(
                record,
                "family={};name={};revision={}",
                family_tag(semantic.family()),
                semantic.name(),
                semantic.revision()
            );
        }
        record
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

/// Fail-closed registry construction errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticRegistryError {
    /// A future semantic family is not yet assigned a canonical registry tag.
    UnsupportedFamily {
        /// Identity that could not be canonically registered.
        semantic: SemanticId,
    },
    /// The same semantic rule identity appeared more than once.
    DuplicateSemantic {
        /// Duplicated stable identity.
        semantic: SemanticId,
    },
}

impl Display for SemanticRegistryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFamily { semantic } => write!(
                formatter,
                "semantic family for {} revision {} is not supported by registry v{}",
                semantic.name(),
                semantic.revision(),
                REGISTRY_CONTRACT_VERSION
            ),
            Self::DuplicateSemantic { semantic } => write!(
                formatter,
                "semantic {} revision {} is registered more than once",
                semantic.name(),
                semantic.revision()
            ),
        }
    }
}

impl std::error::Error for SemanticRegistryError {}

const fn family_rank(family: SemanticFamily) -> u8 {
    match family {
        SemanticFamily::StandardSoftmax => 0,
        SemanticFamily::DifferentialSigned => 1,
        SemanticFamily::ToeplitzStructured => 2,
        SemanticFamily::ProlateConcentration => 3,
        SemanticFamily::GroundStateGreen => 4,
        SemanticFamily::SpectralFlow => 5,
        SemanticFamily::RecurrentMemory => 6,
        SemanticFamily::Hybrid => 7,
        SemanticFamily::Experimental => 8,
        _ => u8::MAX,
    }
}

const fn family_tag(family: SemanticFamily) -> &'static str {
    match family {
        SemanticFamily::StandardSoftmax => "standard-softmax",
        SemanticFamily::DifferentialSigned => "differential-signed",
        SemanticFamily::ToeplitzStructured => "toeplitz-structured",
        SemanticFamily::ProlateConcentration => "prolate-concentration",
        SemanticFamily::GroundStateGreen => "ground-state-green",
        SemanticFamily::SpectralFlow => "spectral-flow",
        SemanticFamily::RecurrentMemory => "recurrent-memory",
        SemanticFamily::Hybrid => "hybrid",
        SemanticFamily::Experimental => "experimental",
        _ => "unsupported",
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
    use flat_semantic::v1::StandardSoftmaxSemantic;

    fn semantic(family: SemanticFamily, name: &str, revision: u32) -> SemanticId {
        SemanticId::new(family, name, revision).unwrap()
    }

    #[test]
    fn canonical_registry_is_independent_of_input_order() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let toeplitz = semantic(SemanticFamily::ToeplitzStructured, "toeplitz-research", 3);
        let recurrent = semantic(SemanticFamily::RecurrentMemory, "delta-memory", 2);

        let forward = SemanticRegistry::new([
            recurrent.clone(),
            standard.clone(),
            toeplitz.clone(),
        ])
        .unwrap();
        let reversed = SemanticRegistry::new([toeplitz, standard, recurrent]).unwrap();

        assert_eq!(forward.entries(), reversed.entries());
        assert_eq!(forward.canonical_record(), reversed.canonical_record());
        assert_eq!(forward.stable_fingerprint(), reversed.stable_fingerprint());
    }

    #[test]
    fn duplicate_identity_fails_closed() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let error = SemanticRegistry::new([standard.clone(), standard.clone()]).unwrap_err();
        assert_eq!(
            error,
            SemanticRegistryError::DuplicateSemantic { semantic: standard }
        );
    }

    #[test]
    fn semantic_revisions_can_coexist_and_resolve_exactly() {
        let first = semantic(SemanticFamily::Experimental, "candidate", 1);
        let second = semantic(SemanticFamily::Experimental, "candidate", 2);
        let registry = SemanticRegistry::new([second.clone(), first.clone()]).unwrap();

        assert_eq!(registry.len(), 2);
        assert!(registry.contains(&first));
        assert!(registry.contains(&second));
        assert_eq!(registry.get(&first), Some(&first));
    }

    #[test]
    fn registry_indexes_rule_identity_not_instance_configuration() {
        let bidirectional = StandardSoftmaxSemantic::new(false, 0.5).unwrap();
        let causal = StandardSoftmaxSemantic::new(true, 0.25).unwrap();
        let bidirectional_id = bidirectional.descriptor().id().clone();
        let causal_id = causal.descriptor().id().clone();

        assert_eq!(bidirectional_id, causal_id);
        assert_ne!(
            bidirectional.stable_fingerprint(),
            causal.stable_fingerprint()
        );

        let registry = SemanticRegistry::new([bidirectional_id.clone()]).unwrap();
        assert!(registry.contains(&causal_id));
        assert_eq!(registry.get(&causal_id), Some(&bidirectional_id));
    }

    #[test]
    fn execution_choice_is_not_part_of_registry_identity() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum TestKernel {
            Portable,
            Optimized,
        }

        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let registry = SemanticRegistry::new([standard]).unwrap();
        let fingerprint = registry.stable_fingerprint();
        let plans = [
            (fingerprint, TestKernel::Portable),
            (fingerprint, TestKernel::Optimized),
        ];

        assert_eq!(plans[0].0, plans[1].0);
        assert_ne!(plans[0].1, plans[1].1);
        let record = registry.canonical_record();
        assert!(!record.contains("Portable"));
        assert!(!record.contains("Optimized"));
    }
}
