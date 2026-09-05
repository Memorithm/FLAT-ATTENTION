//! Backend-neutral decomposition of FLAT semantic mechanisms.
//!
//! This crate makes the mathematical pieces of a semantic rule independently
//! observable without turning them into a kernel dispatch table. Execution,
//! device identity, kernel identity and benchmark evidence deliberately remain
//! outside this contract.
//!
//! The historical StandardSoftmax fast path does not depend on this crate. A
//! caller opts into this metadata explicitly, so constructing the decomposition
//! cannot add hidden allocation or dynamic-dispatch cost to existing execution.

#![forbid(unsafe_code)]

use core::fmt;

use flat_semantic::v1::{SemanticDescriptor, StandardSoftmaxSemantic};

/// Version of the backend-neutral mechanism-decomposition contract.
pub const MECHANISM_CONTRACT_VERSION: u16 = 1;

/// Mathematical role of one decomposed semantic component.
///
/// This classification is metadata only. A component kind never selects or
/// implies an execution kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MechanismComponentKind {
    /// How Q/K/V or alternative state tensors enter the semantic rule.
    Projection,
    /// How interaction scores or structured coefficients are formed.
    Score,
    /// How scores/coefficients are normalized, gated or constrained.
    Normalization,
    /// How weighted/structured information is combined into the output/state.
    Mixing,
    /// Numerical stability, accumulation and conditioning semantics.
    NumericalPolicy,
}

impl MechanismComponentKind {
    const fn stable_name(self) -> &'static str {
        match self {
            Self::Projection => "projection",
            Self::Score => "score",
            Self::Normalization => "normalization",
            Self::Mixing => "mixing",
            Self::NumericalPolicy => "numerical-policy",
        }
    }
}

/// Stable identity of one mathematical mechanism component.
///
/// Names are lowercase ASCII slugs. Revisions start at one. Kernel/device
/// identifiers must not be encoded into component names; execution identity is
/// intentionally a separate concern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MechanismComponentId {
    kind: MechanismComponentKind,
    name: String,
    revision: u32,
}

impl MechanismComponentId {
    /// Construct and validate one component identity.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error when the name is empty/invalid or the
    /// revision is zero.
    pub fn new(
        kind: MechanismComponentKind,
        name: impl Into<String>,
        revision: u32,
    ) -> Result<Self, MechanismComponentIdentityError> {
        let name = name.into();
        if name.is_empty() {
            return Err(MechanismComponentIdentityError::EmptyName);
        }
        if !name.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'-' | b'_' | b'.')
        }) {
            return Err(MechanismComponentIdentityError::InvalidName);
        }
        if revision == 0 {
            return Err(MechanismComponentIdentityError::ZeroRevision);
        }
        Ok(Self {
            kind,
            name,
            revision,
        })
    }

    /// Mathematical role represented by this identity.
    #[must_use]
    pub const fn kind(&self) -> MechanismComponentKind {
        self.kind
    }

    /// Stable component slug.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Component revision.
    #[must_use]
    pub const fn revision(&self) -> u32 {
        self.revision
    }

    fn canonical_fragment(&self) -> String {
        format!(
            "{}={}@{}",
            self.kind.stable_name(),
            self.name,
            self.revision
        )
    }
}

/// Validation errors for mechanism component identity metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MechanismComponentIdentityError {
    /// Component slug is empty.
    EmptyName,
    /// Component slug contains unsupported characters.
    InvalidName,
    /// Revision zero is reserved and invalid.
    ZeroRevision,
}

impl fmt::Display for MechanismComponentIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => formatter.write_str("mechanism component name must not be empty"),
            Self::InvalidName => {
                formatter.write_str("mechanism component name is not a valid stable slug")
            }
            Self::ZeroRevision => {
                formatter.write_str("mechanism component revision must be non-zero")
            }
        }
    }
}

impl std::error::Error for MechanismComponentIdentityError {}

/// Complete backend-neutral mechanism decomposition of one semantic rule.
///
/// State semantics, mask semantics, saved-state schema and weight semantics stay
/// authoritative in [`SemanticDescriptor`]. This structure adds the five
/// decomposition axes that were previously implicit. Keeping the semantic
/// descriptor intact avoids duplicating or silently reconciling those existing
/// contracts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MechanismDescriptor {
    semantic: SemanticDescriptor,
    projection: MechanismComponentId,
    score: MechanismComponentId,
    normalization: MechanismComponentId,
    mixing: MechanismComponentId,
    numerical_policy: MechanismComponentId,
}

impl MechanismDescriptor {
    /// Construct a decomposition after verifying that every supplied component
    /// occupies the correct mathematical role.
    ///
    /// # Errors
    ///
    /// Fails closed when a component identity is supplied in the wrong slot.
    pub fn new(
        semantic: SemanticDescriptor,
        projection: MechanismComponentId,
        score: MechanismComponentId,
        normalization: MechanismComponentId,
        mixing: MechanismComponentId,
        numerical_policy: MechanismComponentId,
    ) -> Result<Self, MechanismDescriptorError> {
        require_kind(&projection, MechanismComponentKind::Projection)?;
        require_kind(&score, MechanismComponentKind::Score)?;
        require_kind(&normalization, MechanismComponentKind::Normalization)?;
        require_kind(&mixing, MechanismComponentKind::Mixing)?;
        require_kind(
            &numerical_policy,
            MechanismComponentKind::NumericalPolicy,
        )?;
        Ok(Self {
            semantic,
            projection,
            score,
            normalization,
            mixing,
            numerical_policy,
        })
    }

    /// Existing semantic identity/state/mask/saved-state contract.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticDescriptor {
        &self.semantic
    }

    /// Projection contract identity.
    #[must_use]
    pub const fn projection(&self) -> &MechanismComponentId {
        &self.projection
    }

    /// Score-rule identity.
    #[must_use]
    pub const fn score(&self) -> &MechanismComponentId {
        &self.score
    }

    /// Normalization/weighting-rule identity.
    #[must_use]
    pub const fn normalization(&self) -> &MechanismComponentId {
        &self.normalization
    }

    /// Mixing-operator identity.
    #[must_use]
    pub const fn mixing(&self) -> &MechanismComponentId {
        &self.mixing
    }

    /// Numerical-policy identity.
    #[must_use]
    pub const fn numerical_policy(&self) -> &MechanismComponentId {
        &self.numerical_policy
    }

    /// Deterministic component-only provenance record.
    ///
    /// This record deliberately excludes kernel/device/benchmark identity. The
    /// semantic instance may carry additional parameters, so callers that need
    /// a complete instance fingerprint should use a semantic-specific wrapper
    /// such as [`StandardSoftmaxMechanism::canonical_record`].
    #[must_use]
    pub fn canonical_component_record(&self) -> String {
        format!(
            "flat-mechanism-v{MECHANISM_CONTRACT_VERSION};semantic={}@{};{};{};{};{};{}",
            self.semantic.id().name(),
            self.semantic.id().revision(),
            self.projection.canonical_fragment(),
            self.score.canonical_fragment(),
            self.normalization.canonical_fragment(),
            self.mixing.canonical_fragment(),
            self.numerical_policy.canonical_fragment(),
        )
    }
}

/// Validation errors for a complete mechanism descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MechanismDescriptorError {
    /// A component was supplied in a slot belonging to another role.
    ComponentKindMismatch {
        /// Role required by the descriptor field.
        expected: MechanismComponentKind,
        /// Role carried by the supplied component identity.
        actual: MechanismComponentKind,
    },
}

impl fmt::Display for MechanismDescriptorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComponentKindMismatch { expected, actual } => write!(
                formatter,
                "mechanism component kind mismatch: expected {}, got {}",
                expected.stable_name(),
                actual.stable_name(),
            ),
        }
    }
}

impl std::error::Error for MechanismDescriptorError {}

fn require_kind(
    component: &MechanismComponentId,
    expected: MechanismComponentKind,
) -> Result<(), MechanismDescriptorError> {
    if component.kind() != expected {
        return Err(MechanismDescriptorError::ComponentKindMismatch {
            expected,
            actual: component.kind(),
        });
    }
    Ok(())
}

/// Explicit Phase-A decomposition of the historical StandardSoftmax semantic.
///
/// The wrapped semantic remains the sole execution authority. This value only
/// supplies independently observable mechanism identities for provenance,
/// research instrumentation and later semantic comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardSoftmaxMechanism {
    semantic: StandardSoftmaxSemantic,
}

impl StandardSoftmaxMechanism {
    /// Wrap an already validated StandardSoftmax semantic instance.
    #[must_use]
    pub const fn new(semantic: StandardSoftmaxSemantic) -> Self {
        Self { semantic }
    }

    /// The unchanged executable StandardSoftmax semantic.
    #[must_use]
    pub const fn semantic(self) -> StandardSoftmaxSemantic {
        self.semantic
    }

    /// Backend-neutral decomposition for StandardSoftmax revision 1.
    ///
    /// Component names describe mathematical/reference behavior, not a WGPU or
    /// other kernel realization.
    #[must_use]
    pub fn descriptor(self) -> MechanismDescriptor {
        MechanismDescriptor::new(
            self.semantic.descriptor(),
            component(MechanismComponentKind::Projection, "direct-qkv", 1),
            component(
                MechanismComponentKind::Score,
                "scaled-dot-product",
                1,
            ),
            component(MechanismComponentKind::Normalization, "row-softmax", 1),
            component(MechanismComponentKind::Mixing, "weighted-value-sum", 1),
            component(
                MechanismComponentKind::NumericalPolicy,
                "legacy-standard-softmax-reference",
                1,
            ),
        )
        .expect("built-in StandardSoftmax mechanism component kinds are valid")
    }

    /// Complete deterministic provenance record for this StandardSoftmax
    /// instance and its mechanism decomposition.
    ///
    /// The existing semantic record binds causal/bidirectional masking and the
    /// exact score-scale bits. The appended mechanism record binds the explicit
    /// Phase-A component decomposition. No execution identity is included.
    #[must_use]
    pub fn canonical_record(self) -> String {
        format!(
            "{};{}",
            self.semantic.canonical_record(),
            self.descriptor().canonical_component_record(),
        )
    }

    /// Stable FNV-1a-64 fingerprint of [`Self::canonical_record`].
    ///
    /// This is a deterministic provenance aid, not a cryptographic digest.
    #[must_use]
    pub fn stable_fingerprint(self) -> u64 {
        fnv1a64(self.canonical_record().as_bytes())
    }
}

fn component(
    kind: MechanismComponentKind,
    name: &'static str,
    revision: u32,
) -> MechanismComponentId {
    MechanismComponentId::new(kind, name, revision)
        .expect("built-in mechanism constants form valid stable identities")
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_semantic::v1::{
        MaskSemantics, SavedStateContract, SemanticFamily, StateSemantics, WeightSemantics,
    };

    #[test]
    fn component_identity_is_strict_and_typed() {
        let score = MechanismComponentId::new(
            MechanismComponentKind::Score,
            "scaled-dot-product",
            1,
        )
        .unwrap();
        assert_eq!(score.kind(), MechanismComponentKind::Score);
        assert_eq!(score.name(), "scaled-dot-product");
        assert_eq!(score.revision(), 1);
        assert_eq!(
            MechanismComponentId::new(MechanismComponentKind::Score, "", 1).unwrap_err(),
            MechanismComponentIdentityError::EmptyName
        );
        assert_eq!(
            MechanismComponentId::new(MechanismComponentKind::Score, "Dot Product", 1)
                .unwrap_err(),
            MechanismComponentIdentityError::InvalidName
        );
        assert_eq!(
            MechanismComponentId::new(MechanismComponentKind::Score, "dot-product", 0)
                .unwrap_err(),
            MechanismComponentIdentityError::ZeroRevision
        );
    }

    #[test]
    fn descriptor_rejects_component_in_wrong_role() {
        let semantic = StandardSoftmaxSemantic::new(true, 0.5).unwrap();
        let err = MechanismDescriptor::new(
            semantic.descriptor(),
            component(MechanismComponentKind::Score, "wrong-role", 1),
            component(MechanismComponentKind::Score, "scaled-dot-product", 1),
            component(MechanismComponentKind::Normalization, "row-softmax", 1),
            component(MechanismComponentKind::Mixing, "weighted-value-sum", 1),
            component(MechanismComponentKind::NumericalPolicy, "reference", 1),
        )
        .unwrap_err();
        assert_eq!(
            err,
            MechanismDescriptorError::ComponentKindMismatch {
                expected: MechanismComponentKind::Projection,
                actual: MechanismComponentKind::Score,
            }
        );
    }

    #[test]
    fn standard_softmax_decomposition_preserves_existing_semantic_contract() {
        let mechanism = StandardSoftmaxMechanism::new(
            StandardSoftmaxSemantic::new(true, 0.625).unwrap(),
        );
        let descriptor = mechanism.descriptor();
        assert_eq!(
            descriptor.semantic().id().family(),
            SemanticFamily::StandardSoftmax
        );
        assert_eq!(descriptor.semantic().id().name(), "standard-softmax");
        assert_eq!(descriptor.semantic().mask(), MaskSemantics::Causal);
        assert_eq!(descriptor.semantic().state(), StateSemantics::Stateless);
        assert_eq!(
            descriptor.semantic().weights(),
            WeightSemantics::ProbabilitySimplex
        );
        assert_eq!(
            descriptor.semantic().saved_state(),
            SavedStateContract::LogSumExp
        );
        assert_eq!(descriptor.projection().name(), "direct-qkv");
        assert_eq!(descriptor.score().name(), "scaled-dot-product");
        assert_eq!(descriptor.normalization().name(), "row-softmax");
        assert_eq!(descriptor.mixing().name(), "weighted-value-sum");
        assert_eq!(
            descriptor.numerical_policy().name(),
            "legacy-standard-softmax-reference"
        );
    }

    #[test]
    fn fingerprint_binds_semantic_parameters_but_not_execution_identity() {
        let causal = StandardSoftmaxMechanism::new(
            StandardSoftmaxSemantic::new(true, 0.625).unwrap(),
        );
        let bidirectional = StandardSoftmaxMechanism::new(
            StandardSoftmaxSemantic::new(false, 0.625).unwrap(),
        );
        let other_scale = StandardSoftmaxMechanism::new(
            StandardSoftmaxSemantic::new(true, 0.5).unwrap(),
        );
        assert_eq!(causal.stable_fingerprint(), causal.stable_fingerprint());
        assert_ne!(causal.stable_fingerprint(), bidirectional.stable_fingerprint());
        assert_ne!(causal.stable_fingerprint(), other_scale.stable_fingerprint());
        let record = causal.canonical_record();
        assert!(!record.contains("wgpu"));
        assert!(!record.contains("device"));
        assert!(!record.contains("kernel"));
        assert!(!record.contains("benchmark"));
    }
}
