//! Backend-neutral semantic control-plane contracts for FLAT-ATTENTION.
//!
//! This crate separates **what an interaction rule computes** from **which
//! kernel executes it**. The existing `flat-attention` crate remains the
//! specialized execution layer and is not required to construct or dispatch
//! through these control-plane objects on its historical fast path.
//!
//! Family enumeration is classification only. A variant does not imply that a
//! scalar oracle, GPU implementation, or production route exists for that
//! family. Executable support must be represented by a concrete semantic type
//! such as [`v1::StandardSoftmaxSemantic`] and qualified independently.

#![forbid(unsafe_code)]

/// Versioned backend-neutral semantic contracts.
pub mod v1 {
    use core::fmt;

    use flat_attention::{
        forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionError,
        FlatAttentionOutput,
    };

    /// Version of this semantic control-plane schema.
    pub const SEMANTIC_CONTRACT_VERSION: u16 = 1;

    /// Broad semantic family classification.
    ///
    /// This enum is not a dispatch table and does not imply executable support.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum SemanticFamily {
        /// Canonical probability-simplex softmax attention.
        StandardSoftmax,
        /// Differential or otherwise signed attention/mixing.
        DifferentialSigned,
        /// Toeplitz or relative structured operators.
        ToeplitzStructured,
        /// Prolate or spectrally concentrated operators.
        ProlateConcentration,
        /// Ground-state / Green-kernel operators.
        GroundStateGreen,
        /// Spectral-flow-derived interaction rules.
        SpectralFlow,
        /// Explicit recurrent or delta-memory semantics.
        RecurrentMemory,
        /// Explicit composition of semantic mechanisms.
        Hybrid,
        /// Research candidate without a stable family assignment.
        Experimental,
    }

    /// Visibility semantics that are part of the mathematical contract.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum MaskSemantics {
        /// Every declared key is visible.
        Bidirectional,
        /// Keys strictly after the query position are invisible.
        Causal,
        /// Visibility is supplied by an external semantic artifact.
        External,
    }

    /// State carried by the mathematical interaction rule.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum StateSemantics {
        /// No semantic state is carried between declared steps.
        Stateless,
        /// Explicit recurrent state is part of the semantic definition.
        Recurrent,
    }

    /// High-level constraint on coefficients used to mix information.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum WeightSemantics {
        /// Non-negative coefficients normalized to unit row sum.
        ProbabilitySimplex,
        /// Signed coefficients are permitted.
        Signed,
        /// Structured linear coefficients without a simplex requirement.
        StructuredLinear,
        /// Coefficients depend on explicit carried state.
        StateDependent,
    }

    /// Saved-state schema required by a semantic forward result.
    ///
    /// `None` is first-class: generic FLAT semantics do not implicitly own an
    /// LSE statistic merely because StandardSoftmax does.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum SavedStateContract {
        /// No saved semantic statistic is required.
        None,
        /// Per-query log-sum-exp values are retained.
        LogSumExp,
    }

    /// Stable identity of one FLAT semantic rule.
    ///
    /// Kernel names, device identity, benchmark results, and external evidence
    /// are intentionally excluded.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct SemanticId {
        family: SemanticFamily,
        name: String,
        revision: u32,
    }

    impl SemanticId {
        /// Construct a validated semantic identity.
        ///
        /// Names are stable lowercase ASCII slugs using letters, digits, `-`,
        /// `_`, or `.`. Revisions start at one.
        ///
        /// # Errors
        ///
        /// Returns a typed validation error for an invalid name or revision.
        pub fn new(
            family: SemanticFamily,
            name: impl Into<String>,
            revision: u32,
        ) -> Result<Self, SemanticIdentityError> {
            let name = name.into();
            if name.is_empty() {
                return Err(SemanticIdentityError::EmptyName);
            }
            if !name.bytes().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'-' | b'_' | b'.')
            }) {
                return Err(SemanticIdentityError::InvalidName);
            }
            if revision == 0 {
                return Err(SemanticIdentityError::ZeroRevision);
            }
            Ok(Self {
                family,
                name,
                revision,
            })
        }

        /// Semantic family classification.
        #[must_use]
        pub const fn family(&self) -> SemanticFamily {
            self.family
        }

        /// Stable semantic slug.
        #[must_use]
        pub fn name(&self) -> &str {
            &self.name
        }

        /// Semantic rule revision.
        #[must_use]
        pub const fn revision(&self) -> u32 {
            self.revision
        }
    }

    /// Reference-level properties of a semantic rule, independent of execution.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct SemanticDescriptor {
        id: SemanticId,
        mask: MaskSemantics,
        state: StateSemantics,
        weights: WeightSemantics,
        saved_state: SavedStateContract,
    }

    impl SemanticDescriptor {
        /// Construct a descriptor from already-validated semantic components.
        #[must_use]
        pub const fn new(
            id: SemanticId,
            mask: MaskSemantics,
            state: StateSemantics,
            weights: WeightSemantics,
            saved_state: SavedStateContract,
        ) -> Self {
            Self {
                id,
                mask,
                state,
                weights,
                saved_state,
            }
        }

        /// Stable rule identity.
        #[must_use]
        pub const fn id(&self) -> &SemanticId {
            &self.id
        }

        /// Visibility semantics.
        #[must_use]
        pub const fn mask(&self) -> MaskSemantics {
            self.mask
        }

        /// State semantics.
        #[must_use]
        pub const fn state(&self) -> StateSemantics {
            self.state
        }

        /// Mixing-weight semantics.
        #[must_use]
        pub const fn weights(&self) -> WeightSemantics {
            self.weights
        }

        /// Saved forward-state schema.
        #[must_use]
        pub const fn saved_state(&self) -> SavedStateContract {
            self.saved_state
        }
    }

    /// Validation errors for semantic identity metadata.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SemanticIdentityError {
        /// Semantic slug is empty.
        EmptyName,
        /// Semantic slug contains unsupported characters.
        InvalidName,
        /// Revision zero is reserved and invalid.
        ZeroRevision,
    }

    impl fmt::Display for SemanticIdentityError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::EmptyName => formatter.write_str("semantic name must not be empty"),
                Self::InvalidName => {
                    formatter.write_str("semantic name is not a valid stable slug")
                }
                Self::ZeroRevision => formatter.write_str("semantic revision must be non-zero"),
            }
        }
    }

    impl std::error::Error for SemanticIdentityError {}

    /// Concrete research semantic backed by FLAT's qualified scalar
    /// structured-history oracle.
    ///
    /// This value is reference/control-plane support only. Constructing or
    /// executing it does not register a WGPU candidate, select a kernel, or
    /// alter the historical StandardSoftmax path.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NonlocalHistorySoftmaxSemantic {
        attention: FlatAttentionConfig,
        history: flat_attention::api::research_nonlocal::NonlocalAttentionConfig,
    }

    impl NonlocalHistorySoftmaxSemantic {
        /// Construct revision 1 after validating its causal/history contract.
        ///
        /// # Errors
        ///
        /// Rejects non-causal requests and invalid structured-history
        /// configuration exactly as the scalar research oracle does.
        pub fn new(
            attention: FlatAttentionConfig,
            history: flat_attention::api::research_nonlocal::NonlocalAttentionConfig,
        ) -> Result<Self, flat_attention::api::research_nonlocal::NonlocalAttentionError> {
            use flat_attention::api::research_nonlocal::NonlocalAttentionError;

            history.validate()?;
            if !attention.causal {
                return Err(NonlocalAttentionError::NonCausalUnsupported);
            }
            Ok(Self { attention, history })
        }

        /// Stable rule identity for research semantic revision 1.
        ///
        /// The identity contains no execution, device, kernel, or benchmark
        /// choice and can therefore be used by the registry without depending
        /// on the scalar-oracle request types.
        #[must_use]
        pub fn semantic_id() -> SemanticId {
            use flat_attention::api::research_nonlocal::{
                NONLOCAL_ATTENTION_SEMANTIC_NAME, NONLOCAL_ATTENTION_SEMANTIC_REVISION,
            };

            SemanticId::new(
                SemanticFamily::Experimental,
                NONLOCAL_ATTENTION_SEMANTIC_NAME,
                NONLOCAL_ATTENTION_SEMANTIC_REVISION,
            )
            .expect("FLAT nonlocal semantic constants form a valid stable identity")
        }

        /// Stable reference-level descriptor for the research semantic.
        #[must_use]
        pub fn descriptor(self) -> SemanticDescriptor {
            SemanticDescriptor::new(
                Self::semantic_id(),
                MaskSemantics::Causal,
                StateSemantics::Stateless,
                WeightSemantics::ProbabilitySimplex,
                SavedStateContract::LogSumExp,
            )
        }

        /// Execute only the deterministic scalar research oracle.
        ///
        /// # Errors
        ///
        /// Propagates the research oracle's validation and numerical errors.
        pub fn execute(
            self,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            shape: flat_attention::AsymmetricGroupedAttentionShape,
        ) -> Result<
            flat_attention::api::research_nonlocal::NonlocalAttentionOutput,
            flat_attention::api::research_nonlocal::NonlocalAttentionError,
        > {
            flat_attention::api::research_nonlocal::forward_reference_nonlocal_history(
                q,
                k,
                v,
                shape,
                self.attention,
                self.history,
            )
        }
    }

    /// Concrete executable S0 semantic backed by FLAT's historical scalar
    /// StandardSoftmax oracle.
    ///
    /// The resolved score scale is stored explicitly so two parameterizations
    /// remain distinguishable even though they share the same rule identity.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct StandardSoftmaxSemantic {
        causal: bool,
        score_scale: f32,
    }

    impl StandardSoftmaxSemantic {
        /// Construct an explicit StandardSoftmax parameterization.
        ///
        /// # Errors
        ///
        /// Rejects non-finite or non-positive score scales using the historical
        /// FLAT error contract.
        pub fn new(causal: bool, score_scale: f32) -> Result<Self, FlatAttentionError> {
            if !score_scale.is_finite() || score_scale <= 0.0 {
                return Err(FlatAttentionError::InvalidScale(score_scale));
            }
            Ok(Self {
                causal,
                score_scale,
            })
        }

        /// Resolve the historical FLAT configuration into an explicit semantic
        /// parameterization for a declared head dimension.
        ///
        /// # Errors
        ///
        /// Propagates the historical configuration validation errors.
        pub fn from_flat_config(
            config: FlatAttentionConfig,
            head_dim: usize,
        ) -> Result<Self, FlatAttentionError> {
            Self::new(config.causal, config.resolved_scale(head_dim)?)
        }

        /// Stable reference-level descriptor. Constructing this control-plane
        /// value allocates its small identity string; the existing specialized
        /// execution path never calls this method implicitly.
        #[must_use]
        pub fn descriptor(self) -> SemanticDescriptor {
            SemanticDescriptor::new(
                SemanticId {
                    family: SemanticFamily::StandardSoftmax,
                    name: "standard-softmax".into(),
                    revision: 1,
                },
                if self.causal {
                    MaskSemantics::Causal
                } else {
                    MaskSemantics::Bidirectional
                },
                StateSemantics::Stateless,
                WeightSemantics::ProbabilitySimplex,
                SavedStateContract::LogSumExp,
            )
        }

        /// Whether autoregressive visibility is part of this instance.
        #[must_use]
        pub const fn causal(self) -> bool {
            self.causal
        }

        /// Explicit score scale used by this instance.
        #[must_use]
        pub const fn score_scale(self) -> f32 {
            self.score_scale
        }

        /// Convert back to the historical execution configuration without
        /// changing its resolved mathematical parameters.
        #[must_use]
        pub const fn to_flat_config(self) -> FlatAttentionConfig {
            FlatAttentionConfig {
                causal: self.causal,
                softmax_scale: Some(self.score_scale),
            }
        }

        /// Deterministic canonical semantic-instance record suitable for
        /// provenance/cache keys. It contains no kernel or device identity.
        #[must_use]
        pub fn canonical_record(self) -> String {
            format!(
                "flat-semantic-v{SEMANTIC_CONTRACT_VERSION};family=standard-softmax;revision=1;mask={};scale_bits={:08x}",
                if self.causal { "causal" } else { "bidirectional" },
                self.score_scale.to_bits(),
            )
        }

        /// Stable FNV-1a-64 fingerprint of [`Self::canonical_record`].
        ///
        /// This is a deterministic identity aid, not a cryptographic digest.
        #[must_use]
        pub fn stable_fingerprint(self) -> u64 {
            fnv1a64(self.canonical_record().as_bytes())
        }

        /// Execute through the historical scalar oracle and expose its result
        /// through the generic semantic output contract.
        ///
        /// # Errors
        ///
        /// Propagates every historical FLAT scalar-reference validation error.
        pub fn forward_reference(
            self,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            shape: AttentionShape,
        ) -> Result<SemanticForwardOutput<f32>, FlatAttentionError> {
            forward_reference(q, k, v, shape, self.to_flat_config()).map(Into::into)
        }
    }

    /// Runtime saved state carried by a generic semantic forward result.
    ///
    /// Additional semantic-specific variants may be added in later contract
    /// revisions. Generic callers must not assume LSE is universally present.
    #[derive(Debug, Clone, PartialEq)]
    #[non_exhaustive]
    pub enum SemanticSavedState<T> {
        /// No saved semantic state.
        None,
        /// Per-query log-sum-exp values for StandardSoftmax.
        LogSumExp(Vec<T>),
    }

    impl<T> SemanticSavedState<T> {
        /// Saved-state schema represented by this value.
        #[must_use]
        pub const fn contract(&self) -> SavedStateContract {
            match self {
                Self::None => SavedStateContract::None,
                Self::LogSumExp(_) => SavedStateContract::LogSumExp,
            }
        }
    }

    /// Generic semantic forward result: primary output plus semantic-specific
    /// saved state.
    #[derive(Debug, Clone, PartialEq)]
    pub struct SemanticForwardOutput<T> {
        output: Vec<T>,
        saved_state: SemanticSavedState<T>,
    }

    impl<T> SemanticForwardOutput<T> {
        /// Construct a semantic result from independently-owned output/state.
        #[must_use]
        pub const fn new(output: Vec<T>, saved_state: SemanticSavedState<T>) -> Self {
            Self {
                output,
                saved_state,
            }
        }

        /// Primary semantic output tensor.
        #[must_use]
        pub fn output(&self) -> &[T] {
            &self.output
        }

        /// Semantic-specific saved state.
        #[must_use]
        pub const fn saved_state(&self) -> &SemanticSavedState<T> {
            &self.saved_state
        }

        /// Consume the wrapper without copying either vector.
        #[must_use]
        pub fn into_parts(self) -> (Vec<T>, SemanticSavedState<T>) {
            (self.output, self.saved_state)
        }
    }

    impl From<FlatAttentionOutput> for SemanticForwardOutput<f32> {
        fn from(value: FlatAttentionOutput) -> Self {
            Self::new(value.output, SemanticSavedState::LogSumExp(value.lse))
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
        use flat_attention::RuntimeKernelId;

        fn shape() -> AttentionShape {
            AttentionShape {
                batch: 1,
                heads: 1,
                seq_len: 3,
                head_dim: 2,
            }
        }

        #[test]
        fn identity_validation_is_fail_closed() {
            assert_eq!(
                SemanticId::new(SemanticFamily::Experimental, "", 1).unwrap_err(),
                SemanticIdentityError::EmptyName
            );
            assert_eq!(
                SemanticId::new(SemanticFamily::Experimental, "Bad Name", 1).unwrap_err(),
                SemanticIdentityError::InvalidName
            );
            assert_eq!(
                SemanticId::new(SemanticFamily::Experimental, "candidate", 0).unwrap_err(),
                SemanticIdentityError::ZeroRevision
            );
        }

        #[test]
        fn standard_softmax_adapter_is_bit_exact_with_legacy_scalar_oracle() {
            let shape = shape();
            let q = vec![0.2, -0.3, 0.1, 0.5, -0.4, 0.7];
            let k = vec![0.6, -0.2, -0.8, 0.3, 0.4, 0.9];
            let v = vec![1.0, -2.0, 0.5, 0.25, -0.75, 1.5];
            let config = FlatAttentionConfig {
                causal: true,
                softmax_scale: Some(0.625),
            };
            let legacy = forward_reference(&q, &k, &v, shape, config).unwrap();
            let semantic =
                StandardSoftmaxSemantic::from_flat_config(config, shape.head_dim).unwrap();
            let generic = semantic.forward_reference(&q, &k, &v, shape).unwrap();
            let (output, saved) = generic.into_parts();
            assert_eq!(output, legacy.output);
            match saved {
                SemanticSavedState::LogSumExp(lse) => assert_eq!(lse, legacy.lse),
                _ => panic!("StandardSoftmax must retain LSE"),
            }
        }

        #[test]
        fn generic_saved_state_does_not_require_lse() {
            let output = SemanticForwardOutput::new(vec![1.0_f32], SemanticSavedState::None);
            assert_eq!(output.saved_state().contract(), SavedStateContract::None);
            let semantic = StandardSoftmaxSemantic::new(false, 1.0).unwrap();
            assert_eq!(
                semantic.descriptor().saved_state(),
                SavedStateContract::LogSumExp
            );
        }

        #[test]
        fn semantic_identity_is_independent_of_kernel_selection() {
            let semantic = StandardSoftmaxSemantic::new(false, 0.5).unwrap();
            let record = semantic.canonical_record();
            let bindings = [
                (semantic.stable_fingerprint(), RuntimeKernelId::Q4Portable),
                (semantic.stable_fingerprint(), RuntimeKernelId::Q4Subgroup),
            ];
            assert_eq!(bindings[0].0, bindings[1].0);
            assert_ne!(bindings[0].1, bindings[1].1);
            assert!(!record.contains("Q4Portable"));
            assert!(!record.contains("Q4Subgroup"));
        }

        #[test]
        fn concrete_parameterization_changes_instance_fingerprint() {
            let baseline = StandardSoftmaxSemantic::new(false, 0.5).unwrap();
            let causal = StandardSoftmaxSemantic::new(true, 0.5).unwrap();
            let rescaled = StandardSoftmaxSemantic::new(false, 0.25).unwrap();
            assert_ne!(baseline.stable_fingerprint(), causal.stable_fingerprint());
            assert_ne!(baseline.stable_fingerprint(), rescaled.stable_fingerprint());
            assert_eq!(
                baseline.descriptor().id().family(),
                SemanticFamily::StandardSoftmax
            );
        }
    }
}
