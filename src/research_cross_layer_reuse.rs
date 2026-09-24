//! MAA-15a research-only cross-layer reuse identity and validation.
//!
//! This module implements the structural contract frozen by
//! `MAA_15A_CROSS_LAYER_REUSE_PREREGISTRATION.md`. It does not execute
//! attention, select candidates, reuse device buffers, or make a performance
//! claim. Its purpose is to make Full/Reindex/Reuse requests explicit and
//! fail closed when source representation, epoch, geometry, causal domain, or
//! candidate identity drifts.

use core::fmt;

use crate::{
    api::research_structural_routing::StructuralCandidateSet, AttentionShape, FlatAttentionError,
};

/// Version of the MAA-15a cross-layer reuse contract.
pub const CROSS_LAYER_REUSE_SCHEMA_VERSION: u32 = 1;

/// Maximum UTF-8 bytes accepted for opaque representation/policy identifiers.
pub const MAX_CROSS_LAYER_ID_BYTES: usize = 128;

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Coarse execution mode frozen by MAA-15a.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossLayerReuseMode {
    /// Current layer owns its representation and computes its own candidates.
    Full,
    /// Current layer reuses a declared representation but computes fresh candidates.
    Reindex,
    /// Current layer reuses both a declared representation and candidate identity.
    Reuse,
}

/// Reuse axes retained separately for evidence/accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossLayerReuseAxes {
    /// Reuse the declared main structural K/V representation.
    pub main_kv: bool,
    /// Reuse an indexer/selection-key representation where the caller has one.
    pub indexer_k: bool,
    /// Reuse the exact canonical candidate set.
    pub candidates: bool,
}

impl CrossLayerReuseAxes {
    /// No reuse: canonical `Full` axes.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            main_kv: false,
            indexer_k: false,
            candidates: false,
        }
    }

    /// Canonical `Reindex` axes. Candidate selection remains fresh.
    #[must_use]
    pub const fn reindex(reuse_indexer_k: bool) -> Self {
        Self {
            main_kv: true,
            indexer_k: reuse_indexer_k,
            candidates: false,
        }
    }

    /// Canonical `Reuse` axes. Candidate identity is reused exactly.
    #[must_use]
    pub const fn reuse(reuse_indexer_k: bool) -> Self {
        Self {
            main_kv: true,
            indexer_k: reuse_indexer_k,
            candidates: true,
        }
    }
}

/// Exact host-side identity of one layer materialization eligible for reuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerMaterializationIdentity {
    layer_id: u32,
    representation_id: String,
    representation_schema_version: u32,
    materialization_epoch: u64,
    shape: AttentionShape,
    causal: bool,
}

impl LayerMaterializationIdentity {
    /// Build a validated materialization identity.
    pub fn new(
        layer_id: u32,
        representation_id: impl Into<String>,
        representation_schema_version: u32,
        materialization_epoch: u64,
        shape: AttentionShape,
        causal: bool,
    ) -> Result<Self, CrossLayerReuseError> {
        validate_shape(shape)?;
        let representation_id = representation_id.into();
        validate_id("representation_id", &representation_id)?;
        if representation_schema_version == 0 {
            return Err(CrossLayerReuseError::ZeroRepresentationSchemaVersion);
        }
        Ok(Self {
            layer_id,
            representation_id,
            representation_schema_version,
            materialization_epoch,
            shape,
            causal,
        })
    }

    #[must_use]
    pub const fn layer_id(&self) -> u32 {
        self.layer_id
    }

    #[must_use]
    pub fn representation_id(&self) -> &str {
        &self.representation_id
    }

    #[must_use]
    pub const fn representation_schema_version(&self) -> u32 {
        self.representation_schema_version
    }

    #[must_use]
    pub const fn materialization_epoch(&self) -> u64 {
        self.materialization_epoch
    }

    #[must_use]
    pub const fn shape(&self) -> AttentionShape {
        self.shape
    }

    #[must_use]
    pub const fn causal(&self) -> bool {
        self.causal
    }
}

/// Destination-owned requirements for accepting a reused source materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerReuseRequirement {
    layer_id: u32,
    representation_id: String,
    representation_schema_version: u32,
    required_source_epoch: u64,
    shape: AttentionShape,
    causal: bool,
    reused_candidate_policy: Option<(String, u32)>,
}

impl LayerReuseRequirement {
    /// Construct a destination requirement.
    pub fn new(
        layer_id: u32,
        representation_id: impl Into<String>,
        representation_schema_version: u32,
        required_source_epoch: u64,
        shape: AttentionShape,
        causal: bool,
    ) -> Result<Self, CrossLayerReuseError> {
        validate_shape(shape)?;
        let representation_id = representation_id.into();
        validate_id("representation_id", &representation_id)?;
        if representation_schema_version == 0 {
            return Err(CrossLayerReuseError::ZeroRepresentationSchemaVersion);
        }
        Ok(Self {
            layer_id,
            representation_id,
            representation_schema_version,
            required_source_epoch,
            shape,
            causal,
            reused_candidate_policy: None,
        })
    }

    /// Require one exact policy identity when candidate reuse is requested.
    pub fn require_reused_candidate_policy(
        mut self,
        policy_id: impl Into<String>,
        policy_schema_version: u32,
    ) -> Result<Self, CrossLayerReuseError> {
        let policy_id = policy_id.into();
        validate_id("candidate_policy_id", &policy_id)?;
        if policy_schema_version == 0 {
            return Err(CrossLayerReuseError::ZeroCandidatePolicySchemaVersion);
        }
        self.reused_candidate_policy = Some((policy_id, policy_schema_version));
        Ok(self)
    }

    #[must_use]
    pub const fn layer_id(&self) -> u32 {
        self.layer_id
    }
}

/// Exact identity of one canonical structural candidate set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralCandidateIdentity {
    source_layer_id: u32,
    policy_id: String,
    policy_schema_version: u32,
    representation_epoch: u64,
    seq_len: usize,
    query_rows: usize,
    admitted_count: usize,
    fingerprint_fnv1a64: u64,
}

impl StructuralCandidateIdentity {
    /// Bind a candidate identity to the actual canonical host candidate set.
    pub fn from_candidates(
        source_layer_id: u32,
        policy_id: impl Into<String>,
        policy_schema_version: u32,
        representation_epoch: u64,
        candidates: &StructuralCandidateSet,
    ) -> Result<Self, CrossLayerReuseError> {
        let policy_id = policy_id.into();
        validate_id("candidate_policy_id", &policy_id)?;
        if policy_schema_version == 0 {
            return Err(CrossLayerReuseError::ZeroCandidatePolicySchemaVersion);
        }
        Ok(Self {
            source_layer_id,
            policy_id,
            policy_schema_version,
            representation_epoch,
            seq_len: candidates.seq_len(),
            query_rows: candidates.query_rows(),
            admitted_count: candidates.admitted_count(),
            fingerprint_fnv1a64: candidate_fingerprint(candidates),
        })
    }

    #[must_use]
    pub const fn source_layer_id(&self) -> u32 {
        self.source_layer_id
    }

    #[must_use]
    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    #[must_use]
    pub const fn policy_schema_version(&self) -> u32 {
        self.policy_schema_version
    }

    #[must_use]
    pub const fn representation_epoch(&self) -> u64 {
        self.representation_epoch
    }

    #[must_use]
    pub const fn fingerprint_fnv1a64(&self) -> u64 {
        self.fingerprint_fnv1a64
    }

    #[must_use]
    pub const fn admitted_count(&self) -> usize {
        self.admitted_count
    }
}

/// One explicit cross-layer request. Construction performs all MAA-15a
/// structural compatibility checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossLayerReusePlan {
    mode: CrossLayerReuseMode,
    axes: CrossLayerReuseAxes,
    source: Option<LayerMaterializationIdentity>,
    destination: LayerReuseRequirement,
    candidates: Option<StructuralCandidateIdentity>,
}

impl CrossLayerReusePlan {
    /// Validate one Full/Reindex/Reuse request.
    pub fn new(
        mode: CrossLayerReuseMode,
        axes: CrossLayerReuseAxes,
        source: Option<LayerMaterializationIdentity>,
        destination: LayerReuseRequirement,
        candidates: Option<StructuralCandidateIdentity>,
    ) -> Result<Self, CrossLayerReuseError> {
        validate_mode_axes(mode, axes)?;

        match mode {
            CrossLayerReuseMode::Full => {
                if source.is_some() {
                    return Err(CrossLayerReuseError::FullModeHasSource);
                }
                if candidates.is_some() {
                    return Err(CrossLayerReuseError::FullModeHasCandidateReuse);
                }
                if destination.reused_candidate_policy.is_some() {
                    return Err(CrossLayerReuseError::CandidatePolicyWithoutReuse);
                }
            }
            CrossLayerReuseMode::Reindex => {
                let source = source
                    .as_ref()
                    .ok_or(CrossLayerReuseError::ReuseModeMissingSource)?;
                validate_source_destination(source, &destination)?;
                if candidates.is_some() {
                    return Err(CrossLayerReuseError::ReindexModeHasCandidateReuse);
                }
                if destination.reused_candidate_policy.is_some() {
                    return Err(CrossLayerReuseError::CandidatePolicyWithoutReuse);
                }
            }
            CrossLayerReuseMode::Reuse => {
                let source = source
                    .as_ref()
                    .ok_or(CrossLayerReuseError::ReuseModeMissingSource)?;
                validate_source_destination(source, &destination)?;
                let candidates = candidates
                    .as_ref()
                    .ok_or(CrossLayerReuseError::ReuseModeMissingCandidates)?;
                validate_candidate_reuse(source, &destination, candidates)?;
            }
        }

        Ok(Self {
            mode,
            axes,
            source,
            destination,
            candidates,
        })
    }

    #[must_use]
    pub const fn mode(&self) -> CrossLayerReuseMode {
        self.mode
    }

    #[must_use]
    pub const fn axes(&self) -> CrossLayerReuseAxes {
        self.axes
    }

    #[must_use]
    pub const fn source(&self) -> Option<&LayerMaterializationIdentity> {
        self.source.as_ref()
    }

    #[must_use]
    pub const fn destination(&self) -> &LayerReuseRequirement {
        &self.destination
    }

    #[must_use]
    pub const fn candidates(&self) -> Option<&StructuralCandidateIdentity> {
        self.candidates.as_ref()
    }
}

fn validate_mode_axes(
    mode: CrossLayerReuseMode,
    axes: CrossLayerReuseAxes,
) -> Result<(), CrossLayerReuseError> {
    let valid = match mode {
        CrossLayerReuseMode::Full => axes == CrossLayerReuseAxes::full(),
        CrossLayerReuseMode::Reindex => axes.main_kv && !axes.candidates,
        CrossLayerReuseMode::Reuse => axes.main_kv && axes.candidates,
    };
    if !valid {
        return Err(CrossLayerReuseError::ModeAxesMismatch { mode, axes });
    }
    Ok(())
}

fn validate_source_destination(
    source: &LayerMaterializationIdentity,
    destination: &LayerReuseRequirement,
) -> Result<(), CrossLayerReuseError> {
    if source.layer_id == destination.layer_id {
        return Err(CrossLayerReuseError::SameLayerReuse {
            layer_id: source.layer_id,
        });
    }
    if source.representation_id != destination.representation_id {
        return Err(CrossLayerReuseError::RepresentationIdMismatch);
    }
    if source.representation_schema_version != destination.representation_schema_version {
        return Err(CrossLayerReuseError::RepresentationSchemaMismatch);
    }
    if source.materialization_epoch != destination.required_source_epoch {
        return Err(CrossLayerReuseError::RepresentationEpochMismatch {
            source: source.materialization_epoch,
            required: destination.required_source_epoch,
        });
    }
    if source.shape != destination.shape {
        return Err(CrossLayerReuseError::GeometryMismatch);
    }
    if source.causal != destination.causal {
        return Err(CrossLayerReuseError::CausalDomainMismatch);
    }
    Ok(())
}

fn validate_candidate_reuse(
    source: &LayerMaterializationIdentity,
    destination: &LayerReuseRequirement,
    candidates: &StructuralCandidateIdentity,
) -> Result<(), CrossLayerReuseError> {
    if candidates.source_layer_id != source.layer_id {
        return Err(CrossLayerReuseError::CandidateSourceLayerMismatch {
            candidate_layer: candidates.source_layer_id,
            source_layer: source.layer_id,
        });
    }
    if candidates.representation_epoch != source.materialization_epoch {
        return Err(CrossLayerReuseError::CandidateEpochMismatch {
            candidate: candidates.representation_epoch,
            source: source.materialization_epoch,
        });
    }
    if candidates.seq_len != source.shape.seq_len
        || candidates.query_rows != source.shape.lse_len()?
    {
        return Err(CrossLayerReuseError::CandidateGeometryMismatch);
    }
    let (required_id, required_version) = destination
        .reused_candidate_policy
        .as_ref()
        .ok_or(CrossLayerReuseError::ReuseModeMissingCandidatePolicy)?;
    if candidates.policy_id != *required_id {
        return Err(CrossLayerReuseError::CandidatePolicyIdMismatch);
    }
    if candidates.policy_schema_version != *required_version {
        return Err(CrossLayerReuseError::CandidatePolicySchemaMismatch);
    }
    Ok(())
}

fn candidate_fingerprint(candidates: &StructuralCandidateSet) -> u64 {
    let mut fingerprint = FNV_OFFSET;
    hash_u64(&mut fingerprint, CROSS_LAYER_REUSE_SCHEMA_VERSION as u64);
    hash_u64(&mut fingerprint, candidates.seq_len() as u64);
    hash_u64(&mut fingerprint, candidates.query_rows() as u64);
    hash_u64(&mut fingerprint, candidates.admitted_count() as u64);
    for &offset in candidates.offsets() {
        hash_u64(&mut fingerprint, offset as u64);
    }
    for &key in candidates.key_positions() {
        hash_u64(&mut fingerprint, key as u64);
    }
    fingerprint
}

fn hash_u64(fingerprint: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *fingerprint ^= u64::from(byte);
        *fingerprint = fingerprint.wrapping_mul(FNV_PRIME);
    }
}

fn validate_id(field: &'static str, value: &str) -> Result<(), CrossLayerReuseError> {
    if value.is_empty() {
        return Err(CrossLayerReuseError::EmptyId { field });
    }
    if value.len() > MAX_CROSS_LAYER_ID_BYTES {
        return Err(CrossLayerReuseError::IdTooLong {
            field,
            bytes: value.len(),
        });
    }
    Ok(())
}

fn validate_shape(shape: AttentionShape) -> Result<(), CrossLayerReuseError> {
    if shape.batch == 0 || shape.heads == 0 || shape.seq_len == 0 || shape.head_dim == 0 {
        return Err(FlatAttentionError::ZeroDimension.into());
    }
    shape.tensor_len()?;
    shape.lse_len()?;
    Ok(())
}

/// Fail-closed MAA-15a structural contract errors.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CrossLayerReuseError {
    Attention(FlatAttentionError),
    EmptyId {
        field: &'static str,
    },
    IdTooLong {
        field: &'static str,
        bytes: usize,
    },
    ZeroRepresentationSchemaVersion,
    ZeroCandidatePolicySchemaVersion,
    ModeAxesMismatch {
        mode: CrossLayerReuseMode,
        axes: CrossLayerReuseAxes,
    },
    FullModeHasSource,
    FullModeHasCandidateReuse,
    ReuseModeMissingSource,
    ReindexModeHasCandidateReuse,
    ReuseModeMissingCandidates,
    CandidatePolicyWithoutReuse,
    ReuseModeMissingCandidatePolicy,
    SameLayerReuse {
        layer_id: u32,
    },
    RepresentationIdMismatch,
    RepresentationSchemaMismatch,
    RepresentationEpochMismatch {
        source: u64,
        required: u64,
    },
    GeometryMismatch,
    CausalDomainMismatch,
    CandidateSourceLayerMismatch {
        candidate_layer: u32,
        source_layer: u32,
    },
    CandidateEpochMismatch {
        candidate: u64,
        source: u64,
    },
    CandidateGeometryMismatch,
    CandidatePolicyIdMismatch,
    CandidatePolicySchemaMismatch,
}

impl From<FlatAttentionError> for CrossLayerReuseError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Attention(value)
    }
}

impl fmt::Display for CrossLayerReuseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attention(error) => write!(formatter, "{error}"),
            Self::EmptyId { field } => write!(formatter, "{field} must not be empty"),
            Self::IdTooLong { field, bytes } => write!(
                formatter,
                "{field} uses {bytes} UTF-8 bytes, maximum is {MAX_CROSS_LAYER_ID_BYTES}"
            ),
            Self::ZeroRepresentationSchemaVersion => {
                write!(formatter, "representation schema version must be non-zero")
            }
            Self::ZeroCandidatePolicySchemaVersion => {
                write!(formatter, "candidate policy schema version must be non-zero")
            }
            Self::ModeAxesMismatch { mode, axes } => {
                write!(formatter, "reuse mode {mode:?} is incompatible with axes {axes:?}")
            }
            Self::FullModeHasSource => write!(formatter, "Full mode must not reuse a source"),
            Self::FullModeHasCandidateReuse => {
                write!(formatter, "Full mode must not carry reused candidates")
            }
            Self::ReuseModeMissingSource => write!(formatter, "reuse mode requires a source layer"),
            Self::ReindexModeHasCandidateReuse => {
                write!(formatter, "Reindex mode must compute fresh candidates")
            }
            Self::ReuseModeMissingCandidates => {
                write!(formatter, "Reuse mode requires exact candidate identity")
            }
            Self::CandidatePolicyWithoutReuse => {
                write!(formatter, "reused candidate policy declared without candidate reuse")
            }
            Self::ReuseModeMissingCandidatePolicy => {
                write!(formatter, "Reuse mode requires destination candidate-policy identity")
            }
            Self::SameLayerReuse { layer_id } => {
                write!(formatter, "cross-layer reuse cannot source destination layer {layer_id}")
            }
            Self::RepresentationIdMismatch => write!(formatter, "representation id mismatch"),
            Self::RepresentationSchemaMismatch => {
                write!(formatter, "representation schema mismatch")
            }
            Self::RepresentationEpochMismatch { source, required } => write!(
                formatter,
                "source materialization epoch {source} differs from required epoch {required}"
            ),
            Self::GeometryMismatch => write!(formatter, "attention geometry mismatch"),
            Self::CausalDomainMismatch => write!(formatter, "causal-domain mismatch"),
            Self::CandidateSourceLayerMismatch {
                candidate_layer,
                source_layer,
            } => write!(
                formatter,
                "candidate source layer {candidate_layer} differs from representation source {source_layer}"
            ),
            Self::CandidateEpochMismatch { candidate, source } => write!(
                formatter,
                "candidate representation epoch {candidate} differs from source epoch {source}"
            ),
            Self::CandidateGeometryMismatch => write!(formatter, "candidate geometry mismatch"),
            Self::CandidatePolicyIdMismatch => write!(formatter, "candidate policy id mismatch"),
            Self::CandidatePolicySchemaMismatch => {
                write!(formatter, "candidate policy schema mismatch")
            }
        }
    }
}

impl std::error::Error for CrossLayerReuseError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::research_structural_routing::StructuralCandidateSet;

    fn shape() -> AttentionShape {
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 4,
            head_dim: 2,
        }
    }

    fn source(epoch: u64) -> LayerMaterializationIdentity {
        LayerMaterializationIdentity::new(3, "flat.structural-kv", 1, epoch, shape(), false).unwrap()
    }

    fn destination(epoch: u64) -> LayerReuseRequirement {
        LayerReuseRequirement::new(7, "flat.structural-kv", 1, epoch, shape(), false).unwrap()
    }

    fn candidates(epoch: u64) -> (StructuralCandidateSet, StructuralCandidateIdentity) {
        let set = StructuralCandidateSet::from_rows(
            shape(),
            vec![vec![0, 1], vec![1], vec![0, 2], vec![3]],
        )
        .unwrap();
        let identity =
            StructuralCandidateIdentity::from_candidates(3, "tiered-recall", 1, epoch, &set)
                .unwrap();
        (set, identity)
    }

    #[test]
    fn full_mode_carries_no_source_or_candidate_reuse() {
        let plan = CrossLayerReusePlan::new(
            CrossLayerReuseMode::Full,
            CrossLayerReuseAxes::full(),
            None,
            destination(4),
            None,
        )
        .unwrap();
        assert_eq!(plan.mode(), CrossLayerReuseMode::Full);
        assert!(plan.source().is_none());
        assert!(plan.candidates().is_none());
    }

    #[test]
    fn reindex_accepts_explicit_compatible_source_and_fresh_candidates() {
        let plan = CrossLayerReusePlan::new(
            CrossLayerReuseMode::Reindex,
            CrossLayerReuseAxes::reindex(true),
            Some(source(9)),
            destination(9),
            None,
        )
        .unwrap();
        assert!(plan.axes().main_kv);
        assert!(plan.axes().indexer_k);
        assert!(!plan.axes().candidates);
    }

    #[test]
    fn reuse_requires_exact_source_epoch_and_candidate_policy() {
        let (_, candidate) = candidates(11);
        let destination = destination(11)
            .require_reused_candidate_policy("tiered-recall", 1)
            .unwrap();
        let plan = CrossLayerReusePlan::new(
            CrossLayerReuseMode::Reuse,
            CrossLayerReuseAxes::reuse(true),
            Some(source(11)),
            destination,
            Some(candidate.clone()),
        )
        .unwrap();
        assert_eq!(
            plan.candidates().unwrap().fingerprint_fnv1a64(),
            candidate.fingerprint_fnv1a64()
        );

        let bad_destination = destination(12)
            .require_reused_candidate_policy("tiered-recall", 1)
            .unwrap();
        assert!(matches!(
            CrossLayerReusePlan::new(
                CrossLayerReuseMode::Reuse,
                CrossLayerReuseAxes::reuse(false),
                Some(source(11)),
                bad_destination,
                Some(candidate),
            ),
            Err(CrossLayerReuseError::RepresentationEpochMismatch {
                source: 11,
                required: 12
            })
        ));
    }

    #[test]
    fn candidate_identity_is_deterministic_and_selection_sensitive() {
        let (set, a) = candidates(5);
        let b = StructuralCandidateIdentity::from_candidates(3, "tiered-recall", 1, 5, &set)
            .unwrap();
        let other = StructuralCandidateSet::from_rows(
            shape(),
            vec![vec![0, 1], vec![1], vec![0, 3], vec![3]],
        )
        .unwrap();
        let c = StructuralCandidateIdentity::from_candidates(3, "tiered-recall", 1, 5, &other)
            .unwrap();
        assert_eq!(a.fingerprint_fnv1a64(), b.fingerprint_fnv1a64());
        assert_ne!(a.fingerprint_fnv1a64(), c.fingerprint_fnv1a64());
    }

    #[test]
    fn mode_axis_substitution_fails_closed() {
        assert!(matches!(
            CrossLayerReusePlan::new(
                CrossLayerReuseMode::Reindex,
                CrossLayerReuseAxes::reuse(false),
                Some(source(2)),
                destination(2),
                None,
            ),
            Err(CrossLayerReuseError::ModeAxesMismatch { .. })
        ));
    }

    #[test]
    fn same_layer_and_candidate_source_drift_fail_closed() {
        let same_destination =
            LayerReuseRequirement::new(3, "flat.structural-kv", 1, 4, shape(), false).unwrap();
        assert!(matches!(
            CrossLayerReusePlan::new(
                CrossLayerReuseMode::Reindex,
                CrossLayerReuseAxes::reindex(false),
                Some(source(4)),
                same_destination,
                None,
            ),
            Err(CrossLayerReuseError::SameLayerReuse { layer_id: 3 })
        ));

        let set = StructuralCandidateSet::all(shape()).unwrap();
        let wrong_layer =
            StructuralCandidateIdentity::from_candidates(2, "tiered-recall", 1, 4, &set).unwrap();
        let destination = destination(4)
            .require_reused_candidate_policy("tiered-recall", 1)
            .unwrap();
        assert!(matches!(
            CrossLayerReusePlan::new(
                CrossLayerReuseMode::Reuse,
                CrossLayerReuseAxes::reuse(false),
                Some(source(4)),
                destination,
                Some(wrong_layer),
            ),
            Err(CrossLayerReuseError::CandidateSourceLayerMismatch {
                candidate_layer: 2,
                source_layer: 3
            })
        ));
    }
}
