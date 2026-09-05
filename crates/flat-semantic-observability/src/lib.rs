//! Opt-in research observability around FLAT scalar/reference semantics.
//!
//! This crate is deliberately absent from the historical `flat-attention` fast
//! path. Production execution therefore pays no instrumentation branch,
//! allocation, callback, registry lookup or dynamic-dispatch cost when research
//! observability is disabled: disabled means this crate is not invoked.
//!
//! The first concrete adapter is StandardSoftmax reference execution. It can run
//! one baseline and one one-shot Q/K/V-perturbed reference invocation under the
//! exact same semantic/mechanism identity and return output/LSE observations.
//! It never materializes an N x N probability matrix and does not compute ITD or
//! TDI diagnostics itself.

#![forbid(unsafe_code)]

use core::fmt;

use flat_attention::{AttentionShape, FlatAttentionError};
use flat_semantic::v1::{SemanticSavedState, StandardSoftmaxSemantic};
use flat_semantic_mechanism::StandardSoftmaxMechanism;

/// Version of the research-observability contract.
pub const RESEARCH_OBSERVABILITY_VERSION: u16 = 1;

/// Addressable class of one-shot research intervention.
///
/// The enum is intentionally broader than StandardSoftmax. Concrete semantic
/// adapters must fail closed on sites they do not support instead of pretending
/// that all semantics own Q/K/V or recurrent state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResearchInterventionSiteKind {
    /// Upstream token/input representation coordinate.
    TokenInputElement,
    /// Query tensor coordinate.
    QueryElement,
    /// Key tensor coordinate.
    KeyElement,
    /// Value tensor coordinate.
    ValueElement,
    /// Generic semantic carried-state coordinate.
    SemanticStateElement,
    /// Explicit recurrent-memory coordinate.
    RecurrentMemoryElement,
    /// Semantic operator/component parameter coordinate.
    OperatorParameter,
}

/// Stable one-shot intervention site within one reference invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResearchInterventionSite {
    kind: ResearchInterventionSiteKind,
    flat_index: usize,
}

impl ResearchInterventionSite {
    /// Construct one flat coordinate in a declared semantic site.
    #[must_use]
    pub const fn new(kind: ResearchInterventionSiteKind, flat_index: usize) -> Self {
        Self { kind, flat_index }
    }

    /// Site class.
    #[must_use]
    pub const fn kind(self) -> ResearchInterventionSiteKind {
        self.kind
    }

    /// Flat coordinate within the selected semantic site.
    #[must_use]
    pub const fn flat_index(self) -> usize {
        self.flat_index
    }
}

/// Scalar mutation used by the first bounded reference adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarInterventionMode {
    /// Add one finite scalar delta to the selected coordinate.
    Add(f32),
    /// Replace the selected coordinate with one finite scalar value.
    Replace(f32),
}

impl ScalarInterventionMode {
    fn value(self) -> f32 {
        match self {
            Self::Add(value) | Self::Replace(value) => value,
        }
    }
}

/// One one-shot scalar intervention applied to a reference invocation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarResearchIntervention {
    site: ResearchInterventionSite,
    mode: ScalarInterventionMode,
}

impl ScalarResearchIntervention {
    /// Construct a finite scalar intervention.
    ///
    /// # Errors
    ///
    /// Rejects NaN and infinity before reference execution.
    pub fn new(
        site: ResearchInterventionSite,
        mode: ScalarInterventionMode,
    ) -> Result<Self, ResearchInterventionError> {
        if !mode.value().is_finite() {
            return Err(ResearchInterventionError::NonFiniteValue);
        }
        Ok(Self { site, mode })
    }

    /// Addressed semantic site.
    #[must_use]
    pub const fn site(self) -> ResearchInterventionSite {
        self.site
    }

    /// Mutation rule.
    #[must_use]
    pub const fn mode(self) -> ScalarInterventionMode {
        self.mode
    }
}

/// Stable caller-owned depth/step/layer label for one reference observation.
///
/// FLAT does not interpret this number. A model/layer harness may use it to bind
/// observations from several semantic invocations without forcing FLAT to own
/// the model graph or TDI trajectory semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResearchObservationDepth(u32);

impl ResearchObservationDepth {
    /// Construct an externally meaningful depth label.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Raw caller-owned depth label.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Semantic/mechanism identity shared by reference and perturbed trajectories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResearchExecutionIdentity {
    semantic_fingerprint: u64,
    mechanism_fingerprint: u64,
}

impl ResearchExecutionIdentity {
    /// Build identity from one validated StandardSoftmax mechanism instance.
    #[must_use]
    pub fn standard_softmax(mechanism: StandardSoftmaxMechanism) -> Self {
        Self {
            semantic_fingerprint: mechanism.semantic().stable_fingerprint(),
            mechanism_fingerprint: mechanism.stable_fingerprint(),
        }
    }

    /// Exact semantic-instance fingerprint.
    #[must_use]
    pub const fn semantic_fingerprint(self) -> u64 {
        self.semantic_fingerprint
    }

    /// Exact mechanism-instance fingerprint.
    #[must_use]
    pub const fn mechanism_fingerprint(self) -> u64 {
        self.mechanism_fingerprint
    }
}

/// Raw reference observation from one StandardSoftmax invocation.
///
/// Only the primary output and semantic-specific LSE are retained. No score or
/// probability matrix is materialized for observability.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardSoftmaxReferenceObservation {
    depth: ResearchObservationDepth,
    output: Vec<f32>,
    lse: Vec<f32>,
}

impl StandardSoftmaxReferenceObservation {
    /// Caller-owned model/layer depth label.
    #[must_use]
    pub const fn depth(&self) -> ResearchObservationDepth {
        self.depth
    }

    /// Primary semantic output.
    #[must_use]
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// StandardSoftmax log-sum-exp saved statistic.
    #[must_use]
    pub fn lse(&self) -> &[f32] {
        &self.lse
    }
}

/// Baseline/perturbed observation pair under one exact semantic identity.
///
/// The same `StandardSoftmaxSemantic` instance executes both arms. The record is
/// raw FLAT evidence only: ITD static summaries and TDI dynamic-recovery metrics
/// remain separate downstream evidence blocks and are not computed here.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardSoftmaxPairedReference {
    identity: ResearchExecutionIdentity,
    intervention: ScalarResearchIntervention,
    reference: StandardSoftmaxReferenceObservation,
    perturbed: StandardSoftmaxReferenceObservation,
}

impl StandardSoftmaxPairedReference {
    /// Shared semantic/mechanism identity for both trajectories.
    #[must_use]
    pub const fn identity(&self) -> ResearchExecutionIdentity {
        self.identity
    }

    /// Applied one-shot intervention.
    #[must_use]
    pub const fn intervention(&self) -> ScalarResearchIntervention {
        self.intervention
    }

    /// Unperturbed reference observation.
    #[must_use]
    pub const fn reference(&self) -> &StandardSoftmaxReferenceObservation {
        &self.reference
    }

    /// Perturbed observation after exactly one declared input-site mutation.
    #[must_use]
    pub const fn perturbed(&self) -> &StandardSoftmaxReferenceObservation {
        &self.perturbed
    }
}

/// Execute one research-only baseline/perturbed StandardSoftmax pair.
///
/// This function is intentionally outside the historical specialized path. It
/// validates the intervention before either arm executes, applies the mutation
/// to a research-owned copy of exactly one Q/K/V tensor, then executes both arms
/// through the same validated [`StandardSoftmaxSemantic`] scalar reference.
///
/// # Errors
///
/// Fails closed for unsupported sites, out-of-range coordinates, non-finite or
/// overflowing mutations, historical scalar-reference validation/numerical
/// errors, or an unexpected saved-state contract.
pub fn run_standard_softmax_paired_reference(
    mechanism: StandardSoftmaxMechanism,
    depth: ResearchObservationDepth,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    intervention: ScalarResearchIntervention,
) -> Result<StandardSoftmaxPairedReference, ResearchObservabilityError> {
    validate_standard_softmax_intervention(q, k, v, intervention)?;

    let semantic = mechanism.semantic();
    let reference = execute_observation(semantic, depth, q, k, v, shape)?;

    let perturbed = match intervention.site().kind() {
        ResearchInterventionSiteKind::QueryElement => {
            let mut changed = q.to_vec();
            apply_scalar(&mut changed, intervention)?;
            execute_observation(semantic, depth, &changed, k, v, shape)?
        }
        ResearchInterventionSiteKind::KeyElement => {
            let mut changed = k.to_vec();
            apply_scalar(&mut changed, intervention)?;
            execute_observation(semantic, depth, q, &changed, v, shape)?
        }
        ResearchInterventionSiteKind::ValueElement => {
            let mut changed = v.to_vec();
            apply_scalar(&mut changed, intervention)?;
            execute_observation(semantic, depth, q, k, &changed, shape)?
        }
        _ => {
            return Err(ResearchInterventionError::UnsupportedSite {
                kind: intervention.site().kind(),
            }
            .into())
        }
    };

    Ok(StandardSoftmaxPairedReference {
        identity: ResearchExecutionIdentity::standard_softmax(mechanism),
        intervention,
        reference,
        perturbed,
    })
}

fn validate_standard_softmax_intervention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    intervention: ScalarResearchIntervention,
) -> Result<(), ResearchInterventionError> {
    let len = match intervention.site().kind() {
        ResearchInterventionSiteKind::QueryElement => q.len(),
        ResearchInterventionSiteKind::KeyElement => k.len(),
        ResearchInterventionSiteKind::ValueElement => v.len(),
        kind => return Err(ResearchInterventionError::UnsupportedSite { kind }),
    };
    if intervention.site().flat_index() >= len {
        return Err(ResearchInterventionError::IndexOutOfBounds {
            index: intervention.site().flat_index(),
            len,
        });
    }
    Ok(())
}

fn apply_scalar(
    values: &mut [f32],
    intervention: ScalarResearchIntervention,
) -> Result<(), ResearchInterventionError> {
    let index = intervention.site().flat_index();
    let current = values[index];
    let next = match intervention.mode() {
        ScalarInterventionMode::Add(delta) => current + delta,
        ScalarInterventionMode::Replace(value) => value,
    };
    if !next.is_finite() {
        return Err(ResearchInterventionError::NonFiniteResult);
    }
    values[index] = next;
    Ok(())
}

fn execute_observation(
    semantic: StandardSoftmaxSemantic,
    depth: ResearchObservationDepth,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
) -> Result<StandardSoftmaxReferenceObservation, ResearchObservabilityError> {
    let (output, saved) = semantic
        .forward_reference(q, k, v, shape)
        .map_err(ResearchObservabilityError::Reference)?
        .into_parts();
    let lse = match saved {
        SemanticSavedState::LogSumExp(lse) => lse,
        _ => return Err(ResearchObservabilityError::UnexpectedSavedState),
    };
    Ok(StandardSoftmaxReferenceObservation { depth, output, lse })
}

/// Fail-closed intervention validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResearchInterventionError {
    /// Mutation value was NaN or infinity.
    NonFiniteValue,
    /// The selected semantic does not expose this intervention site.
    UnsupportedSite {
        /// Rejected site kind.
        kind: ResearchInterventionSiteKind,
    },
    /// Flat coordinate lies outside the selected site tensor/state.
    IndexOutOfBounds {
        /// Rejected flat coordinate.
        index: usize,
        /// Available element count.
        len: usize,
    },
    /// Applying a finite delta overflowed or otherwise produced non-finite data.
    NonFiniteResult,
}

impl fmt::Display for ResearchInterventionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteValue => formatter.write_str("research intervention value is non-finite"),
            Self::UnsupportedSite { kind } => {
                write!(formatter, "research intervention site {kind:?} is unsupported")
            }
            Self::IndexOutOfBounds { index, len } => write!(
                formatter,
                "research intervention index {index} is outside site length {len}"
            ),
            Self::NonFiniteResult => {
                formatter.write_str("research intervention produced a non-finite value")
            }
        }
    }
}

impl std::error::Error for ResearchInterventionError {}

/// Reference-observability execution errors.
#[derive(Debug)]
pub enum ResearchObservabilityError {
    /// One-shot intervention contract failed before evidence could be emitted.
    Intervention(ResearchInterventionError),
    /// Historical scalar/reference execution failed.
    Reference(FlatAttentionError),
    /// The wrapped semantic returned a saved-state variant inconsistent with
    /// its StandardSoftmax descriptor.
    UnexpectedSavedState,
}

impl From<ResearchInterventionError> for ResearchObservabilityError {
    fn from(value: ResearchInterventionError) -> Self {
        Self::Intervention(value)
    }
}

impl fmt::Display for ResearchObservabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Intervention(error) => write!(formatter, "{error}"),
            Self::Reference(error) => write!(formatter, "reference execution failed: {error}"),
            Self::UnexpectedSavedState => {
                formatter.write_str("StandardSoftmax reference returned unexpected saved state")
            }
        }
    }
}

impl std::error::Error for ResearchObservabilityError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> AttentionShape {
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 3,
            head_dim: 2,
        }
    }

    fn mechanism() -> StandardSoftmaxMechanism {
        StandardSoftmaxMechanism::new(StandardSoftmaxSemantic::new(true, 0.625).unwrap())
    }

    fn fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![0.2, -0.3, 0.1, 0.5, -0.4, 0.7],
            vec![0.6, -0.2, -0.8, 0.3, 0.4, 0.9],
            vec![1.0, -2.0, 0.5, 0.25, -0.75, 1.5],
        )
    }

    #[test]
    fn paired_run_uses_one_exact_semantic_identity() {
        let (q, k, v) = fixture();
        let intervention = ScalarResearchIntervention::new(
            ResearchInterventionSite::new(ResearchInterventionSiteKind::KeyElement, 2),
            ScalarInterventionMode::Add(0.25),
        )
        .unwrap();
        let result = run_standard_softmax_paired_reference(
            mechanism(),
            ResearchObservationDepth::new(7),
            &q,
            &k,
            &v,
            shape(),
            intervention,
        )
        .unwrap();

        assert_eq!(result.reference().depth().get(), 7);
        assert_eq!(result.perturbed().depth().get(), 7);
        assert_eq!(
            result.identity(),
            ResearchExecutionIdentity::standard_softmax(mechanism())
        );
        assert_ne!(result.reference().output(), result.perturbed().output());
        assert!(!result.reference().lse().is_empty());
        assert!(!result.perturbed().lse().is_empty());
    }

    #[test]
    fn q_k_and_v_sites_are_independently_addressable() {
        let (q, k, v) = fixture();
        for kind in [
            ResearchInterventionSiteKind::QueryElement,
            ResearchInterventionSiteKind::KeyElement,
            ResearchInterventionSiteKind::ValueElement,
        ] {
            let intervention = ScalarResearchIntervention::new(
                ResearchInterventionSite::new(kind, 0),
                ScalarInterventionMode::Add(0.125),
            )
            .unwrap();
            run_standard_softmax_paired_reference(
                mechanism(),
                ResearchObservationDepth::new(0),
                &q,
                &k,
                &v,
                shape(),
                intervention,
            )
            .unwrap();
        }
    }

    #[test]
    fn unsupported_recurrent_site_fails_closed_for_stateless_softmax() {
        let (q, k, v) = fixture();
        let intervention = ScalarResearchIntervention::new(
            ResearchInterventionSite::new(
                ResearchInterventionSiteKind::RecurrentMemoryElement,
                0,
            ),
            ScalarInterventionMode::Replace(1.0),
        )
        .unwrap();
        let error = run_standard_softmax_paired_reference(
            mechanism(),
            ResearchObservationDepth::new(0),
            &q,
            &k,
            &v,
            shape(),
            intervention,
        )
        .unwrap_err();
        match error {
            ResearchObservabilityError::Intervention(
                ResearchInterventionError::UnsupportedSite { kind },
            ) => assert_eq!(kind, ResearchInterventionSiteKind::RecurrentMemoryElement),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn invalid_intervention_fails_before_reference_evidence() {
        assert_eq!(
            ScalarResearchIntervention::new(
                ResearchInterventionSite::new(ResearchInterventionSiteKind::QueryElement, 0),
                ScalarInterventionMode::Add(f32::NAN),
            )
            .unwrap_err(),
            ResearchInterventionError::NonFiniteValue
        );

        let (q, k, v) = fixture();
        let intervention = ScalarResearchIntervention::new(
            ResearchInterventionSite::new(ResearchInterventionSiteKind::ValueElement, 99),
            ScalarInterventionMode::Replace(0.0),
        )
        .unwrap();
        let error = run_standard_softmax_paired_reference(
            mechanism(),
            ResearchObservationDepth::new(0),
            &q,
            &k,
            &v,
            shape(),
            intervention,
        )
        .unwrap_err();
        match error {
            ResearchObservabilityError::Intervention(
                ResearchInterventionError::IndexOutOfBounds { index, len },
            ) => {
                assert_eq!(index, 99);
                assert_eq!(len, v.len());
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn additive_overflow_fails_closed() {
        let (mut q, k, v) = fixture();
        q[0] = f32::MAX;
        let intervention = ScalarResearchIntervention::new(
            ResearchInterventionSite::new(ResearchInterventionSiteKind::QueryElement, 0),
            ScalarInterventionMode::Add(f32::MAX),
        )
        .unwrap();
        let error = run_standard_softmax_paired_reference(
            mechanism(),
            ResearchObservationDepth::new(0),
            &q,
            &k,
            &v,
            shape(),
            intervention,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ResearchObservabilityError::Intervention(
                ResearchInterventionError::NonFiniteResult
            )
        ));
    }
}
