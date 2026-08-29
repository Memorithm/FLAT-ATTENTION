//! Context-bound FLAT -> ElasticXxx kernel selection.
//!
//! This module adds recommendation freshness to the existing adapter without
//! weakening or replacing any FLAT safety gate. Candidate translation,
//! correctness/timing evidence, and workload-dependent
//! [`elastic_kernel::DispatchGrid`] validation are identical to the ordinary
//! adapter path. The only additional step is binding the generic Elastic
//! selection to a caller-supplied [`RecommendationContext`].

use super::{
    adapt_candidate, capability_snapshot, evidence_for_candidate, logical_resource_id,
    realization_identity, workload_fingerprint, AdapterError, DispatchRejection,
};
use elastic_core::{FreshnessSnapshot, RecommendationContext, RecommendationFreshnessError};
use elastic_kernel::{
    plan_with_context, ContextualSelection, KernelCandidate as ElasticKernelCandidate,
    SelectionPolicy as ElasticSelectionPolicy,
};
use flat_attention::kernel_autotune::SelectionRecord as FlatTuningRecord;
use flat_attention::kernel_candidates::{
    generate_candidates, KernelCandidate as FlatKernelCandidate,
    SelectionPolicy as FlatSelectionPolicy,
};
use flat_attention::kernel_ir::AttentionProblem;
use flat_attention::RuntimeDeviceCapabilities;

/// Context-bound result of one Elastic selection over a stable FLAT candidate
/// set.
#[derive(Debug, Clone)]
pub struct ContextualAdapterPlan {
    /// Candidates offered at the FLAT boundary, in deterministic FLAT order.
    pub flat_candidates: Vec<FlatKernelCandidate>,
    /// Candidates offered to Elastic after explicit M26 and dispatch-grid
    /// rejections are removed.
    pub elastic_candidates: Vec<ElasticKernelCandidate>,
    /// Workload-dependent dispatch rejections preserved by the bridge.
    pub dispatch_rejections: Vec<DispatchRejection>,
    /// Generic Elastic planner outcome bound to recommendation freshness.
    pub selection: ContextualSelection,
}
