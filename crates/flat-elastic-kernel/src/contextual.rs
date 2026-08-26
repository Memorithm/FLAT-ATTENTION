//! Context-bound FLAT -> ElasticXxx kernel selection.
//!
//! This module adds recommendation freshness to the existing adapter without
//! weakening or replacing any FLAT safety gate. Candidate translation,
//! correctness/timing evidence, and workload-dependent [`DispatchGrid`]
//! validation are identical to the ordinary adapter path. The only additional
//! step is binding the generic Elastic selection to a caller-supplied
//! [`RecommendationContext`].

use super::{
    adapt_candidate, capability_snapshot, evidence_for_candidate, logical_resource_id,
    realization_identity, workload_fingerprint, AdapterError, DispatchRejection,
};
use elastic_core::{
    FreshnessSnapshot, RecommendationContext, RecommendationFreshnessError,
};
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

impl ContextualAdapterPlan {
    /// Resolve the selected Elastic realization back to its FLAT candidate only
    /// after the recommendation context has been revalidated.
    ///
    /// `Ok(None)` preserves all honest non-selected Elastic outcomes.
    ///
    /// # Errors
    ///
    /// Returns [`RecommendationFreshnessError`] when the planner epoch,
    /// observation epoch, or any tracked resource generation changed after
    /// planning.
    pub fn selected_flat_candidate_if_fresh(
        &self,
        current: &FreshnessSnapshot,
    ) -> Result<Option<&FlatKernelCandidate>, RecommendationFreshnessError> {
        let Some(record) = self.selection.selected_record_if_fresh(current)? else {
            return Ok(None);
        };
        Ok(self.flat_candidates.iter().find(|candidate| {
            realization_identity(candidate) == record.selected_realization().as_str()
        }))
    }
}

/// Adapt an already-generated FLAT candidate list and bind Elastic selection to
/// explicit recommendation freshness assumptions.
///
/// FLAT M26 correctness/timing rejection and the workload-dependent
/// `DispatchGrid` gate remain in front of Elastic planning exactly as on the
/// non-contextual path.
pub fn plan_adapted_with_context(
    problem: &AttentionProblem,
    capabilities: &RuntimeDeviceCapabilities,
    flat_candidates: &[FlatKernelCandidate],
    tuning: Option<&FlatTuningRecord>,
    policy: &ElasticSelectionPolicy,
    context: RecommendationContext,
) -> Result<ContextualAdapterPlan, AdapterError> {
    let snapshot = capability_snapshot(capabilities)?;
    let logical_resource_id = logical_resource_id(problem)?;
    let workload = workload_fingerprint(problem);
    let mut elastic_candidates = Vec::with_capacity(flat_candidates.len());
    let mut dispatch_rejections = Vec::new();

    for candidate in flat_candidates {
        let Some(evidence) = evidence_for_candidate(candidate, tuning)? else {
            continue;
        };
        let (elastic_candidate, dispatch_grid) =
            adapt_candidate(problem, candidate, logical_resource_id.clone(), evidence)?;
        if let Err(reason) = dispatch_grid.check_against(&snapshot) {
            dispatch_rejections.push(DispatchRejection {
                realization: realization_identity(candidate),
                reason,
            });
            continue;
        }
        elastic_candidates.push(elastic_candidate);
    }

    let selection = plan_with_context(
        &logical_resource_id,
        workload,
        &snapshot,
        policy,
        &elastic_candidates,
        context,
    );

    Ok(ContextualAdapterPlan {
        flat_candidates: flat_candidates.to_vec(),
        elastic_candidates,
        dispatch_rejections,
        selection,
    })
}

/// Generate FLAT's bounded M25 candidate set, preserve its M24 prefilter, and
/// bind generic Elastic selection to explicit recommendation freshness.
pub fn generate_and_plan_with_context(
    problem: &AttentionProblem,
    capabilities: &RuntimeDeviceCapabilities,
    flat_policy: &FlatSelectionPolicy,
    tuning: Option<&FlatTuningRecord>,
    elastic_policy: &ElasticSelectionPolicy,
    context: RecommendationContext,
) -> Result<ContextualAdapterPlan, AdapterError> {
    let candidates = generate_candidates(problem, capabilities, flat_policy);
    plan_adapted_with_context(
        problem,
        capabilities,
        &candidates,
        tuning,
        elastic_policy,
        context,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{latency_policy, plan_adapted};
    use elastic_core::{ObservationEpoch, PlannerEpoch, ResourceGeneration};
    use elastic_kernel::{CapabilityRejectionReason, SelectionOutcome};

    fn problem() -> AttentionProblem {
        AttentionProblem {
            batch_heads: 4,
            seq_len: 128,
            head_dim: 64,
            causal: true,
        }
    }

    fn caps() -> RuntimeDeviceCapabilities {
        RuntimeDeviceCapabilities {
            max_workgroups_per_dimension: 65_535,
            max_workgroup_size_x: 64,
            max_workgroup_size_y: 1_024,
            max_workgroup_size_z: 64,
            max_workgroup_storage_bytes: 32_768,
            max_binding_entries: 8,
            max_storage_buffer_binding_size: 1 << 30,
            subgroup_supported: true,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f16_supported: true,
        }
    }

    fn one_candidate() -> FlatKernelCandidate {
        generate_candidates(&problem(), &caps(), &FlatSelectionPolicy::default())
            .into_iter()
            .next()
            .expect("candidate")
    }

    fn context() -> RecommendationContext {
        RecommendationContext::new(PlannerEpoch::new(7), ObservationEpoch::new(11))
            .with_resource_generation(
                logical_resource_id(&problem()).expect("logical resource"),
                ResourceGeneration::new(5),
            )
    }

    fn current() -> FreshnessSnapshot {
        FreshnessSnapshot::new(PlannerEpoch::new(7), ObservationEpoch::new(11))
            .with_resource_generation(
                logical_resource_id(&problem()).expect("logical resource"),
                ResourceGeneration::new(5),
            )
    }

    #[test]
    fn fresh_context_exposes_the_same_uncontested_flat_candidate() {
        let candidate = one_candidate();
        let policy = latency_policy(true).expect("policy");
        let ordinary = plan_adapted(&problem(), &caps(), &[candidate], None, &policy)
            .expect("ordinary plan");
        let contextual = plan_adapted_with_context(
            &problem(),
            &caps(),
            &[candidate],
            None,
            &policy,
            context(),
        )
        .expect("contextual plan");

        assert_eq!(
            ordinary.selected_flat_candidate().map(|candidate| candidate.id),
            contextual
                .selected_flat_candidate_if_fresh(&current())
                .expect("fresh context")
                .map(|candidate| candidate.id)
        );
    }

    #[test]
    fn stale_planner_epoch_blocks_selected_flat_candidate() {
        let candidate = one_candidate();
        let policy = latency_policy(true).expect("policy");
        let plan = plan_adapted_with_context(
            &problem(),
            &caps(),
            &[candidate],
            None,
            &policy,
            context(),
        )
        .expect("contextual plan");
        let stale = FreshnessSnapshot::new(PlannerEpoch::new(8), ObservationEpoch::new(11))
            .with_resource_generation(
                logical_resource_id(&problem()).expect("logical resource"),
                ResourceGeneration::new(5),
            );

        assert_eq!(
            plan.selected_flat_candidate_if_fresh(&stale),
            Err(RecommendationFreshnessError::PlannerEpochMismatch {
                recommended: PlannerEpoch::new(7),
                current: PlannerEpoch::new(8),
            })
        );
    }

    #[test]
    fn changed_resource_generation_blocks_selected_flat_candidate() {
        let candidate = one_candidate();
        let policy = latency_policy(true).expect("policy");
        let plan = plan_adapted_with_context(
            &problem(),
            &caps(),
            &[candidate],
            None,
            &policy,
            context(),
        )
        .expect("contextual plan");
        let resource = logical_resource_id(&problem()).expect("logical resource");
        let stale = FreshnessSnapshot::new(PlannerEpoch::new(7), ObservationEpoch::new(11))
            .with_resource_generation(resource.clone(), ResourceGeneration::new(6));

        assert_eq!(
            plan.selected_flat_candidate_if_fresh(&stale),
            Err(RecommendationFreshnessError::ResourceGenerationMismatch {
                resource,
                recommended: ResourceGeneration::new(5),
                current: ResourceGeneration::new(6),
            })
        );
    }

    #[test]
    fn contextual_path_preserves_dispatch_grid_rejection() {
        let candidate = one_candidate();
        let policy = latency_policy(true).expect("policy");
        let mut limited = caps();
        limited.max_workgroups_per_dimension = 31;
        let plan = plan_adapted_with_context(
            &problem(),
            &limited,
            &[candidate],
            None,
            &policy,
            context(),
        )
        .expect("contextual dispatch plan");

        assert_eq!(plan.dispatch_rejections.len(), 1);
        assert_eq!(
            plan.dispatch_rejections[0].reason,
            CapabilityRejectionReason::DispatchGridExceeded {
                axis: 0,
                required_workgroups: 32,
                available_workgroups: 31,
            }
        );
        assert!(matches!(
            plan.selection.outcome(),
            SelectionOutcome::NoCandidate { offered: 0, .. }
        ));
    }
}
