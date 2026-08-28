//! Device-aware admissibility planning for an already-selected FLAT semantic.
//!
//! This crate is deliberately narrower than routing or autotuning. It consumes
//! three facts that have already been established elsewhere:
//!
//! 1. a semantic-only [`SemanticSelectionDecision`];
//! 2. the semantic/runtime compatibility declarations from
//!    [`SemanticExecutionCatalog`];
//! 3. FLAT's existing dense-Q4 [`AttentionProblem`] and
//!    [`RuntimeDeviceCapabilities`] model.
//!
//! It then asks one bounded question for the currently generic forward family:
//!
//! > Which existing dense-Q4 [`KernelCandidate`] values are both executable on
//! > this device/problem and declared compatible with the selected semantic?
//!
//! The result is an admissibility plan, not an execution choice. No candidate
//! is timed, no autotuner is invoked, no backend is opened and the mathematical
//! semantic can never be substituted because a different kernel happens to be
//! available.

#![forbid(unsafe_code)]

use flat_attention::{
    kernel_candidates::{generate_candidates, KernelCandidate, SelectionPolicy},
    kernel_ir::AttentionProblem,
    RuntimeDeviceCapabilities, RuntimeKernelId,
};
use flat_semantic::v1::SemanticId;
use flat_semantic_control::SemanticSelectionDecision;
use flat_semantic_execution::{ExecutionRole, SemanticExecutionCatalog};

/// Version of the device-admissible semantic planning contract.
pub const SEMANTIC_FORWARD_PLANNING_VERSION: u16 = 1;

/// Deterministic forward admissibility plan for one exact selected semantic.
///
/// Every candidate in this record already survived FLAT's existing dense-Q4
/// problem/module validation, lifecycle policy and static device-capability
/// prefilter. That does **not** make the candidate benchmark-selected or prove
/// model/scientific quality; correctness/timing/autotuning remain later gates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardExecutionPlan {
    semantic: SemanticId,
    problem: AttentionProblem,
    device_capability_fingerprint: u64,
    selection_policy: SelectionPolicy,
    compatible_runtime_kernels: Vec<RuntimeKernelId>,
    candidates: Vec<KernelCandidate>,
}

impl ForwardExecutionPlan {
    /// Exact mathematical semantic preserved from the caller's selection.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }

    /// Dense-Q4 semantic geometry supplied to the existing candidate generator.
    #[must_use]
    pub const fn problem(&self) -> AttentionProblem {
        self.problem
    }

    /// Stable fingerprint of the explicit device capabilities used for this
    /// plan. Adapter marketing names are intentionally absent.
    #[must_use]
    pub const fn device_capability_fingerprint(&self) -> u64 {
        self.device_capability_fingerprint
    }

    /// Existing FLAT candidate-generation policy used for this plan.
    #[must_use]
    pub const fn selection_policy(&self) -> SelectionPolicy {
        self.selection_policy
    }

    /// Runtime families declared compatible by the semantic execution catalog
    /// before problem/device filtering.
    #[must_use]
    pub fn compatible_runtime_kernels(&self) -> &[RuntimeKernelId] {
        &self.compatible_runtime_kernels
    }

    /// Device/problem-admissible dense-Q4 candidates, still unranked by timing.
    #[must_use]
    pub fn candidates(&self) -> &[KernelCandidate] {
        &self.candidates
    }
}

/// Explicit ordinary outcomes of forward semantic execution planning.
///
/// These are not runtime errors. Empty surfaces stay distinguishable so a
/// caller can tell whether semantic selection failed, FLAT has no compatible
/// implementation declaration, or implementations exist but none are
/// admissible for the supplied device/problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForwardPlanningOutcome {
    /// At least one dense-Q4 candidate is admissible.
    Ready(ForwardExecutionPlan),
    /// No mathematical semantic was selected upstream.
    NoSemanticSelection,
    /// The selected semantic has no forward runtime compatibility declaration.
    NoCompatibleRuntimeFamily {
        /// Exact semantic that remains selected; no fallback occurred.
        semantic: SemanticId,
    },
    /// Compatible runtime families exist, but the existing dense-Q4 candidate
    /// generator admitted none for this problem/device/policy.
    NoDeviceAdmissibleCandidate {
        /// Exact semantic that remains selected; no fallback occurred.
        semantic: SemanticId,
        /// Catalog-declared forward runtime families considered by the plan.
        compatible_runtime_kernels: Vec<RuntimeKernelId>,
        /// Stable capability fingerprint of the rejected device world.
        device_capability_fingerprint: u64,
    },
}

impl ForwardPlanningOutcome {
    /// Selected semantic preserved by this outcome, when one existed upstream.
    #[must_use]
    pub const fn semantic(&self) -> Option<&SemanticId> {
        match self {
            Self::Ready(plan) => Some(plan.semantic()),
            Self::NoSemanticSelection => None,
            Self::NoCompatibleRuntimeFamily { semantic }
            | Self::NoDeviceAdmissibleCandidate { semantic, .. } => Some(semantic),
        }
    }
}

/// Intersect semantic/runtime compatibility with FLAT's existing dense-Q4
/// forward candidate generation and device capability prefilter.
///
/// This function performs no timing, correctness replay, benchmark ranking,
/// pipeline creation, WGPU dispatch or semantic substitution. The deterministic
/// order of surviving candidates is inherited from [`generate_candidates`].
#[must_use]
pub fn plan_forward_execution(
    catalog: &SemanticExecutionCatalog,
    selection: &SemanticSelectionDecision,
    problem: &AttentionProblem,
    capabilities: &RuntimeDeviceCapabilities,
    policy: &SelectionPolicy,
) -> ForwardPlanningOutcome {
    let Some(semantic) = selection.semantic() else {
        return ForwardPlanningOutcome::NoSemanticSelection;
    };

    let compatible_runtime_kernels = catalog.compatible_kernels(selection, ExecutionRole::Forward);
    if compatible_runtime_kernels.is_empty() {
        return ForwardPlanningOutcome::NoCompatibleRuntimeFamily {
            semantic: semantic.clone(),
        };
    }

    let candidates = generate_candidates(problem, capabilities, policy)
        .into_iter()
        .filter(|candidate| {
            candidate
                .runtime_kernel_id()
                .is_some_and(|kernel| compatible_runtime_kernels.contains(&kernel))
        })
        .collect::<Vec<_>>();

    let device_capability_fingerprint = capabilities.stable_fingerprint();
    if candidates.is_empty() {
        return ForwardPlanningOutcome::NoDeviceAdmissibleCandidate {
            semantic: semantic.clone(),
            compatible_runtime_kernels,
            device_capability_fingerprint,
        };
    }

    ForwardPlanningOutcome::Ready(ForwardExecutionPlan {
        semantic: semantic.clone(),
        problem: *problem,
        device_capability_fingerprint,
        selection_policy: *policy,
        compatible_runtime_kernels,
        candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_attention::{AttentionShape, FlatAttentionConfig};
    use flat_semantic::v1::{SemanticFamily, SemanticId};
    use flat_semantic_execution::{
        standard_softmax_runtime_catalog, ExecutionBinding, SemanticExecutionCatalog,
    };

    fn semantic(family: SemanticFamily, name: &str, revision: u32) -> SemanticId {
        SemanticId::new(family, name, revision).unwrap()
    }

    fn standard_selection() -> SemanticSelectionDecision {
        SemanticSelectionDecision::Selected {
            semantic: semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1),
            preference_rank: 0,
        }
    }

    fn problem() -> AttentionProblem {
        AttentionProblem::from_shape(
            &AttentionShape {
                batch: 2,
                heads: 4,
                seq_len: 129,
                head_dim: 64,
            },
            FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
        )
        .unwrap()
    }

    fn capabilities(subgroup_supported: bool) -> RuntimeDeviceCapabilities {
        RuntimeDeviceCapabilities {
            max_workgroups_per_dimension: 65_535,
            max_workgroup_size_x: 64,
            max_workgroup_size_y: 1_024,
            max_workgroup_size_z: 64,
            max_workgroup_storage_bytes: 32_768,
            max_binding_entries: 8,
            max_storage_buffer_binding_size: 1 << 30,
            subgroup_supported,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f16_supported: true,
        }
    }

    #[test]
    fn selected_standard_softmax_intersects_catalog_and_device_candidates() {
        let catalog = standard_softmax_runtime_catalog();
        let selection = standard_selection();
        let problem = problem();
        let capabilities = capabilities(true);
        let policy = SelectionPolicy::default();

        let outcome =
            plan_forward_execution(&catalog, &selection, &problem, &capabilities, &policy);
        let ForwardPlanningOutcome::Ready(plan) = outcome else {
            panic!("expected a ready StandardSoftmax forward plan");
        };

        assert_eq!(plan.semantic(), selection.semantic().unwrap());
        assert_eq!(plan.problem(), problem);
        assert_eq!(
            plan.device_capability_fingerprint(),
            capabilities.stable_fingerprint()
        );
        assert_eq!(plan.selection_policy(), policy);
        assert!(!plan.candidates().is_empty());
        assert!(plan.candidates().iter().all(|candidate| {
            candidate
                .runtime_kernel_id()
                .is_some_and(|kernel| plan.compatible_runtime_kernels().contains(&kernel))
        }));
        assert!(plan
            .candidates()
            .iter()
            .any(|candidate| candidate.runtime_kernel_id() == Some(RuntimeKernelId::Q4Subgroup)));
    }

    #[test]
    fn device_capability_change_prunes_subgroup_without_changing_semantic() {
        let catalog = standard_softmax_runtime_catalog();
        let selection = standard_selection();
        let problem = problem();
        let policy = SelectionPolicy::default();

        let with_subgroup =
            plan_forward_execution(&catalog, &selection, &problem, &capabilities(true), &policy);
        let without_subgroup = plan_forward_execution(
            &catalog,
            &selection,
            &problem,
            &capabilities(false),
            &policy,
        );

        let ForwardPlanningOutcome::Ready(with_subgroup) = with_subgroup else {
            panic!("subgroup-capable profile should be plannable");
        };
        let ForwardPlanningOutcome::Ready(without_subgroup) = without_subgroup else {
            panic!("portable profile should remain plannable");
        };

        assert_eq!(with_subgroup.semantic(), without_subgroup.semantic());
        assert!(with_subgroup
            .candidates()
            .iter()
            .any(|candidate| candidate.runtime_kernel_id() == Some(RuntimeKernelId::Q4Subgroup)));
        assert!(without_subgroup
            .candidates()
            .iter()
            .all(|candidate| candidate.runtime_kernel_id() != Some(RuntimeKernelId::Q4Subgroup)));
    }

    #[test]
    fn unsupported_semantic_has_no_runtime_fallback() {
        let recurrent = semantic(SemanticFamily::RecurrentMemory, "delta-memory", 1);
        let selection = SemanticSelectionDecision::Selected {
            semantic: recurrent.clone(),
            preference_rank: 0,
        };
        let original = selection.clone();
        let outcome = plan_forward_execution(
            &standard_softmax_runtime_catalog(),
            &selection,
            &problem(),
            &capabilities(true),
            &SelectionPolicy::default(),
        );

        assert_eq!(
            outcome,
            ForwardPlanningOutcome::NoCompatibleRuntimeFamily {
                semantic: recurrent.clone()
            }
        );
        assert_eq!(outcome.semantic(), Some(&recurrent));
        assert_eq!(selection, original);
    }

    #[test]
    fn no_semantic_selection_cannot_manufacture_execution_candidates() {
        let outcome = plan_forward_execution(
            &standard_softmax_runtime_catalog(),
            &SemanticSelectionDecision::NoRegisteredPreference,
            &problem(),
            &capabilities(true),
            &SelectionPolicy::default(),
        );
        assert_eq!(outcome, ForwardPlanningOutcome::NoSemanticSelection);
        assert_eq!(outcome.semantic(), None);
    }

    #[test]
    fn compatible_runtime_can_still_be_device_inadmissible() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let selection = SemanticSelectionDecision::Selected {
            semantic: standard.clone(),
            preference_rank: 0,
        };
        let catalog = SemanticExecutionCatalog::new([ExecutionBinding::new(
            standard.clone(),
            ExecutionRole::Forward,
            RuntimeKernelId::Q4Subgroup,
        )])
        .unwrap();
        let capabilities = capabilities(false);

        let outcome = plan_forward_execution(
            &catalog,
            &selection,
            &problem(),
            &capabilities,
            &SelectionPolicy::default(),
        );
        assert_eq!(
            outcome,
            ForwardPlanningOutcome::NoDeviceAdmissibleCandidate {
                semantic: standard,
                compatible_runtime_kernels: vec![RuntimeKernelId::Q4Subgroup],
                device_capability_fingerprint: capabilities.stable_fingerprint(),
            }
        );
    }

    #[test]
    fn catalog_projection_can_restrict_candidate_surface_without_ranking() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let selection = SemanticSelectionDecision::Selected {
            semantic: standard.clone(),
            preference_rank: 0,
        };
        let catalog = SemanticExecutionCatalog::new([ExecutionBinding::new(
            standard,
            ExecutionRole::Forward,
            RuntimeKernelId::Q4Portable,
        )])
        .unwrap();

        let outcome = plan_forward_execution(
            &catalog,
            &selection,
            &problem(),
            &capabilities(true),
            &SelectionPolicy::default(),
        );
        let ForwardPlanningOutcome::Ready(plan) = outcome else {
            panic!("portable binding should remain device admissible");
        };

        assert_eq!(
            plan.compatible_runtime_kernels(),
            &[RuntimeKernelId::Q4Portable]
        );
        assert_eq!(plan.candidates().len(), 1);
        assert_eq!(
            plan.candidates()[0].runtime_kernel_id(),
            Some(RuntimeKernelId::Q4Portable)
        );
    }
}
