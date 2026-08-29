//! Exact registered-semantic provenance bridge for forward planning and tuning.
//!
//! This module is additive to the older preference-ranked control-plane bridge.
//! It accepts the exact registered selection decision produced by
//! `flat-semantic-selection`, preserves that decision unchanged, and delegates
//! implementation admissibility and benchmark ranking to the existing planner
//! and autotuner. The exact selection is never replaced by a runtime outcome.

use flat_attention::{
    kernel_autotune::{
        BenchmarkProtocol, CorrectnessGate, ExplicitCandidateSetError, TimingHarness,
    },
    kernel_candidates::SelectionPolicy,
    kernel_ir::AttentionProblem,
    RuntimeDeviceCapabilities, RuntimeKernelId,
};
use flat_semantic_control::SemanticSelectionDecision as PreferenceSelectionDecision;
use flat_semantic_execution::SemanticExecutionCatalog;
use flat_semantic_selection::SemanticSelectionDecision as ExactSemanticSelectionDecision;

use crate::{
    plan_forward_execution, tune_forward_execution_plan, ForwardExecutionPlan,
    ForwardPlanningOutcome, ForwardTuningRecord,
};

/// Forward plan whose semantic provenance is the exact registered-selection
/// decision from `flat-semantic-selection`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactForwardExecutionPlan {
    selection: ExactSemanticSelectionDecision,
    execution: ForwardExecutionPlan,
}

impl ExactForwardExecutionPlan {
    /// Exact registered semantic-selection decision that authorized the plan.
    #[must_use]
    pub const fn selection(&self) -> &ExactSemanticSelectionDecision {
        &self.selection
    }

    /// Existing device-admissible execution plan for that exact semantic.
    #[must_use]
    pub const fn execution(&self) -> &ForwardExecutionPlan {
        &self.execution
    }
}

/// Explicit result of planning from an exact registered semantic decision.
///
/// No variant can substitute another semantic. Empty implementation surfaces
/// retain the exact selection decision that produced them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactForwardPlanningOutcome {
    /// At least one implementation candidate survived semantic/runtime and
    /// device-admissibility filtering.
    Ready(ExactForwardExecutionPlan),
    /// The exact selected semantic has no compatible forward runtime family.
    NoCompatibleRuntimeFamily {
        selection: ExactSemanticSelectionDecision,
    },
    /// Compatible runtime families exist, but none are admissible for the
    /// supplied problem/device/policy.
    NoDeviceAdmissibleCandidate {
        selection: ExactSemanticSelectionDecision,
        compatible_runtime_kernels: Vec<RuntimeKernelId>,
        device_capability_fingerprint: u64,
    },
    /// Defensive representation of an impossible mismatch between an exact
    /// registered selection and the older planner's no-selection outcome.
    SelectionInvariantViolation {
        selection: ExactSemanticSelectionDecision,
    },
}

impl ExactForwardPlanningOutcome {
    /// Exact registered semantic-selection provenance for every outcome.
    #[must_use]
    pub const fn selection(&self) -> &ExactSemanticSelectionDecision {
        match self {
            Self::Ready(plan) => plan.selection(),
            Self::NoCompatibleRuntimeFamily { selection }
            | Self::NoDeviceAdmissibleCandidate { selection, .. }
            | Self::SelectionInvariantViolation { selection } => selection,
        }
    }
}

/// Tuning evidence paired with the exact registered semantic decision that
/// authorized its implementation surface.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactForwardTuningRecord {
    selection: ExactSemanticSelectionDecision,
    tuning: ForwardTuningRecord,
}

impl ExactForwardTuningRecord {
    /// Exact registered semantic-selection provenance.
    #[must_use]
    pub const fn selection(&self) -> &ExactSemanticSelectionDecision {
        &self.selection
    }

    /// Existing correctness-gated benchmark selection evidence.
    #[must_use]
    pub const fn tuning(&self) -> &ForwardTuningRecord {
        &self.tuning
    }
}

/// Plan implementations for one exact registered semantic decision.
///
/// The implementation path is intentionally delegated to the existing E4g
/// planner by projecting only the already-selected semantic identity into its
/// compatibility API. The exact decision itself is retained unchanged in the
/// returned provenance wrapper.
#[must_use]
pub fn plan_exact_forward_execution(
    catalog: &SemanticExecutionCatalog,
    selection: &ExactSemanticSelectionDecision,
    problem: &AttentionProblem,
    capabilities: &RuntimeDeviceCapabilities,
    policy: &SelectionPolicy,
) -> ExactForwardPlanningOutcome {
    let preference_projection = PreferenceSelectionDecision::Selected {
        semantic: selection.semantic().clone(),
        preference_rank: 0,
    };
    match plan_forward_execution(
        catalog,
        &preference_projection,
        problem,
        capabilities,
        policy,
    ) {
        ForwardPlanningOutcome::Ready(execution) => {
            ExactForwardPlanningOutcome::Ready(ExactForwardExecutionPlan {
                selection: selection.clone(),
                execution,
            })
        }
        ForwardPlanningOutcome::NoCompatibleRuntimeFamily { .. } => {
            ExactForwardPlanningOutcome::NoCompatibleRuntimeFamily {
                selection: selection.clone(),
            }
        }
        ForwardPlanningOutcome::NoDeviceAdmissibleCandidate {
            compatible_runtime_kernels,
            device_capability_fingerprint,
            ..
        } => ExactForwardPlanningOutcome::NoDeviceAdmissibleCandidate {
            selection: selection.clone(),
            compatible_runtime_kernels,
            device_capability_fingerprint,
        },
        ForwardPlanningOutcome::NoSemanticSelection => {
            ExactForwardPlanningOutcome::SelectionInvariantViolation {
                selection: selection.clone(),
            }
        }
    }
}

/// Tune exactly the candidates in an exact-selection forward plan.
///
/// # Errors
///
/// Returns the same structural explicit-candidate error as the existing E4i
/// tuning bridge. Correctness, timing and ranking semantics are unchanged.
pub fn tune_exact_forward_execution_plan(
    plan: &ExactForwardExecutionPlan,
    protocol: BenchmarkProtocol,
    gate: &mut dyn CorrectnessGate,
    harness: &mut dyn TimingHarness,
) -> Result<ExactForwardTuningRecord, ExplicitCandidateSetError> {
    let tuning = tune_forward_execution_plan(plan.execution(), protocol, gate, harness)?;
    Ok(ExactForwardTuningRecord {
        selection: plan.selection().clone(),
        tuning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_attention::kernel_autotune::{TimingHarness, TimingSample};
    use flat_attention::kernel_candidates::KernelCandidate;
    use flat_attention::{AttentionShape, FlatAttentionConfig};
    use flat_semantic::v1::{SemanticFamily, SemanticId};
    use flat_semantic_execution::{
        standard_softmax_runtime_catalog, ExecutionBinding, ExecutionRole,
    };
    use flat_semantic_registry::SemanticRegistry;
    use flat_semantic_selection::{ExactSemanticSelectionPolicy, SemanticSelectionRequest};

    fn semantic() -> SemanticId {
        SemanticId::new(SemanticFamily::StandardSoftmax, "standard-softmax", 1).unwrap()
    }

    fn exact_selection() -> ExactSemanticSelectionDecision {
        let semantic = semantic();
        let registry = SemanticRegistry::new([semantic.clone()]).unwrap();
        ExactSemanticSelectionPolicy
            .select(&registry, &SemanticSelectionRequest::new(semantic))
            .unwrap()
    }

    fn problem() -> AttentionProblem {
        AttentionProblem::from_shape(
            &AttentionShape {
                batch: 1,
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

    fn capabilities() -> RuntimeDeviceCapabilities {
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

    struct PassGate;
    impl CorrectnessGate for PassGate {
        fn verify(&mut self, _: &KernelCandidate, _: &AttentionProblem) -> Result<(), String> {
            Ok(())
        }
    }

    struct Harness;
    impl TimingHarness for Harness {
        fn measure(
            &mut self,
            _: &KernelCandidate,
            _: &AttentionProblem,
            protocol: &BenchmarkProtocol,
        ) -> Result<TimingSample, String> {
            Ok(TimingSample {
                median_us: 1.0,
                p95_us: 1.5,
                iterations: protocol.iterations,
            })
        }
    }

    #[test]
    fn exact_selection_provenance_survives_planning_and_tuning() {
        let selection = exact_selection();
        let selection_fingerprint = selection.stable_fingerprint();
        let outcome = plan_exact_forward_execution(
            &standard_softmax_runtime_catalog(),
            &selection,
            &problem(),
            &capabilities(),
            &SelectionPolicy::default(),
        );
        let ExactForwardPlanningOutcome::Ready(plan) = outcome else {
            panic!("exact StandardSoftmax plan expected");
        };
        assert_eq!(plan.selection().stable_fingerprint(), selection_fingerprint);
        assert_eq!(plan.execution().semantic(), selection.semantic());

        let mut gate = PassGate;
        let mut harness = Harness;
        let record = tune_exact_forward_execution_plan(
            &plan,
            BenchmarkProtocol {
                warmups: 1,
                iterations: 2,
            },
            &mut gate,
            &mut harness,
        )
        .unwrap();
        assert_eq!(
            record.selection().stable_fingerprint(),
            selection_fingerprint
        );
        assert_eq!(record.tuning().semantic(), selection.semantic());
    }

    #[test]
    fn unsupported_exact_semantic_never_falls_back() {
        let selected = SemanticId::new(SemanticFamily::RecurrentMemory, "delta-memory", 1).unwrap();
        let registry = SemanticRegistry::new([selected.clone()]).unwrap();
        let selection = ExactSemanticSelectionPolicy
            .select(&registry, &SemanticSelectionRequest::new(selected.clone()))
            .unwrap();
        let outcome = plan_exact_forward_execution(
            &standard_softmax_runtime_catalog(),
            &selection,
            &problem(),
            &capabilities(),
            &SelectionPolicy::default(),
        );
        assert_eq!(outcome.selection().semantic(), &selected);
        assert!(matches!(
            outcome,
            ExactForwardPlanningOutcome::NoCompatibleRuntimeFamily { .. }
        ));
    }

    #[test]
    fn exact_catalog_restriction_is_not_widened_by_tuning() {
        let selection = exact_selection();
        let catalog = SemanticExecutionCatalog::new([ExecutionBinding::new(
            semantic(),
            ExecutionRole::Forward,
            RuntimeKernelId::Q4Portable,
        )])
        .unwrap();
        let outcome = plan_exact_forward_execution(
            &catalog,
            &selection,
            &problem(),
            &capabilities(),
            &SelectionPolicy::default(),
        );
        let ExactForwardPlanningOutcome::Ready(plan) = outcome else {
            panic!("portable exact plan expected");
        };
        assert_eq!(plan.execution().candidates().len(), 1);
        assert_eq!(
            plan.execution().candidates()[0].runtime_kernel_id(),
            Some(RuntimeKernelId::Q4Portable)
        );
    }
}
