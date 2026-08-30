//! Fail-closed provenance revalidation before exact semantic tuning.
//!
//! Planning-time validation is not sufficient when an exact execution plan is
//! retained across registry or execution-catalog updates. These bridges
//! revalidate the exact semantic selection immediately before
//! correctness-gated timing and can additionally require the current forward
//! execution surface to match the one recorded in the plan. A stale decision
//! or changed compatible-kernel surface therefore cannot reach either the
//! correctness gate or the timing harness, and no replacement semantic or
//! candidate set is synthesized.

use flat_attention::kernel_autotune::{
    BenchmarkProtocol, CorrectnessGate, ExplicitCandidateSetError, TimingHarness,
};
use flat_attention::RuntimeKernelId;
use flat_semantic_execution::{ExecutionRole, SemanticExecutionCatalog};
use flat_semantic_registry::SemanticRegistry;
use flat_semantic_selection::SemanticSelectionValidationError;

use crate::exact_selection::{
    tune_exact_forward_execution_plan, ExactForwardExecutionPlan, ExactForwardTuningRecord,
};

/// Failure while revalidating or tuning one exact semantic execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidatedExactForwardTuningError {
    /// The exact semantic decision embedded in the plan no longer matches the
    /// current registry.
    Selection(SemanticSelectionValidationError),
    /// The already-planned explicit candidate set is structurally invalid.
    CandidateSet(ExplicitCandidateSetError),
}

impl From<SemanticSelectionValidationError> for ValidatedExactForwardTuningError {
    fn from(value: SemanticSelectionValidationError) -> Self {
        Self::Selection(value)
    }
}

impl From<ExplicitCandidateSetError> for ValidatedExactForwardTuningError {
    fn from(value: ExplicitCandidateSetError) -> Self {
        Self::CandidateSet(value)
    }
}

/// Failure while validating the current execution catalog and then performing
/// the existing exact semantic tuning checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogValidatedExactForwardTuningError {
    /// Forward compatibility for the exact selected semantic changed since the
    /// execution plan was produced.
    ExecutionCatalogDrift {
        /// Canonical compatible runtime surface recorded in the plan.
        planned: Vec<RuntimeKernelId>,
        /// Canonical compatible runtime surface declared by the current
        /// execution catalog.
        current: Vec<RuntimeKernelId>,
    },
    /// Registry provenance or explicit candidate-set validation failed.
    Tuning(ValidatedExactForwardTuningError),
}

impl From<ValidatedExactForwardTuningError> for CatalogValidatedExactForwardTuningError {
    fn from(value: ValidatedExactForwardTuningError) -> Self {
        Self::Tuning(value)
    }
}

/// Revalidate the exact semantic decision immediately before tuning the
/// already-admitted implementation candidates.
///
/// Validation happens before either correctness verification or measurement.
/// The function delegates candidate validation, correctness gating, timing and
/// deterministic implementation ranking to the existing exact tuning bridge.
/// It never regenerates candidates and never changes the selected semantic.
///
/// # Errors
///
/// Returns [`ValidatedExactForwardTuningError::Selection`] if registry
/// provenance changed since selection, or
/// [`ValidatedExactForwardTuningError::CandidateSet`] if the explicit planned
/// candidate set is structurally invalid.
pub fn tune_validated_exact_forward_execution_plan(
    registry: &SemanticRegistry,
    plan: &ExactForwardExecutionPlan,
    protocol: BenchmarkProtocol,
    gate: &mut dyn CorrectnessGate,
    harness: &mut dyn TimingHarness,
) -> Result<ExactForwardTuningRecord, ValidatedExactForwardTuningError> {
    plan.selection().validate_against_registry(registry)?;
    Ok(tune_exact_forward_execution_plan(
        plan, protocol, gate, harness,
    )?)
}

/// Require the current execution catalog to preserve the exact forward runtime
/// surface recorded in the plan, then perform the existing registry-validated
/// tuning path.
///
/// Only bindings for the already-selected semantic and forward role are
/// compared. Unrelated catalog changes do not invalidate the plan. A changed
/// relevant surface fails closed before correctness verification or timing; it
/// is never widened, regenerated, or interpreted as permission to select a
/// different semantic.
///
/// # Errors
///
/// Returns [`CatalogValidatedExactForwardTuningError::ExecutionCatalogDrift`]
/// when the current compatible forward runtime surface differs from the one
/// recorded during planning, or wraps the existing validated tuning errors.
pub fn tune_catalog_validated_exact_forward_execution_plan(
    registry: &SemanticRegistry,
    catalog: &SemanticExecutionCatalog,
    plan: &ExactForwardExecutionPlan,
    protocol: BenchmarkProtocol,
    gate: &mut dyn CorrectnessGate,
    harness: &mut dyn TimingHarness,
) -> Result<ExactForwardTuningRecord, CatalogValidatedExactForwardTuningError> {
    let current = catalog
        .bindings()
        .iter()
        .filter(|binding| {
            binding.semantic() == plan.selection().semantic()
                && binding.role() == ExecutionRole::Forward
        })
        .map(|binding| binding.kernel())
        .collect::<Vec<_>>();
    let planned = plan.execution().compatible_runtime_kernels();
    if current != planned {
        return Err(
            CatalogValidatedExactForwardTuningError::ExecutionCatalogDrift {
                planned: planned.to_vec(),
                current,
            },
        );
    }
    Ok(tune_validated_exact_forward_execution_plan(
        registry, plan, protocol, gate, harness,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_attention::kernel_autotune::TimingSample;
    use flat_attention::kernel_candidates::{KernelCandidate, SelectionPolicy};
    use flat_attention::kernel_ir::AttentionProblem;
    use flat_attention::{
        AttentionShape, FlatAttentionConfig, RuntimeDeviceCapabilities, RuntimeKernelId,
    };
    use flat_semantic::v1::{SemanticFamily, SemanticId};
    use flat_semantic_execution::{
        standard_softmax_runtime_catalog, ExecutionBinding, ExecutionRole,
        SemanticExecutionCatalog,
    };
    use flat_semantic_registry::SemanticRegistry;
    use flat_semantic_selection::{
        ExactSemanticSelectionPolicy, SemanticSelectionRequest, SemanticSelectionValidationError,
    };

    use crate::exact_selection::{plan_exact_forward_execution, ExactForwardPlanningOutcome};

    fn semantic() -> SemanticId {
        SemanticId::new(SemanticFamily::StandardSoftmax, "standard-softmax", 1).unwrap()
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

    fn ready_plan(registry: &SemanticRegistry) -> ExactForwardExecutionPlan {
        let selected = semantic();
        let selection = ExactSemanticSelectionPolicy
            .select(registry, &SemanticSelectionRequest::new(selected))
            .unwrap();
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
        plan
    }

    #[derive(Default)]
    struct CountingGate {
        calls: usize,
    }

    impl CorrectnessGate for CountingGate {
        fn verify(&mut self, _: &KernelCandidate, _: &AttentionProblem) -> Result<(), String> {
            self.calls += 1;
            Ok(())
        }
    }

    #[derive(Default)]
    struct CountingHarness {
        calls: usize,
    }

    impl TimingHarness for CountingHarness {
        fn measure(
            &mut self,
            _: &KernelCandidate,
            _: &AttentionProblem,
            protocol: &BenchmarkProtocol,
        ) -> Result<TimingSample, String> {
            self.calls += 1;
            Ok(TimingSample {
                median_us: 1.0,
                p95_us: 1.5,
                iterations: protocol.iterations,
            })
        }
    }

    #[test]
    fn unchanged_registry_allows_exact_tuning() {
        let registry = SemanticRegistry::new([semantic()]).unwrap();
        let plan = ready_plan(&registry);
        let mut gate = CountingGate::default();
        let mut harness = CountingHarness::default();

        let record = tune_validated_exact_forward_execution_plan(
            &registry,
            &plan,
            BenchmarkProtocol {
                warmups: 1,
                iterations: 2,
            },
            &mut gate,
            &mut harness,
        )
        .unwrap();

        assert_eq!(record.selection(), plan.selection());
        assert!(gate.calls > 0);
        assert!(harness.calls > 0);
    }

    #[test]
    fn registry_drift_fails_before_correctness_or_timing() {
        let original = SemanticRegistry::new([semantic()]).unwrap();
        let plan = ready_plan(&original);
        let extra = SemanticId::new(SemanticFamily::RecurrentMemory, "delta-memory", 1).unwrap();
        let current = SemanticRegistry::new([semantic(), extra]).unwrap();
        let mut gate = CountingGate::default();
        let mut harness = CountingHarness::default();

        let error = tune_validated_exact_forward_execution_plan(
            &current,
            &plan,
            BenchmarkProtocol {
                warmups: 1,
                iterations: 2,
            },
            &mut gate,
            &mut harness,
        )
        .unwrap_err();

        assert_eq!(
            error,
            ValidatedExactForwardTuningError::Selection(
                SemanticSelectionValidationError::RegistryFingerprintMismatch {
                    decision: original.stable_fingerprint(),
                    current: current.stable_fingerprint(),
                }
            )
        );
        assert_eq!(gate.calls, 0);
        assert_eq!(harness.calls, 0);
    }

    #[test]
    fn unchanged_execution_catalog_allows_validated_tuning() {
        let registry = SemanticRegistry::new([semantic()]).unwrap();
        let catalog = standard_softmax_runtime_catalog();
        let plan = ready_plan(&registry);
        let mut gate = CountingGate::default();
        let mut harness = CountingHarness::default();

        let record = tune_catalog_validated_exact_forward_execution_plan(
            &registry,
            &catalog,
            &plan,
            BenchmarkProtocol {
                warmups: 1,
                iterations: 2,
            },
            &mut gate,
            &mut harness,
        )
        .unwrap();

        assert_eq!(record.selection(), plan.selection());
        assert!(gate.calls > 0);
        assert!(harness.calls > 0);
    }

    #[test]
    fn execution_catalog_drift_fails_before_correctness_or_timing() {
        let registry = SemanticRegistry::new([semantic()]).unwrap();
        let plan = ready_plan(&registry);
        let current = SemanticExecutionCatalog::new([ExecutionBinding::new(
            semantic(),
            ExecutionRole::Forward,
            RuntimeKernelId::Q4Portable,
        )])
        .unwrap();
        let mut gate = CountingGate::default();
        let mut harness = CountingHarness::default();

        let error = tune_catalog_validated_exact_forward_execution_plan(
            &registry,
            &current,
            &plan,
            BenchmarkProtocol {
                warmups: 1,
                iterations: 2,
            },
            &mut gate,
            &mut harness,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CatalogValidatedExactForwardTuningError::ExecutionCatalogDrift {
                planned,
                current,
            } if planned.len() > current.len() && current == vec![RuntimeKernelId::Q4Portable]
        ));
        assert_eq!(gate.calls, 0);
        assert_eq!(harness.calls, 0);
    }
}
