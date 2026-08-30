//! Fail-closed device-capability provenance validation before exact tuning.
//!
//! Exact forward plans already record the device-capability fingerprint that
//! admitted their candidate surface. A retained plan must not be tuned after
//! the effective runtime device capabilities change without detecting that
//! provenance drift first.

use flat_attention::kernel_autotune::{BenchmarkProtocol, CorrectnessGate, TimingHarness};
use flat_attention::RuntimeDeviceCapabilities;
use flat_semantic_execution::SemanticExecutionCatalog;
use flat_semantic_registry::SemanticRegistry;

use crate::exact_selection::{ExactForwardExecutionPlan, ExactForwardTuningRecord};
use crate::validated_tuning::{
    tune_catalog_validated_exact_forward_execution_plan, CatalogValidatedExactForwardTuningError,
};

/// Failure while validating current device provenance and then performing the
/// existing catalog/registry-validated exact tuning path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceValidatedExactForwardTuningError {
    /// The effective runtime-device capabilities differ from those that admitted
    /// the retained plan.
    DeviceCapabilityDrift {
        /// Fingerprint recorded when the plan was produced.
        planned: u64,
        /// Fingerprint of the capabilities supplied at tuning time.
        current: u64,
    },
    /// Existing execution-catalog, registry, or candidate-set validation failed.
    Tuning(CatalogValidatedExactForwardTuningError),
}

impl From<CatalogValidatedExactForwardTuningError> for DeviceValidatedExactForwardTuningError {
    fn from(value: CatalogValidatedExactForwardTuningError) -> Self {
        Self::Tuning(value)
    }
}

/// Require the current device-capability fingerprint to match the one recorded
/// in the exact forward plan, then perform the existing catalog/registry-
/// validated tuning path.
///
/// Device drift is rejected before correctness verification or measurement.
/// The function never regenerates candidates, widens the execution surface, or
/// changes the selected semantic.
///
/// # Errors
///
/// Returns [`DeviceValidatedExactForwardTuningError::DeviceCapabilityDrift`]
/// when current device capabilities differ from planning-time capabilities, or
/// wraps the existing catalog/registry/candidate validation errors.
pub fn tune_device_validated_exact_forward_execution_plan(
    registry: &SemanticRegistry,
    catalog: &SemanticExecutionCatalog,
    capabilities: &RuntimeDeviceCapabilities,
    plan: &ExactForwardExecutionPlan,
    protocol: BenchmarkProtocol,
    gate: &mut dyn CorrectnessGate,
    harness: &mut dyn TimingHarness,
) -> Result<ExactForwardTuningRecord, DeviceValidatedExactForwardTuningError> {
    let planned = plan.execution().device_capability_fingerprint();
    let current = capabilities.stable_fingerprint();
    if current != planned {
        return Err(
            DeviceValidatedExactForwardTuningError::DeviceCapabilityDrift { planned, current },
        );
    }
    Ok(tune_catalog_validated_exact_forward_execution_plan(
        registry, catalog, plan, protocol, gate, harness,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_attention::kernel_autotune::TimingSample;
    use flat_attention::kernel_candidates::{KernelCandidate, SelectionPolicy};
    use flat_attention::kernel_ir::AttentionProblem;
    use flat_attention::{AttentionShape, FlatAttentionConfig};
    use flat_semantic::v1::{SemanticFamily, SemanticId};
    use flat_semantic_execution::standard_softmax_runtime_catalog;
    use flat_semantic_selection::{ExactSemanticSelectionPolicy, SemanticSelectionRequest};

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
    fn unchanged_device_capabilities_allow_validated_tuning() {
        let registry = SemanticRegistry::new([semantic()]).unwrap();
        let catalog = standard_softmax_runtime_catalog();
        let capabilities = capabilities();
        let plan = ready_plan(&registry);
        let mut gate = CountingGate::default();
        let mut harness = CountingHarness::default();

        let record = tune_device_validated_exact_forward_execution_plan(
            &registry,
            &catalog,
            &capabilities,
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
    fn device_capability_drift_fails_before_correctness_or_timing() {
        let registry = SemanticRegistry::new([semantic()]).unwrap();
        let catalog = standard_softmax_runtime_catalog();
        let plan = ready_plan(&registry);
        let mut current = capabilities();
        current.subgroup_supported = false;
        let mut gate = CountingGate::default();
        let mut harness = CountingHarness::default();

        let error = tune_device_validated_exact_forward_execution_plan(
            &registry,
            &catalog,
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
            DeviceValidatedExactForwardTuningError::DeviceCapabilityDrift {
                planned: plan.execution().device_capability_fingerprint(),
                current: current.stable_fingerprint(),
            }
        );
        assert_eq!(gate.calls, 0);
        assert_eq!(harness.calls, 0);
    }
}
