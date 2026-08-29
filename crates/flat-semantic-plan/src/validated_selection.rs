//! Fail-closed registry revalidation before exact semantic execution planning.
//!
//! This bridge composes E4k selection-provenance validation with the existing
//! exact-selection planner. Validation happens before runtime-family or device
//! candidate admissibility is considered, so a stale semantic decision cannot
//! be executed or silently replaced by another semantic.

use flat_attention::{
    kernel_candidates::SelectionPolicy, kernel_ir::AttentionProblem, RuntimeDeviceCapabilities,
};
use flat_semantic_execution::SemanticExecutionCatalog;
use flat_semantic_registry::SemanticRegistry;
use flat_semantic_selection::{SemanticSelectionDecision, SemanticSelectionValidationError};

use crate::exact_selection::{plan_exact_forward_execution, ExactForwardPlanningOutcome};

/// Revalidate one exact semantic decision against the current registry before
/// producing any execution plan.
///
/// # Errors
///
/// Returns the exact selection-provenance validation error when the selected
/// semantic was removed, its identity fingerprint changed, or the complete
/// registry fingerprint differs from the registry that authorized the
/// decision. No execution planning occurs after a validation failure.
pub fn plan_validated_exact_forward_execution(
    registry: &SemanticRegistry,
    catalog: &SemanticExecutionCatalog,
    selection: &SemanticSelectionDecision,
    problem: &AttentionProblem,
    capabilities: &RuntimeDeviceCapabilities,
    policy: &SelectionPolicy,
) -> Result<ExactForwardPlanningOutcome, SemanticSelectionValidationError> {
    selection.validate_against_registry(registry)?;
    Ok(plan_exact_forward_execution(
        catalog,
        selection,
        problem,
        capabilities,
        policy,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_attention::{AttentionShape, FlatAttentionConfig};
    use flat_semantic::v1::{SemanticFamily, SemanticId};
    use flat_semantic_execution::standard_softmax_runtime_catalog;
    use flat_semantic_selection::{
        ExactSemanticSelectionPolicy, SemanticSelectionRequest, SemanticSelectionValidationError,
    };

    fn semantic(family: SemanticFamily, name: &str, revision: u32) -> SemanticId {
        SemanticId::new(family, name, revision).unwrap()
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

    #[test]
    fn current_registry_allows_exact_planning() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let registry = SemanticRegistry::new([standard.clone()]).unwrap();
        let selection = ExactSemanticSelectionPolicy
            .select(&registry, &SemanticSelectionRequest::new(standard))
            .unwrap();

        let outcome = plan_validated_exact_forward_execution(
            &registry,
            &standard_softmax_runtime_catalog(),
            &selection,
            &problem(),
            &capabilities(),
            &SelectionPolicy::default(),
        )
        .unwrap();

        assert!(matches!(outcome, ExactForwardPlanningOutcome::Ready(_)));
    }

    #[test]
    fn registry_drift_fails_before_exact_planning_without_fallback() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let extra = semantic(SemanticFamily::RecurrentMemory, "delta-memory", 1);
        let original = SemanticRegistry::new([standard.clone()]).unwrap();
        let current = SemanticRegistry::new([extra, standard.clone()]).unwrap();
        let selection = ExactSemanticSelectionPolicy
            .select(&original, &SemanticSelectionRequest::new(standard))
            .unwrap();

        let error = plan_validated_exact_forward_execution(
            &current,
            &standard_softmax_runtime_catalog(),
            &selection,
            &problem(),
            &capabilities(),
            &SelectionPolicy::default(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            SemanticSelectionValidationError::RegistryFingerprintMismatch {
                decision: original.stable_fingerprint(),
                current: current.stable_fingerprint(),
            }
        );
    }

    #[test]
    fn removed_semantic_fails_without_revision_substitution() {
        let selected = semantic(SemanticFamily::Experimental, "candidate", 1);
        let replacement = semantic(SemanticFamily::Experimental, "candidate", 2);
        let original = SemanticRegistry::new([selected.clone()]).unwrap();
        let current = SemanticRegistry::new([replacement]).unwrap();
        let selection = ExactSemanticSelectionPolicy
            .select(&original, &SemanticSelectionRequest::new(selected.clone()))
            .unwrap();

        let error = plan_validated_exact_forward_execution(
            &current,
            &standard_softmax_runtime_catalog(),
            &selection,
            &problem(),
            &capabilities(),
            &SelectionPolicy::default(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            SemanticSelectionValidationError::Selection(
                flat_semantic_selection::SemanticSelectionError::UnregisteredSemantic {
                    requested
                }
            ) if requested == selected
        ));
    }
}
