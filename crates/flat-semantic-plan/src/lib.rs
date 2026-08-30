//! Device-aware admissibility planning and bounded tuning for an already-selected FLAT semantic.
//!
//! Planning intersects semantic/runtime compatibility with FLAT's existing
//! device-admissible candidate generator. Tuning may then consume exactly that
//! plan; it cannot regenerate or widen the candidate surface and it never
//! changes the selected mathematical semantic.

#![forbid(unsafe_code)]

pub mod device_validated_tuning;
pub mod exact_selection;
pub mod validated_selection;
pub mod validated_tuning;

use flat_attention::{
    kernel_autotune::{
        tune_candidates, BenchmarkProtocol, CorrectnessGate, ExplicitCandidateSetError,
        SelectionRecord, TimingHarness,
    },
    kernel_candidates::{generate_candidates, KernelCandidate, SelectionPolicy},
    kernel_ir::AttentionProblem,
    RuntimeDeviceCapabilities, RuntimeKernelId,
};
use flat_semantic::v1::SemanticId;
use flat_semantic_control::SemanticSelectionDecision;
use flat_semantic_execution::{ExecutionRole, SemanticExecutionCatalog};

/// Version of the device-admissible semantic planning contract.
pub const SEMANTIC_FORWARD_PLANNING_VERSION: u16 = 1;
/// Version of the semantic-bound tuning provenance contract.
pub const SEMANTIC_FORWARD_TUNING_VERSION: u16 = 1;

/// Deterministic forward admissibility plan for one exact selected semantic.
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
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }
    #[must_use]
    pub const fn problem(&self) -> AttentionProblem {
        self.problem
    }
    #[must_use]
    pub const fn device_capability_fingerprint(&self) -> u64 {
        self.device_capability_fingerprint
    }
    #[must_use]
    pub const fn selection_policy(&self) -> SelectionPolicy {
        self.selection_policy
    }
    #[must_use]
    pub fn compatible_runtime_kernels(&self) -> &[RuntimeKernelId] {
        &self.compatible_runtime_kernels
    }
    #[must_use]
    pub fn candidates(&self) -> &[KernelCandidate] {
        &self.candidates
    }
}

/// Tuning evidence bound to the exact semantic plan that admitted its candidates.
///
/// `selection` is benchmark evidence among the plan's implementations. It is
/// never a semantic ranking and cannot replace `semantic`. The exact benchmark
/// protocol is retained even when no candidate produces timing evidence, so
/// warm-up and measured-iteration provenance is not inferred from outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardTuningRecord {
    semantic: SemanticId,
    problem: AttentionProblem,
    device_capability_fingerprint: u64,
    selection_policy: SelectionPolicy,
    compatible_runtime_kernels: Vec<RuntimeKernelId>,
    benchmark_protocol: BenchmarkProtocol,
    selection: SelectionRecord,
}

impl ForwardTuningRecord {
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }
    #[must_use]
    pub const fn problem(&self) -> AttentionProblem {
        self.problem
    }
    #[must_use]
    pub const fn device_capability_fingerprint(&self) -> u64 {
        self.device_capability_fingerprint
    }
    #[must_use]
    pub const fn selection_policy(&self) -> SelectionPolicy {
        self.selection_policy
    }
    #[must_use]
    pub fn compatible_runtime_kernels(&self) -> &[RuntimeKernelId] {
        &self.compatible_runtime_kernels
    }
    /// Exact warm-up/measurement protocol requested for this tuning session.
    #[must_use]
    pub const fn benchmark_protocol(&self) -> BenchmarkProtocol {
        self.benchmark_protocol
    }
    #[must_use]
    pub const fn selection(&self) -> &SelectionRecord {
        &self.selection
    }
}

/// Tune exactly the candidates admitted by a forward semantic execution plan.
///
/// This delegates correctness, timing and deterministic benchmark ranking to
/// FLAT's existing explicit-candidate autotuning seam. No candidates are
/// generated here, so the semantic planner's candidate restriction cannot be
/// silently widened. The requested benchmark protocol is retained verbatim in
/// the returned provenance record.
///
/// # Errors
/// Returns the explicit-candidate structural error if a malformed plan reaches
/// the autotuning boundary.
pub fn tune_forward_execution_plan(
    plan: &ForwardExecutionPlan,
    protocol: BenchmarkProtocol,
    gate: &mut dyn CorrectnessGate,
    harness: &mut dyn TimingHarness,
) -> Result<ForwardTuningRecord, ExplicitCandidateSetError> {
    let problem = plan.problem();
    let selection = tune_candidates(&problem, plan.candidates(), protocol, gate, harness)?;
    Ok(ForwardTuningRecord {
        semantic: plan.semantic().clone(),
        problem,
        device_capability_fingerprint: plan.device_capability_fingerprint(),
        selection_policy: plan.selection_policy(),
        compatible_runtime_kernels: plan.compatible_runtime_kernels().to_vec(),
        benchmark_protocol: protocol,
        selection,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForwardPlanningOutcome {
    Ready(ForwardExecutionPlan),
    NoSemanticSelection,
    NoCompatibleRuntimeFamily {
        semantic: SemanticId,
    },
    NoDeviceAdmissibleCandidate {
        semantic: SemanticId,
        compatible_runtime_kernels: Vec<RuntimeKernelId>,
        device_capability_fingerprint: u64,
    },
}

impl ForwardPlanningOutcome {
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
    use flat_attention::kernel_autotune::TimingSample;
    use flat_attention::{AttentionShape, FlatAttentionConfig};
    use flat_semantic::v1::{SemanticFamily, SemanticId};
    use flat_semantic_execution::{standard_softmax_runtime_catalog, ExecutionBinding};

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
        let selection = standard_selection();
        let outcome = plan_forward_execution(
            &standard_softmax_runtime_catalog(),
            &selection,
            &problem(),
            &capabilities(true),
            &SelectionPolicy::default(),
        );
        let ForwardPlanningOutcome::Ready(plan) = outcome else {
            panic!("expected ready plan");
        };
        assert_eq!(plan.semantic(), selection.semantic().unwrap());
        assert!(!plan.candidates().is_empty());
        assert!(plan.candidates().iter().all(|candidate| candidate
            .runtime_kernel_id()
            .is_some_and(|kernel| plan.compatible_runtime_kernels().contains(&kernel))));
    }

    #[test]
    fn unsupported_semantic_has_no_runtime_fallback() {
        let recurrent = semantic(SemanticFamily::RecurrentMemory, "delta-memory", 1);
        let selection = SemanticSelectionDecision::Selected {
            semantic: recurrent.clone(),
            preference_rank: 0,
        };
        let outcome = plan_forward_execution(
            &standard_softmax_runtime_catalog(),
            &selection,
            &problem(),
            &capabilities(true),
            &SelectionPolicy::default(),
        );
        assert_eq!(outcome.semantic(), Some(&recurrent));
        assert!(matches!(
            outcome,
            ForwardPlanningOutcome::NoCompatibleRuntimeFamily { .. }
        ));
    }

    struct PassGate;
    impl CorrectnessGate for PassGate {
        fn verify(&mut self, _: &KernelCandidate, _: &AttentionProblem) -> Result<(), String> {
            Ok(())
        }
    }
    struct ScriptedHarness {
        measured: Vec<RuntimeKernelId>,
    }
    impl TimingHarness for ScriptedHarness {
        fn measure(
            &mut self,
            candidate: &KernelCandidate,
            _: &AttentionProblem,
            protocol: &BenchmarkProtocol,
        ) -> Result<TimingSample, String> {
            let kernel = candidate
                .runtime_kernel_id()
                .ok_or_else(|| "missing runtime kernel".to_owned())?;
            self.measured.push(kernel);
            Ok(TimingSample {
                median_us: if kernel == RuntimeKernelId::Q4Portable {
                    2.0
                } else {
                    1.0
                },
                p95_us: 3.0,
                iterations: protocol.iterations,
            })
        }
    }

    #[test]
    fn tuning_consumes_exact_plan_and_preserves_semantic_provenance() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let selection = SemanticSelectionDecision::Selected {
            semantic: standard.clone(),
            preference_rank: 0,
        };
        let catalog = SemanticExecutionCatalog::new([ExecutionBinding::new(
            standard.clone(),
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
            panic!("portable plan expected");
        };
        assert_eq!(plan.candidates().len(), 1);
        let mut gate = PassGate;
        let mut harness = ScriptedHarness {
            measured: Vec::new(),
        };
        let protocol = BenchmarkProtocol {
            warmups: 7,
            iterations: 11,
        };
        let record = tune_forward_execution_plan(
            &plan,
            protocol,
            &mut gate,
            &mut harness,
        )
        .unwrap();
        assert_eq!(record.semantic(), &standard);
        assert_eq!(
            record.device_capability_fingerprint(),
            plan.device_capability_fingerprint()
        );
        assert_eq!(record.benchmark_protocol(), protocol);
        assert_eq!(harness.measured, vec![RuntimeKernelId::Q4Portable]);
        assert_eq!(record.selection().per_candidate.len(), 1);
        assert_eq!(
            record
                .selection()
                .selected
                .as_ref()
                .unwrap()
                .candidate
                .runtime_kernel_id(),
            Some(RuntimeKernelId::Q4Portable)
        );
    }
}
