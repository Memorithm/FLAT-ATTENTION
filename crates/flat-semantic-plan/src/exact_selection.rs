//! Exact registered-semantic provenance bridge for forward planning and tuning.
//!
//! This module is additive to the older preference-ranked control-plane bridge.
//! It accepts the exact registered selection decision produced by
//! `flat-semantic-selection`, preserves that decision unchanged, and delegates
//! implementation admissibility and benchmark ranking to the existing planner
//! and autotuner. The exact selection is never replaced by a runtime outcome.

use std::fmt::Write as _;

use flat_attention::{
    kernel_autotune::{
        BenchmarkProtocol, CandidateEvidence, CorrectnessGate, CorrectnessOutcome,
        ExplicitCandidateSetError, MeasurementRejection, TimingHarness,
    },
    kernel_candidates::{CandidateLifecycle, SelectionPolicy},
    kernel_ir::AttentionProblem,
    RuntimeDeviceCapabilities, RuntimeKernelId,
};
use flat_semantic_control::SemanticSelectionDecision as PreferenceSelectionDecision;
use flat_semantic_execution::SemanticExecutionCatalog;
use flat_semantic_registry::SemanticRegistry;
use flat_semantic_selection::SemanticSelectionDecision as ExactSemanticSelectionDecision;

use crate::{
    plan_forward_execution, tune_forward_execution_plan, ForwardExecutionPlan,
    ForwardPlanningOutcome, ForwardTuningRecord,
};

/// Version of the canonical exact-forward tuning provenance record.
pub const EXACT_FORWARD_TUNING_PROVENANCE_VERSION: u16 = 1;

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

    /// Canonical exportable record for exact semantic tuning provenance.
    ///
    /// The record binds mathematical semantic identity, exact registered
    /// selection provenance, semantic problem geometry, planning-time device
    /// capabilities, candidate-selection policy, benchmark protocol, compatible
    /// runtime surface, every candidate outcome, and the final implementation
    /// selection. Floating-point timings are serialized by IEEE-754 bit pattern
    /// to avoid textual float ambiguity.
    ///
    /// This is evidence provenance only. It does not claim physical hardware
    /// performance and does not alter semantic or implementation selection.
    #[must_use]
    pub fn canonical_provenance_record(&self) -> String {
        let tuning = self.tuning();
        let semantic_registry = SemanticRegistry::new([self.selection.semantic().clone()])
            .expect("exact selection originated from a valid semantic registry");
        let policy = tuning.selection_policy();
        let protocol = tuning.benchmark_protocol();
        let evidence = tuning.selection();
        let mut record = format!(
            "flat-exact-forward-tuning-v{EXACT_FORWARD_TUNING_PROVENANCE_VERSION}\nselection={}\nsemantic_identity={}\nproblem={}\ndevice_capabilities={:016x}\npolicy=allow_experimental:{},max_candidates:{}\nprotocol=warmups:{},iterations:{}\n",
            escape_component(&self.selection.canonical_record()),
            escape_component(&semantic_registry.canonical_record()),
            escape_component(&tuning.problem().canonical_record()),
            tuning.device_capability_fingerprint(),
            u8::from(policy.allow_experimental),
            policy.max_candidates,
            protocol.warmups,
            protocol.iterations,
        );

        let compatible = tuning.compatible_runtime_kernels();
        let _ = writeln!(record, "compatible_runtime_count={}", compatible.len());
        for (index, kernel) in compatible.iter().enumerate() {
            let _ = writeln!(
                record,
                "compatible_runtime[{index}]={}",
                runtime_kernel_tag(*kernel)
            );
        }

        let _ = writeln!(record, "candidate_count={}", evidence.per_candidate.len());
        for (index, (candidate, outcome)) in evidence.per_candidate.iter().enumerate() {
            let runtime = candidate
                .runtime_kernel_id()
                .map_or("none", runtime_kernel_tag);
            let lifecycle = match candidate.lifecycle {
                CandidateLifecycle::Qualified => "qualified",
                CandidateLifecycle::Experimental => "experimental",
            };
            let _ = write!(
                record,
                "candidate[{index}]=id:{:016x},runtime:{runtime},lifecycle:{lifecycle},",
                candidate.id.get()
            );
            write_candidate_evidence(&mut record, outcome);
            record.push('\n');
        }

        match &evidence.selected {
            Some(selected) => {
                let _ = writeln!(
                    record,
                    "selected=id:{:016x},median_bits:{:016x},p95_bits:{:016x},iterations:{}",
                    selected.candidate.id.get(),
                    selected.timing.median_us.to_bits(),
                    selected.timing.p95_us.to_bits(),
                    selected.timing.iterations,
                );
            }
            None => record.push_str("selected=none\n"),
        }
        record
    }

    /// Stable FNV-1a-64 fingerprint of [`Self::canonical_provenance_record`].
    ///
    /// This is a deterministic provenance/cache-key aid, not a cryptographic
    /// authenticity primitive.
    #[must_use]
    pub fn stable_provenance_fingerprint(&self) -> u64 {
        fnv1a64(self.canonical_provenance_record().as_bytes())
    }
}

fn write_candidate_evidence(record: &mut String, evidence: &CandidateEvidence) {
    match evidence {
        CandidateEvidence::Measured { timing } => {
            let _ = write!(
                record,
                "evidence:measured,median_bits:{:016x},p95_bits:{:016x},iterations:{}",
                timing.median_us.to_bits(),
                timing.p95_us.to_bits(),
                timing.iterations,
            );
        }
        CandidateEvidence::Rejected { rejection } => match rejection {
            MeasurementRejection::Correctness(CorrectnessOutcome::Passed) => {
                record.push_str("evidence:rejected,reason:correctness-passed-without-measurement");
            }
            MeasurementRejection::Correctness(CorrectnessOutcome::Failed(reason)) => {
                let _ = write!(
                    record,
                    "evidence:rejected,reason:correctness-failed,message:{}",
                    escape_component(reason)
                );
            }
            MeasurementRejection::Harness(reason) => {
                let _ = write!(
                    record,
                    "evidence:rejected,reason:harness,message:{}",
                    escape_component(reason)
                );
            }
        },
    }
}

const fn runtime_kernel_tag(kernel: RuntimeKernelId) -> &'static str {
    match kernel {
        RuntimeKernelId::Q4Portable => "q4-portable",
        RuntimeKernelId::Q4Vec4Portable => "q4-vec4-portable",
        RuntimeKernelId::Q4Vec4DoubleBuffered => "q4-vec4-double-buffered",
        RuntimeKernelId::Q4Subgroup => "q4-subgroup",
        RuntimeKernelId::GroupedForwardPortable => "grouped-forward-portable",
        RuntimeKernelId::ResidentDecodePortable => "resident-decode-portable",
        RuntimeKernelId::PagedDecodePortable => "paged-decode-portable",
        RuntimeKernelId::GroupedBackwardRecomputePortable => "grouped-backward-recompute-portable",
    }
}

fn escape_component(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '%' => escaped.push_str("%25"),
            '\n' => escaped.push_str("%0a"),
            '\r' => escaped.push_str("%0d"),
            ';' => escaped.push_str("%3b"),
            '=' => escaped.push_str("%3d"),
            ',' => escaped.push_str("%2c"),
            ':' => escaped.push_str("%3a"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
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

    fn ready_plan() -> ExactForwardExecutionPlan {
        let selection = exact_selection();
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

    fn tune(
        plan: &ExactForwardExecutionPlan,
        protocol: BenchmarkProtocol,
    ) -> ExactForwardTuningRecord {
        let mut gate = PassGate;
        let mut harness = Harness;
        tune_exact_forward_execution_plan(plan, protocol, &mut gate, &mut harness).unwrap()
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

        let record = tune(
            &plan,
            BenchmarkProtocol {
                warmups: 1,
                iterations: 2,
            },
        );
        assert_eq!(
            record.selection().stable_fingerprint(),
            selection_fingerprint
        );
        assert_eq!(record.tuning().semantic(), selection.semantic());
    }

    #[test]
    fn canonical_tuning_provenance_is_deterministic_and_semantic_visible() {
        let plan = ready_plan();
        let protocol = BenchmarkProtocol {
            warmups: 7,
            iterations: 11,
        };
        let first = tune(&plan, protocol);
        let second = tune(&plan, protocol);

        assert_eq!(
            first.canonical_provenance_record(),
            second.canonical_provenance_record()
        );
        assert_eq!(
            first.stable_provenance_fingerprint(),
            second.stable_provenance_fingerprint()
        );
        let record = first.canonical_provenance_record();
        assert!(record.contains(
            "semantic_identity=flat-semantic-registry-v1%3bcount%3d1%0afamily%3dstandard-softmax%3bname%3dstandard-softmax%3brevision%3d1%0a"
        ));
        assert!(record.contains("protocol=warmups:7,iterations:11"));
        assert!(record.contains("problem=bh%3d4%3bn%3d129%3bd%3d64%3bcausal%3d1"));
        assert!(record.contains("candidate_count="));
        assert!(record.contains("selected=id:"));
    }

    #[test]
    fn canonical_tuning_provenance_changes_with_measurement_protocol() {
        let plan = ready_plan();
        let first = tune(
            &plan,
            BenchmarkProtocol {
                warmups: 1,
                iterations: 2,
            },
        );
        let second = tune(
            &plan,
            BenchmarkProtocol {
                warmups: 3,
                iterations: 5,
            },
        );

        assert_ne!(
            first.stable_provenance_fingerprint(),
            second.stable_provenance_fingerprint()
        );
        assert_ne!(
            first.canonical_provenance_record(),
            second.canonical_provenance_record()
        );
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
