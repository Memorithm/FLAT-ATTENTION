//! Explicit adapter from FLAT-ATTENTION kernel planning facts to the generic
//! `elastic-kernel` planner in ElasticXxx.
//!
//! Ownership stays one-way and explicit:
//! - FLAT owns attention semantics, Kernel IR, candidate generation and the
//!   correctness/measurement protocol;
//! - ElasticXxx owns generic capability filtering and deterministic selection;
//! - this crate owns only the translation boundary.
//!
//! The adapter deliberately has no WGPU dependency. It consumes the host-side
//! [`flat_attention::RuntimeDeviceCapabilities`] record and the deterministic
//! M25/M26 candidate/evidence surfaces.

#![forbid(unsafe_code)]

use elastic_core::{BuiltinObjective, ContractId, LogicalResourceId, ObjectiveId};
use elastic_eir::Fingerprint;
use elastic_kernel::{
    plan, BindingLimits, CapabilityRejectionReason, CapabilitySnapshot, DispatchGrid, Evidence,
    EvidenceUnit, FeatureRequirement, FeatureSupport, KernelCandidate as ElasticKernelCandidate,
    KernelRequirements, MeasuredQuantity, ObjectiveEvidence, RealizationIdentity, SelectionOutcome,
    SelectionPolicy as ElasticSelectionPolicy, SubgroupSupport, WorkgroupLimits,
};
use flat_attention::kernel_autotune::{CandidateEvidence, SelectionRecord as FlatTuningRecord};
use flat_attention::kernel_candidates::{
    generate_candidates, KernelCandidate as FlatKernelCandidate,
    SelectionPolicy as FlatSelectionPolicy,
};
use flat_attention::kernel_ir::{AttentionProblem, CapabilityRequirement};
use flat_attention::RuntimeDeviceCapabilities;
use std::fmt;

/// Exact ElasticXxx revision this adapter was compiled and reviewed against.
pub const ELASTICXXX_REVISION: &str = "b20e062c091ed82f51ddd690053490be60fda5c7";
/// Adapter schema. Bump when identity/evidence/capability translation semantics change.
pub const ADAPTER_SCHEMA_VERSION: u32 = 2;
/// Version tag attached to latency evidence translated from FLAT M26 records.
pub const FLAT_M26_EVIDENCE_PROTOCOL_VERSION: u32 = 1;

const CONTRACT_ID: &str = "flat-attention/dense-q4-forward-v1";
const RESOURCE_PREFIX: &str = "flat-attention/dense-q4/";
const REALIZATION_PREFIX: &str = "flat-candidate/";
const WORKLOAD_FINGERPRINT_DOMAIN: &str = "flat-elastic-kernel/workload/v1";

/// Translation failures. No adapter error silently substitutes a candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdapterError {
    /// FLAT Kernel IR could not materialize the candidate/problem pair.
    FlatKernel(String),
    /// FLAT capability facts could not form a valid Elastic snapshot.
    ElasticCapability(String),
    /// The translated Elastic candidate failed construction.
    ElasticCandidate(String),
    /// A static adapter-owned Elastic identifier failed validation.
    InvalidElasticIdentity,
    /// Measured M26 evidence was non-finite, negative, empty, or too large.
    InvalidTimingEvidence,
    /// Checked byte/count conversion overflowed.
    IntegerOverflow,
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FlatKernel(message) => write!(f, "FLAT kernel adaptation failed: {message}"),
            Self::ElasticCapability(message) => {
                write!(f, "Elastic capability adaptation failed: {message}")
            }
            Self::ElasticCandidate(message) => {
                write!(f, "Elastic candidate adaptation failed: {message}")
            }
            Self::InvalidElasticIdentity => {
                write!(f, "adapter-generated Elastic identity is invalid")
            }
            Self::InvalidTimingEvidence => write!(f, "FLAT timing evidence is invalid"),
            Self::IntegerOverflow => write!(f, "adapter integer conversion overflowed"),
        }
    }
}

impl std::error::Error for AdapterError {}

/// One workload-dependent dispatch rejection performed by the bridge before
/// generic Elastic candidate selection.
///
/// `DispatchGrid` is intentionally separate from intrinsic
/// [`KernelRequirements`], so these rejections are preserved explicitly here
/// instead of being hidden or mislabeled as static candidate requirements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchRejection {
    /// Stable Elastic realization identity of the rejected FLAT candidate.
    pub realization: String,
    /// Typed Elastic capability reason.
    pub reason: CapabilityRejectionReason,
}

/// Result of one Elastic selection over a stable FLAT candidate set.
#[derive(Debug, Clone)]
pub struct AdapterPlan {
    /// Candidates offered at the FLAT boundary, in deterministic FLAT order.
    pub flat_candidates: Vec<FlatKernelCandidate>,
    /// Candidates actually offered to Elastic after explicit M26 correctness
    /// rejections and workload dispatch-grid rejections are removed.
    pub elastic_candidates: Vec<ElasticKernelCandidate>,
    /// Workload-dependent dispatch rejections performed before generic
    /// selection, in deterministic FLAT candidate order.
    pub dispatch_rejections: Vec<DispatchRejection>,
    /// Generic Elastic selection result, including its own static capability
    /// rejection/evidence record for the candidates it received.
    pub selection: SelectionOutcome,
}

impl AdapterPlan {
    /// Resolve the selected Elastic realization back to the originating FLAT
    /// candidate. Returns `None` for non-selected outcomes.
    #[must_use]
    pub fn selected_flat_candidate(&self) -> Option<&FlatKernelCandidate> {
        let SelectionOutcome::Selected(record) = &self.selection else {
            return None;
        };
        self.flat_candidates.iter().find(|candidate| {
            realization_identity(candidate) == record.selected_realization().as_str()
        })
    }
}

/// Convert FLAT's host-side device model into a generic Elastic capability
/// snapshot.
///
/// FLAT's current M25 candidates are `[x, 1, 1]` workgroups. The FLAT device
/// model exposes per-axis maxima but not a separate total-invocations limit;
/// therefore this adapter conservatively uses the reported X-axis maximum as
/// the total maximum. That is sufficient and non-overclaiming for the current
/// candidate family, and must be revisited before admitting a candidate with
/// Y/Z workgroup dimensions greater than one.
pub fn capability_snapshot(
    capabilities: &RuntimeDeviceCapabilities,
) -> Result<CapabilitySnapshot, AdapterError> {
    let subgroup_support = if capabilities.subgroup_supported {
        SubgroupSupport::supported(
            capabilities.subgroup_min_size,
            capabilities.subgroup_max_size,
        )
        .map_err(|error| AdapterError::ElasticCapability(error.to_string()))?
    } else {
        SubgroupSupport::unsupported()
    };

    CapabilitySnapshot::new(CapabilitySnapshot {
        workgroup_limits: WorkgroupLimits {
            max_invocations_per_axis: [
                capabilities.max_workgroup_size_x,
                capabilities.max_workgroup_size_y,
                capabilities.max_workgroup_size_z,
            ],
            max_invocations_per_workgroup: capabilities.max_workgroup_size_x,
            max_workgroups_per_axis: capabilities.max_workgroups_per_dimension,
            max_workgroup_storage_bytes: u64::from(capabilities.max_workgroup_storage_bytes),
        },
        binding_limits: BindingLimits {
            max_bind_groups: capabilities.max_binding_entries,
            max_storage_buffer_binding_bytes: u64::from(
                capabilities.max_storage_buffer_binding_size,
            ),
        },
        subgroup_support,
        shader_f16: FeatureSupport::Known(capabilities.f16_supported),
        // FLAT RuntimeDeviceCapabilities intentionally has no matrix-op claim
        // today. Unknown must remain unknown, not be fabricated as false.
        matrix_ops: FeatureSupport::Unknown,
    })
    .map_err(|error| AdapterError::ElasticCapability(error.to_string()))
}

/// Standard Elastic latency policy for the FLAT forward contract.
///
/// Static latency estimates are not accepted by this bridge. When
/// `accept_uncontested_fallback` is true, one sole surviving safe realization
/// may be selected without fabricating comparative evidence; Elastic records
/// that decision as `UncontestedFallback`.
pub fn latency_policy(
    accept_uncontested_fallback: bool,
) -> Result<ElasticSelectionPolicy, AdapterError> {
    ElasticSelectionPolicy::with_options(
        vec![latency_objective()],
        contract_id()?,
        false,
        accept_uncontested_fallback,
    )
    .map_err(|error| AdapterError::ElasticCandidate(error.to_string()))
}

/// Adapt an already-generated FLAT candidate list and let ElasticXxx perform
/// generic capability filtering and selection.
///
/// Supplying an M26 tuning record attaches real measured latency evidence.
/// Candidates explicitly rejected by the M26 correctness/timing pipeline are
/// not offered to Elastic; absence of a measurement remains `Unknown`.
/// Workload-dependent dispatch grids are validated against the same Elastic
/// capability snapshot before selection and retained as typed bridge-level
/// rejections when they exceed a boundary.
pub fn plan_adapted(
    problem: &AttentionProblem,
    capabilities: &RuntimeDeviceCapabilities,
    flat_candidates: &[FlatKernelCandidate],
    tuning: Option<&FlatTuningRecord>,
    policy: &ElasticSelectionPolicy,
) -> Result<AdapterPlan, AdapterError> {
    let snapshot = capability_snapshot(capabilities)?;
    let logical_resource_id = logical_resource_id(problem)?;
    let workload = workload_fingerprint(problem);
    let mut elastic_candidates = Vec::with_capacity(flat_candidates.len());
    let mut dispatch_rejections = Vec::new();

    for candidate in flat_candidates {
        let Some(evidence) = evidence_for_candidate(candidate, tuning)? else {
            // Explicit FLAT M26 rejection: correctness before timing is part
            // of the candidate's admissibility, not merely missing evidence.
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

    let selection = plan(
        &logical_resource_id,
        workload,
        &snapshot,
        policy,
        &elastic_candidates,
    );

    Ok(AdapterPlan {
        flat_candidates: flat_candidates.to_vec(),
        elastic_candidates,
        dispatch_rejections,
        selection,
    })
}

/// Generate the bounded M25 candidate set with FLAT's execution-safety
/// prefilter, then delegate generic selection to ElasticXxx.
///
/// FLAT keeps its own M24 prefilter as defense in depth before any executable
/// pipeline is built. [`plan_adapted`] is separately exposed so tests and
/// higher-level orchestration can demonstrate Elastic capability filtering on
/// a common unpruned candidate set.
pub fn generate_and_plan(
    problem: &AttentionProblem,
    capabilities: &RuntimeDeviceCapabilities,
    flat_policy: &FlatSelectionPolicy,
    tuning: Option<&FlatTuningRecord>,
    elastic_policy: &ElasticSelectionPolicy,
) -> Result<AdapterPlan, AdapterError> {
    let candidates = generate_candidates(problem, capabilities, flat_policy);
    plan_adapted(problem, capabilities, &candidates, tuning, elastic_policy)
}

fn adapt_candidate(
    problem: &AttentionProblem,
    candidate: &FlatKernelCandidate,
    logical_resource_id: LogicalResourceId,
    evidence: ObjectiveEvidence,
) -> Result<(ElasticKernelCandidate, DispatchGrid), AdapterError> {
    let module = candidate
        .module_for(problem)
        .map_err(|error| AdapterError::FlatKernel(error.to_string()))?;
    let resources = module.resources();

    // Current generated FLAT modules are X-only workgroups. Refuse to encode a
    // future geometry under this adapter schema instead of silently guessing.
    let invocations_per_axis = [resources.invocations_per_workgroup, 1, 1];

    let qkv_bytes = problem
        .tensor_elements()
        .map_err(|error| AdapterError::FlatKernel(error.to_string()))?
        .checked_mul(4)
        .ok_or(AdapterError::IntegerOverflow)?;
    let output_bytes = problem
        .output_elements()
        .map_err(|error| AdapterError::FlatKernel(error.to_string()))?
        .checked_mul(4)
        .ok_or(AdapterError::IntegerOverflow)?;

    let subgroup_min_width = candidate
        .static_requirements()
        .iter()
        .any(|requirement| matches!(requirement, CapabilityRequirement::SubgroupOperations))
        .then_some(1);

    let requirements = KernelRequirements {
        invocations_per_workgroup: resources.invocations_per_workgroup,
        invocations_per_axis,
        workgroup_storage_bytes: resources.workgroup_storage_bytes,
        bind_groups: resources.binding_entries,
        max_storage_buffer_binding_bytes: qkv_bytes.max(output_bytes),
        subgroup_min_width,
        shader_f16: FeatureRequirement::NotRequired,
        matrix_ops: FeatureRequirement::NotRequired,
    };

    let elastic_candidate = ElasticKernelCandidate::new(
        logical_resource_id,
        RealizationIdentity::new(realization_identity(candidate))
            .map_err(|_| AdapterError::InvalidElasticIdentity)?,
        module.ir_version().major,
        requirements,
        contract_id()?,
        evidence,
    )
    .map_err(|error| AdapterError::ElasticCandidate(error.to_string()))?;

    Ok((
        elastic_candidate,
        DispatchGrid::new(resources.dispatch_extents),
    ))
}

fn evidence_for_candidate(
    candidate: &FlatKernelCandidate,
    tuning: Option<&FlatTuningRecord>,
) -> Result<Option<ObjectiveEvidence>, AdapterError> {
    let Some(tuning) = tuning else {
        return Ok(Some(ObjectiveEvidence::new()));
    };
    let Some((_, evidence)) = tuning
        .per_candidate
        .iter()
        .find(|(recorded, _)| recorded.id == candidate.id)
    else {
        return Ok(Some(ObjectiveEvidence::new()));
    };

    match evidence {
        CandidateEvidence::Rejected { .. } => Ok(None),
        CandidateEvidence::Measured { timing } => {
            if !timing.median_us.is_finite()
                || !timing.p95_us.is_finite()
                || timing.median_us < 0.0
                || timing.p95_us < 0.0
                || timing.iterations == 0
            {
                return Err(AdapterError::InvalidTimingEvidence);
            }
            let nanoseconds = timing.median_us * 1_000.0;
            if nanoseconds > u64::MAX as f64 {
                return Err(AdapterError::InvalidTimingEvidence);
            }
            let samples = u32::try_from(timing.iterations)
                .map_err(|_| AdapterError::InvalidTimingEvidence)?;
            Ok(Some(ObjectiveEvidence::new().with(
                latency_objective(),
                Evidence::Measured(MeasuredQuantity {
                    magnitude: nanoseconds as u64,
                    unit: EvidenceUnit::Nanoseconds,
                    protocol_version: FLAT_M26_EVIDENCE_PROTOCOL_VERSION,
                    samples,
                }),
            )))
        }
    }
}

fn logical_resource_id(problem: &AttentionProblem) -> Result<LogicalResourceId, AdapterError> {
    LogicalResourceId::new(format!("{RESOURCE_PREFIX}{}", problem.canonical_record()))
        .map_err(|_| AdapterError::InvalidElasticIdentity)
}

fn contract_id() -> Result<ContractId, AdapterError> {
    ContractId::new(CONTRACT_ID).map_err(|_| AdapterError::InvalidElasticIdentity)
}

fn latency_objective() -> ObjectiveId {
    ObjectiveId::builtin(BuiltinObjective::Latency)
}

fn realization_identity(candidate: &FlatKernelCandidate) -> String {
    format!("{REALIZATION_PREFIX}{}", candidate.id)
}

fn workload_fingerprint(problem: &AttentionProblem) -> Fingerprint {
    Fingerprint::EMPTY
        .text(WORKLOAD_FINGERPRINT_DOMAIN)
        .text(&problem.canonical_record())
}

#[cfg(test)]
mod tests {
    use super::*;
    use elastic_kernel::{DecisiveEvidence, Feature, RejectedReason};
    use flat_attention::kernel_autotune::{CorrectnessOutcome, MeasurementRejection, TimingSample};
    use flat_attention::kernel_ir::ScoreReduction;

    fn problem() -> AttentionProblem {
        AttentionProblem {
            batch_heads: 4,
            seq_len: 128,
            head_dim: 64,
            causal: true,
        }
    }

    fn caps(subgroup: bool) -> RuntimeDeviceCapabilities {
        RuntimeDeviceCapabilities {
            max_workgroups_per_dimension: 65_535,
            max_workgroup_size_x: 64,
            max_workgroup_size_y: 1_024,
            max_workgroup_size_z: 64,
            max_workgroup_storage_bytes: 32_768,
            max_binding_entries: 8,
            max_storage_buffer_binding_size: 1 << 30,
            subgroup_supported: subgroup,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f16_supported: true,
        }
    }

    fn candidates() -> Vec<FlatKernelCandidate> {
        generate_candidates(&problem(), &caps(true), &FlatSelectionPolicy::default())
    }

    fn tuning_for(candidates: &[FlatKernelCandidate]) -> FlatTuningRecord {
        let per_candidate = candidates
            .iter()
            .copied()
            .map(|candidate| {
                let median_us =
                    if candidate.config.score_reduction == ScoreReduction::SubgroupAssisted {
                        40.0
                    } else if candidate.config.vector_width.components() == 4 {
                        60.0
                    } else {
                        100.0
                    };
                (
                    candidate,
                    CandidateEvidence::Measured {
                        timing: TimingSample {
                            median_us,
                            p95_us: median_us + 5.0,
                            iterations: 30,
                        },
                    },
                )
            })
            .collect();
        FlatTuningRecord {
            selected: None,
            per_candidate,
        }
    }

    #[test]
    fn capability_mapping_keeps_unreported_matrix_ops_unknown() {
        let snapshot = capability_snapshot(&caps(true)).expect("valid snapshot");
        assert_eq!(
            snapshot.feature_support(Feature::MatrixOps),
            FeatureSupport::Unknown
        );
        assert_eq!(
            snapshot.feature_support(Feature::ShaderF16),
            FeatureSupport::Known(true)
        );
    }

    #[test]
    fn elastic_selection_changes_with_subgroup_capability() {
        let candidates = candidates();
        let tuning = tuning_for(&candidates);
        let policy = latency_policy(false).expect("valid policy");

        let rich = plan_adapted(&problem(), &caps(true), &candidates, Some(&tuning), &policy)
            .expect("rich plan");
        let rich_selected = rich.selected_flat_candidate().expect("rich selection");
        assert_eq!(
            rich_selected.config.score_reduction,
            ScoreReduction::SubgroupAssisted
        );
        assert!(rich.dispatch_rejections.is_empty());

        let portable = plan_adapted(
            &problem(),
            &caps(false),
            &candidates,
            Some(&tuning),
            &policy,
        )
        .expect("portable plan");
        let portable_selected = portable
            .selected_flat_candidate()
            .expect("portable selection");
        assert_ne!(portable_selected.id, rich_selected.id);
        assert_ne!(
            portable_selected.config.score_reduction,
            ScoreReduction::SubgroupAssisted
        );

        let SelectionOutcome::Selected(record) = &portable.selection else {
            panic!("portable profile must select a legal measured candidate");
        };
        assert!(record.rejected().iter().any(|rejection| matches!(
            rejection.reason(),
            RejectedReason::Infeasible(
                elastic_kernel::CapabilityRejectionReason::SubgroupUnsupported
            )
        )));
    }

    #[test]
    fn dispatch_grid_exact_boundary_is_admitted_and_limit_minus_one_is_rejected() {
        let candidate = candidates().into_iter().next().expect("candidate");
        let policy = latency_policy(true).expect("policy");

        let mut exact = caps(true);
        exact.max_workgroups_per_dimension = 32;
        let exact_plan = plan_adapted(&problem(), &exact, &[candidate], None, &policy)
            .expect("exact boundary plan");
        assert!(exact_plan.dispatch_rejections.is_empty());
        assert_eq!(exact_plan.elastic_candidates.len(), 1);
        assert!(matches!(
            exact_plan.selection,
            SelectionOutcome::Selected(_)
        ));

        let mut too_small = exact;
        too_small.max_workgroups_per_dimension = 31;
        let rejected = plan_adapted(&problem(), &too_small, &[candidate], None, &policy)
            .expect("typed dispatch rejection plan");
        assert!(rejected.elastic_candidates.is_empty());
        assert_eq!(rejected.dispatch_rejections.len(), 1);
        assert_eq!(
            rejected.dispatch_rejections[0].reason,
            CapabilityRejectionReason::DispatchGridExceeded {
                axis: 0,
                required_workgroups: 32,
                available_workgroups: 31,
            }
        );
        assert!(matches!(
            rejected.selection,
            SelectionOutcome::NoCandidate { offered: 0, .. }
        ));
    }

    #[test]
    fn one_unknown_candidate_requires_explicit_uncontested_policy() {
        let candidate = candidates().into_iter().next().expect("candidate");
        let strict = latency_policy(false).expect("strict policy");
        let strict_plan = plan_adapted(&problem(), &caps(true), &[candidate], None, &strict)
            .expect("strict plan");
        assert!(matches!(
            strict_plan.selection,
            SelectionOutcome::InsufficientEvidence { .. }
        ));

        let permissive = latency_policy(true).expect("permissive policy");
        let permissive_plan =
            plan_adapted(&problem(), &caps(true), &[candidate], None, &permissive)
                .expect("permissive plan");
        let SelectionOutcome::Selected(record) = permissive_plan.selection else {
            panic!("uncontested fallback must be explicitly selectable");
        };
        assert_eq!(
            record.decisive_evidence(),
            Some(&DecisiveEvidence::UncontestedFallback)
        );
    }

    #[test]
    fn explicit_m26_rejection_is_never_offered_to_elastic() {
        let candidates = candidates();
        let rejected = candidates[0];
        let tuning = FlatTuningRecord {
            selected: None,
            per_candidate: vec![(
                rejected,
                CandidateEvidence::Rejected {
                    rejection: MeasurementRejection::Correctness(CorrectnessOutcome::Failed(
                        "oracle mismatch".into(),
                    )),
                },
            )],
        };
        let policy = latency_policy(true).expect("policy");
        let plan = plan_adapted(&problem(), &caps(true), &candidates, Some(&tuning), &policy)
            .expect("plan");
        assert!(plan
            .elastic_candidates
            .iter()
            .all(|candidate| candidate.realization().as_str() != realization_identity(&rejected)));
    }

    #[test]
    fn invalid_timing_is_refused_instead_of_becoming_evidence() {
        let candidate = candidates().into_iter().next().expect("candidate");
        let tuning = FlatTuningRecord {
            selected: None,
            per_candidate: vec![(
                candidate,
                CandidateEvidence::Measured {
                    timing: TimingSample {
                        median_us: f64::NAN,
                        p95_us: 1.0,
                        iterations: 30,
                    },
                },
            )],
        };
        let policy = latency_policy(false).expect("policy");
        assert_eq!(
            plan_adapted(
                &problem(),
                &caps(true),
                &[candidate],
                Some(&tuning),
                &policy
            )
            .expect_err("NaN evidence must fail"),
            AdapterError::InvalidTimingEvidence
        );
    }
}
