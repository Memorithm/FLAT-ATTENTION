//! Adapter between FLAT-ATTENTION kernel candidates and the generic
//! Elastic kernel-planning contracts (`elastic-kernel`).
//!
//! Responsibility split:
//! - FLAT owns attention knowledge: kernel IR, candidate families,
//!   capability requirements, WGSL codegen, the scalar oracle;
//! - Elastic owns generic selection: capability filtering, objective-ordered
//!   deterministic planning, auditable evidence, lifecycle;
//! - this crate translates between them and contains no new kernel knowledge.

#![forbid(unsafe_code)]

use std::fmt;

use elastic_core::ObjectiveId;
use elastic_core::{ContractId, LogicalResourceId};
use elastic_eir::Fingerprint;
use elastic_kernel::{
    plan, CapabilitySnapshot, Evidence, EvidenceUnit, FeatureRequirement, FeatureSupport,
    KernelCandidate, KernelRequirements, MeasuredQuantity, ObjectiveEvidence, RealizationIdentity,
    SelectionOutcome, SelectionPolicy, StaticQuantity, SubgroupSupport, WorkgroupLimits,
};
use flat_attention::{AttentionProblem, DeviceLimitsView, KernelCandidateSpec};

pub const BRIDGE_VERSION: u32 = 1;
const WORKLOAD_FINGERPRINT_DOMAIN: &str = "flat-elastic-bridge/workload/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BridgeObjective {
    Latency,
    MemoryFootprint,
}

impl BridgeObjective {
    fn objective_id(self) -> ObjectiveId {
        match self {
            Self::Latency => ObjectiveId::builtin(elastic_core::BuiltinObjective::Latency),
            Self::MemoryFootprint => {
                ObjectiveId::builtin(elastic_core::BuiltinObjective::MemoryFootprint)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectiveOrdering {
    first: BridgeObjective,
    second: Option<BridgeObjective>,
}

impl ObjectiveOrdering {
    #[must_use]
    pub const fn solo(first: BridgeObjective) -> Self {
        Self {
            first,
            second: None,
        }
    }
    #[must_use]
    pub const fn pair(first: BridgeObjective, second: BridgeObjective) -> Self {
        Self {
            first,
            second: Some(second),
        }
    }
    fn ids(&self) -> Vec<ObjectiveId> {
        let mut ids = vec![self.first.objective_id()];
        if let Some(second) = self.second {
            ids.push(second.objective_id());
        }
        ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BridgeError {
    InvalidCapabilitySnapshot(String),
    InvalidCandidate(String),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapabilitySnapshot(d) => write!(f, "capability translation failed: {d}"),
            Self::InvalidCandidate(d) => write!(f, "candidate translation failed: {d}"),
        }
    }
}
impl std::error::Error for BridgeError {}

pub fn capability_snapshot(limits: &DeviceLimitsView) -> Result<CapabilitySnapshot, BridgeError> {
    let subgroup_support = if limits.subgroup_supported {
        SubgroupSupport::supported(1, u32::MAX).expect("range 1..MAX is always valid")
    } else {
        SubgroupSupport::unsupported()
    };
    let max_inv = limits.max_workgroup_size.iter().copied().min().unwrap_or(0);
    CapabilitySnapshot::new(CapabilitySnapshot {
        workgroup_limits: WorkgroupLimits {
            max_invocations_per_axis: limits.max_workgroup_size,
            max_invocations_per_workgroup: max_inv,
            max_workgroups_per_axis: limits.max_workgroups_per_dimension,
            max_workgroup_storage_bytes: limits.max_workgroup_storage_bytes,
        },
        binding_limits: elastic_kernel::BindingLimits {
            max_bind_groups: limits.max_bind_groups,
            max_storage_buffer_binding_bytes: limits.max_storage_buffer_binding_bytes,
        },
        subgroup_support,
        shader_f16: FeatureSupport::Unknown,
        matrix_ops: FeatureSupport::Unknown,
    })
    .map_err(|e| BridgeError::InvalidCapabilitySnapshot(e.to_string()))
}

#[must_use]
pub fn workload_fingerprint(problem: &AttentionProblem, contract: &ContractId) -> Fingerprint {
    Fingerprint::EMPTY
        .text(WORKLOAD_FINGERPRINT_DOMAIN)
        .number(u64::from(BRIDGE_VERSION))
        .text(problem.canonical_record().as_str())
        .text(contract.as_str())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementFixture {
    pub magnitude: u64,
    pub protocol_version: u32,
    pub samples: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Measurements<'a> {
    entries: &'a [(&'static str, BridgeObjective, MeasurementFixture)],
}
impl<'a> Measurements<'a> {
    #[must_use]
    pub fn none() -> Self {
        Self { entries: &[] }
    }
    #[must_use]
    pub fn new(entries: &'a [(&'static str, BridgeObjective, MeasurementFixture)]) -> Self {
        Self { entries }
    }
}

fn evidence_for_spec(spec: &KernelCandidateSpec, m: Measurements<'_>) -> ObjectiveEvidence {
    let mut e = ObjectiveEvidence::new();
    let v = spec.ir().identity().variant;
    e.attach(
        BridgeObjective::MemoryFootprint.objective_id(),
        Evidence::StaticEstimate(StaticQuantity {
            magnitude: spec.static_kv_storage_scalar_loads(),
            unit: EvidenceUnit::Dimensionless,
        }),
    );
    for (cv, obj, f) in m.entries {
        if *cv != v {
            continue;
        }
        let unit = match obj {
            BridgeObjective::Latency => EvidenceUnit::Nanoseconds,
            BridgeObjective::MemoryFootprint => EvidenceUnit::Bytes,
        };
        e.attach(
            obj.objective_id(),
            Evidence::Measured(MeasuredQuantity {
                magnitude: f.magnitude,
                unit,
                protocol_version: f.protocol_version,
                samples: f.samples,
            }),
        );
    }
    e
}

pub fn to_elastic_candidate(
    spec: &KernelCandidateSpec,
    lid: &LogicalResourceId,
    cid: &ContractId,
    m: Measurements<'_>,
) -> Result<KernelCandidate, BridgeError> {
    let id = spec.ir().identity();
    let rid = RealizationIdentity::new(format!(
        "{}:{}@v{}",
        id.family, id.variant, id.schema_version
    ))
    .map_err(|e| BridgeError::InvalidCandidate(e.to_string()))?;
    let r = KernelRequirements {
        invocations_per_workgroup: spec.requirements().workgroup_size_x,
        invocations_per_axis: [spec.requirements().workgroup_size_x, 1, 1],
        workgroup_storage_bytes: spec.requirements().workgroup_storage_bytes,
        bind_groups: spec.requirements().bind_groups,
        max_storage_buffer_binding_bytes: spec.requirements().workgroup_storage_bytes.max(4),
        subgroup_min_width: spec.requirements().requires_subgroup.then_some(1),
        shader_f16: FeatureRequirement::NotRequired,
        matrix_ops: FeatureRequirement::NotRequired,
    };
    KernelCandidate::new(
        lid.clone(),
        rid,
        u32::from(id.schema_version),
        r,
        cid.clone(),
        evidence_for_spec(spec, m),
    )
    .map_err(|e| BridgeError::InvalidCandidate(e.to_string()))
}

#[derive(Clone)]
pub struct SelectionRequest<'a> {
    pub problem: AttentionProblem,
    pub contract: ContractId,
    pub capabilities: CapabilitySnapshot,
    pub candidates: &'a [KernelCandidateSpec],
    pub objectives: ObjectiveOrdering,
    pub allow_static_estimates: bool,
    pub accept_uncontested_fallback: bool,
    pub measurements: Measurements<'a>,
}
impl fmt::Debug for SelectionRequest<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectionRequest")
            .field("problem", &self.problem.canonical_record())
            .field("contract", &self.contract.as_str())
            .field("candidates", &self.candidates.len())
            .finish_non_exhaustive()
    }
}

pub fn select_realization(
    lid: &LogicalResourceId,
    req: &SelectionRequest<'_>,
) -> Result<SelectionOutcome, BridgeError> {
    let policy = SelectionPolicy::with_options(
        req.objectives.ids(),
        req.contract.clone(),
        req.allow_static_estimates,
        req.accept_uncontested_fallback,
    )
    .map_err(|e| BridgeError::InvalidCandidate(e.to_string()))?;
    let mut elastic = Vec::with_capacity(req.candidates.len());
    for spec in req.candidates {
        elastic.push(to_elastic_candidate(
            spec,
            lid,
            &req.contract,
            req.measurements,
        )?);
    }
    Ok(plan(
        lid,
        workload_fingerprint(&req.problem, &req.contract),
        &req.capabilities,
        &policy,
        &elastic,
    ))
}
