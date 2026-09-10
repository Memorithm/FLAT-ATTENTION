//! Research-only binding between DA-LUC tier plans and observed paged KV topology.
//!
//! FDAL6 turns a validated logical [`DalucTierRoutingPlan`] into explicit
//! logical-page/physical-page metadata for one observed [`WgpuPagedKvStateObservation`]
//! state. It does not inspect, transcode, copy, move, evict, promote, or mutate
//! K/V payload bytes. The result is evidence and execution-planning metadata
//! only; a backend remains responsible for every physical representation or
//! residency transition.
//!
//! The binding deliberately requires one DA-LUC segment per KV page. This keeps
//! each tier assignment aligned with FLAT's authoritative logical-to-physical
//! page mapping and avoids inventing sub-page movement semantics.

use core::fmt;

use super::research_da_luc::{DalucKvViewContract, DalucKvViewError, DalucStorageTopology};
use super::research_da_luc_oracle::tiering::{
    DalucPrecisionTier, DalucTierId, DalucTierRoutingError, DalucTierRoutingPlan,
};
use crate::paged_kv::WgpuPagedKvStateObservation;

/// Version of the metadata-only FDAL6 paged tier binding semantics.
pub const DA_LUC_PAGED_TIER_BINDING_VERSION: u16 = 1;

/// One DA-LUC tier assignment bound to the physical page observed by FLAT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucPagedTierPageAssignment {
    pub logical_page: usize,
    pub physical_page: usize,
    pub start_token: usize,
    pub end_token_exclusive: usize,
    pub generation: u64,
    pub tier_id: DalucTierId,
}

/// Versioned metadata-only binding of a tier plan to one paged-cache observation.
///
/// Equality means only that the exposed metadata is equal. It does not establish
/// K/V content identity, prove that a declared compressed representation has
/// been materialized, or authorize a backend to move bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DalucPagedTierBinding {
    pub binding_version: u16,
    pub observation_schema_version: u32,
    pub kv_view_schema_version: u16,
    pub routing_version: u16,
    pub kv_len: usize,
    pub page_size: usize,
    pub physical_pages: usize,
    pub generation: u64,
    pub branch_epoch: u64,
    pub assignments: Vec<DalucPagedTierPageAssignment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DalucPagedTierBindingError {
    Contract(DalucKvViewError),
    Routing(DalucTierRoutingError),
    UnsubmittedRecordedWrites,
    UnsupportedBatch {
        actual: usize,
    },
    KvHeadsMismatch {
        contract: usize,
        observation: usize,
    },
    KeyHeadDimMismatch {
        contract: usize,
        observation: usize,
    },
    ValueHeadDimMismatch {
        contract: usize,
        observation: usize,
    },
    KvLenMismatch {
        contract: usize,
        observation: usize,
    },
    ExpectedPagedTopology,
    PageGeometryMismatch {
        contract_page_size: usize,
        observation_page_size: usize,
        contract_physical_pages: usize,
        observation_physical_pages: usize,
    },
    SegmentSizeMismatch {
        segment_size: usize,
        page_size: usize,
    },
    AssignmentCountMismatch {
        plan: usize,
        observation: usize,
    },
    AssignmentPageMismatch {
        logical_page: usize,
    },
    AllocationFailure,
}

impl fmt::Display for DalucPagedTierBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(formatter, "{error}"),
            Self::Routing(error) => write!(formatter, "{error}"),
            Self::UnsubmittedRecordedWrites => write!(
                formatter,
                "FDAL6 refuses a paged KV observation whose externally recorded GPU writes may still be submit-able"
            ),
            Self::UnsupportedBatch { actual } => write!(
                formatter,
                "FDAL6 WGPU paged binding requires batch=1, got {actual}"
            ),
            Self::KvHeadsMismatch {
                contract,
                observation,
            } => write!(
                formatter,
                "FDAL6 KV head count mismatch: contract={contract}, observation={observation}"
            ),
            Self::KeyHeadDimMismatch {
                contract,
                observation,
            } => write!(
                formatter,
                "FDAL6 key head dimension mismatch: contract={contract}, observation={observation}"
            ),
            Self::ValueHeadDimMismatch {
                contract,
                observation,
            } => write!(
                formatter,
                "FDAL6 value head dimension mismatch: contract={contract}, observation={observation}"
            ),
            Self::KvLenMismatch {
                contract,
                observation,
            } => write!(
                formatter,
                "FDAL6 KV length mismatch: contract={contract}, observation={observation}"
            ),
            Self::ExpectedPagedTopology => write!(
                formatter,
                "FDAL6 paged binding requires a DA-LUC paged storage topology"
            ),
            Self::PageGeometryMismatch {
                contract_page_size,
                observation_page_size,
                contract_physical_pages,
                observation_physical_pages,
            } => write!(
                formatter,
                "FDAL6 page geometry mismatch: contract page_size={contract_page_size}, physical_pages={contract_physical_pages}; observation page_size={observation_page_size}, physical_pages={observation_physical_pages}"
            ),
            Self::SegmentSizeMismatch {
                segment_size,
                page_size,
            } => write!(
                formatter,
                "FDAL6 requires one tier segment per KV page: segment_size={segment_size}, page_size={page_size}"
            ),
            Self::AssignmentCountMismatch { plan, observation } => write!(
                formatter,
                "FDAL6 assignment count mismatch: plan={plan}, observed mapped pages={observation}"
            ),
            Self::AssignmentPageMismatch { logical_page } => write!(
                formatter,
                "FDAL6 plan bounds do not match observed logical page {logical_page}"
            ),
            Self::AllocationFailure => write!(
                formatter,
                "FDAL6 could not allocate the paged tier binding assignment vector"
            ),
        }
    }
}

impl std::error::Error for DalucPagedTierBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            Self::Routing(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DalucKvViewError> for DalucPagedTierBindingError {
    fn from(value: DalucKvViewError) -> Self {
        Self::Contract(value)
    }
}

impl From<DalucTierRoutingError> for DalucPagedTierBindingError {
    fn from(value: DalucTierRoutingError) -> Self {
        Self::Routing(value)
    }
}

/// Bind a validated DA-LUC tier plan to one observed WGPU paged-KV topology.
///
/// The binding is fail-closed when externally recorded writes may still be
/// submit-able. It validates the DA-LUC contract and routing plan, then requires
/// the cache-facing dimensions, live length and paged geometry to match exactly.
/// One routing segment must equal one page so every logical tier assignment has
/// one unambiguous observed physical-page target.
///
/// This function does not validate that `contract.keys` or `contract.values`
/// describe bytes currently stored in the WGPU cache. Those fields define the
/// representation requested by the research plan; materialization and payload
/// identity require independent backend evidence.
pub fn bind_paged_tier_plan(
    observation: &WgpuPagedKvStateObservation,
    contract: DalucKvViewContract,
    tiers: &[DalucPrecisionTier],
    plan: &DalucTierRoutingPlan,
) -> Result<DalucPagedTierBinding, DalucPagedTierBindingError> {
    if observation.has_unsubmitted_recorded_writes() {
        return Err(DalucPagedTierBindingError::UnsubmittedRecordedWrites);
    }

    contract.validate()?;
    plan.validate_against(contract, tiers)?;

    if contract.shape.batch != 1 {
        return Err(DalucPagedTierBindingError::UnsupportedBatch {
            actual: contract.shape.batch,
        });
    }
    if contract.shape.kv_heads != observation.kv_heads() {
        return Err(DalucPagedTierBindingError::KvHeadsMismatch {
            contract: contract.shape.kv_heads,
            observation: observation.kv_heads(),
        });
    }
    if contract.shape.key_head_dim != observation.head_dim() {
        return Err(DalucPagedTierBindingError::KeyHeadDimMismatch {
            contract: contract.shape.key_head_dim,
            observation: observation.head_dim(),
        });
    }
    if contract.shape.value_head_dim != observation.head_dim() {
        return Err(DalucPagedTierBindingError::ValueHeadDimMismatch {
            contract: contract.shape.value_head_dim,
            observation: observation.head_dim(),
        });
    }

    let topology = observation.topology();
    let telemetry = topology.telemetry();
    let config = topology.config();
    if contract.shape.kv_len != telemetry.live_tokens {
        return Err(DalucPagedTierBindingError::KvLenMismatch {
            contract: contract.shape.kv_len,
            observation: telemetry.live_tokens,
        });
    }

    let DalucStorageTopology::Paged {
        page_size: contract_page_size,
        physical_pages_per_batch: contract_physical_pages,
    } = contract.layout.topology
    else {
        return Err(DalucPagedTierBindingError::ExpectedPagedTopology);
    };
    if contract_page_size != config.page_size || contract_physical_pages != config.physical_pages {
        return Err(DalucPagedTierBindingError::PageGeometryMismatch {
            contract_page_size,
            observation_page_size: config.page_size,
            contract_physical_pages,
            observation_physical_pages: config.physical_pages,
        });
    }
    if plan.segment_size != config.page_size {
        return Err(DalucPagedTierBindingError::SegmentSizeMismatch {
            segment_size: plan.segment_size,
            page_size: config.page_size,
        });
    }

    let pages = topology.pages();
    if plan.assignments.len() != pages.len() {
        return Err(DalucPagedTierBindingError::AssignmentCountMismatch {
            plan: plan.assignments.len(),
            observation: pages.len(),
        });
    }

    let mut assignments = Vec::new();
    assignments
        .try_reserve_exact(pages.len())
        .map_err(|_| DalucPagedTierBindingError::AllocationFailure)?;

    for (page, assignment) in pages.iter().copied().zip(&plan.assignments) {
        if assignment.segment_index != page.logical_page()
            || assignment.start_token != page.logical_token_start()
            || assignment.end_token_exclusive != page.logical_token_end()
        {
            return Err(DalucPagedTierBindingError::AssignmentPageMismatch {
                logical_page: page.logical_page(),
            });
        }
        assignments.push(DalucPagedTierPageAssignment {
            logical_page: page.logical_page(),
            physical_page: page.physical_page(),
            start_token: page.logical_token_start(),
            end_token_exclusive: page.logical_token_end(),
            generation: page.generation(),
            tier_id: assignment.tier_id,
        });
    }

    Ok(DalucPagedTierBinding {
        binding_version: DA_LUC_PAGED_TIER_BINDING_VERSION,
        observation_schema_version: observation.schema_version(),
        kv_view_schema_version: contract.schema_version,
        routing_version: plan.routing_version,
        kv_len: telemetry.live_tokens,
        page_size: config.page_size,
        physical_pages: config.physical_pages,
        generation: telemetry.generation,
        branch_epoch: observation.branch_epoch(),
        assignments,
    })
}
