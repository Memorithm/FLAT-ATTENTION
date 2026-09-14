//! Research-only cache-instance-scoped companion for FDAL6 paged tier bindings.
//!
//! [`crate::api::research_da_luc_paged_binding::DalucPagedTierBinding`]
//! intentionally proves metadata freshness only. This module adds an opt-in
//! stronger scope for callers that also need to prove that the binding is being
//! checked against the same [`crate::paged_kv::WgpuPagedKvCache`] instance and
//! append-only lineage from which it was created.
//!
//! The scope retains the validated view contract, tier catalog and FLAT's opaque
//! [`crate::paged_kv::WgpuPagedKvCheckpoint`]. Transition plans compare declared
//! tier assignments, not materialized representations or codebook contents.
//! No cache ID is serialized, no K/V bytes are inspected, and no payload movement,
//! residency change, synchronization or representation transition is authorized.

use core::fmt;

use super::research_da_luc::DalucKvViewContract;
use super::research_da_luc_oracle::tiering::{DalucPrecisionTier, DalucTierId, DalucTierRoutingPlan};
use super::research_da_luc_paged_binding::{
    bind_paged_tier_plan, DalucPagedTierBinding, DalucPagedTierBindingError,
};
use crate::paged_kv::{WgpuPagedKvCache, WgpuPagedKvCacheError, WgpuPagedKvCheckpoint};

/// Version of the research-only page-aware tier-assignment transition contract.
pub const DA_LUC_PAGED_TIER_TRANSITION_VERSION: u16 = 1;

/// Opaque FDAL6 binding scoped to one resident paged-cache instance and lineage.
///
/// The checkpoint, validated view contract and tier catalog are private.
/// Cloning preserves the same cache scope and immutable representation metadata.
/// Neither the catalog nor the checkpoint attests K/V or codebook payload bytes.
#[derive(Debug, Clone)]
pub struct DalucCacheScopedPagedTierBinding {
    binding: DalucPagedTierBinding,
    contract: DalucKvViewContract,
    tiers: Vec<DalucPrecisionTier>,
    checkpoint: WgpuPagedKvCheckpoint,
}

/// Failure while creating or validating a cache-scoped FDAL6 binding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DalucCacheScopedPagedTierBindingError {
    /// The metadata-only FDAL6 binding contract failed.
    Binding(DalucPagedTierBindingError),
    /// The resident cache observation or checkpoint provenance contract failed.
    Cache(WgpuPagedKvCacheError),
    /// The validated tier catalog could not be retained.
    AllocationFailure,
}

impl fmt::Display for DalucCacheScopedPagedTierBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binding(error) => write!(formatter, "{error}"),
            Self::Cache(error) => write!(formatter, "{error}"),
            Self::AllocationFailure => {
                write!(formatter, "FDAL6 cache-scoped tier catalog allocation failed")
            }
        }
    }
}

impl std::error::Error for DalucCacheScopedPagedTierBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Binding(error) => Some(error),
            Self::Cache(error) => Some(error),
            Self::AllocationFailure => None,
        }
    }
}

impl From<DalucPagedTierBindingError> for DalucCacheScopedPagedTierBindingError {
    fn from(value: DalucPagedTierBindingError) -> Self {
        Self::Binding(value)
    }
}

impl From<WgpuPagedKvCacheError> for DalucCacheScopedPagedTierBindingError {
    fn from(value: WgpuPagedKvCacheError) -> Self {
        Self::Cache(value)
    }
}

impl DalucCacheScopedPagedTierBinding {
    /// Borrow the underlying versioned metadata-only FDAL6 binding.
    #[must_use]
    pub fn binding(&self) -> &DalucPagedTierBinding {
        &self.binding
    }

    /// Return the exact validated view contract captured at binding creation.
    #[must_use]
    pub fn contract(&self) -> DalucKvViewContract {
        self.contract
    }

    /// Borrow the exact validated tier catalog, preserving FDAL5 priority order.
    #[must_use]
    pub fn tiers(&self) -> &[DalucPrecisionTier] {
        &self.tiers
    }

    /// Validate exact cache-instance/lineage scope and current FDAL6 metadata.
    ///
    /// Checkpoint validation rejects foreign instances first. Append-only growth
    /// preserves checkpoint provenance but makes the fixed-length binding stale.
    /// Truncate/reset lineage changes fail through the checkpoint contract.
    /// This checks metadata/lifecycle only, not K/V byte identity or GPU completion.
    pub fn validate_cache(
        &self,
        cache: &WgpuPagedKvCache,
    ) -> Result<(), DalucCacheScopedPagedTierBindingError> {
        cache.validate_checkpoint(&self.checkpoint)?;
        let observation = cache.observation()?;
        self.binding.validate_observation(&observation)?;
        Ok(())
    }

    /// Plan tier-assignment changes from `previous` to `self` on one live cache.
    ///
    /// Both bindings must validate against `cache` now. The complete captured
    /// view contracts must be equal; even layout-only changes are rejected.
    /// Catalogs must define exactly the same IDs and K/V representations, but
    /// their priority order may differ. Every changed tier ID emits one record
    /// in logical-page order, including an exact partial final-page span.
    ///
    /// This is not an execution permit or proof of materialization. In particular,
    /// equal tier IDs/descriptors do not prove equal codebook or K/V bytes. An
    /// empty plan means no declared tier-ID change, not that payloads are equal.
    /// The returned plan borrows both immutable bindings and must be revalidated
    /// against the live cache before it is used by a separate execution layer.
    pub fn transitions_from<'a>(
        &'a self,
        previous: &'a Self,
        cache: &WgpuPagedKvCache,
    ) -> Result<DalucPagedTierTransitionPlan<'a>, DalucPagedTierTransitionError> {
        validate_transition_pair(previous, self, cache)?;
        let changed = previous
            .binding
            .assignments
            .iter()
            .zip(&self.binding.assignments)
            .filter(|(prior, next)| prior.tier_id != next.tier_id)
            .count();
        let mut transitions = Vec::new();
        transitions
            .try_reserve_exact(changed)
            .map_err(|_| DalucPagedTierTransitionError::AllocationFailure)?;
        for (prior, next) in previous
            .binding
            .assignments
            .iter()
            .zip(&self.binding.assignments)
        {
            if prior.tier_id != next.tier_id {
                transitions.push(DalucPagedTierTransition {
                    logical_page: next.logical_page,
                    physical_page: next.physical_page,
                    start_token: next.start_token,
                    end_token_exclusive: next.end_token_exclusive,
                    generation: next.generation,
                    branch_epoch: self.binding.branch_epoch,
                    from_tier: prior.tier_id,
                    to_tier: next.tier_id,
                });
            }
        }
        Ok(DalucPagedTierTransitionPlan {
            previous,
            next: self,
            transitions,
        })
    }
}

/// Bind a DA-LUC paged tier plan to one exact resident WGPU cache instance.
///
/// Retains the validated contract and tier catalog, then captures the checkpoint
/// under the same immutable cache borrow. No K/V payload is read, copied, moved,
/// transcoded or synchronized. Representations remain declarations only.
pub fn bind_paged_tier_plan_to_cache(
    cache: &WgpuPagedKvCache,
    contract: DalucKvViewContract,
    tiers: &[DalucPrecisionTier],
    plan: &DalucTierRoutingPlan,
) -> Result<DalucCacheScopedPagedTierBinding, DalucCacheScopedPagedTierBindingError> {
    let observation = cache.observation()?;
    let binding = bind_paged_tier_plan(&observation, contract, tiers, plan)?;
    let mut retained_tiers = Vec::new();
    retained_tiers
        .try_reserve_exact(tiers.len())
        .map_err(|_| DalucCacheScopedPagedTierBindingError::AllocationFailure)?;
    retained_tiers.extend_from_slice(tiers);
    let checkpoint = cache.checkpoint();

    Ok(DalucCacheScopedPagedTierBinding {
        binding,
        contract,
        tiers: retained_tiers,
        checkpoint,
    })
}

/// One declared tier-ID change on a live physical page, not a residency event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucPagedTierTransition {
    pub logical_page: usize,
    pub physical_page: usize,
    pub start_token: usize,
    pub end_token_exclusive: usize,
    pub generation: u64,
    pub branch_epoch: u64,
    pub from_tier: DalucTierId,
    pub to_tier: DalucTierId,
}

/// Immutable, revalidatable page-aware planning metadata.
///
/// References keep the original bindings and their catalogs alive without
/// cloning their allocations. The cache itself is not locked or pinned by this
/// plan. A successful validation is a point-in-time metadata check, not a lease,
/// GPU fence, content attestation or authority to mutate memory.
#[derive(Debug)]
pub struct DalucPagedTierTransitionPlan<'a> {
    previous: &'a DalucCacheScopedPagedTierBinding,
    next: &'a DalucCacheScopedPagedTierBinding,
    transitions: Vec<DalucPagedTierTransition>,
}

impl DalucPagedTierTransitionPlan<'_> {
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        DA_LUC_PAGED_TIER_TRANSITION_VERSION
    }

    #[must_use]
    pub fn transitions(&self) -> &[DalucPagedTierTransition] {
        &self.transitions
    }

    #[must_use]
    pub fn previous(&self) -> &DalucCacheScopedPagedTierBinding {
        self.previous
    }

    #[must_use]
    pub fn next(&self) -> &DalucCacheScopedPagedTierBinding {
        self.next
    }

    /// Reject a foreign, tainted, grown, reset or branched cache at use time.
    pub fn validate_cache(
        &self,
        cache: &WgpuPagedKvCache,
    ) -> Result<(), DalucPagedTierTransitionError> {
        validate_transition_pair(self.previous, self.next, cache)
    }
}

/// Failure to compare two declared paged tier assignments safely.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DalucPagedTierTransitionError {
    PreviousBinding(DalucCacheScopedPagedTierBindingError),
    NextBinding(DalucCacheScopedPagedTierBindingError),
    /// The complete view contracts differ, including layout or base descriptors.
    IncompatibleViewContracts,
    /// Catalog ID sets or the K/V definitions attached to an ID differ.
    IncompatibleTierCatalogs,
    AllocationFailure,
}

impl fmt::Display for DalucPagedTierTransitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreviousBinding(error) => write!(formatter, "previous FDAL6 binding: {error}"),
            Self::NextBinding(error) => write!(formatter, "next FDAL6 binding: {error}"),
            Self::IncompatibleViewContracts => {
                write!(formatter, "FDAL6 transition view contracts differ")
            }
            Self::IncompatibleTierCatalogs => {
                write!(formatter, "FDAL6 transition tier catalogs differ")
            }
            Self::AllocationFailure => {
                write!(formatter, "FDAL6 page transition allocation failed")
            }
        }
    }
}

impl std::error::Error for DalucPagedTierTransitionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PreviousBinding(error) | Self::NextBinding(error) => Some(error),
            _ => None,
        }
    }
}

fn validate_transition_pair(
    previous: &DalucCacheScopedPagedTierBinding,
    next: &DalucCacheScopedPagedTierBinding,
    cache: &WgpuPagedKvCache,
) -> Result<(), DalucPagedTierTransitionError> {
    previous
        .validate_cache(cache)
        .map_err(DalucPagedTierTransitionError::PreviousBinding)?;
    next.validate_cache(cache)
        .map_err(DalucPagedTierTransitionError::NextBinding)?;
    if previous.contract != next.contract {
        return Err(DalucPagedTierTransitionError::IncompatibleViewContracts);
    }
    // Each catalog was validated for unique IDs at construction. Reordering
    // selection priority is allowed, but redefining or dropping any ID is not.
    if previous.tiers.len() != next.tiers.len()
        || previous.tiers.iter().any(|tier| !next.tiers.contains(tier))
    {
        return Err(DalucPagedTierTransitionError::IncompatibleTierCatalogs);
    }
    Ok(())
}
