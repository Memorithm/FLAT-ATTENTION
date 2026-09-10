//! Research-only cache-instance-scoped companion for FDAL6 paged tier bindings.
//!
//! [`crate::api::research_da_luc_paged_binding::DalucPagedTierBinding`]
//! intentionally proves metadata freshness only. This module adds an opt-in
//! stronger scope for callers that also need to prove that the binding is being
//! checked against the same [`crate::paged_kv::WgpuPagedKvCache`] instance and
//! append-only lineage from which it was created.
//!
//! The scope is implemented with FLAT's existing opaque
//! [`crate::paged_kv::WgpuPagedKvCheckpoint`] provenance token. It also retains
//! the validated tier catalog so a `tier_id` can be interpreted without relying
//! on mutable or out-of-band caller state. It does not invent a serializable
//! cache ID, hash or inspect K/V payload bytes, attest device memory, or authorize
//! any payload movement or representation transition.

use core::fmt;

use super::research_da_luc::DalucKvViewContract;
use super::research_da_luc_oracle::tiering::{DalucPrecisionTier, DalucTierRoutingPlan};
use super::research_da_luc_paged_binding::{
    bind_paged_tier_plan, DalucPagedTierBinding, DalucPagedTierBindingError,
};
use crate::paged_kv::{WgpuPagedKvCache, WgpuPagedKvCacheError, WgpuPagedKvCheckpoint};

/// Opaque FDAL6 binding scoped to one resident paged-cache instance and lineage.
///
/// The embedded checkpoint and validated tier catalog are deliberately private.
/// This type is not a global cache identity and does not expose or serialize the
/// checkpoint's private provenance token. Cloning the value preserves the same
/// cache scope and the same immutable tier definitions.
#[derive(Debug, Clone)]
pub struct DalucCacheScopedPagedTierBinding {
    binding: DalucPagedTierBinding,
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

    /// Borrow the exact tier catalog validated when this binding was created.
    ///
    /// Catalog order is preserved because FDAL5 uses caller order as selection
    /// priority. Each stable `tier_id` therefore remains attached to the K/V
    /// representation semantics that were validated for this binding.
    #[must_use]
    pub fn tiers(&self) -> &[DalucPrecisionTier] {
        &self.tiers
    }

    /// Validate exact cache-instance/lineage scope and current FDAL6 metadata.
    ///
    /// The checkpoint validation runs first so a different cache instance fails
    /// closed even when it exposes byte-for-byte equal metadata observations.
    /// Append-only growth preserves checkpoint provenance but still makes the
    /// fixed-length FDAL6 binding stale, which is then rejected by
    /// [`DalucPagedTierBinding::validate_observation`]. Truncate/reset lineage
    /// changes fail through the checkpoint contract.
    ///
    /// This remains a metadata/lifecycle proof only. It does not inspect or hash
    /// K/V bytes and therefore does not establish device-payload identity.
    pub fn validate_cache(
        &self,
        cache: &WgpuPagedKvCache,
    ) -> Result<(), DalucCacheScopedPagedTierBindingError> {
        cache.validate_checkpoint(&self.checkpoint)?;
        let observation = cache.observation()?;
        self.binding.validate_observation(&observation)?;
        Ok(())
    }
}

/// Bind a DA-LUC paged tier plan to one exact resident WGPU cache instance.
///
/// This first creates the existing metadata-only FDAL6 binding, preserving all
/// of its contract/geometry/taint checks. It then retains the already-validated
/// tier catalog and captures FLAT's opaque checkpoint under the same immutable
/// cache borrow. The returned scope can later be checked with
/// [`DalucCacheScopedPagedTierBinding::validate_cache`].
///
/// No K/V payload is read, copied, moved, transcoded or synchronized.
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
        tiers: retained_tiers,
        checkpoint,
    })
}
