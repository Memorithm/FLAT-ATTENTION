//! Metadata-only observation of one resident WGPU paged-KV cache state.
//!
//! This layer joins the backend-neutral page-table topology with WGPU cache
//! shape and lifecycle metadata already owned by [`WgpuPagedKvCache`]. It does
//! not read K/V bytes, create buffers, submit commands, or choose a residency
//! policy.
//!
//! The metadata is intentionally not a content hash or globally unique cache
//! identity. In particular, two different cache instances may have equal
//! observations, and callers that need to prove content identity must carry
//! independent evidence.

use super::{PagedKvObservation, WgpuPagedKvCache, WgpuPagedKvCacheError};

/// Current schema version for [`WgpuPagedKvStateObservation`].
pub const WGPU_PAGED_KV_STATE_OBSERVATION_SCHEMA_VERSION: u32 = 1;

/// Read-only metadata describing one WGPU paged-KV cache state.
///
/// `branch_epoch` distinguishes destructive lineage changes within one cache
/// instance and table generation. It is not globally unique: equality of two
/// observations means only that their exposed metadata is equal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuPagedKvStateObservation {
    schema_version: u32,
    topology: PagedKvObservation,
    kv_heads: usize,
    head_dim: usize,
    branch_epoch: u64,
    unsubmitted_recorded_writes: bool,
}

impl WgpuPagedKvStateObservation {
    /// Capture metadata without reading or synchronizing device K/V storage.
    ///
    /// # Errors
    ///
    /// Propagates page-table observation failures through
    /// [`WgpuPagedKvCacheError::Table`].
    pub fn capture(cache: &WgpuPagedKvCache) -> Result<Self, WgpuPagedKvCacheError> {
        Ok(Self {
            schema_version: WGPU_PAGED_KV_STATE_OBSERVATION_SCHEMA_VERSION,
            topology: PagedKvObservation::capture(cache.table())?,
            kv_heads: cache.kv_heads(),
            head_dim: cache.head_dim(),
            branch_epoch: cache.branch_epoch(),
            unsubmitted_recorded_writes: cache.has_unsubmitted_recorded_writes(),
        })
    }

    /// Observation schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Backend-neutral page-table topology captured at the same instant.
    #[must_use]
    pub const fn topology(&self) -> &PagedKvObservation {
        &self.topology
    }

    /// Native K/V head count retained by the resident cache.
    #[must_use]
    pub const fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    /// Elements per K/V head row.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Append-only branch epoch within this cache instance and generation.
    #[must_use]
    pub const fn branch_epoch(&self) -> u64 {
        self.branch_epoch
    }

    /// Whether externally recorded writes may still be submit-able.
    #[must_use]
    pub const fn has_unsubmitted_recorded_writes(&self) -> bool {
        self.unsubmitted_recorded_writes
    }
}

impl WgpuPagedKvCache {
    /// Capture topology plus WGPU cache lifecycle metadata without device I/O.
    ///
    /// This observation does not identify resident K/V bytes. `branch_epoch`
    /// is meaningful only within the owning cache instance; use a checkpoint
    /// when an operation must prove that it belongs to that exact cache.
    pub fn observation(&self) -> Result<WgpuPagedKvStateObservation, WgpuPagedKvCacheError> {
        WgpuPagedKvStateObservation::capture(self)
    }
}
