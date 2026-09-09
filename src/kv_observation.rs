//! Backend-neutral observation records for paged KV experiments.
//!
//! This module is metadata-only. It does not move K/V bytes, choose retention
//! policy, or assign scientific meaning to an event. FLAT owns the page/cache
//! semantics; consumers such as KVLab may record these snapshots and events as
//! experimental evidence without reimplementing the page table.
//!
//! An observation identifies page-table topology, not K/V content identity.
//! In particular, truncate followed by append may reuse the same physical pages
//! in the same table generation while storing different K/V bytes. Consumers
//! that need content/branch identity must carry the owning cache's lineage or
//! independent content evidence in addition to this table observation.

use crate::paged_kv::{PagedKvConfig, PagedKvError, PagedKvTable, PagedKvTelemetry};

/// Current schema version for [`PagedKvObservation`].
pub const PAGED_KV_OBSERVATION_SCHEMA_VERSION: u32 = 1;

/// Exact logical span backed by one physical page at observation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvPageObservation {
    logical_page: usize,
    physical_page: usize,
    logical_token_start: usize,
    logical_token_end: usize,
    generation: u64,
}

impl PagedKvPageObservation {
    /// Zero-based logical page index.
    #[must_use]
    pub const fn logical_page(self) -> usize {
        self.logical_page
    }

    /// Physical page backing this logical page.
    #[must_use]
    pub const fn physical_page(self) -> usize {
        self.physical_page
    }

    /// Inclusive logical token start.
    #[must_use]
    pub const fn logical_token_start(self) -> usize {
        self.logical_token_start
    }

    /// Exclusive logical token end.
    #[must_use]
    pub const fn logical_token_end(self) -> usize {
        self.logical_token_end
    }

    /// Page-table generation owning this mapping.
    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

/// Versioned, read-only view of one paged KV table topology.
///
/// The page geometry is captured explicitly so consumers never need to infer it
/// from telemetry or populated pages. Equality means the captured page-table
/// metadata is equal. It does not prove that the resident K/V bytes or
/// speculative cache branch are the same.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvObservation {
    schema_version: u32,
    config: PagedKvConfig,
    telemetry: PagedKvTelemetry,
    pages: Vec<PagedKvPageObservation>,
}

impl PagedKvObservation {
    /// Capture the current logical page layout through the public page-table API.
    ///
    /// # Errors
    ///
    /// Propagates telemetry/addressability failures from [`PagedKvTable`].
    pub fn capture(table: &PagedKvTable) -> Result<Self, PagedKvError> {
        let telemetry = table.telemetry()?;
        let config = table.config();
        let mut pages = Vec::new();
        pages
            .try_reserve_exact(telemetry.mapped_pages)
            .map_err(|_| PagedKvError::CapacityOverflow)?;

        for logical_page in 0..telemetry.mapped_pages {
            let logical_token_start = logical_page
                .checked_mul(config.page_size)
                .ok_or(PagedKvError::CapacityOverflow)?;
            let logical_token_end = logical_token_start
                .checked_add(config.page_size)
                .ok_or(PagedKvError::CapacityOverflow)?
                .min(telemetry.live_tokens);
            let address = table
                .address(logical_token_start)
                .ok_or(PagedKvError::CapacityOverflow)?;
            pages.push(PagedKvPageObservation {
                logical_page,
                physical_page: address.physical_page,
                logical_token_start,
                logical_token_end,
                generation: address.generation,
            });
        }

        Ok(Self {
            schema_version: PAGED_KV_OBSERVATION_SCHEMA_VERSION,
            config,
            telemetry,
            pages,
        })
    }

    /// Observation schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Exact page geometry captured with this observation.
    #[must_use]
    pub const fn config(&self) -> PagedKvConfig {
        self.config
    }

    /// Aggregate page-table telemetry captured with the page spans.
    #[must_use]
    pub const fn telemetry(&self) -> PagedKvTelemetry {
        self.telemetry
    }

    /// Mapped pages in deterministic logical-page order.
    #[must_use]
    pub fn pages(&self) -> &[PagedKvPageObservation] {
        &self.pages
    }
}

/// Policy-neutral vocabulary for residency transitions observed by a backend.
///
/// These variants describe evidence only. This enum does not prescribe when a
/// transition should occur or how a backend implements it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvResidencyEventKind {
    Retain,
    Demote,
    Promote,
    Evict,
}

/// One ordered residency event for a physical KV page.
///
/// Generation disambiguates page reuse across resets, but the event still does
/// not identify the bytes stored in that page or a speculative cache branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvResidencyEvent {
    sequence: u64,
    physical_page: usize,
    generation: u64,
    kind: KvResidencyEventKind,
}

impl KvResidencyEvent {
    /// Construct a metadata-only event record.
    #[must_use]
    pub const fn new(
        sequence: u64,
        physical_page: usize,
        generation: u64,
        kind: KvResidencyEventKind,
    ) -> Self {
        Self {
            sequence,
            physical_page,
            generation,
            kind,
        }
    }

    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    #[must_use]
    pub const fn physical_page(self) -> usize {
        self.physical_page
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }

    #[must_use]
    pub const fn kind(self) -> KvResidencyEventKind {
        self.kind
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_kv::{PagedKvConfig, PagedKvTable};

    #[test]
    fn capture_preserves_page_geometry_spans_and_fragmentation() {
        let config = PagedKvConfig {
            page_size: 4,
            physical_pages: 3,
        };
        let mut table = PagedKvTable::new(config).unwrap();
        table.append(6).unwrap();

        let observation = PagedKvObservation::capture(&table).unwrap();
        assert_eq!(observation.schema_version(), 1);
        assert_eq!(observation.config(), config);
        assert_eq!(observation.telemetry().live_tokens, 6);
        assert_eq!(observation.telemetry().mapped_pages, 2);
        assert_eq!(observation.telemetry().internal_fragmentation_tokens, 2);
        assert_eq!(observation.pages().len(), 2);
        assert_eq!(observation.pages()[0].logical_token_start(), 0);
        assert_eq!(observation.pages()[0].logical_token_end(), 4);
        assert_eq!(observation.pages()[1].logical_token_start(), 4);
        assert_eq!(observation.pages()[1].logical_token_end(), 6);
        assert_eq!(observation.pages()[0].physical_page(), 0);
        assert_eq!(observation.pages()[1].physical_page(), 1);
    }

    #[test]
    fn empty_observation_retains_declared_page_geometry() {
        let config = PagedKvConfig {
            page_size: 16,
            physical_pages: 8,
        };
        let table = PagedKvTable::new(config).unwrap();
        let observation = PagedKvObservation::capture(&table).unwrap();

        assert_eq!(observation.config(), config);
        assert!(observation.pages().is_empty());
        assert_eq!(observation.telemetry().capacity_tokens, 128);
    }

    #[test]
    fn reset_changes_generation_and_clears_observed_pages() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 2,
            physical_pages: 2,
        })
        .unwrap();
        table.append(1).unwrap();
        let before = PagedKvObservation::capture(&table).unwrap();
        assert_eq!(before.pages()[0].generation(), 0);

        table.reset().unwrap();
        let after = PagedKvObservation::capture(&table).unwrap();
        assert_eq!(after.telemetry().generation, 1);
        assert!(after.pages().is_empty());
    }

    #[test]
    fn reused_physical_page_has_a_new_generation_after_reset() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 2,
            physical_pages: 2,
        })
        .unwrap();
        table.append(1).unwrap();
        let before = PagedKvObservation::capture(&table).unwrap();
        let before_page = before.pages()[0];

        table.reset().unwrap();
        table.append(1).unwrap();
        let after = PagedKvObservation::capture(&table).unwrap();
        let after_page = after.pages()[0];

        assert_eq!(before_page.physical_page(), after_page.physical_page());
        assert_ne!(before_page.generation(), after_page.generation());
        assert_eq!(after_page.generation(), after.telemetry().generation);
    }

    #[test]
    fn identical_topology_does_not_imply_identical_kv_lineage() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 3,
        })
        .unwrap();
        table.append(6).unwrap();
        let before = PagedKvObservation::capture(&table).unwrap();

        // The table deliberately has no K/V-content identity. A cache may
        // truncate the suffix and write replacement K/V into the same rows.
        table.truncate(4).unwrap();
        table.append(2).unwrap();
        let after = PagedKvObservation::capture(&table).unwrap();

        assert_eq!(before, after);
    }

    #[test]
    fn residency_event_is_policy_neutral_metadata() {
        let event = KvResidencyEvent::new(7, 3, 2, KvResidencyEventKind::Promote);
        assert_eq!(event.sequence(), 7);
        assert_eq!(event.physical_page(), 3);
        assert_eq!(event.generation(), 2);
        assert_eq!(event.kind(), KvResidencyEventKind::Promote);
    }
}
