//! Plane-epoch paged-KV shadow with a W64-capable page map.
//!
//! This v2 shadow tests a stronger representation hypothesis than v1:
//! the table generation is a property of the whole mapping epoch and therefore
//! does not need to be duplicated inside every logical-page word.
//!
//! Per-page words contain only physical page identity in lane 0. Wider words
//! reserve higher lanes as zero. The current generation remains a single table-
//! level field. This makes W64 structurally admissible without truncating either
//! physical-page identity or generation.
//!
//! The production `PagedKvTable` remains authoritative. This module is a
//! differential research implementation and makes no performance claim.

use elastic_core::{ElasticWordError, ElasticWordPlaneV1, ElasticWordWidthV1};
use flat_attention::paged_kv::{
    PagedKvAddress, PagedKvConfig, PagedKvError, PagedKvTable, PagedKvTelemetry,
};
use std::fmt;

/// Versioned plane-epoch shadow contract.
pub const FLAT_ELASTIC_PAGED_KV_PLANE_EPOCH_V2: &str = "flat.elastic-paged-kv-plane-epoch@2.0.0";

/// One lossless semantic lane is required per page: physical-page identity.
pub const FLAT_ELASTIC_PAGED_KV_REQUIRED_LANES_V2: u8 = 1;

/// Snapshot pairing the table-level generation with the flat page-map plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElasticPagedKvSnapshotV2 {
    /// Generation shared by every mapping in this snapshot.
    pub generation: u64,
    /// Width-tagged flat physical-page map.
    pub plane: ElasticWordPlaneV1,
}

/// Fail-closed errors for the v2 shadow table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElasticPagedKvPlaneEpochError {
    /// Authoritative paged-KV semantics rejected the operation.
    Paged(PagedKvError),
    /// Physical page identity cannot be encoded in one u64 lane.
    PhysicalPageOutOfRange { physical_page: usize },
    /// Stored physical page cannot be represented as usize on this target.
    StoredPhysicalPageOutOfRange { physical_page: u64 },
    /// Lane count or buffer size arithmetic overflowed.
    StorageOverflow,
    /// A reserved lane contains data and cannot be silently discarded.
    ReservedLaneNonZero {
        logical_page: usize,
        lane_index: usize,
        value: u64,
    },
    /// Generic ElasticWord validation failed.
    Elastic(ElasticWordError),
}

impl fmt::Display for ElasticPagedKvPlaneEpochError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Paged(error) => write!(f, "paged-KV semantic error: {error}"),
            Self::PhysicalPageOutOfRange { physical_page } => write!(
                f,
                "physical page {physical_page} cannot be represented in one u64 lane"
            ),
            Self::StoredPhysicalPageOutOfRange { physical_page } => write!(
                f,
                "stored physical page {physical_page} cannot be represented as usize"
            ),
            Self::StorageOverflow => {
                f.write_str("elastic plane-epoch paged-KV storage arithmetic overflowed")
            }
            Self::ReservedLaneNonZero {
                logical_page,
                lane_index,
                value,
            } => write!(
                f,
                "logical page {logical_page} reserved lane {lane_index} is non-zero ({value:#x})"
            ),
            Self::Elastic(error) => write!(f, "ElasticWord error: {error}"),
        }
    }
}

impl std::error::Error for ElasticPagedKvPlaneEpochError {}

impl From<PagedKvError> for ElasticPagedKvPlaneEpochError {
    fn from(value: PagedKvError) -> Self {
        Self::Paged(value)
    }
}

impl From<ElasticWordError> for ElasticPagedKvPlaneEpochError {
    fn from(value: ElasticWordError) -> Self {
        Self::Elastic(value)
    }
}

/// Experimental flat paged-KV table whose generation is carried once per epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElasticPagedKvPlaneEpochV2 {
    config: PagedKvConfig,
    live_tokens: usize,
    generation: u64,
    width: ElasticWordWidthV1,
    lanes: Vec<u64>,
    next_free_page: usize,
}

impl ElasticPagedKvPlaneEpochV2 {
    /// Create an empty table at any ElasticWord v1 width, including W64.
    pub fn new(
        config: PagedKvConfig,
        width: ElasticWordWidthV1,
    ) -> Result<Self, ElasticPagedKvPlaneEpochError> {
        config.capacity_tokens()?;
        debug_assert!(width.lanes() >= FLAT_ELASTIC_PAGED_KV_REQUIRED_LANES_V2);
        Ok(Self {
            config,
            live_tokens: 0,
            generation: 0,
            width,
            lanes: Vec::new(),
            next_free_page: 0,
        })
    }

    /// Current width carried once for the complete page map.
    #[must_use]
    pub const fn width(&self) -> ElasticWordWidthV1 {
        self.width
    }

    /// Current configuration.
    #[must_use]
    pub const fn config(&self) -> PagedKvConfig {
        self.config
    }

    /// Live token count.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.live_tokens
    }

    /// Whether the logical cache is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live_tokens == 0
    }

    /// Generation carried once for the whole mapping epoch.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of mapped logical pages.
    #[must_use]
    pub fn mapped_pages(&self) -> usize {
        self.lanes.len() / usize::from(self.width.lanes())
    }

    /// Borrow flat native-lane storage.
    #[must_use]
    pub fn as_lanes(&self) -> &[u64] {
        &self.lanes
    }

    /// Snapshot generation plus width-tagged mapping plane.
    pub fn snapshot(&self) -> Result<ElasticPagedKvSnapshotV2, ElasticPagedKvPlaneEpochError> {
        Ok(ElasticPagedKvSnapshotV2 {
            generation: self.generation,
            plane: ElasticWordPlaneV1::new(self.width, self.lanes.clone())?,
        })
    }

    /// Append tokens with the same deterministic lowest-free-page assignment as
    /// the authoritative table.
    pub fn append(&mut self, tokens: usize) -> Result<(), ElasticPagedKvPlaneEpochError> {
        if tokens == 0 {
            return Ok(());
        }

        let new_len = self
            .live_tokens
            .checked_add(tokens)
            .ok_or(PagedKvError::CapacityOverflow)?;
        let capacity = self.config.capacity_tokens()?;
        if new_len > capacity {
            return Err(PagedKvError::CapacityExceeded {
                requested: new_len,
                capacity,
            }
            .into());
        }

        let required_pages = new_len.div_ceil(self.config.page_size);
        while self.mapped_pages() < required_pages {
            let physical_page = self.next_free_page;
            let physical_lane = u64::try_from(physical_page).map_err(|_| {
                ElasticPagedKvPlaneEpochError::PhysicalPageOutOfRange { physical_page }
            })?;
            self.push_page_word(physical_lane)?;
            self.next_free_page = physical_page
                .checked_add(1)
                .ok_or(ElasticPagedKvPlaneEpochError::StorageOverflow)?;
        }
        self.live_tokens = new_len;
        Ok(())
    }

    /// Truncate the live prefix and release complete tail page-map words.
    pub fn truncate(&mut self, new_len: usize) -> Result<(), ElasticPagedKvPlaneEpochError> {
        if new_len > self.live_tokens {
            return Err(PagedKvError::TruncateOutOfBounds {
                requested_len: new_len,
                current_len: self.live_tokens,
            }
            .into());
        }
        if new_len == self.live_tokens {
            return Ok(());
        }

        let required_pages = if new_len == 0 {
            0
        } else {
            new_len.div_ceil(self.config.page_size)
        };
        let target_lanes = required_pages
            .checked_mul(usize::from(self.width.lanes()))
            .ok_or(ElasticPagedKvPlaneEpochError::StorageOverflow)?;
        self.lanes.truncate(target_lanes);

        self.next_free_page = if required_pages == 0 {
            0
        } else {
            let physical_lane = self.page_word(required_pages - 1)?[0];
            usize::try_from(physical_lane)
                .map_err(
                    |_| ElasticPagedKvPlaneEpochError::StoredPhysicalPageOutOfRange {
                        physical_page: physical_lane,
                    },
                )?
                .checked_add(1)
                .ok_or(ElasticPagedKvPlaneEpochError::StorageOverflow)?
        };
        self.live_tokens = new_len;
        Ok(())
    }

    /// Resolve one logical token using plane-level generation.
    pub fn address(
        &self,
        logical_token: usize,
    ) -> Result<Option<PagedKvAddress>, ElasticPagedKvPlaneEpochError> {
        if logical_token >= self.live_tokens {
            return Ok(None);
        }

        let logical_page = logical_token / self.config.page_size;
        let offset_in_page = logical_token % self.config.page_size;
        let physical_lane = self.page_word(logical_page)?[0];
        let physical_page = usize::try_from(physical_lane).map_err(|_| {
            ElasticPagedKvPlaneEpochError::StoredPhysicalPageOutOfRange {
                physical_page: physical_lane,
            }
        })?;

        Ok(Some(PagedKvAddress {
            physical_page,
            offset_in_page,
            generation: self.generation,
        }))
    }

    /// Logical telemetry parity surface.
    pub fn telemetry(&self) -> Result<PagedKvTelemetry, ElasticPagedKvPlaneEpochError> {
        let capacity_tokens = self.config.capacity_tokens()?;
        let mapped_pages = self.mapped_pages();
        let allocated_tokens = mapped_pages
            .checked_mul(self.config.page_size)
            .ok_or(PagedKvError::CapacityOverflow)?;

        Ok(PagedKvTelemetry {
            live_tokens: self.live_tokens,
            capacity_tokens,
            mapped_pages,
            free_pages: self.config.physical_pages - mapped_pages,
            internal_fragmentation_tokens: allocated_tokens - self.live_tokens,
            generation: self.generation,
        })
    }

    /// Invalidate the prior mapping epoch, clear every mapping and restart page
    /// assignment from physical page zero.
    pub fn reset(&mut self) -> Result<(), ElasticPagedKvPlaneEpochError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(PagedKvError::GenerationOverflow)?;
        self.live_tokens = 0;
        self.lanes.clear();
        self.next_free_page = 0;
        Ok(())
    }

    /// Change the page-map word width without duplicating generation per page.
    pub fn reencode_width(
        &mut self,
        target: ElasticWordWidthV1,
    ) -> Result<(), ElasticPagedKvPlaneEpochError> {
        if target == self.width {
            return Ok(());
        }
        self.validate_reserved_lanes_zero()?;

        let mapped_pages = self.mapped_pages();
        let target_lanes = usize::from(target.lanes());
        let target_len = mapped_pages
            .checked_mul(target_lanes)
            .ok_or(ElasticPagedKvPlaneEpochError::StorageOverflow)?;
        let mut reencoded = vec![0_u64; target_len];

        for logical_page in 0..mapped_pages {
            let physical_page = self.page_word(logical_page)?[0];
            let start = logical_page
                .checked_mul(target_lanes)
                .ok_or(ElasticPagedKvPlaneEpochError::StorageOverflow)?;
            reencoded[start] = physical_page;
        }

        self.width = target;
        self.lanes = reencoded;
        Ok(())
    }

    /// Differentially verify complete observable semantics against the
    /// authoritative table for the current live domain and first out-of-range
    /// token.
    pub fn verify_against(
        &self,
        reference: &PagedKvTable,
    ) -> Result<bool, ElasticPagedKvPlaneEpochError> {
        if self.config != reference.config()
            || self.live_tokens != reference.len()
            || self.generation != reference.generation()
            || self.telemetry()? != reference.telemetry()?
        {
            return Ok(false);
        }

        for logical_token in 0..self.live_tokens {
            if self.address(logical_token)? != reference.address(logical_token) {
                return Ok(false);
            }
        }

        if self.address(self.live_tokens)? != reference.address(self.live_tokens) {
            return Ok(false);
        }

        Ok(true)
    }

    fn push_page_word(&mut self, physical_page: u64) -> Result<(), ElasticPagedKvPlaneEpochError> {
        let lanes_per_word = usize::from(self.width.lanes());
        let new_len = self
            .lanes
            .len()
            .checked_add(lanes_per_word)
            .ok_or(ElasticPagedKvPlaneEpochError::StorageOverflow)?;
        self.lanes.resize(new_len, 0);
        self.lanes[new_len - lanes_per_word] = physical_page;
        Ok(())
    }

    fn page_word(&self, logical_page: usize) -> Result<&[u64], ElasticPagedKvPlaneEpochError> {
        let lanes_per_word = usize::from(self.width.lanes());
        let start = logical_page
            .checked_mul(lanes_per_word)
            .ok_or(ElasticPagedKvPlaneEpochError::StorageOverflow)?;
        let end = start
            .checked_add(lanes_per_word)
            .ok_or(ElasticPagedKvPlaneEpochError::StorageOverflow)?;
        self.lanes
            .get(start..end)
            .ok_or(ElasticPagedKvPlaneEpochError::StorageOverflow)
    }

    fn validate_reserved_lanes_zero(&self) -> Result<(), ElasticPagedKvPlaneEpochError> {
        let lanes_per_word = usize::from(self.width.lanes());
        for (logical_page, word) in self.lanes.chunks_exact(lanes_per_word).enumerate() {
            for (lane_index, value) in word.iter().copied().enumerate().skip(1) {
                if value != 0 {
                    return Err(ElasticPagedKvPlaneEpochError::ReservedLaneNonZero {
                        logical_page,
                        lane_index,
                        value,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    enum Operation {
        Append(usize),
        Truncate(usize),
        Reset,
    }

    fn widths() -> [ElasticWordWidthV1; 6] {
        [64_u16, 128, 256, 512, 1024, 2048].map(|bits| ElasticWordWidthV1::from_bits(bits).unwrap())
    }

    fn assert_equivalent(reference: &PagedKvTable, elastic: &ElasticPagedKvPlaneEpochV2) {
        assert!(elastic.verify_against(reference).unwrap());
        assert_eq!(elastic.telemetry().unwrap(), reference.telemetry().unwrap());
    }

    fn apply_pair(
        reference: &mut PagedKvTable,
        elastic: &mut ElasticPagedKvPlaneEpochV2,
        operation: Operation,
    ) {
        let reference_before = reference.clone();
        let elastic_before = elastic.clone();

        let reference_result = match operation {
            Operation::Append(tokens) => reference.append(tokens),
            Operation::Truncate(len) => reference.truncate(len),
            Operation::Reset => reference.reset(),
        };
        let elastic_result = match operation {
            Operation::Append(tokens) => elastic.append(tokens),
            Operation::Truncate(len) => elastic.truncate(len),
            Operation::Reset => elastic.reset(),
        };

        match (reference_result, elastic_result) {
            (Ok(()), Ok(())) => assert_equivalent(reference, elastic),
            (Err(reference_error), Err(ElasticPagedKvPlaneEpochError::Paged(elastic_error))) => {
                assert_eq!(reference_error, elastic_error);
                assert_eq!(*reference, reference_before);
                assert_eq!(*elastic, elastic_before);
            }
            (reference_result, elastic_result) => panic!(
                "differential result mismatch: reference={reference_result:?} elastic={elastic_result:?}"
            ),
        }
    }

    #[test]
    fn w64_matches_authoritative_table_across_lifecycle() {
        let config = PagedKvConfig {
            page_size: 4,
            physical_pages: 5,
        };
        let mut reference = PagedKvTable::new(config).unwrap();
        let mut elastic =
            ElasticPagedKvPlaneEpochV2::new(config, ElasticWordWidthV1::from_bits(64).unwrap())
                .unwrap();

        for operation in [
            Operation::Append(1),
            Operation::Append(8),
            Operation::Truncate(5),
            Operation::Append(7),
            Operation::Reset,
            Operation::Append(6),
            Operation::Truncate(0),
            Operation::Append(3),
        ] {
            apply_pair(&mut reference, &mut elastic, operation);
        }
    }

    #[test]
    fn every_frozen_width_matches_the_same_semantics() {
        let config = PagedKvConfig {
            page_size: 3,
            physical_pages: 5,
        };
        for width in widths() {
            let mut reference = PagedKvTable::new(config).unwrap();
            let mut elastic = ElasticPagedKvPlaneEpochV2::new(config, width).unwrap();
            for operation in [
                Operation::Append(10),
                Operation::Truncate(4),
                Operation::Append(6),
                Operation::Reset,
                Operation::Append(15),
            ] {
                apply_pair(&mut reference, &mut elastic, operation);
            }
        }
    }

    #[test]
    fn reencoding_w64_through_w2048_preserves_reference_semantics() {
        let config = PagedKvConfig {
            page_size: 2,
            physical_pages: 6,
        };
        let mut reference = PagedKvTable::new(config).unwrap();
        reference.append(9).unwrap();

        let mut elastic =
            ElasticPagedKvPlaneEpochV2::new(config, ElasticWordWidthV1::from_bits(64).unwrap())
                .unwrap();
        elastic.append(9).unwrap();
        assert_equivalent(&reference, &elastic);

        for bits in [2048_u16, 128, 1024, 256, 512, 64] {
            elastic
                .reencode_width(ElasticWordWidthV1::from_bits(bits).unwrap())
                .unwrap();
            assert_eq!(elastic.width().bits(), bits);
            assert_equivalent(&reference, &elastic);
        }
    }

    #[test]
    fn snapshot_carries_generation_once_and_w64_one_lane_per_page() {
        let config = PagedKvConfig {
            page_size: 2,
            physical_pages: 4,
        };
        let mut table =
            ElasticPagedKvPlaneEpochV2::new(config, ElasticWordWidthV1::from_bits(64).unwrap())
                .unwrap();
        table.append(5).unwrap();
        table.reset().unwrap();
        table.append(3).unwrap();

        let snapshot = table.snapshot().unwrap();
        assert_eq!(snapshot.generation, 1);
        assert_eq!(snapshot.plane.width().bits(), 64);
        assert_eq!(snapshot.plane.word_count(), 2);
        assert_eq!(snapshot.plane.as_lanes(), &[0, 1]);
    }

    #[test]
    fn bounded_operation_sequences_match_authoritative_table() {
        let config = PagedKvConfig {
            page_size: 2,
            physical_pages: 3,
        };
        let operations = [
            Operation::Append(0),
            Operation::Append(1),
            Operation::Append(2),
            Operation::Append(3),
            Operation::Truncate(0),
            Operation::Truncate(1),
            Operation::Truncate(2),
            Operation::Truncate(3),
            Operation::Truncate(4),
            Operation::Reset,
        ];

        let initial_reference = PagedKvTable::new(config).unwrap();
        let initial_elastic =
            ElasticPagedKvPlaneEpochV2::new(config, ElasticWordWidthV1::from_bits(64).unwrap())
                .unwrap();

        let mut frontier = vec![(initial_reference, initial_elastic)];
        for _depth in 0..4 {
            let mut next = Vec::new();
            for (reference, elastic) in frontier {
                for operation in operations {
                    let mut next_reference = reference.clone();
                    let mut next_elastic = elastic.clone();
                    apply_pair(&mut next_reference, &mut next_elastic, operation);
                    next.push((next_reference, next_elastic));
                }
            }
            frontier = next;
        }

        assert_eq!(frontier.len(), operations.len().pow(4));
    }

    #[test]
    fn reserved_high_lane_blocks_narrowing() {
        let config = PagedKvConfig {
            page_size: 1,
            physical_pages: 2,
        };
        let mut table =
            ElasticPagedKvPlaneEpochV2::new(config, ElasticWordWidthV1::from_bits(128).unwrap())
                .unwrap();
        table.append(1).unwrap();
        table.lanes[1] = 0x55;

        assert_eq!(
            table.reencode_width(ElasticWordWidthV1::from_bits(64).unwrap()),
            Err(ElasticPagedKvPlaneEpochError::ReservedLaneNonZero {
                logical_page: 0,
                lane_index: 1,
                value: 0x55,
            })
        );
    }
}
