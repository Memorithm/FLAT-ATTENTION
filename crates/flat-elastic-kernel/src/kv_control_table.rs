//! Experimental elastic-word implementation of FLAT paged-KV metadata.
//!
//! The authoritative production table remains `flat_attention::paged_kv::PagedKvTable`.
//! This shadow implementation exists to prove that the same semantics can be
//! represented by one flat `u64` lane plane whose width is carried once for the
//! complete table and can be re-encoded between qualified widths.
//!
//! No performance, allocation, cache-locality or production-promotion claim is
//! made by this module.

use elastic_core::{ElasticWordError, ElasticWordPlaneV1, ElasticWordWidthV1};
use flat_attention::paged_kv::{
    PagedKvAddress, PagedKvConfig, PagedKvError, PagedKvTable, PagedKvTelemetry,
};
use std::fmt;

/// Schema identity for the first FLAT elastic page-table shadow.
pub const FLAT_ELASTIC_PAGED_KV_TABLE_V1: &str = "flat.elastic-paged-kv-table@1.0.0";

/// Number of semantic lanes currently required by FLAT page metadata.
///
/// Lane 0 = physical page identity.
/// Lane 1 = table generation.
/// Lanes 2+ are reserved and must remain zero in this schema.
pub const FLAT_ELASTIC_PAGED_KV_REQUIRED_LANES_V1: u8 = 2;

/// Shadow-table failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElasticPagedKvTableError {
    /// Existing FLAT configuration or semantic operation failed.
    Paged(PagedKvError),
    /// Selected representation width cannot preserve the v1 metadata schema.
    WidthTooNarrow { bits: u16, required_bits: u16 },
    /// Physical page identity cannot be represented in the u64 lane schema.
    PhysicalPageOutOfRange { physical_page: usize },
    /// A stored physical-page lane cannot be represented as usize on this target.
    StoredPhysicalPageOutOfRange { physical_page: u64 },
    /// Lane-count or buffer-length arithmetic overflowed.
    StorageOverflow,
    /// Reserved high lanes are non-zero, so narrowing/re-encoding is ambiguous.
    ReservedLaneNonZero {
        logical_page: usize,
        lane_index: usize,
        value: u64,
    },
    /// Generic ElasticWord plane validation failed.
    Elastic(ElasticWordError),
}

impl fmt::Display for ElasticPagedKvTableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Paged(error) => write!(f, "paged-KV semantic error: {error}"),
            Self::WidthTooNarrow {
                bits,
                required_bits,
            } => write!(
                f,
                "elastic paged-KV width {bits} bits is narrower than the {required_bits}-bit v1 semantic requirement"
            ),
            Self::PhysicalPageOutOfRange { physical_page } => write!(
                f,
                "physical page {physical_page} cannot be represented in a u64 lane"
            ),
            Self::StoredPhysicalPageOutOfRange { physical_page } => write!(
                f,
                "stored physical page {physical_page} cannot be represented as usize on this target"
            ),
            Self::StorageOverflow => {
                f.write_str("elastic paged-KV lane storage arithmetic overflowed")
            }
            Self::ReservedLaneNonZero {
                logical_page,
                lane_index,
                value,
            } => write!(
                f,
                "elastic paged-KV logical page {logical_page} reserved lane {lane_index} is non-zero ({value:#x})"
            ),
            Self::Elastic(error) => write!(f, "ElasticWord error: {error}"),
        }
    }
}

impl std::error::Error for ElasticPagedKvTableError {}

impl From<PagedKvError> for ElasticPagedKvTableError {
    fn from(value: PagedKvError) -> Self {
        Self::Paged(value)
    }
}

impl From<ElasticWordError> for ElasticPagedKvTableError {
    fn from(value: ElasticWordError) -> Self {
        Self::Elastic(value)
    }
}

/// Flat elastic shadow of `PagedKvTable`.
///
/// The table carries width once and stores every logical page consecutively:
///
/// ```text
/// width = W256
/// [physical,generation,0,0][physical,generation,0,0]...
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElasticPagedKvTableV1 {
    config: PagedKvConfig,
    live_tokens: usize,
    generation: u64,
    width: ElasticWordWidthV1,
    lanes: Vec<u64>,
    next_free_page: usize,
}

impl ElasticPagedKvTableV1 {
    /// Construct an empty shadow table at one qualified width.
    pub fn new(
        config: PagedKvConfig,
        width: ElasticWordWidthV1,
    ) -> Result<Self, ElasticPagedKvTableError> {
        config.capacity_tokens()?;
        validate_width(width)?;
        Ok(Self {
            config,
            live_tokens: 0,
            generation: 0,
            width,
            lanes: Vec::new(),
            next_free_page: 0,
        })
    }

    /// Current representation width carried once for the complete table.
    #[must_use]
    pub const fn width(&self) -> ElasticWordWidthV1 {
        self.width
    }

    /// Current FLAT table configuration.
    #[must_use]
    pub const fn config(&self) -> PagedKvConfig {
        self.config
    }

    /// Number of live logical tokens.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.live_tokens
    }

    /// Whether no logical token is live.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live_tokens == 0
    }

    /// Current table generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of currently mapped logical pages.
    #[must_use]
    pub fn mapped_pages(&self) -> usize {
        self.lanes.len() / usize::from(self.width.lanes())
    }

    /// Borrow the flat native-lane storage.
    #[must_use]
    pub fn as_lanes(&self) -> &[u64] {
        &self.lanes
    }

    /// Materialize the current state as the generic ElasticWord plane contract.
    pub fn elastic_plane(&self) -> Result<ElasticWordPlaneV1, ElasticPagedKvTableError> {
        Ok(ElasticWordPlaneV1::new(self.width, self.lanes.clone())?)
    }

    /// Append logical tokens using the same deterministic page assignment as
    /// the authoritative FLAT table.
    pub fn append(&mut self, tokens: usize) -> Result<(), ElasticPagedKvTableError> {
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
            let physical_lane = u64::try_from(physical_page)
                .map_err(|_| ElasticPagedKvTableError::PhysicalPageOutOfRange { physical_page })?;
            self.push_page_word(physical_lane, self.generation)?;
            self.next_free_page = physical_page
                .checked_add(1)
                .ok_or(ElasticPagedKvTableError::StorageOverflow)?;
        }

        self.live_tokens = new_len;
        Ok(())
    }

    /// Rewind the live logical prefix, releasing complete tail page metadata.
    pub fn truncate(&mut self, new_len: usize) -> Result<(), ElasticPagedKvTableError> {
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
            .ok_or(ElasticPagedKvTableError::StorageOverflow)?;
        self.lanes.truncate(target_lanes);

        self.next_free_page = if required_pages == 0 {
            0
        } else {
            let last_physical = self.page_lanes(required_pages - 1)?[0];
            usize::try_from(last_physical)
                .map_err(|_| ElasticPagedKvTableError::StoredPhysicalPageOutOfRange {
                    physical_page: last_physical,
                })?
                .checked_add(1)
                .ok_or(ElasticPagedKvTableError::StorageOverflow)?
        };
        self.live_tokens = new_len;
        Ok(())
    }

    /// Resolve one logical token through the flat elastic metadata plane.
    pub fn address(
        &self,
        logical_token: usize,
    ) -> Result<Option<PagedKvAddress>, ElasticPagedKvTableError> {
        if logical_token >= self.live_tokens {
            return Ok(None);
        }

        let logical_page = logical_token / self.config.page_size;
        let offset_in_page = logical_token % self.config.page_size;
        let word = self.page_lanes(logical_page)?;
        let physical_page_u64 = word[0];
        let stored_generation = word[1];
        if stored_generation != self.generation {
            return Ok(None);
        }
        let physical_page = usize::try_from(physical_page_u64).map_err(|_| {
            ElasticPagedKvTableError::StoredPhysicalPageOutOfRange {
                physical_page: physical_page_u64,
            }
        })?;

        Ok(Some(PagedKvAddress {
            physical_page,
            offset_in_page,
            generation: stored_generation,
        }))
    }

    /// Report the same logical telemetry surface as `PagedKvTable`.
    pub fn telemetry(&self) -> Result<PagedKvTelemetry, ElasticPagedKvTableError> {
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

    /// Reset the table and advance the generation before physical-page reuse.
    pub fn reset(&mut self) -> Result<(), ElasticPagedKvTableError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(PagedKvError::GenerationOverflow)?;
        self.live_tokens = 0;
        self.lanes.clear();
        self.next_free_page = 0;
        Ok(())
    }

    /// Re-encode the table to another lossless ElasticWord width.
    ///
    /// The v1 semantics occupy only the first two lanes. Every reserved lane is
    /// checked to remain zero before a width change, so future schema extensions
    /// cannot be silently discarded.
    pub fn reencode_width(
        &mut self,
        target: ElasticWordWidthV1,
    ) -> Result<(), ElasticPagedKvTableError> {
        validate_width(target)?;
        if target == self.width {
            return Ok(());
        }

        self.validate_reserved_lanes_zero()?;

        let mapped_pages = self.mapped_pages();
        let target_lanes = usize::from(target.lanes());
        let target_len = mapped_pages
            .checked_mul(target_lanes)
            .ok_or(ElasticPagedKvTableError::StorageOverflow)?;
        let mut reencoded = vec![0_u64; target_len];

        for logical_page in 0..mapped_pages {
            let source = self.page_lanes(logical_page)?;
            let start = logical_page
                .checked_mul(target_lanes)
                .ok_or(ElasticPagedKvTableError::StorageOverflow)?;
            reencoded[start] = source[0];
            reencoded[start + 1] = source[1];
        }

        self.width = target;
        self.lanes = reencoded;
        Ok(())
    }

    /// Compare the shadow state against the current authoritative FLAT table.
    ///
    /// This is a differential qualification helper, not a promotion gate by
    /// itself.
    pub fn verify_against(
        &self,
        reference: &PagedKvTable,
    ) -> Result<bool, ElasticPagedKvTableError> {
        if self.config != reference.config()
            || self.live_tokens != reference.len()
            || self.generation != reference.generation()
            || self.telemetry()? != reference.telemetry()?
        {
            return Ok(false);
        }

        let capacity = self.config.capacity_tokens()?;
        for logical_token in 0..capacity {
            if self.address(logical_token)? != reference.address(logical_token) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn push_page_word(
        &mut self,
        physical_page: u64,
        generation: u64,
    ) -> Result<(), ElasticPagedKvTableError> {
        let lanes_per_word = usize::from(self.width.lanes());
        let new_len = self
            .lanes
            .len()
            .checked_add(lanes_per_word)
            .ok_or(ElasticPagedKvTableError::StorageOverflow)?;
        self.lanes.resize(new_len, 0);
        let start = new_len - lanes_per_word;
        self.lanes[start] = physical_page;
        self.lanes[start + 1] = generation;
        Ok(())
    }

    fn page_lanes(&self, logical_page: usize) -> Result<&[u64], ElasticPagedKvTableError> {
        let lanes_per_word = usize::from(self.width.lanes());
        let start = logical_page
            .checked_mul(lanes_per_word)
            .ok_or(ElasticPagedKvTableError::StorageOverflow)?;
        let end = start
            .checked_add(lanes_per_word)
            .ok_or(ElasticPagedKvTableError::StorageOverflow)?;
        self.lanes
            .get(start..end)
            .ok_or(ElasticPagedKvTableError::StorageOverflow)
    }

    fn validate_reserved_lanes_zero(&self) -> Result<(), ElasticPagedKvTableError> {
        let lanes_per_word = usize::from(self.width.lanes());
        for (logical_page, word) in self.lanes.chunks_exact(lanes_per_word).enumerate() {
            for (lane_index, value) in word.iter().copied().enumerate().skip(2) {
                if value != 0 {
                    return Err(ElasticPagedKvTableError::ReservedLaneNonZero {
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

fn validate_width(width: ElasticWordWidthV1) -> Result<(), ElasticPagedKvTableError> {
    if width.lanes() < FLAT_ELASTIC_PAGED_KV_REQUIRED_LANES_V1 {
        return Err(ElasticPagedKvTableError::WidthTooNarrow {
            bits: width.bits(),
            required_bits: u16::from(FLAT_ELASTIC_PAGED_KV_REQUIRED_LANES_V1) * 64,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn widths() -> [ElasticWordWidthV1; 5] {
        [128_u16, 256, 512, 1024, 2048].map(|bits| ElasticWordWidthV1::from_bits(bits).unwrap())
    }

    fn assert_equivalent(reference: &PagedKvTable, elastic: &ElasticPagedKvTableV1) {
        assert!(elastic.verify_against(reference).unwrap());
        assert_eq!(elastic.len(), reference.len());
        assert_eq!(elastic.generation(), reference.generation());
        assert_eq!(elastic.telemetry().unwrap(), reference.telemetry().unwrap());
    }

    #[test]
    fn every_lossless_width_matches_authoritative_operation_sequence() {
        let config = PagedKvConfig {
            page_size: 4,
            physical_pages: 6,
        };

        for width in widths() {
            let mut reference = PagedKvTable::new(config).unwrap();
            let mut elastic = ElasticPagedKvTableV1::new(config, width).unwrap();
            assert_equivalent(&reference, &elastic);

            reference.append(1).unwrap();
            elastic.append(1).unwrap();
            assert_equivalent(&reference, &elastic);

            reference.append(8).unwrap();
            elastic.append(8).unwrap();
            assert_equivalent(&reference, &elastic);

            reference.truncate(5).unwrap();
            elastic.truncate(5).unwrap();
            assert_equivalent(&reference, &elastic);

            reference.append(7).unwrap();
            elastic.append(7).unwrap();
            assert_equivalent(&reference, &elastic);

            reference.reset().unwrap();
            elastic.reset().unwrap();
            assert_equivalent(&reference, &elastic);

            reference.append(6).unwrap();
            elastic.append(6).unwrap();
            assert_equivalent(&reference, &elastic);
        }
    }

    #[test]
    fn width_reencoding_preserves_authoritative_semantics() {
        let config = PagedKvConfig {
            page_size: 3,
            physical_pages: 5,
        };
        let mut reference = PagedKvTable::new(config).unwrap();
        reference.append(10).unwrap();

        let mut elastic =
            ElasticPagedKvTableV1::new(config, ElasticWordWidthV1::from_bits(128).unwrap())
                .unwrap();
        elastic.append(10).unwrap();
        assert_equivalent(&reference, &elastic);

        for bits in [2048_u16, 512, 1024, 256, 128] {
            elastic
                .reencode_width(ElasticWordWidthV1::from_bits(bits).unwrap())
                .unwrap();
            assert_eq!(elastic.width().bits(), bits);
            assert_equivalent(&reference, &elastic);
        }
    }

    #[test]
    fn w64_is_rejected_until_a_lossless_packing_contract_exists() {
        let config = PagedKvConfig {
            page_size: 2,
            physical_pages: 2,
        };
        let error = ElasticPagedKvTableV1::new(config, ElasticWordWidthV1::from_bits(64).unwrap())
            .unwrap_err();
        assert_eq!(
            error,
            ElasticPagedKvTableError::WidthTooNarrow {
                bits: 64,
                required_bits: 128,
            }
        );
    }

    #[test]
    fn failed_capacity_append_matches_reference_and_is_non_mutating() {
        let config = PagedKvConfig {
            page_size: 2,
            physical_pages: 2,
        };
        let mut reference = PagedKvTable::new(config).unwrap();
        let mut elastic =
            ElasticPagedKvTableV1::new(config, ElasticWordWidthV1::from_bits(128).unwrap())
                .unwrap();

        reference.append(3).unwrap();
        elastic.append(3).unwrap();
        let before = elastic.clone();

        assert_eq!(
            reference.append(2),
            Err(PagedKvError::CapacityExceeded {
                requested: 5,
                capacity: 4,
            })
        );
        assert_eq!(
            elastic.append(2),
            Err(ElasticPagedKvTableError::Paged(
                PagedKvError::CapacityExceeded {
                    requested: 5,
                    capacity: 4,
                }
            ))
        );
        assert_eq!(elastic, before);
    }

    #[test]
    fn reserved_lane_data_blocks_reencoding_fail_closed() {
        let config = PagedKvConfig {
            page_size: 2,
            physical_pages: 2,
        };
        let mut elastic =
            ElasticPagedKvTableV1::new(config, ElasticWordWidthV1::from_bits(256).unwrap())
                .unwrap();
        elastic.append(1).unwrap();

        elastic.lanes[2] = 0x55;
        assert_eq!(
            elastic.reencode_width(ElasticWordWidthV1::from_bits(128).unwrap()),
            Err(ElasticPagedKvTableError::ReservedLaneNonZero {
                logical_page: 0,
                lane_index: 2,
                value: 0x55,
            })
        );
    }

    #[test]
    fn elastic_plane_carries_width_once_for_all_mappings() {
        let config = PagedKvConfig {
            page_size: 1,
            physical_pages: 3,
        };
        let mut elastic =
            ElasticPagedKvTableV1::new(config, ElasticWordWidthV1::from_bits(512).unwrap())
                .unwrap();
        elastic.append(3).unwrap();

        let plane = elastic.elastic_plane().unwrap();
        assert_eq!(plane.width().bits(), 512);
        assert_eq!(plane.word_count(), 3);
        assert_eq!(
            plane.as_lanes(),
            &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,]
        );
    }
}
