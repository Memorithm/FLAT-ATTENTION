//! Projection of FLAT paged-KV mapping metadata into ElasticWord v1.
//!
//! This bridge is deliberately read-only. FLAT keeps ownership of paged-KV
//! semantics and mutation; ElasticXxx owns the generic elastic representation
//! contract. The bridge only materializes the current authoritative mapping
//! metadata as a flat lane plane for planning/evidence.

use elastic_kv::{ElasticWordError, ElasticWordPlaneV1, ElasticWordWidthV1};
use flat_attention::paged_kv::PagedKvTable;
use std::fmt;

/// Versioned FLAT projection contract for paged-KV control metadata.
pub const FLAT_PAGED_KV_ELASTIC_WORD_V1: &str = "flat.paged-kv-elastic-word@1.0.0";

/// Current lossless generic width for one FLAT page mapping.
///
/// The authoritative FLAT mapping currently contains two independent values:
/// physical page identity and table generation. The v1 projection therefore
/// uses two native 64-bit lanes (W128). No W64 packing policy is implied.
pub const FLAT_PAGED_KV_CONTROL_LANES_V1: u8 = 2;

/// Errors from the read-only FLAT -> ElasticWord projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagedKvElasticWordError {
    /// A target cannot represent a FLAT physical page identity as a u64 lane.
    PhysicalPageOutOfRange {
        logical_page: usize,
        physical_page: usize,
    },
    /// Flat lane capacity arithmetic overflowed.
    LaneCapacityOverflow,
    /// Generic ElasticWord validation rejected the produced plane.
    Elastic(ElasticWordError),
}

impl fmt::Display for PagedKvElasticWordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PhysicalPageOutOfRange {
                logical_page,
                physical_page,
            } => write!(
                f,
                "FLAT logical page {logical_page} physical page {physical_page} cannot be represented as a u64 lane"
            ),
            Self::LaneCapacityOverflow => {
                f.write_str("FLAT paged-KV ElasticWord lane capacity overflowed")
            }
            Self::Elastic(error) => write!(f, "ElasticWord projection failed: {error}"),
        }
    }
}

impl std::error::Error for PagedKvElasticWordError {}

impl From<ElasticWordError> for PagedKvElasticWordError {
    fn from(value: ElasticWordError) -> Self {
        Self::Elastic(value)
    }
}

/// Return the minimum v1 width that preserves the complete current FLAT page
/// mapping without assigning a narrower packing policy.
pub fn current_paged_kv_lossless_width() -> Result<ElasticWordWidthV1, PagedKvElasticWordError> {
    Ok(ElasticWordWidthV1::from_lanes(
        FLAT_PAGED_KV_CONTROL_LANES_V1,
    )?)
}

/// Project current FLAT page mappings into one contiguous ElasticWord plane.
///
/// Each logical page contributes exactly two lanes in logical-page order:
///
///     [physical_page, generation][physical_page, generation]...
///
/// The width is carried once by the ElasticWord plane. The function does not
/// mutate FLAT, authorize a representation transition, or claim that W128 is
/// optimal. It is the lossless baseline from which narrower/wider candidates
/// can be evaluated.
pub fn project_paged_kv_control_plane(
    table: &PagedKvTable,
) -> Result<ElasticWordPlaneV1, PagedKvElasticWordError> {
    let mappings = table.mapped_page_metadata();
    let mapped_pages = mappings.len();
    let capacity = mapped_pages
        .checked_mul(usize::from(FLAT_PAGED_KV_CONTROL_LANES_V1))
        .ok_or(PagedKvElasticWordError::LaneCapacityOverflow)?;
    let mut lanes = Vec::with_capacity(capacity);

    for (logical_page, (physical_page, generation)) in mappings.enumerate() {
        let physical_page = u64::try_from(physical_page).map_err(|_| {
            PagedKvElasticWordError::PhysicalPageOutOfRange {
                logical_page,
                physical_page,
            }
        })?;
        lanes.push(physical_page);
        lanes.push(generation);
    }

    Ok(ElasticWordPlaneV1::new(
        current_paged_kv_lossless_width()?,
        lanes,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_attention::paged_kv::PagedKvConfig;

    #[test]
    fn projection_is_flat_w128_in_logical_page_order() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 2,
            physical_pages: 4,
        })
        .unwrap();
        table.append(5).unwrap();

        let plane = project_paged_kv_control_plane(&table).unwrap();
        assert_eq!(plane.width().bits(), 128);
        assert_eq!(plane.word_count(), 3);
        assert_eq!(plane.as_lanes(), &[0, 0, 1, 0, 2, 0]);
    }

    #[test]
    fn projection_preserves_generation_after_reset_and_page_reuse() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 2,
            physical_pages: 3,
        })
        .unwrap();
        table.append(3).unwrap();
        table.reset().unwrap();
        table.append(3).unwrap();

        let plane = project_paged_kv_control_plane(&table).unwrap();
        assert_eq!(plane.as_lanes(), &[0, 1, 1, 1]);
    }

    #[test]
    fn wider_elastic_plane_is_structurally_expandable_without_semantic_rewrite() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 2,
        })
        .unwrap();
        table.append(5).unwrap();

        let baseline = project_paged_kv_control_plane(&table).unwrap();
        let w512 = ElasticWordWidthV1::from_bits(512).unwrap();
        let expanded = baseline.reference_repack_zero_extended(w512).unwrap();

        assert_eq!(expanded.word_count(), 2);
        assert_eq!(expanded.word(0).unwrap(), &[0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(expanded.word(1).unwrap(), &[1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn baseline_does_not_silently_authorize_w64() {
        assert_eq!(current_paged_kv_lossless_width().unwrap().bits(), 128);
    }
}
