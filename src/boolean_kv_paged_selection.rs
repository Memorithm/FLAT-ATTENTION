use core::fmt;

use crate::api::boolean_kv::{
    BooleanKvCache, BooleanKvError, BooleanKvMatch, PackedBooleanSignature,
};
use crate::paged_kv::{PagedKvError, PagedKvTable};

/// Numerical K/V storage geometry used only for exact traffic accounting.
///
/// `scalar_bytes` is the stored byte width of one numerical K or V scalar
/// (for example, 2 for fp16/bf16 and 4 for fp32). The Boolean index never
/// changes the authority of the numerical K/V payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericalKvPageGeometry {
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scalar_bytes: usize,
}

impl NumericalKvPageGeometry {
    pub fn bytes_per_token(self) -> Result<usize, BooleanKvPagedSelectionError> {
        if self.kv_heads == 0 || self.head_dim == 0 || self.scalar_bytes == 0 {
            return Err(BooleanKvPagedSelectionError::ZeroNumericalGeometry);
        }
        // K + V, hence the leading factor of two.
        2usize
            .checked_mul(self.kv_heads)
            .and_then(|value| value.checked_mul(self.head_dim))
            .and_then(|value| value.checked_mul(self.scalar_bytes))
            .ok_or(BooleanKvPagedSelectionError::ArithmeticOverflow)
    }
}

/// One Boolean-selected numerical KV page.
///
/// `logical_page` preserves the original sequence/page identity. Consumers
/// must not reinterpret the selected pages as a dense renumbered sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanKvSelectedPage {
    pub logical_page: usize,
    pub physical_page: usize,
    pub live_tokens: usize,
    pub hamming_distance: usize,
    pub xnor_matches: usize,
}

/// Correctness-first BIKV handoff from the Boolean page index to numerical KV.
///
/// This plan is metadata only. It does not stage/copy numerical K/V, submit GPU
/// work, compact positions, or claim an end-to-end speedup. Selected pages are
/// sorted by original logical page so the consumer retains sequence identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanIndexedKvSelection {
    pub generation: u64,
    pub signature_bits: usize,
    pub live_tokens: usize,
    pub mapped_pages: usize,
    pub boolean_pages_scanned: usize,
    pub boolean_key_bytes_read: usize,
    pub full_numerical_kv_bytes: usize,
    pub selected_numerical_kv_bytes: usize,
    pub avoided_numerical_kv_bytes: usize,
    pub selected_pages: Vec<BooleanKvSelectedPage>,
}

impl BooleanIndexedKvSelection {
    #[must_use]
    pub fn selected_page_count(&self) -> usize {
        self.selected_pages.len()
    }

    #[must_use]
    pub fn selected_page_ids(&self) -> Vec<usize> {
        self.selected_pages
            .iter()
            .map(|page| page.logical_page)
            .collect()
    }

    #[must_use]
    pub fn candidate_density(&self) -> f64 {
        if self.mapped_pages == 0 {
            0.0
        } else {
            self.selected_pages.len() as f64 / self.mapped_pages as f64
        }
    }

    #[must_use]
    pub fn numerical_bytes_avoided_per_boolean_byte(&self) -> Option<f64> {
        if self.boolean_key_bytes_read == 0 {
            None
        } else {
            Some(self.avoided_numerical_kv_bytes as f64 / self.boolean_key_bytes_read as f64)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BooleanKvPagedSelectionError {
    Boolean(BooleanKvError),
    Table(PagedKvError),
    ZeroNumericalGeometry,
    ArithmeticOverflow,
    GenerationMismatch {
        boolean_generation: u64,
        paged_generation: u64,
    },
    PageCountMismatch {
        boolean_pages: usize,
        mapped_pages: usize,
    },
    CandidateOutOfRange {
        logical_page: usize,
        mapped_pages: usize,
    },
    MissingPagedAddress {
        logical_page: usize,
    },
}

impl fmt::Display for BooleanKvPagedSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boolean(error) => write!(f, "{error}"),
            Self::Table(error) => write!(f, "{error}"),
            Self::ZeroNumericalGeometry => write!(
                f,
                "BIKV numerical geometry requires non-zero kv_heads, head_dim and scalar_bytes"
            ),
            Self::ArithmeticOverflow => write!(f, "BIKV traffic accounting overflows usize"),
            Self::GenerationMismatch {
                boolean_generation,
                paged_generation,
            } => write!(
                f,
                "Boolean KV generation {boolean_generation} does not match paged KV generation {paged_generation}"
            ),
            Self::PageCountMismatch {
                boolean_pages,
                mapped_pages,
            } => write!(
                f,
                "Boolean KV contains {boolean_pages} pages but paged numerical KV maps {mapped_pages} pages"
            ),
            Self::CandidateOutOfRange {
                logical_page,
                mapped_pages,
            } => write!(
                f,
                "Boolean KV candidate page {logical_page} is outside {mapped_pages} mapped numerical pages"
            ),
            Self::MissingPagedAddress { logical_page } => write!(
                f,
                "numerical KV has no live address for selected logical page {logical_page}"
            ),
        }
    }
}

impl std::error::Error for BooleanKvPagedSelectionError {}

impl From<BooleanKvError> for BooleanKvPagedSelectionError {
    fn from(value: BooleanKvError) -> Self {
        Self::Boolean(value)
    }
}

impl From<PagedKvError> for BooleanKvPagedSelectionError {
    fn from(value: PagedKvError) -> Self {
        Self::Table(value)
    }
}

/// Build a BIKV page-selection plan while keeping numerical K/V authoritative.
///
/// The Boolean cache and numerical paged table must have the same generation
/// and exactly one Boolean signature per mapped numerical page. The Boolean
/// search may rank by Hamming distance, but the returned pages are reordered by
/// original logical page before handoff to any numerical consumer.
pub fn build_boolean_indexed_kv_selection(
    boolean_cache: &BooleanKvCache,
    paged_table: &PagedKvTable,
    query: &PackedBooleanSignature,
    max_distance: usize,
    limit: Option<usize>,
    numerical_geometry: NumericalKvPageGeometry,
) -> Result<BooleanIndexedKvSelection, BooleanKvPagedSelectionError> {
    let telemetry = paged_table.telemetry()?;
    if boolean_cache.generation() != telemetry.generation {
        return Err(BooleanKvPagedSelectionError::GenerationMismatch {
            boolean_generation: boolean_cache.generation(),
            paged_generation: telemetry.generation,
        });
    }
    if boolean_cache.len() != telemetry.mapped_pages {
        return Err(BooleanKvPagedSelectionError::PageCountMismatch {
            boolean_pages: boolean_cache.len(),
            mapped_pages: telemetry.mapped_pages,
        });
    }

    let bytes_per_token = numerical_geometry.bytes_per_token()?;
    let full_numerical_kv_bytes = telemetry
        .live_tokens
        .checked_mul(bytes_per_token)
        .ok_or(BooleanKvPagedSelectionError::ArithmeticOverflow)?;
    let accounting = boolean_cache.accounting()?;
    let matches = boolean_cache.search_hamming(query, max_distance, limit)?;
    let mut selected_pages = map_matches_to_pages(paged_table, telemetry.mapped_pages, &matches)?;
    selected_pages.sort_unstable_by_key(|page| page.logical_page);

    let selected_live_tokens = selected_pages.iter().try_fold(0usize, |total, page| {
        total
            .checked_add(page.live_tokens)
            .ok_or(BooleanKvPagedSelectionError::ArithmeticOverflow)
    })?;
    let selected_numerical_kv_bytes = selected_live_tokens
        .checked_mul(bytes_per_token)
        .ok_or(BooleanKvPagedSelectionError::ArithmeticOverflow)?;
    let avoided_numerical_kv_bytes = full_numerical_kv_bytes
        .checked_sub(selected_numerical_kv_bytes)
        .ok_or(BooleanKvPagedSelectionError::ArithmeticOverflow)?;

    Ok(BooleanIndexedKvSelection {
        generation: telemetry.generation,
        signature_bits: boolean_cache.signature_bits(),
        live_tokens: telemetry.live_tokens,
        mapped_pages: telemetry.mapped_pages,
        boolean_pages_scanned: boolean_cache.len(),
        boolean_key_bytes_read: accounting.key_physical_bytes,
        full_numerical_kv_bytes,
        selected_numerical_kv_bytes,
        avoided_numerical_kv_bytes,
        selected_pages,
    })
}

fn map_matches_to_pages(
    paged_table: &PagedKvTable,
    mapped_pages: usize,
    matches: &[BooleanKvMatch],
) -> Result<Vec<BooleanKvSelectedPage>, BooleanKvPagedSelectionError> {
    let config = paged_table.config();
    let live_tokens = paged_table.len();
    let mut selected = Vec::with_capacity(matches.len());

    for candidate in matches {
        if candidate.logical_page >= mapped_pages {
            return Err(BooleanKvPagedSelectionError::CandidateOutOfRange {
                logical_page: candidate.logical_page,
                mapped_pages,
            });
        }
        let first_token = candidate
            .logical_page
            .checked_mul(config.page_size)
            .ok_or(BooleanKvPagedSelectionError::ArithmeticOverflow)?;
        let address = paged_table.address(first_token).ok_or(
            BooleanKvPagedSelectionError::MissingPagedAddress {
                logical_page: candidate.logical_page,
            },
        )?;
        let remaining = live_tokens
            .checked_sub(first_token)
            .ok_or(BooleanKvPagedSelectionError::ArithmeticOverflow)?;
        let live_tokens_in_page = remaining.min(config.page_size);
        selected.push(BooleanKvSelectedPage {
            logical_page: candidate.logical_page,
            physical_page: address.physical_page,
            live_tokens: live_tokens_in_page,
            hamming_distance: candidate.hamming_distance,
            xnor_matches: candidate.xnor_matches,
        });
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_kv::PagedKvConfig;

    fn signature(byte: u8) -> PackedBooleanSignature {
        let bits = (0..8).map(|bit| byte & (1 << bit) != 0).collect::<Vec<_>>();
        PackedBooleanSignature::from_bools(&bits).unwrap()
    }

    fn table_with_ten_tokens() -> PagedKvTable {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 4,
        })
        .unwrap();
        table.append(10).unwrap();
        table
    }

    fn cache_with_three_pages() -> BooleanKvCache {
        let mut cache = BooleanKvCache::new(8).unwrap();
        // Distances from query 0b0000_0000 are 2, 0, 1. The Boolean search
        // therefore ranks page 1 before page 2 before page 0.
        cache.append(signature(0b0000_0011), None).unwrap();
        cache.append(signature(0), None).unwrap();
        cache.append(signature(0b0000_0001), None).unwrap();
        cache
    }

    #[test]
    fn builds_exact_bikv_handoff_and_restores_logical_page_order() {
        let table = table_with_ten_tokens();
        let cache = cache_with_three_pages();
        let plan = build_boolean_indexed_kv_selection(
            &cache,
            &table,
            &signature(0),
            2,
            None,
            NumericalKvPageGeometry {
                kv_heads: 2,
                head_dim: 8,
                scalar_bytes: 2,
            },
        )
        .unwrap();

        assert_eq!(plan.selected_page_ids(), vec![0, 1, 2]);
        assert_eq!(plan.selected_pages[0].hamming_distance, 2);
        assert_eq!(plan.selected_pages[1].hamming_distance, 0);
        assert_eq!(plan.selected_pages[2].hamming_distance, 1);
        assert_eq!(plan.selected_pages[2].live_tokens, 2);
        assert_eq!(plan.boolean_key_bytes_read, 24);
        assert_eq!(plan.full_numerical_kv_bytes, 640);
        assert_eq!(plan.selected_numerical_kv_bytes, 640);
        assert_eq!(plan.avoided_numerical_kv_bytes, 0);
    }

    #[test]
    fn limit_keeps_hamming_winners_but_handoff_is_sequence_ordered() {
        let table = table_with_ten_tokens();
        let cache = cache_with_three_pages();
        let plan = build_boolean_indexed_kv_selection(
            &cache,
            &table,
            &signature(0),
            2,
            Some(2),
            NumericalKvPageGeometry {
                kv_heads: 2,
                head_dim: 8,
                scalar_bytes: 2,
            },
        )
        .unwrap();

        // Hamming winners are page 1 (distance 0) and page 2 (distance 1).
        assert_eq!(plan.selected_page_ids(), vec![1, 2]);
        assert_eq!(plan.selected_pages[0].live_tokens, 4);
        assert_eq!(plan.selected_pages[1].live_tokens, 2);
        assert_eq!(plan.full_numerical_kv_bytes, 640);
        assert_eq!(plan.selected_numerical_kv_bytes, 384);
        assert_eq!(plan.avoided_numerical_kv_bytes, 256);
        assert_eq!(plan.candidate_density(), 2.0 / 3.0);
        assert_eq!(
            plan.numerical_bytes_avoided_per_boolean_byte(),
            Some(256.0 / 24.0)
        );
    }

    #[test]
    fn rejects_generation_drift_before_routing() {
        let mut table = table_with_ten_tokens();
        let cache = cache_with_three_pages();
        table.reset().unwrap();
        assert_eq!(
            build_boolean_indexed_kv_selection(
                &cache,
                &table,
                &signature(0),
                2,
                None,
                NumericalKvPageGeometry {
                    kv_heads: 2,
                    head_dim: 8,
                    scalar_bytes: 2,
                },
            ),
            Err(BooleanKvPagedSelectionError::GenerationMismatch {
                boolean_generation: 0,
                paged_generation: 1,
            })
        );
    }

    #[test]
    fn rejects_missing_boolean_page_metadata() {
        let table = table_with_ten_tokens();
        let mut cache = BooleanKvCache::new(8).unwrap();
        cache.append(signature(0), None).unwrap();
        cache.append(signature(1), None).unwrap();
        assert_eq!(
            build_boolean_indexed_kv_selection(
                &cache,
                &table,
                &signature(0),
                2,
                None,
                NumericalKvPageGeometry {
                    kv_heads: 2,
                    head_dim: 8,
                    scalar_bytes: 2,
                },
            ),
            Err(BooleanKvPagedSelectionError::PageCountMismatch {
                boolean_pages: 2,
                mapped_pages: 3,
            })
        );
    }

    #[test]
    fn rejects_zero_numerical_geometry() {
        let table = table_with_ten_tokens();
        let cache = cache_with_three_pages();
        assert_eq!(
            build_boolean_indexed_kv_selection(
                &cache,
                &table,
                &signature(0),
                2,
                None,
                NumericalKvPageGeometry {
                    kv_heads: 2,
                    head_dim: 8,
                    scalar_bytes: 0,
                },
            ),
            Err(BooleanKvPagedSelectionError::ZeroNumericalGeometry)
        );
    }
}
