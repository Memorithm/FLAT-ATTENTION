use core::fmt;
use std::fmt::Write as _;

use crate::api::boolean_kv::{
    BooleanKvCache, BooleanKvError, BooleanKvMatch, PackedBooleanSignature,
};
use crate::paged_kv::{PagedKvError, PagedKvTable};

/// Canonical schema emitted when retaining a Boolean KV page-selection decision.
///
/// This is research evidence only. It records the logical selection and exact
/// accounting already computed by the router; it does not claim physical DRAM
/// traffic, latency, model quality, or a performance improvement.
pub const BOOLEAN_KV_SELECTION_EVIDENCE_SCHEMA: &str = "flat.boolean-kv-selection.v1";

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

    /// Validate the retained selection evidence independently from construction.
    ///
    /// `BooleanIndexedKvSelection` is intentionally a transparent research
    /// record. Callers can therefore deserialize or reconstruct one outside the
    /// canonical builder. Evidence export revalidates the invariants needed to
    /// prevent a forged/mutated aggregate from being retained as if it came from
    /// the authoritative page router.
    pub fn validate_evidence(&self) -> Result<(), BooleanKvSelectionEvidenceError> {
        if self.signature_bits == 0 {
            return Err(BooleanKvSelectionEvidenceError::ZeroSignatureBits);
        }
        if self.boolean_pages_scanned != self.mapped_pages {
            return Err(BooleanKvSelectionEvidenceError::ScannedPageCountMismatch {
                scanned_pages: self.boolean_pages_scanned,
                mapped_pages: self.mapped_pages,
            });
        }
        if self.selected_pages.len() > self.mapped_pages {
            return Err(BooleanKvSelectionEvidenceError::TooManySelectedPages {
                selected_pages: self.selected_pages.len(),
                mapped_pages: self.mapped_pages,
            });
        }
        if self
            .selected_numerical_kv_bytes
            .checked_add(self.avoided_numerical_kv_bytes)
            != Some(self.full_numerical_kv_bytes)
        {
            return Err(BooleanKvSelectionEvidenceError::NumericalByteAccountingMismatch);
        }
        if (self.live_tokens == 0) != (self.mapped_pages == 0) {
            return Err(
                BooleanKvSelectionEvidenceError::LivePageCardinalityMismatch {
                    live_tokens: self.live_tokens,
                    mapped_pages: self.mapped_pages,
                },
            );
        }
        if self.live_tokens == 0 {
            if self.full_numerical_kv_bytes != 0
                || self.selected_numerical_kv_bytes != 0
                || self.avoided_numerical_kv_bytes != 0
                || !self.selected_pages.is_empty()
            {
                return Err(BooleanKvSelectionEvidenceError::EmptySelectionAccountingMismatch);
            }
        } else {
            if self.full_numerical_kv_bytes % self.live_tokens != 0 {
                return Err(BooleanKvSelectionEvidenceError::NumericalByteAccountingMismatch);
            }
            let bytes_per_token = self.full_numerical_kv_bytes / self.live_tokens;
            let selected_live_tokens =
                self.selected_pages.iter().try_fold(0usize, |sum, page| {
                    sum.checked_add(page.live_tokens)
                        .ok_or(BooleanKvSelectionEvidenceError::NumericalByteAccountingMismatch)
                })?;
            if selected_live_tokens > self.live_tokens
                || selected_live_tokens.checked_mul(bytes_per_token)
                    != Some(self.selected_numerical_kv_bytes)
            {
                return Err(
                    BooleanKvSelectionEvidenceError::SelectedByteAccountingMismatch {
                        selected_live_tokens,
                        bytes_per_token,
                        selected_numerical_kv_bytes: self.selected_numerical_kv_bytes,
                    },
                );
            }
        }
        if self.mapped_pages > 0 && self.boolean_key_bytes_read == 0 {
            return Err(BooleanKvSelectionEvidenceError::MissingBooleanBytesRead);
        }

        let mut previous_logical = None;
        let mut physical_pages = std::collections::BTreeSet::new();
        for page in &self.selected_pages {
            if page.logical_page >= self.mapped_pages {
                return Err(BooleanKvSelectionEvidenceError::LogicalPageOutOfRange {
                    logical_page: page.logical_page,
                    mapped_pages: self.mapped_pages,
                });
            }
            if previous_logical.is_some_and(|previous| page.logical_page <= previous) {
                return Err(BooleanKvSelectionEvidenceError::LogicalPagesNotStrictlyOrdered);
            }
            if page.live_tokens == 0 {
                return Err(BooleanKvSelectionEvidenceError::ZeroLiveTokens {
                    logical_page: page.logical_page,
                });
            }
            if page.hamming_distance.checked_add(page.xnor_matches) != Some(self.signature_bits) {
                return Err(
                    BooleanKvSelectionEvidenceError::SignatureAccountingMismatch {
                        logical_page: page.logical_page,
                    },
                );
            }
            if !physical_pages.insert(page.physical_page) {
                return Err(BooleanKvSelectionEvidenceError::DuplicatePhysicalPage {
                    physical_page: page.physical_page,
                });
            }
            previous_logical = Some(page.logical_page);
        }
        Ok(())
    }

    /// Deterministic, dependency-free JSON for cross-project evidence retention.
    ///
    /// The checksum detects accidental mutation and gives KVLab a stable content
    /// identity to retain. It is not a cryptographic authenticity primitive.
    pub fn canonical_evidence_json(&self) -> Result<String, BooleanKvSelectionEvidenceError> {
        self.validate_evidence()?;
        let mut payload = String::with_capacity(512 + self.selected_pages.len() * 128);
        write!(
            payload,
            "{{\"schema\":\"{}\",\"generation\":{},\"signature_bits\":{},\"live_tokens\":{},\"mapped_pages\":{},\"boolean_pages_scanned\":{},\"boolean_key_bytes_read\":{},\"full_numerical_kv_bytes\":{},\"selected_numerical_kv_bytes\":{},\"avoided_numerical_kv_bytes\":{},\"selected_pages\":[",
            BOOLEAN_KV_SELECTION_EVIDENCE_SCHEMA,
            self.generation,
            self.signature_bits,
            self.live_tokens,
            self.mapped_pages,
            self.boolean_pages_scanned,
            self.boolean_key_bytes_read,
            self.full_numerical_kv_bytes,
            self.selected_numerical_kv_bytes,
            self.avoided_numerical_kv_bytes,
        )
        .expect("writing to String cannot fail");
        for (index, page) in self.selected_pages.iter().enumerate() {
            if index != 0 {
                payload.push(',');
            }
            write!(
                payload,
                "{{\"logical_page\":{},\"physical_page\":{},\"live_tokens\":{},\"hamming_distance\":{},\"xnor_matches\":{}}}",
                page.logical_page,
                page.physical_page,
                page.live_tokens,
                page.hamming_distance,
                page.xnor_matches,
            )
            .expect("writing to String cannot fail");
        }
        payload.push(']');
        let checksum = fnv1a64(payload.as_bytes());
        write!(
            payload,
            ",\"evidence_checksum\":{{\"algorithm\":\"fnv1a64\",\"value\":\"{checksum:016x}\"}}}}"
        )
        .expect("writing to String cannot fail");
        Ok(payload)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BooleanKvSelectionEvidenceError {
    ZeroSignatureBits,
    ScannedPageCountMismatch {
        scanned_pages: usize,
        mapped_pages: usize,
    },
    TooManySelectedPages {
        selected_pages: usize,
        mapped_pages: usize,
    },
    NumericalByteAccountingMismatch,
    LivePageCardinalityMismatch {
        live_tokens: usize,
        mapped_pages: usize,
    },
    EmptySelectionAccountingMismatch,
    SelectedByteAccountingMismatch {
        selected_live_tokens: usize,
        bytes_per_token: usize,
        selected_numerical_kv_bytes: usize,
    },
    MissingBooleanBytesRead,
    LogicalPageOutOfRange {
        logical_page: usize,
        mapped_pages: usize,
    },
    LogicalPagesNotStrictlyOrdered,
    ZeroLiveTokens {
        logical_page: usize,
    },
    SignatureAccountingMismatch {
        logical_page: usize,
    },
    DuplicatePhysicalPage {
        physical_page: usize,
    },
}

impl fmt::Display for BooleanKvSelectionEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSignatureBits => write!(f, "Boolean KV selection evidence requires non-zero signature_bits"),
            Self::ScannedPageCountMismatch { scanned_pages, mapped_pages } => write!(f, "Boolean KV selection scanned {scanned_pages} pages for {mapped_pages} mapped pages"),
            Self::TooManySelectedPages { selected_pages, mapped_pages } => write!(f, "Boolean KV selection retains {selected_pages} pages from only {mapped_pages} mapped pages"),
            Self::NumericalByteAccountingMismatch => write!(f, "Boolean KV selection numerical byte accounting is inconsistent"),
            Self::LivePageCardinalityMismatch { live_tokens, mapped_pages } => write!(f, "Boolean KV selection live-token/page cardinality is inconsistent: {live_tokens} live tokens across {mapped_pages} mapped pages"),
            Self::EmptySelectionAccountingMismatch => write!(f, "empty Boolean KV selection must retain zero numerical bytes and no selected pages"),
            Self::SelectedByteAccountingMismatch { selected_live_tokens, bytes_per_token, selected_numerical_kv_bytes } => write!(f, "Boolean KV selection selected-byte accounting is inconsistent: {selected_live_tokens} live tokens at {bytes_per_token} bytes/token but {selected_numerical_kv_bytes} selected bytes recorded"),
            Self::MissingBooleanBytesRead => write!(f, "Boolean KV selection with mapped pages must retain non-zero Boolean key bytes read"),
            Self::LogicalPageOutOfRange { logical_page, mapped_pages } => write!(f, "Boolean KV selection logical page {logical_page} is outside {mapped_pages} mapped pages"),
            Self::LogicalPagesNotStrictlyOrdered => write!(f, "Boolean KV selection logical pages must be strictly increasing"),
            Self::ZeroLiveTokens { logical_page } => write!(f, "Boolean KV selection page {logical_page} cannot retain zero live tokens"),
            Self::SignatureAccountingMismatch { logical_page } => write!(f, "Boolean KV selection page {logical_page} has inconsistent Hamming/XNOR accounting"),
            Self::DuplicatePhysicalPage { physical_page } => write!(f, "Boolean KV selection repeats physical page {physical_page}"),
        }
    }
}

impl std::error::Error for BooleanKvSelectionEvidenceError {}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
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
    #[test]
    fn canonical_selection_evidence_is_deterministic_and_retains_page_identity() {
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

        let first = plan.canonical_evidence_json().unwrap();
        let second = plan.canonical_evidence_json().unwrap();
        assert_eq!(first, second);
        assert!(first.contains("\"schema\":\"flat.boolean-kv-selection.v1\""));
        assert!(first.contains("\"logical_page\":1"));
        assert!(first.contains("\"logical_page\":2"));
        assert!(first.contains("\"evidence_checksum\":{\"algorithm\":\"fnv1a64\""));
    }

    #[test]
    fn selection_evidence_rejects_mutated_page_order_and_signature_accounting() {
        let table = table_with_ten_tokens();
        let cache = cache_with_three_pages();
        let mut plan = build_boolean_indexed_kv_selection(
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

        plan.selected_pages.swap(0, 1);
        assert_eq!(
            plan.validate_evidence(),
            Err(BooleanKvSelectionEvidenceError::LogicalPagesNotStrictlyOrdered)
        );

        plan.selected_pages
            .sort_unstable_by_key(|page| page.logical_page);
        plan.selected_pages[0].xnor_matches = 0;
        assert_eq!(
            plan.validate_evidence(),
            Err(BooleanKvSelectionEvidenceError::SignatureAccountingMismatch { logical_page: 0 })
        );
    }
    #[test]
    fn selection_evidence_rejects_selected_byte_drift_even_when_total_is_conserved() {
        let table = table_with_ten_tokens();
        let cache = cache_with_three_pages();
        let mut plan = build_boolean_indexed_kv_selection(
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

        plan.selected_numerical_kv_bytes = 320;
        plan.avoided_numerical_kv_bytes = 320;
        assert_eq!(
            plan.validate_evidence(),
            Err(
                BooleanKvSelectionEvidenceError::SelectedByteAccountingMismatch {
                    selected_live_tokens: 6,
                    bytes_per_token: 64,
                    selected_numerical_kv_bytes: 320,
                }
            )
        );
    }
}
