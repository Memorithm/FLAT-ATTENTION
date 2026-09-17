//! BKV-5 cooperative Boolean-selection handoff contract.
//!
//! This research-only module defines the bounded handoff between an already
//! qualified first-token prefix and one Boolean-indexed numerical-KV selection.
//! It deliberately does not implement a scheduler, worker queue, GPU submission,
//! lease, retry loop, or artifact store.  Its only job is to prevent stale or
//! mismatched selection metadata from crossing the CPU/GPU cooperation boundary
//! and to make backpressure fallback explicit.

use core::fmt;

use crate::api::boolean_kv_paged_selection::{
    BooleanIndexedKvSelection, BooleanKvSelectionEvidenceError,
};
use crate::api::research_boolean_overlap::FirstDecodeReadyPrefix;

/// Version of the bounded BKV-5 handoff semantics.
pub const BKV5_COOPERATIVE_HANDOFF_SCHEMA_VERSION: u32 = 1;

/// Outcome of admitting one already-computed Boolean selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CooperativeSelectionDisposition {
    /// The selection is generation/prefix aligned and bounded capacity exists.
    ConsumeBooleanSelection,
    /// The selection is valid, but the bounded cooperation slot is saturated.
    /// The caller must use its existing dense numerical fallback rather than
    /// queueing unbounded Boolean work or consuming stale metadata later.
    DenseFallbackBackpressure,
}

/// Immutable admission evidence for one cooperative handoff decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CooperativeSelectionTicket {
    generation: u64,
    prefix_tokens: usize,
    mapped_pages: usize,
    selected_pages: usize,
    pending_before: usize,
    max_pending: usize,
    disposition: CooperativeSelectionDisposition,
}

impl CooperativeSelectionTicket {
    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }

    #[must_use]
    pub const fn prefix_tokens(self) -> usize {
        self.prefix_tokens
    }

    #[must_use]
    pub const fn mapped_pages(self) -> usize {
        self.mapped_pages
    }

    #[must_use]
    pub const fn selected_pages(self) -> usize {
        self.selected_pages
    }

    #[must_use]
    pub const fn pending_before(self) -> usize {
        self.pending_before
    }

    #[must_use]
    pub const fn max_pending(self) -> usize {
        self.max_pending
    }

    #[must_use]
    pub const fn disposition(self) -> CooperativeSelectionDisposition {
        self.disposition
    }
}

/// Fail-closed errors before any cooperative consumer may use a selection.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CooperativeSelectionHandoffError {
    InvalidSelection(BooleanKvSelectionEvidenceError),
    ZeroPendingCapacity,
    PendingCountExceedsCapacity {
        pending: usize,
        max_pending: usize,
    },
    GenerationMismatch {
        ready_generation: u64,
        selection_generation: u64,
    },
    PrefixTokenMismatch {
        ready_prefix_tokens: usize,
        selection_live_tokens: usize,
    },
    PageCoverageMismatch {
        ready_pages: usize,
        selection_mapped_pages: usize,
    },
}

impl fmt::Display for CooperativeSelectionHandoffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSelection(error) => write!(f, "invalid Boolean selection: {error}"),
            Self::ZeroPendingCapacity => {
                write!(f, "BKV-5 cooperative handoff max_pending must be non-zero")
            }
            Self::PendingCountExceedsCapacity {
                pending,
                max_pending,
            } => write!(
                f,
                "BKV-5 pending count {pending} exceeds declared capacity {max_pending}"
            ),
            Self::GenerationMismatch {
                ready_generation,
                selection_generation,
            } => write!(
                f,
                "BKV-5 selection generation {selection_generation} does not match ready generation {ready_generation}"
            ),
            Self::PrefixTokenMismatch {
                ready_prefix_tokens,
                selection_live_tokens,
            } => write!(
                f,
                "BKV-5 selection covers {selection_live_tokens} live tokens, but ready prefix contains {ready_prefix_tokens}"
            ),
            Self::PageCoverageMismatch {
                ready_pages,
                selection_mapped_pages,
            } => write!(
                f,
                "BKV-5 selection maps {selection_mapped_pages} pages, but ready prefix covers {ready_pages}"
            ),
        }
    }
}

impl std::error::Error for CooperativeSelectionHandoffError {}

/// Admit one Boolean selection at the CPU/GPU cooperation boundary.
///
/// `pending_before` and `max_pending` are observations supplied by the caller's
/// existing execution mechanism. This function does not own or mutate a queue.
/// Saturation produces an explicit dense-fallback disposition. A count greater
/// than the declared bound is rejected because it indicates broken accounting,
/// not ordinary backpressure.
pub fn qualify_cooperative_selection(
    ready: FirstDecodeReadyPrefix,
    selection: &BooleanIndexedKvSelection,
    pending_before: usize,
    max_pending: usize,
) -> Result<CooperativeSelectionTicket, CooperativeSelectionHandoffError> {
    selection
        .validate_evidence()
        .map_err(CooperativeSelectionHandoffError::InvalidSelection)?;

    if max_pending == 0 {
        return Err(CooperativeSelectionHandoffError::ZeroPendingCapacity);
    }
    if pending_before > max_pending {
        return Err(
            CooperativeSelectionHandoffError::PendingCountExceedsCapacity {
                pending: pending_before,
                max_pending,
            },
        );
    }
    if selection.generation != ready.generation() {
        return Err(CooperativeSelectionHandoffError::GenerationMismatch {
            ready_generation: ready.generation(),
            selection_generation: selection.generation,
        });
    }
    if selection.live_tokens != ready.prefix_tokens() {
        return Err(CooperativeSelectionHandoffError::PrefixTokenMismatch {
            ready_prefix_tokens: ready.prefix_tokens(),
            selection_live_tokens: selection.live_tokens,
        });
    }
    if selection.mapped_pages != ready.pages() {
        return Err(CooperativeSelectionHandoffError::PageCoverageMismatch {
            ready_pages: ready.pages(),
            selection_mapped_pages: selection.mapped_pages,
        });
    }

    let disposition = if pending_before == max_pending {
        CooperativeSelectionDisposition::DenseFallbackBackpressure
    } else {
        CooperativeSelectionDisposition::ConsumeBooleanSelection
    };

    Ok(CooperativeSelectionTicket {
        generation: selection.generation,
        prefix_tokens: selection.live_tokens,
        mapped_pages: selection.mapped_pages,
        selected_pages: selection.selected_page_count(),
        pending_before,
        max_pending,
        disposition,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::boolean_kv_paged_selection::BooleanKvSelectedPage;
    use crate::api::research_boolean_overlap::{
        qualify_first_decode_prefix, PrefillReadinessSnapshot,
    };

    fn ready() -> FirstDecodeReadyPrefix {
        qualify_first_decode_prefix(PrefillReadinessSnapshot {
            numerical_generation: 7,
            boolean_generation: 7,
            prefix_tokens: 10,
            page_size: 4,
            numerical_mapped_pages: 3,
            boolean_signature_pages: 3,
        })
        .unwrap()
    }

    fn selection() -> BooleanIndexedKvSelection {
        BooleanIndexedKvSelection {
            generation: 7,
            signature_bits: 64,
            max_distance: 3,
            live_tokens: 10,
            mapped_pages: 3,
            boolean_pages_scanned: 3,
            boolean_key_bytes_read: 24,
            numerical_kv_bytes_per_token: 16,
            full_numerical_kv_bytes: 160,
            selected_numerical_kv_bytes: 128,
            avoided_numerical_kv_bytes: 32,
            selected_pages: vec![
                BooleanKvSelectedPage {
                    logical_page: 0,
                    physical_page: 10,
                    live_tokens: 4,
                    hamming_distance: 1,
                    xnor_matches: 63,
                },
                BooleanKvSelectedPage {
                    logical_page: 1,
                    physical_page: 11,
                    live_tokens: 4,
                    hamming_distance: 2,
                    xnor_matches: 62,
                },
            ],
        }
    }

    #[test]
    fn admits_aligned_selection_when_capacity_exists() {
        let ticket = qualify_cooperative_selection(ready(), &selection(), 1, 2).unwrap();
        assert_eq!(ticket.generation(), 7);
        assert_eq!(ticket.prefix_tokens(), 10);
        assert_eq!(ticket.mapped_pages(), 3);
        assert_eq!(ticket.selected_pages(), 2);
        assert_eq!(ticket.pending_before(), 1);
        assert_eq!(ticket.max_pending(), 2);
        assert_eq!(
            ticket.disposition(),
            CooperativeSelectionDisposition::ConsumeBooleanSelection
        );
    }

    #[test]
    fn saturation_is_explicit_dense_fallback_not_hidden_queueing() {
        let ticket = qualify_cooperative_selection(ready(), &selection(), 2, 2).unwrap();
        assert_eq!(
            ticket.disposition(),
            CooperativeSelectionDisposition::DenseFallbackBackpressure
        );
    }

    #[test]
    fn rejects_stale_generation_before_backpressure_decision() {
        let mut stale = selection();
        stale.generation = 6;
        assert_eq!(
            qualify_cooperative_selection(ready(), &stale, 2, 2),
            Err(CooperativeSelectionHandoffError::GenerationMismatch {
                ready_generation: 7,
                selection_generation: 6,
            })
        );
    }

    #[test]
    fn rejects_prefix_or_page_coverage_drift() {
        let mut prefix_drift = selection();
        prefix_drift.live_tokens = 9;
        prefix_drift.full_numerical_kv_bytes = 144;
        prefix_drift.avoided_numerical_kv_bytes = 16;
        assert_eq!(
            qualify_cooperative_selection(ready(), &prefix_drift, 0, 1),
            Err(CooperativeSelectionHandoffError::PrefixTokenMismatch {
                ready_prefix_tokens: 10,
                selection_live_tokens: 9,
            })
        );

        let mut page_drift = selection();
        page_drift.mapped_pages = 4;
        page_drift.boolean_pages_scanned = 4;
        page_drift.boolean_key_bytes_read = 32;
        assert_eq!(
            qualify_cooperative_selection(ready(), &page_drift, 0, 1),
            Err(CooperativeSelectionHandoffError::PageCoverageMismatch {
                ready_pages: 3,
                selection_mapped_pages: 4,
            })
        );
    }

    #[test]
    fn rejects_invalid_capacity_accounting() {
        assert_eq!(
            qualify_cooperative_selection(ready(), &selection(), 0, 0),
            Err(CooperativeSelectionHandoffError::ZeroPendingCapacity)
        );
        assert_eq!(
            qualify_cooperative_selection(ready(), &selection(), 3, 2),
            Err(
                CooperativeSelectionHandoffError::PendingCountExceedsCapacity {
                    pending: 3,
                    max_pending: 2,
                }
            )
        );
    }
}
