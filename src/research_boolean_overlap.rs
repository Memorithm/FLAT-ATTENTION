//! M13B.4 first-token readiness correctness contract.
//!
//! This module is deliberately backend-neutral and research-only. It does not
//! schedule GPU work or claim overlap/performance. It qualifies the metadata
//! invariant that must hold before the first decode token may consume Boolean
//! routing metadata for a committed numerical K/V prefix.

use core::fmt;

/// Backend-neutral event/timestamp contract for preregistered M13B.4 evidence.
#[path = "research_boolean_overlap_trace.rs"]
pub mod trace;

pub const M13B4_READINESS_SCHEMA_VERSION: u32 = 1;

/// Snapshot taken at the prefill/decode boundary.
///
/// `numerical_mapped_pages` and `boolean_signature_pages` must both cover the
/// exact committed prefix under the same page geometry and generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillReadinessSnapshot {
    pub numerical_generation: u64,
    pub boolean_generation: u64,
    pub prefix_tokens: usize,
    pub page_size: usize,
    pub numerical_mapped_pages: usize,
    pub boolean_signature_pages: usize,
}

/// A prefix proven decode-visible for M13B.4 first-token consumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FirstDecodeReadyPrefix {
    generation: u64,
    prefix_tokens: usize,
    page_size: usize,
    pages: usize,
}

impl FirstDecodeReadyPrefix {
    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }

    #[must_use]
    pub const fn prefix_tokens(self) -> usize {
        self.prefix_tokens
    }

    #[must_use]
    pub const fn page_size(self) -> usize {
        self.page_size
    }

    #[must_use]
    pub const fn pages(self) -> usize {
        self.pages
    }

    /// Revalidate the first decode invocation against the qualified prefix.
    ///
    /// The preregistered M13B.4 contract requires the first decode token to use
    /// metadata for the complete committed prefill prefix without rebuilding
    /// historical K/KV signatures on that first-token critical path.
    pub fn validate_first_decode(
        self,
        consumption: FirstDecodeConsumption,
    ) -> Result<(), FirstTokenReadinessError> {
        if consumption.generation != self.generation {
            return Err(FirstTokenReadinessError::ConsumerGenerationMismatch {
                ready_generation: self.generation,
                consumer_generation: consumption.generation,
            });
        }
        if consumption.prefix_tokens != self.prefix_tokens {
            return Err(FirstTokenReadinessError::ConsumerPrefixMismatch {
                ready_prefix_tokens: self.prefix_tokens,
                consumer_prefix_tokens: consumption.prefix_tokens,
            });
        }
        if consumption.historical_signature_rebuilds != 0 {
            return Err(
                FirstTokenReadinessError::HistoricalSignatureRebuildOnFirstToken {
                    rebuilds: consumption.historical_signature_rebuilds,
                },
            );
        }
        Ok(())
    }
}

/// Evidence supplied by the first decode invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FirstDecodeConsumption {
    pub generation: u64,
    pub prefix_tokens: usize,
    /// Number of historical K/KV signature rebuilds performed on the first
    /// decode token critical path. M13B.4 requires this to be zero.
    pub historical_signature_rebuilds: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FirstTokenReadinessError {
    ZeroPageSize,
    ArithmeticOverflow,
    GenerationMismatch {
        numerical_generation: u64,
        boolean_generation: u64,
    },
    NumericalPageCoverageMismatch {
        expected_pages: usize,
        actual_pages: usize,
    },
    BooleanPageCoverageMismatch {
        expected_pages: usize,
        actual_pages: usize,
    },
    ConsumerGenerationMismatch {
        ready_generation: u64,
        consumer_generation: u64,
    },
    ConsumerPrefixMismatch {
        ready_prefix_tokens: usize,
        consumer_prefix_tokens: usize,
    },
    HistoricalSignatureRebuildOnFirstToken {
        rebuilds: usize,
    },
}

impl fmt::Display for FirstTokenReadinessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPageSize => write!(f, "M13B.4 readiness page_size must be non-zero"),
            Self::ArithmeticOverflow => write!(f, "M13B.4 readiness page accounting overflowed"),
            Self::GenerationMismatch {
                numerical_generation,
                boolean_generation,
            } => write!(
                f,
                "M13B.4 numerical generation {numerical_generation} does not match Boolean generation {boolean_generation}"
            ),
            Self::NumericalPageCoverageMismatch {
                expected_pages,
                actual_pages,
            } => write!(
                f,
                "M13B.4 numerical prefix requires {expected_pages} mapped pages, got {actual_pages}"
            ),
            Self::BooleanPageCoverageMismatch {
                expected_pages,
                actual_pages,
            } => write!(
                f,
                "M13B.4 Boolean prefix requires {expected_pages} signature pages, got {actual_pages}"
            ),
            Self::ConsumerGenerationMismatch {
                ready_generation,
                consumer_generation,
            } => write!(
                f,
                "M13B.4 first decode consumes generation {consumer_generation}, but ready prefix is generation {ready_generation}"
            ),
            Self::ConsumerPrefixMismatch {
                ready_prefix_tokens,
                consumer_prefix_tokens,
            } => write!(
                f,
                "M13B.4 first decode consumes {consumer_prefix_tokens} prefix tokens, but {ready_prefix_tokens} were qualified"
            ),
            Self::HistoricalSignatureRebuildOnFirstToken { rebuilds } => write!(
                f,
                "M13B.4 first decode rebuilt {rebuilds} historical Boolean signatures on the critical path"
            ),
        }
    }
}

impl std::error::Error for FirstTokenReadinessError {}

/// Qualify a prefill snapshot for first-token Boolean routing.
///
/// The returned value is evidence of metadata consistency only. It does not
/// prove temporal overlap, latency improvement, physical bandwidth reduction,
/// or downstream model quality.
pub fn qualify_first_decode_prefix(
    snapshot: PrefillReadinessSnapshot,
) -> Result<FirstDecodeReadyPrefix, FirstTokenReadinessError> {
    if snapshot.page_size == 0 {
        return Err(FirstTokenReadinessError::ZeroPageSize);
    }
    if snapshot.numerical_generation != snapshot.boolean_generation {
        return Err(FirstTokenReadinessError::GenerationMismatch {
            numerical_generation: snapshot.numerical_generation,
            boolean_generation: snapshot.boolean_generation,
        });
    }

    let expected_pages = if snapshot.prefix_tokens == 0 {
        0
    } else {
        snapshot
            .prefix_tokens
            .checked_add(snapshot.page_size - 1)
            .ok_or(FirstTokenReadinessError::ArithmeticOverflow)?
            / snapshot.page_size
    };

    if snapshot.numerical_mapped_pages != expected_pages {
        return Err(FirstTokenReadinessError::NumericalPageCoverageMismatch {
            expected_pages,
            actual_pages: snapshot.numerical_mapped_pages,
        });
    }
    if snapshot.boolean_signature_pages != expected_pages {
        return Err(FirstTokenReadinessError::BooleanPageCoverageMismatch {
            expected_pages,
            actual_pages: snapshot.boolean_signature_pages,
        });
    }

    Ok(FirstDecodeReadyPrefix {
        generation: snapshot.numerical_generation,
        prefix_tokens: snapshot.prefix_tokens,
        page_size: snapshot.page_size,
        pages: expected_pages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready_snapshot() -> PrefillReadinessSnapshot {
        PrefillReadinessSnapshot {
            numerical_generation: 7,
            boolean_generation: 7,
            prefix_tokens: 10,
            page_size: 4,
            numerical_mapped_pages: 3,
            boolean_signature_pages: 3,
        }
    }

    #[test]
    fn qualifies_complete_partial_tail_prefix() {
        let ready = qualify_first_decode_prefix(ready_snapshot()).unwrap();
        assert_eq!(ready.generation(), 7);
        assert_eq!(ready.prefix_tokens(), 10);
        assert_eq!(ready.page_size(), 4);
        assert_eq!(ready.pages(), 3);
    }

    #[test]
    fn rejects_generation_drift() {
        let mut snapshot = ready_snapshot();
        snapshot.boolean_generation = 8;
        assert_eq!(
            qualify_first_decode_prefix(snapshot),
            Err(FirstTokenReadinessError::GenerationMismatch {
                numerical_generation: 7,
                boolean_generation: 8,
            })
        );
    }

    #[test]
    fn rejects_incomplete_or_extra_page_coverage() {
        let mut numerical = ready_snapshot();
        numerical.numerical_mapped_pages = 2;
        assert_eq!(
            qualify_first_decode_prefix(numerical),
            Err(FirstTokenReadinessError::NumericalPageCoverageMismatch {
                expected_pages: 3,
                actual_pages: 2,
            })
        );

        let mut boolean = ready_snapshot();
        boolean.boolean_signature_pages = 4;
        assert_eq!(
            qualify_first_decode_prefix(boolean),
            Err(FirstTokenReadinessError::BooleanPageCoverageMismatch {
                expected_pages: 3,
                actual_pages: 4,
            })
        );
    }

    #[test]
    fn first_decode_must_consume_exact_ready_prefix_without_rebuild() {
        let ready = qualify_first_decode_prefix(ready_snapshot()).unwrap();
        assert_eq!(
            ready.validate_first_decode(FirstDecodeConsumption {
                generation: 7,
                prefix_tokens: 10,
                historical_signature_rebuilds: 0,
            }),
            Ok(())
        );
        assert_eq!(
            ready.validate_first_decode(FirstDecodeConsumption {
                generation: 7,
                prefix_tokens: 10,
                historical_signature_rebuilds: 1,
            }),
            Err(FirstTokenReadinessError::HistoricalSignatureRebuildOnFirstToken { rebuilds: 1 })
        );
    }

    #[test]
    fn first_decode_rejects_stale_generation_and_prefix_drift() {
        let ready = qualify_first_decode_prefix(ready_snapshot()).unwrap();
        assert_eq!(
            ready.validate_first_decode(FirstDecodeConsumption {
                generation: 8,
                prefix_tokens: 10,
                historical_signature_rebuilds: 0,
            }),
            Err(FirstTokenReadinessError::ConsumerGenerationMismatch {
                ready_generation: 7,
                consumer_generation: 8,
            })
        );
        assert_eq!(
            ready.validate_first_decode(FirstDecodeConsumption {
                generation: 7,
                prefix_tokens: 9,
                historical_signature_rebuilds: 0,
            }),
            Err(FirstTokenReadinessError::ConsumerPrefixMismatch {
                ready_prefix_tokens: 10,
                consumer_prefix_tokens: 9,
            })
        );
    }

    #[test]
    fn zero_prefix_is_well_defined_and_requires_no_pages() {
        let ready = qualify_first_decode_prefix(PrefillReadinessSnapshot {
            numerical_generation: 0,
            boolean_generation: 0,
            prefix_tokens: 0,
            page_size: 16,
            numerical_mapped_pages: 0,
            boolean_signature_pages: 0,
        })
        .unwrap();
        assert_eq!(ready.pages(), 0);
        assert_eq!(ready.prefix_tokens(), 0);
    }

    #[test]
    fn rejects_zero_page_size() {
        let mut snapshot = ready_snapshot();
        snapshot.page_size = 0;
        assert_eq!(
            qualify_first_decode_prefix(snapshot),
            Err(FirstTokenReadinessError::ZeroPageSize)
        );
    }
}
