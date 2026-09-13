//! Research-only BIKV qualification accounting and promotion gate.
//!
//! This module records the costs required by the Boolean KV roadmap before a
//! Boolean-indexed numerical KV path can be promoted. It deliberately separates
//! logical payload-byte accounting from claims about physical DRAM traffic.
//! No runtime routing is changed by this module.

use core::fmt;

/// Schema version for serialized/external BIKV qualification records.
pub const BIKV_QUALIFICATION_SCHEMA_VERSION: u16 = 1;

/// Exact logical geometry/accounting inputs for one BIKV qualification sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BikvAccountingInput {
    pub live_tokens: usize,
    pub selected_live_tokens: usize,
    pub mapped_pages: usize,
    pub selected_pages: usize,
    pub page_size: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scalar_bytes: usize,
    /// Boolean-index bytes actually read by the measured selection path.
    pub boolean_index_bytes_read: u64,
}

/// End-to-end latency components for the same sample.
///
/// Uploads/readback may be excluded only when both candidate and baseline use
/// the same already-resident buffers and the benchmark declares that scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BikvLatencyInput {
    pub signature_generation_ns: u64,
    pub boolean_search_ns: u64,
    pub synchronization_ns: u64,
    pub selected_attention_ns: u64,
    pub dense_attention_ns: u64,
}

/// Fully validated, exact-integer BIKV qualification record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BikvQualificationRecord {
    accounting: BikvAccountingInput,
    latency: BikvLatencyInput,
    kv_bytes_per_token: u64,
    dense_numerical_kv_bytes: u64,
    selected_numerical_kv_bytes: u64,
    avoided_numerical_kv_bytes: u64,
    total_bikv_latency_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BikvPromotionDecision {
    Promote,
    FallbackQualityGate,
    FallbackCorrectnessGate,
    FallbackNoLatencyWin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BikvQualificationError {
    ZeroDimension(&'static str),
    SelectedPagesExceedMapped {
        selected_pages: usize,
        mapped_pages: usize,
    },
    SelectedTokensExceedLive {
        selected_live_tokens: usize,
        live_tokens: usize,
    },
    LiveTokensExceedMappedCapacity {
        live_tokens: usize,
        mapped_capacity_tokens: usize,
    },
    SelectedTokensExceedSelectedCapacity {
        selected_live_tokens: usize,
        selected_capacity_tokens: usize,
    },
    EmptySelectionMismatch {
        selected_pages: usize,
        selected_live_tokens: usize,
    },
    MissingBooleanIndexTraffic,
    MissingDenseLatency,
    Overflow,
}

impl fmt::Display for BikvQualificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(name) => write!(f, "BIKV qualification dimension {name} must be non-zero"),
            Self::SelectedPagesExceedMapped { selected_pages, mapped_pages } => write!(
                f,
                "BIKV selected pages ({selected_pages}) exceed mapped pages ({mapped_pages})"
            ),
            Self::SelectedTokensExceedLive { selected_live_tokens, live_tokens } => write!(
                f,
                "BIKV selected live tokens ({selected_live_tokens}) exceed live tokens ({live_tokens})"
            ),
            Self::LiveTokensExceedMappedCapacity { live_tokens, mapped_capacity_tokens } => write!(
                f,
                "BIKV live tokens ({live_tokens}) exceed mapped page capacity ({mapped_capacity_tokens})"
            ),
            Self::SelectedTokensExceedSelectedCapacity { selected_live_tokens, selected_capacity_tokens } => write!(
                f,
                "BIKV selected live tokens ({selected_live_tokens}) exceed selected page capacity ({selected_capacity_tokens})"
            ),
            Self::EmptySelectionMismatch { selected_pages, selected_live_tokens } => write!(
                f,
                "BIKV empty-selection accounting is inconsistent: {selected_pages} pages, {selected_live_tokens} live tokens"
            ),
            Self::MissingBooleanIndexTraffic => write!(
                f,
                "BIKV qualification must include non-zero Boolean index bytes read"
            ),
            Self::MissingDenseLatency => write!(
                f,
                "BIKV qualification requires a non-zero dense baseline latency"
            ),
            Self::Overflow => write!(f, "BIKV qualification accounting overflow"),
        }
    }
}

impl std::error::Error for BikvQualificationError {}

impl BikvQualificationRecord {
    pub fn new(
        accounting: BikvAccountingInput,
        latency: BikvLatencyInput,
    ) -> Result<Self, BikvQualificationError> {
        validate_accounting(accounting)?;
        if latency.dense_attention_ns == 0 {
            return Err(BikvQualificationError::MissingDenseLatency);
        }

        let kv_bytes_per_token = checked_product_u64(&[
            2,
            usize_to_u64(accounting.kv_heads)?,
            usize_to_u64(accounting.head_dim)?,
            usize_to_u64(accounting.scalar_bytes)?,
        ])?;
        let dense_numerical_kv_bytes =
            checked_mul_u64(usize_to_u64(accounting.live_tokens)?, kv_bytes_per_token)?;
        let selected_numerical_kv_bytes = checked_mul_u64(
            usize_to_u64(accounting.selected_live_tokens)?,
            kv_bytes_per_token,
        )?;
        let avoided_numerical_kv_bytes = dense_numerical_kv_bytes
            .checked_sub(selected_numerical_kv_bytes)
            .ok_or(BikvQualificationError::Overflow)?;
        let total_bikv_latency_ns = latency
            .signature_generation_ns
            .checked_add(latency.boolean_search_ns)
            .and_then(|value| value.checked_add(latency.synchronization_ns))
            .and_then(|value| value.checked_add(latency.selected_attention_ns))
            .ok_or(BikvQualificationError::Overflow)?;

        Ok(Self {
            accounting,
            latency,
            kv_bytes_per_token,
            dense_numerical_kv_bytes,
            selected_numerical_kv_bytes,
            avoided_numerical_kv_bytes,
            total_bikv_latency_ns,
        })
    }

    #[must_use]
    pub fn accounting(&self) -> BikvAccountingInput {
        self.accounting
    }

    #[must_use]
    pub fn latency(&self) -> BikvLatencyInput {
        self.latency
    }

    #[must_use]
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
    }

    #[must_use]
    pub fn dense_numerical_kv_bytes(&self) -> u64 {
        self.dense_numerical_kv_bytes
    }

    #[must_use]
    pub fn selected_numerical_kv_bytes(&self) -> u64 {
        self.selected_numerical_kv_bytes
    }

    #[must_use]
    pub fn avoided_numerical_kv_bytes(&self) -> u64 {
        self.avoided_numerical_kv_bytes
    }

    #[must_use]
    pub fn total_bikv_latency_ns(&self) -> u64 {
        self.total_bikv_latency_ns
    }

    #[must_use]
    pub fn selected_page_density(&self) -> f64 {
        self.accounting.selected_pages as f64 / self.accounting.mapped_pages as f64
    }

    #[must_use]
    pub fn selected_token_density(&self) -> f64 {
        self.accounting.selected_live_tokens as f64 / self.accounting.live_tokens as f64
    }

    /// Logical numerical-KV bytes avoided per Boolean-index byte read.
    ///
    /// This is an accounting ratio, not a physical DRAM-traffic measurement.
    #[must_use]
    pub fn avoided_bytes_per_boolean_byte_read(&self) -> f64 {
        self.avoided_numerical_kv_bytes as f64 / self.accounting.boolean_index_bytes_read as f64
    }

    #[must_use]
    pub fn dense_over_bikv_latency_ratio(&self) -> Option<f64> {
        if self.total_bikv_latency_ns == 0 {
            None
        } else {
            Some(self.latency.dense_attention_ns as f64 / self.total_bikv_latency_ns as f64)
        }
    }

    /// Apply the roadmap promotion rule for a latency objective.
    ///
    /// Correctness and quality are supplied by independent experiment gates;
    /// this type never infers either from timing or candidate density.
    #[must_use]
    pub fn promotion_decision(
        &self,
        correctness_gate_passed: bool,
        quality_gate_passed: bool,
    ) -> BikvPromotionDecision {
        if !correctness_gate_passed {
            return BikvPromotionDecision::FallbackCorrectnessGate;
        }
        if !quality_gate_passed {
            return BikvPromotionDecision::FallbackQualityGate;
        }
        if self.total_bikv_latency_ns >= self.latency.dense_attention_ns {
            return BikvPromotionDecision::FallbackNoLatencyWin;
        }
        BikvPromotionDecision::Promote
    }
}

fn validate_accounting(input: BikvAccountingInput) -> Result<(), BikvQualificationError> {
    for (name, value) in [
        ("live_tokens", input.live_tokens),
        ("mapped_pages", input.mapped_pages),
        ("page_size", input.page_size),
        ("kv_heads", input.kv_heads),
        ("head_dim", input.head_dim),
        ("scalar_bytes", input.scalar_bytes),
    ] {
        if value == 0 {
            return Err(BikvQualificationError::ZeroDimension(name));
        }
    }
    if input.selected_pages > input.mapped_pages {
        return Err(BikvQualificationError::SelectedPagesExceedMapped {
            selected_pages: input.selected_pages,
            mapped_pages: input.mapped_pages,
        });
    }
    if input.selected_live_tokens > input.live_tokens {
        return Err(BikvQualificationError::SelectedTokensExceedLive {
            selected_live_tokens: input.selected_live_tokens,
            live_tokens: input.live_tokens,
        });
    }
    let mapped_capacity_tokens = input
        .mapped_pages
        .checked_mul(input.page_size)
        .ok_or(BikvQualificationError::Overflow)?;
    if input.live_tokens > mapped_capacity_tokens {
        return Err(BikvQualificationError::LiveTokensExceedMappedCapacity {
            live_tokens: input.live_tokens,
            mapped_capacity_tokens,
        });
    }
    if (input.selected_pages == 0) != (input.selected_live_tokens == 0) {
        return Err(BikvQualificationError::EmptySelectionMismatch {
            selected_pages: input.selected_pages,
            selected_live_tokens: input.selected_live_tokens,
        });
    }
    let selected_capacity_tokens = input
        .selected_pages
        .checked_mul(input.page_size)
        .ok_or(BikvQualificationError::Overflow)?;
    if input.selected_live_tokens > selected_capacity_tokens {
        return Err(
            BikvQualificationError::SelectedTokensExceedSelectedCapacity {
                selected_live_tokens: input.selected_live_tokens,
                selected_capacity_tokens,
            },
        );
    }
    if input.boolean_index_bytes_read == 0 {
        return Err(BikvQualificationError::MissingBooleanIndexTraffic);
    }
    Ok(())
}

fn usize_to_u64(value: usize) -> Result<u64, BikvQualificationError> {
    u64::try_from(value).map_err(|_| BikvQualificationError::Overflow)
}

fn checked_mul_u64(left: u64, right: u64) -> Result<u64, BikvQualificationError> {
    left.checked_mul(right)
        .ok_or(BikvQualificationError::Overflow)
}

fn checked_product_u64(values: &[u64]) -> Result<u64, BikvQualificationError> {
    values.iter().try_fold(1u64, |acc, &value| {
        acc.checked_mul(value)
            .ok_or(BikvQualificationError::Overflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accounting() -> BikvAccountingInput {
        BikvAccountingInput {
            live_tokens: 230,
            selected_live_tokens: 120,
            mapped_pages: 4,
            selected_pages: 2,
            page_size: 64,
            kv_heads: 4,
            head_dim: 64,
            scalar_bytes: 4,
            boolean_index_bytes_read: 128,
        }
    }

    fn latency() -> BikvLatencyInput {
        BikvLatencyInput {
            signature_generation_ns: 100,
            boolean_search_ns: 200,
            synchronization_ns: 50,
            selected_attention_ns: 600,
            dense_attention_ns: 1_200,
        }
    }

    #[test]
    fn accounts_exact_logical_kv_bytes_and_total_latency() {
        let record = BikvQualificationRecord::new(accounting(), latency()).unwrap();
        assert_eq!(record.kv_bytes_per_token(), 2_048);
        assert_eq!(record.dense_numerical_kv_bytes(), 471_040);
        assert_eq!(record.selected_numerical_kv_bytes(), 245_760);
        assert_eq!(record.avoided_numerical_kv_bytes(), 225_280);
        assert_eq!(record.total_bikv_latency_ns(), 950);
        assert_eq!(record.avoided_bytes_per_boolean_byte_read(), 1_760.0);
        assert!((record.selected_page_density() - 0.5).abs() < f64::EPSILON);
        assert!((record.selected_token_density() - (120.0 / 230.0)).abs() < 1.0e-12);
        assert!(
            (record.dense_over_bikv_latency_ratio().unwrap() - (1_200.0 / 950.0)).abs() < 1.0e-12
        );
    }

    #[test]
    fn promotion_requires_independent_gates_and_end_to_end_latency_win() {
        let record = BikvQualificationRecord::new(accounting(), latency()).unwrap();
        assert_eq!(
            record.promotion_decision(true, true),
            BikvPromotionDecision::Promote
        );
        assert_eq!(
            record.promotion_decision(false, true),
            BikvPromotionDecision::FallbackCorrectnessGate
        );
        assert_eq!(
            record.promotion_decision(true, false),
            BikvPromotionDecision::FallbackQualityGate
        );

        let slower = BikvLatencyInput {
            selected_attention_ns: 1_000,
            ..latency()
        };
        let record = BikvQualificationRecord::new(accounting(), slower).unwrap();
        assert_eq!(
            record.promotion_decision(true, true),
            BikvPromotionDecision::FallbackNoLatencyWin
        );
    }

    #[test]
    fn rejects_inconsistent_selection_and_missing_costs() {
        let mut invalid = accounting();
        invalid.selected_pages = 1;
        assert!(matches!(
            BikvQualificationRecord::new(invalid, latency()),
            Err(BikvQualificationError::SelectedTokensExceedSelectedCapacity { .. })
        ));

        let mut invalid = accounting();
        invalid.selected_pages = 0;
        assert!(matches!(
            BikvQualificationRecord::new(invalid, latency()),
            Err(BikvQualificationError::EmptySelectionMismatch { .. })
        ));

        let mut invalid = accounting();
        invalid.boolean_index_bytes_read = 0;
        assert_eq!(
            BikvQualificationRecord::new(invalid, latency()),
            Err(BikvQualificationError::MissingBooleanIndexTraffic)
        );

        let mut invalid_latency = latency();
        invalid_latency.dense_attention_ns = 0;
        assert_eq!(
            BikvQualificationRecord::new(accounting(), invalid_latency),
            Err(BikvQualificationError::MissingDenseLatency)
        );
    }

    #[test]
    fn permits_consistent_all_reject_accounting_without_claiming_quality() {
        let mut all_reject = accounting();
        all_reject.selected_pages = 0;
        all_reject.selected_live_tokens = 0;
        let record = BikvQualificationRecord::new(all_reject, latency()).unwrap();
        assert_eq!(record.selected_numerical_kv_bytes(), 0);
        assert_eq!(record.avoided_numerical_kv_bytes(), 471_040);
        assert_eq!(
            record.promotion_decision(true, false),
            BikvPromotionDecision::FallbackQualityGate
        );
    }
}
