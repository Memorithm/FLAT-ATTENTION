//! RAE-4: scalar dense-logit attention-mass audit of event-selected pages.
//!
//! This is an independent numerical *oracle*, not a sparse attention kernel.
//! Scores for all causally eligible keys are supplied by the caller; therefore
//! this cannot establish any computation or physical K/V access savings.

use core::fmt;

use crate::softmax_mass::{
    output_error_bound, retained_softmax_mass, RetainedSoftmaxMass, SoftmaxMassError,
};

pub const RAE_NUMERIC_MAX_KEYS: usize = 1_048_576;
pub const RAE_NUMERIC_MAX_PAGES: usize = 65_536;

#[derive(Debug, Clone, PartialEq)]
pub enum RaeNumericMassError {
    InvalidDimensions,
    WrongEligibilityWidth { expected: usize, actual: usize },
    WrongKeyPageWidth { expected: usize, actual: usize },
    WrongPageMaskWordCount { expected: usize, actual: usize },
    NonZeroPagePadding,
    OutOfRangeKeyPage { key: usize, page: usize },
    NonFiniteEligibleScore { key: usize },
    NoEligibleKeys,
    InvalidDroppedMassBudget,
    InvalidErrorBudget,
    Softmax(SoftmaxMassError),
}

impl fmt::Display for RaeNumericMassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RaeNumericMassError {}

impl From<SoftmaxMassError> for RaeNumericMassError {
    fn from(value: SoftmaxMassError) -> Self {
        Self::Softmax(value)
    }
}

/// Mass retained after including all and only causally eligible keys whose
/// original page ID occurs in the candidate bitset.
#[derive(Debug, Clone, PartialEq)]
pub struct RaeNumericMassAudit {
    pub eligible_keys: usize,
    pub selected_keys: usize,
    /// Set bits in the caller's page mask, including pages with no eligible key.
    pub selected_pages: usize,
    pub page_count: usize,
    /// Dense *eligible* softmax is the independent reference.
    pub softmax: RetainedSoftmaxMass,
    pub max_abs_value_bound: f64,
    pub worst_case_output_error_bound: f64,
}

impl RaeNumericMassAudit {
    /// Oracle-side diagnostic only: this is NOT a runtime work-elision permit.
    /// A nonempty candidate and both independently specified quality budgets
    /// are required; true model quality and resource cost still need evidence.
    pub fn within_oracle_budgets(
        &self,
        max_dropped_mass: f64,
        max_output_error: f64,
    ) -> Result<bool, RaeNumericMassError> {
        if !max_dropped_mass.is_finite() || !(0.0..=1.0).contains(&max_dropped_mass) {
            return Err(RaeNumericMassError::InvalidDroppedMassBudget);
        }
        if !max_output_error.is_finite() || max_output_error < 0.0 {
            return Err(RaeNumericMassError::InvalidErrorBudget);
        }
        Ok(self.selected_keys > 0
            && self.softmax.dropped_mass() <= max_dropped_mass
            && self.worst_case_output_error_bound <= max_output_error)
    }
}

/// Evaluate actual softmax probability mass discarded by a page mask.
///
/// `eligible` is the authoritative causal/padding/length eligibility mask.
/// It is applied BEFORE constructing the dense numerical reference: a key
/// masked out by causality never contributes to either partition, even if its
/// original logit is huge. Ineligible scores may be non-finite, because they
/// are not read by the oracle; all eligible scores must be finite.
///
/// `key_pages[i]` is the original logical page ID of key `i`, not a ranking.
/// This oracle expands a selected page into *all* its eligible keys. The
/// f64 value bound must independently bound every relevant V component.
/// No SML (Q/K/V-free) semantic data is silently interpreted as QK scores.
pub fn audit_causal_page_mass(
    scores: &[f64],
    eligible: &[bool],
    key_pages: &[usize],
    page_count: usize,
    selected_page_mask: &[u64],
    max_abs_value: f64,
) -> Result<RaeNumericMassAudit, RaeNumericMassError> {
    if scores.is_empty()
        || scores.len() > RAE_NUMERIC_MAX_KEYS
        || page_count == 0
        || page_count > RAE_NUMERIC_MAX_PAGES
    {
        return Err(RaeNumericMassError::InvalidDimensions);
    }
    if eligible.len() != scores.len() {
        return Err(RaeNumericMassError::WrongEligibilityWidth {
            expected: scores.len(),
            actual: eligible.len(),
        });
    }
    if key_pages.len() != scores.len() {
        return Err(RaeNumericMassError::WrongKeyPageWidth {
            expected: scores.len(),
            actual: key_pages.len(),
        });
    }
    let expected_words = page_count.div_ceil(64);
    if selected_page_mask.len() != expected_words {
        return Err(RaeNumericMassError::WrongPageMaskWordCount {
            expected: expected_words,
            actual: selected_page_mask.len(),
        });
    }
    let tail = page_count % 64;
    if tail != 0 && selected_page_mask[expected_words - 1] >> tail != 0 {
        return Err(RaeNumericMassError::NonZeroPagePadding);
    }

    let mut dense_eligible = Vec::with_capacity(scores.len());
    let mut selected_original_eligible = Vec::new();
    for (key, ((score, &is_eligible), &page)) in
        scores.iter().zip(eligible).zip(key_pages).enumerate()
    {
        if page >= page_count {
            return Err(RaeNumericMassError::OutOfRangeKeyPage { key, page });
        }
        if !is_eligible {
            continue;
        }
        if !score.is_finite() {
            return Err(RaeNumericMassError::NonFiniteEligibleScore { key });
        }
        let eligible_index = dense_eligible.len();
        dense_eligible.push(*score);
        if selected_page_mask[page / 64] & (1_u64 << (page % 64)) != 0 {
            selected_original_eligible.push(eligible_index);
        }
    }
    if dense_eligible.is_empty() {
        return Err(RaeNumericMassError::NoEligibleKeys);
    }
    let softmax = retained_softmax_mass(&dense_eligible, &selected_original_eligible)?;
    let worst_case_output_error_bound = output_error_bound(softmax.retained_mass(), max_abs_value)?;
    Ok(RaeNumericMassAudit {
        eligible_keys: dense_eligible.len(),
        selected_keys: selected_original_eligible.len(),
        selected_pages: selected_page_mask
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum(),
        page_count,
        softmax,
        max_abs_value_bound: max_abs_value,
        worst_case_output_error_bound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misses_numerically_dominant_key_even_when_logical_page_counts_look_sparse() {
        let result =
            audit_causal_page_mass(&[100.0, 0.0], &[true, true], &[0, 1], 2, &[0b10], 2.0).unwrap();
        assert_eq!(result.selected_pages, 1);
        assert_eq!(result.selected_keys, 1);
        assert!(result.softmax.dropped_mass() > 0.999999);
        assert!(result.worst_case_output_error_bound > 3.99999);
        assert!(!result.within_oracle_budgets(0.1, 1.0).unwrap());
    }

    #[test]
    fn half_mass_is_detected_on_equal_scores() {
        let result =
            audit_causal_page_mass(&[0.0, 0.0], &[true, true], &[0, 1], 2, &[0b01], 1.0).unwrap();
        assert!((result.softmax.retained_mass() - 0.5).abs() < 1.0e-15);
        assert!((result.worst_case_output_error_bound - 1.0).abs() < 1.0e-15);
        assert!(result.within_oracle_budgets(0.5, 1.0).unwrap());
        assert!(!result.within_oracle_budgets(0.49, 1.0).unwrap());
    }

    #[test]
    fn causal_exclusion_happens_before_softmax_normalization() {
        let result =
            audit_causal_page_mass(&[1000.0, 0.0], &[false, true], &[0, 1], 2, &[0b10], 3.0)
                .unwrap();
        assert_eq!(result.eligible_keys, 1);
        assert_eq!(result.selected_keys, 1);
        assert_eq!(result.softmax.retained_mass(), 1.0);
        assert_eq!(result.worst_case_output_error_bound, 0.0);
    }

    #[test]
    fn dense_selection_and_aliased_pages_retain_unit_mass() {
        let result = audit_causal_page_mass(
            &[10.0, -4.0, 1.0],
            &[true, true, true],
            &[0, 1, 1],
            2,
            &[0b11],
            1.5,
        )
        .unwrap();
        assert_eq!(result.selected_pages, 2);
        assert_eq!(result.selected_keys, 3);
        assert_eq!(result.softmax.dropped_mass(), 0.0);
        assert!(result.within_oracle_budgets(0.0, 0.0).unwrap());
    }

    #[test]
    fn zero_candidate_keys_reports_full_drop_and_fails_budget_admission() {
        let result =
            audit_causal_page_mass(&[0.0, 0.0], &[true, true], &[0, 0], 2, &[0b10], 1.0).unwrap();
        assert_eq!(result.selected_pages, 1);
        assert_eq!(result.selected_keys, 0);
        assert_eq!(result.softmax.dropped_mass(), 1.0);
        assert!(!result.within_oracle_budgets(1.0, 2.0).unwrap());
    }

    #[test]
    fn malformed_mask_geometry_and_nonfinite_eligible_score_fail_closed() {
        assert_eq!(
            audit_causal_page_mass(&[0.0], &[true], &[0], 1, &[2], 1.0),
            Err(RaeNumericMassError::NonZeroPagePadding)
        );
        assert_eq!(
            audit_causal_page_mass(&[0.0], &[true], &[1], 1, &[1], 1.0),
            Err(RaeNumericMassError::OutOfRangeKeyPage { key: 0, page: 1 })
        );
        assert_eq!(
            audit_causal_page_mass(&[0.0], &[false], &[0], 1, &[1], 1.0),
            Err(RaeNumericMassError::NoEligibleKeys)
        );
        assert_eq!(
            audit_causal_page_mass(&[f64::NAN], &[true], &[0], 1, &[1], 1.0),
            Err(RaeNumericMassError::NonFiniteEligibleScore { key: 0 })
        );
        assert_eq!(
            audit_causal_page_mass(&[f64::NAN, 0.0], &[false, true], &[0, 1], 2, &[0b10], 1.0)
                .unwrap()
                .softmax
                .retained_mass(),
            1.0
        );
    }
}
