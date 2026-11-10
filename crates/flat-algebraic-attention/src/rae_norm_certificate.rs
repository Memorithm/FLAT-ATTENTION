//! RAE-6: conservative certificate for omitted attention pages using norms.
//!
//! This deliberately avoids reading OMITTED K vectors or computing their
//! Q·K logits in the certificate itself. The caller must provide independently
//! validated, version-pinned per-page maximum K norms and exact causal
//! eligible-key counts. This does NOT establish net savings: refreshing that
//! metadata, reading selected K/V and computing Q norms all have costs.
//!
//! Research-only bound. A certificate is NOT an admission to modify the
//! default FLAT attention kernel, causal semantics, KV-cache or model output.

use core::fmt;

pub const RAE_CERT_MAX_PAGES: usize = 65_536;
pub const RAE_CERT_MAX_ELIGIBLE_KEYS: usize = 1_048_576;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaeCausalPageBounds {
    /// Number of currently causally eligible keys in this ORIGINAL page.
    /// The source must supply counts valid for the CURRENT request and mask.
    pub eligible_keys: usize,
    /// Independently certified max ||K_i||_2 for all eligible K in the
    /// page, for the exact pinned K generation (not a learned estimate).
    pub max_key_l2_norm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaeBoundVersions {
    /// Identity of keys used for the current attention query.
    pub query_key_generation: u64,
    /// Generation of the precomputed per-page max-key-norm metadata.
    pub norms_key_generation: u64,
    /// Generation of value vectors used by the output-error bound.
    pub query_value_generation: u64,
    /// Generation of the asserted max absolute V-component bound.
    pub value_bound_generation: u64,
    /// Current query-specific causal/padding/length eligibility generation.
    pub query_mask_generation: u64,
    /// Generation under which per-page eligible key counts were computed.
    pub counts_mask_generation: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RaeCertError {
    InvalidDimensions,
    InvalidPageMaskWidth { expected: usize, actual: usize },
    NonZeroPageMaskPadding,
    InvalidQueryNorm,
    InvalidValueBound,
    StaleKeyNorms,
    StaleValueBound,
    StaleCausalCounts,
    InvalidPageNorm { page: usize },
    TooManyEligibleKeys,
    WrongSelectedLogitCount { expected: usize, actual: usize },
    NoSelectedEligibleKeys,
    NonFiniteSelectedLogit { index: usize },
    NonFiniteBoundTerm { page: usize },
    NumericalOverflow,
    InvalidDroppedMassBudget,
    InvalidOutputErrorBudget,
}

impl fmt::Display for RaeCertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RaeCertError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaeNormMassCertificate {
    pub page_count: usize,
    pub selected_pages: usize,
    pub selected_eligible_keys: usize,
    pub omitted_eligible_keys: usize,
    /// Conservative upper bound for the dense causal softmax mass on
    /// omitted pages. Never infer that the bound is a measured dropout.
    pub maximum_dropped_mass: f64,
    /// Conservative absolute componentwise difference bound between full
    /// attention and renormalized selected-key attention, when all true
    /// V components are bounded by `value_abs_bound`.
    pub maximum_output_error: f64,
}

impl RaeNormMassCertificate {
    /// A research-only numerical budget test. Passing is not permission to
    /// skip attention in a real model without additional quality and runtime
    /// safety/fallback qualification.
    pub fn within_budgets(
        self,
        maximum_dropped_mass: f64,
        maximum_output_error: f64,
    ) -> Result<bool, RaeCertError> {
        if !maximum_dropped_mass.is_finite() || !(0.0..=1.0).contains(&maximum_dropped_mass) {
            return Err(RaeCertError::InvalidDroppedMassBudget);
        }
        if !maximum_output_error.is_finite() || maximum_output_error < 0.0 {
            return Err(RaeCertError::InvalidOutputErrorBudget);
        }
        Ok(self.maximum_dropped_mass <= maximum_dropped_mass
            && self.maximum_output_error <= maximum_output_error)
    }
}

/// Upper bound on dropped dense causal attention probability without reading
/// the logits of omitted keys.
///
/// Assume reference score s_i = (q · k_i) * scale, scale >= 0, and
/// scaled_query_l2_norm = ||q||_2 * scale (with no additive per-key bias).
/// By Cauchy-Schwarz, each omitted logit in page p obeys
/// s_i <= scaled_query_l2_norm * max_key_l2_norm[p].
///
/// Let S = sum(exp(s_i)) over all currently eligible SELECTED keys, with
/// exact scores provided by the caller, and
/// U = sum_p eligible_key_count[p] * exp(q_scaled_norm * max_K_norm[p])
/// over omitted pages. Then true dropped mass <= U/(S + U).
/// Both sums are computed in log-space to avoid exp overflow.
///
/// Preconditions which this function CANNOT establish independently:
/// - page-count/mask and selected logits describe the same current query;
/// - page max norms are certified upper bounds, including all updated K;
/// - eligible-key counts reflect current causal/padding/length constraints;
/// - no positional/additive/masking score term violates the upper bound;
/// - V bound is current for the exact referenced generation.
/// - generation IDs are minted by an authoritative source, not caller guesses.
///
/// A mismatch is rejected; matching integers alone are not proof of provenance.
///
/// This certificate does not read OMITTED keys and does not reconstruct dense
/// QK; selected key dot-products are still required. None of the work for
/// preparing/maintaining metadata is included in this host-only cost.
pub fn certify_omitted_page_mass(
    selected_pages: &[u64],
    pages: &[RaeCausalPageBounds],
    selected_eligible_logits: &[f64],
    scaled_query_l2_norm: f64,
    value_abs_bound: f64,
    versions: RaeBoundVersions,
) -> Result<RaeNormMassCertificate, RaeCertError> {
    if pages.is_empty() || pages.len() > RAE_CERT_MAX_PAGES {
        return Err(RaeCertError::InvalidDimensions);
    }
    let words = pages.len().div_ceil(64);
    if selected_pages.len() != words {
        return Err(RaeCertError::InvalidPageMaskWidth {
            expected: words,
            actual: selected_pages.len(),
        });
    }
    let tail = pages.len() % 64;
    if tail != 0 && selected_pages[words - 1] >> tail != 0 {
        return Err(RaeCertError::NonZeroPageMaskPadding);
    }
    if !scaled_query_l2_norm.is_finite() || scaled_query_l2_norm < 0.0 {
        return Err(RaeCertError::InvalidQueryNorm);
    }
    if !value_abs_bound.is_finite() || value_abs_bound < 0.0 {
        return Err(RaeCertError::InvalidValueBound);
    }
    if versions.query_key_generation != versions.norms_key_generation {
        return Err(RaeCertError::StaleKeyNorms);
    }
    if versions.query_value_generation != versions.value_bound_generation {
        return Err(RaeCertError::StaleValueBound);
    }
    if versions.query_mask_generation != versions.counts_mask_generation {
        return Err(RaeCertError::StaleCausalCounts);
    }

    let mut selected_page_count = 0;
    let mut selected_key_count = 0_usize;
    let mut omitted_key_count = 0_usize;
    let mut log_omitted_mass_bound = f64::NEG_INFINITY;

    for (page, info) in pages.iter().enumerate() {
        if !info.max_key_l2_norm.is_finite() || info.max_key_l2_norm < 0.0 {
            return Err(RaeCertError::InvalidPageNorm { page });
        }
        let included = selected_pages[page / 64] & (1_u64 << (page % 64)) != 0;
        if included {
            selected_page_count += 1;
            selected_key_count = selected_key_count
                .checked_add(info.eligible_keys)
                .ok_or(RaeCertError::TooManyEligibleKeys)?;
        } else {
            omitted_key_count = omitted_key_count
                .checked_add(info.eligible_keys)
                .ok_or(RaeCertError::TooManyEligibleKeys)?;
            if info.eligible_keys != 0 {
                let upper_logit = scaled_query_l2_norm * info.max_key_l2_norm;
                if !upper_logit.is_finite() {
                    return Err(RaeCertError::NonFiniteBoundTerm { page });
                }
                let term = (info.eligible_keys as f64).ln() + upper_logit;
                if !term.is_finite() {
                    return Err(RaeCertError::NonFiniteBoundTerm { page });
                }
                log_omitted_mass_bound = log_add_exp(log_omitted_mass_bound, term)
                    .ok_or(RaeCertError::NumericalOverflow)?;
            }
        }
    }
    let eligible_key_count = selected_key_count
        .checked_add(omitted_key_count)
        .ok_or(RaeCertError::TooManyEligibleKeys)?;
    if eligible_key_count > RAE_CERT_MAX_ELIGIBLE_KEYS {
        return Err(RaeCertError::TooManyEligibleKeys);
    }
    if selected_key_count != selected_eligible_logits.len() {
        return Err(RaeCertError::WrongSelectedLogitCount {
            expected: selected_key_count,
            actual: selected_eligible_logits.len(),
        });
    }
    if selected_key_count == 0 {
        return Err(RaeCertError::NoSelectedEligibleKeys);
    }

    let mut max_score = f64::NEG_INFINITY;
    for (index, &score) in selected_eligible_logits.iter().enumerate() {
        if !score.is_finite() {
            return Err(RaeCertError::NonFiniteSelectedLogit { index });
        }
        max_score = max_score.max(score);
    }
    let selected_sum: f64 = selected_eligible_logits
        .iter()
        .map(|score| (*score - max_score).exp())
        .sum();
    let log_selected_mass = max_score + selected_sum.ln();
    if !log_selected_mass.is_finite() {
        return Err(RaeCertError::NumericalOverflow);
    }

    let mass = if omitted_key_count == 0 {
        0.0
    } else if log_omitted_mass_bound >= log_selected_mass {
        let ratio = (log_selected_mass - log_omitted_mass_bound).exp();
        1.0 / (1.0 + ratio)
    } else {
        let ratio = (log_omitted_mass_bound - log_selected_mass).exp();
        ratio / (1.0 + ratio)
    };

    // Round positive upper bounds *up* one representable float to avoid
    // accidentally claiming an exact threshold when numerical evaluation
    // rounded the conservative expression down.
    // An omitted page with eligible keys has strictly positive exp-mass,
    // even when the floating-point sigmoid underflows to numeric zero.
    // Never certify *exact* zero omission merely because exp underflowed.
    let maximum_dropped_mass = if omitted_key_count == 0 {
        0.0
    } else if mass == 0.0 {
        f64::from_bits(1)
    } else {
        round_nonnegative_up(mass.min(1.0))
    };
    let raw_output_bound = (2.0 * maximum_dropped_mass) * value_abs_bound;
    if !raw_output_bound.is_finite() {
        return Err(RaeCertError::NumericalOverflow);
    }
    let maximum_output_error = if raw_output_bound == 0.0
        && maximum_dropped_mass > 0.0
        && value_abs_bound > 0.0
    {
        // Preserve the strictly positive mathematical bound when the
        // double multiplication itself underflows.
        f64::from_bits(1)
    } else {
        round_nonnegative_up(raw_output_bound)
    };

    Ok(RaeNormMassCertificate {
        page_count: pages.len(),
        selected_pages: selected_page_count,
        selected_eligible_keys: selected_key_count,
        omitted_eligible_keys: omitted_key_count,
        maximum_dropped_mass,
        maximum_output_error,
    })
}

fn log_add_exp(first: f64, second: f64) -> Option<f64> {
    if first == f64::NEG_INFINITY {
        return Some(second);
    }
    if second == f64::NEG_INFINITY {
        return Some(first);
    }
    let maximum = first.max(second);
    let minimum = first.min(second);
    let sum = maximum + (minimum - maximum).exp().ln_1p();
    sum.is_finite().then_some(sum)
}

fn round_nonnegative_up(value: f64) -> f64 {
    if value == 0.0 || value == 1.0 {
        return value;
    }
    // A positive finite float's next bit-pattern represents nextUp.
    f64::from_bits(value.to_bits() + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rae_numeric_mass::audit_causal_page_mass;

    fn versions() -> RaeBoundVersions {
        RaeBoundVersions {
            query_key_generation: 7,
            norms_key_generation: 7,
            query_value_generation: 9,
            value_bound_generation: 9,
            query_mask_generation: 11,
            counts_mask_generation: 11,
        }
    }

    #[test]
    fn certifies_small_omission_using_norms_without_omitted_qk_logits() {
        // q=[1], selected K=[10], omitted K=[0]. The selected logit
        // 10 is supplied; the omitted logit is NOT passed to this API.
        let result = certify_omitted_page_mass(
            &[0b01],
            &[
                RaeCausalPageBounds {
                    eligible_keys: 1,
                    max_key_l2_norm: 10.0,
                },
                RaeCausalPageBounds {
                    eligible_keys: 1,
                    max_key_l2_norm: 0.0,
                },
            ],
            &[10.0],
            1.0,
            1.0,
            versions(),
        )
        .unwrap();
        assert_eq!(result.selected_eligible_keys, 1);
        assert_eq!(result.omitted_eligible_keys, 1);
        assert!(result.maximum_dropped_mass > 0.0);
        assert!(result.maximum_dropped_mass < 0.00005);
        assert!(result.maximum_output_error < 0.0001);
        assert!(result.within_budgets(0.00005, 0.0001).unwrap());

        // Independent dense causal oracle ONLY in test code.
        let actual =
            audit_causal_page_mass(&[10.0, 0.0], &[true, true], &[0, 1], 2, &[0b01], 1.0).unwrap();
        assert!(result.maximum_dropped_mass >= actual.softmax.dropped_mass());
    }

    #[test]
    fn dominant_omitted_page_makes_certification_fail() {
        let result = certify_omitted_page_mass(
            &[0b01],
            &[
                RaeCausalPageBounds {
                    eligible_keys: 1,
                    max_key_l2_norm: 0.0,
                },
                RaeCausalPageBounds {
                    eligible_keys: 1,
                    max_key_l2_norm: 20.0,
                },
            ],
            &[0.0],
            1.0,
            1.0,
            versions(),
        )
        .unwrap();
        assert!(result.maximum_dropped_mass > 0.999999);
        assert!(!result.within_budgets(0.01, 0.1).unwrap());
    }

    #[test]
    fn wholly_ineligible_future_page_contributes_no_mass() {
        let result = certify_omitted_page_mass(
            &[0b01],
            &[
                RaeCausalPageBounds {
                    eligible_keys: 1,
                    max_key_l2_norm: 1.0,
                },
                RaeCausalPageBounds {
                    eligible_keys: 0,
                    max_key_l2_norm: 1.0e300,
                },
            ],
            &[1.0],
            1.0e300,
            2.0,
            versions(),
        )
        .unwrap();
        assert_eq!(result.omitted_eligible_keys, 0);
        assert_eq!(result.maximum_dropped_mass, 0.0);
        assert_eq!(result.maximum_output_error, 0.0);
    }

    #[test]
    fn stale_k_v_and_eligibility_metadata_are_independently_rejected() {
        let page = [RaeCausalPageBounds {
            eligible_keys: 1,
            max_key_l2_norm: 1.0,
        }];
        let mut version = versions();
        version.norms_key_generation += 1;
        assert_eq!(
            certify_omitted_page_mass(&[1], &page, &[0.0], 1.0, 1.0, version),
            Err(RaeCertError::StaleKeyNorms)
        );
        version = versions();
        version.value_bound_generation += 1;
        assert_eq!(
            certify_omitted_page_mass(&[1], &page, &[0.0], 1.0, 1.0, version),
            Err(RaeCertError::StaleValueBound)
        );
        version = versions();
        version.counts_mask_generation += 1;
        assert_eq!(
            certify_omitted_page_mass(&[1], &page, &[0.0], 1.0, 1.0, version),
            Err(RaeCertError::StaleCausalCounts)
        );
    }

    #[test]
    fn incorrect_selected_logits_and_bitset_geometry_fail_closed() {
        let pages = [
            RaeCausalPageBounds {
                eligible_keys: 2,
                max_key_l2_norm: 1.0,
            },
            RaeCausalPageBounds {
                eligible_keys: 1,
                max_key_l2_norm: 2.0,
            },
        ];
        assert_eq!(
            certify_omitted_page_mass(&[1], &pages, &[0.0], 1.0, 1.0, versions()),
            Err(RaeCertError::WrongSelectedLogitCount {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            certify_omitted_page_mass(&[4], &pages, &[], 1.0, 1.0, versions()),
            Err(RaeCertError::NonZeroPageMaskPadding)
        );
        assert_eq!(
            certify_omitted_page_mass(&[], &pages, &[], 1.0, 1.0, versions()),
            Err(RaeCertError::InvalidPageMaskWidth {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(
            certify_omitted_page_mass(&[0], &pages, &[], 1.0, 1.0, versions()),
            Err(RaeCertError::NoSelectedEligibleKeys)
        );
    }

    #[test]
    fn rejects_nonfinite_bounds_and_would_be_overflowing_omitted_logit() {
        let pages = [
            RaeCausalPageBounds {
                eligible_keys: 1,
                max_key_l2_norm: 1.0,
            },
            RaeCausalPageBounds {
                eligible_keys: 1,
                max_key_l2_norm: f64::MAX,
            },
        ];
        assert_eq!(
            certify_omitted_page_mass(&[1], &pages, &[0.0], f64::MAX, 1.0, versions()),
            Err(RaeCertError::NonFiniteBoundTerm { page: 1 })
        );
        let mut invalid = pages;
        invalid[1].max_key_l2_norm = -1.0;
        assert_eq!(
            certify_omitted_page_mass(&[1], &invalid, &[0.0], 1.0, 1.0, versions()),
            Err(RaeCertError::InvalidPageNorm { page: 1 })
        );
        assert_eq!(
            certify_omitted_page_mass(&[1], &pages, &[f64::NAN], 1.0, 1.0, versions()),
            Err(RaeCertError::NonFiniteSelectedLogit { index: 0 })
        );
    }

    #[test]
    fn boundary_pages_and_multiple_keys_are_counted_without_dense_omitted_logits() {
        let mut pages = vec![
            RaeCausalPageBounds {
                eligible_keys: 0,
                max_key_l2_norm: 0.0,
            };
            65
        ];
        pages[0].eligible_keys = 2;
        pages[64].eligible_keys = 3;
        pages[64].max_key_l2_norm = 1.0;
        let result =
            certify_omitted_page_mass(&[1, 0], &pages, &[0.0, 0.0], 0.0, 2.0, versions()).unwrap();
        assert_eq!(result.page_count, 65);
        assert_eq!(result.selected_pages, 1);
        assert_eq!(result.selected_eligible_keys, 2);
        assert_eq!(result.omitted_eligible_keys, 3);
        assert!(result.maximum_dropped_mass >= 0.6);
        assert!(result.maximum_dropped_mass < 0.600000000000001);
    }
    #[test]
    fn underflow_does_not_invent_exactly_zero_omitted_mass() {
        let candidate = certify_omitted_page_mass(
            &[0b01],
            &[
                RaeCausalPageBounds {
                    eligible_keys: 1,
                    max_key_l2_norm: 1000.0,
                },
                RaeCausalPageBounds {
                    eligible_keys: 1,
                    max_key_l2_norm: 0.0,
                },
            ],
            &[1000.0],
            1.0,
            1.0,
            versions(),
        )
        .unwrap();
        assert!(candidate.maximum_dropped_mass > 0.0);
        assert!(candidate.maximum_output_error > 0.0);
        assert!(!candidate.within_budgets(0.0, 0.0).unwrap());
    }

}
