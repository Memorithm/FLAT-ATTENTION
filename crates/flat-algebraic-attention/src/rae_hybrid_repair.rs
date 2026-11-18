//! RAE-7: bounded repair of impulse-proposed causal attention pages.
//!
//! A dirty-word or recurrent event router may propose a small packed page set,
//! but an event is NOT a numerical relevance certificate. This host-only
//! planner promotes the omitted pages with the largest conservative upper
//! softmax contributions until a norm-based omission bound passes or the
//! promotion budget is exhausted. A failed plan explicitly requests dense
//! fallback; it never returns an uncertified sparse mask.
//!
//! Unlike RAE-6, newly promoted pages do not need QK logits during planning:
//! Cauchy-Schwarz gives each key a conservative lower logit bound, while
//! omitted keys retain an upper logit bound. Exact selected-page QK and
//! numerical attention must still run after planning. This is not a GPU
//! implementation, formal interval proof, or production skip authorization.

use crate::rae_norm_certificate::{
    certify_omitted_page_mass, RaeBoundVersions, RaeCausalPageBounds, RaeCertError,
    RaeNormMassCertificate,
};

/// Reasons why an impulse candidate is not admitted to the research sparse arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaeHybridDenseReason {
    /// Trusted external state says that the query/context was invalidated.
    ContextInvalidated,
    /// The conservative bounds still exceed the budgets after bounded repair.
    BoundNotMet,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RaeHybridPlan {
    /// The packed mask can be evaluated as a research-only sparse candidate.
    /// Actual K/V reads, softmax and correctness qualification remain external.
    BoundedSparse {
        selected_page_mask: Vec<u64>,
        promoted_pages: Vec<usize>,
        certificate: RaeNormMassCertificate,
    },
    /// No sparse mask is exposed: the consumer must use the dense numerical
    /// reference (including all causally eligible pages).
    DenseFallback {
        reason: RaeHybridDenseReason,
        initial_certificate: RaeNormMassCertificate,
        attempted_promotions: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct RiskPage {
    page: usize,
    eligible_keys: usize,
    log_upper_mass: f64,
    log_lower_mass: f64,
}

/// Repair an event-proposed packed page mask using version-pinned K-norm bounds.
///
/// `selected_eligible_logits` contains exact, finite logits for ALL eligible
/// keys of the initially selected pages in the same query and original page
/// coordinate system. The page metadata must have the exact causal eligibility
/// counts and certified max-K norms for that query's K/mask generations.
///
/// For each initially omitted page p, we bound its total partition contribution
/// above by N_p * exp(||q||*scale*Kmax_p). Once promoted, the same page contributes
/// at least N_p * exp(-||q||*scale*Kmax_p) to the retained partition.
/// These inequalities allow bounded repair WITHOUT computing the promoted
/// page's QK logits until the sparse attention executor runs.
///
/// The risk ordering is deterministic (largest upper bound first, original
/// page ID as tie-break). It is a heuristic ordering, NOT an optimal minimum
/// subset solution. Work is O(P log P) and may exceed the savings from skipping
/// K/V, even when the returned certificate passes.
///
/// `context_invalidated` comes from an independent trusted source, not an
/// inference from missing impulses. A true value forces dense fallback.
/// False is not evidence of data freshness: provenance and versions must be
/// established by the calling model/runtime. Matching generation IDs alone
/// cannot prove that claim.
///
/// This is a research certificate using f64, with one-ULP upward rounding of
/// computed positive bounds; it is not formally interval-certified arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn plan_bounded_hybrid_attention(
    initial_page_mask: &[u64],
    pages: &[RaeCausalPageBounds],
    selected_eligible_logits: &[f64],
    scaled_query_l2_norm: f64,
    value_abs_bound: f64,
    versions: RaeBoundVersions,
    maximum_dropped_mass: f64,
    maximum_output_error: f64,
    max_promotions: usize,
    context_invalidated: bool,
) -> Result<RaeHybridPlan, RaeCertError> {
    // RAE-6 validates widths, padding, counts, versions, logits and norms.
    // No unchecked page/bitset indexing occurs before this call succeeds.
    let initial = certify_omitted_page_mass(
        initial_page_mask,
        pages,
        selected_eligible_logits,
        scaled_query_l2_norm,
        value_abs_bound,
        versions,
    )?;
    // Always validate budgets, even when external invalidation forces dense.
    let initially_within_budget =
        initial.within_budgets(maximum_dropped_mass, maximum_output_error)?;
    if context_invalidated {
        return Ok(RaeHybridPlan::DenseFallback {
            reason: RaeHybridDenseReason::ContextInvalidated,
            initial_certificate: initial,
            attempted_promotions: 0,
        });
    }
    if initially_within_budget {
        return Ok(RaeHybridPlan::BoundedSparse {
            selected_page_mask: initial_page_mask.to_vec(),
            promoted_pages: Vec::new(),
            certificate: initial,
        });
    }

    // All omitted pages with zero eligible keys are irrelevant to softmax.
    // RAE-6 has already rejected non-finite upper terms for nonempty pages.
    let mut risks = Vec::new();
    for (page, metadata) in pages.iter().enumerate() {
        if initial_page_mask[page / 64] & (1_u64 << (page % 64)) != 0
            || metadata.eligible_keys == 0
        {
            continue;
        }
        let log_count = (metadata.eligible_keys as f64).ln();
        let radius = scaled_query_l2_norm * metadata.max_key_l2_norm;
        risks.push(RiskPage {
            page,
            eligible_keys: metadata.eligible_keys,
            log_upper_mass: log_count + radius,
            log_lower_mass: log_count - radius,
        });
    }
    risks.sort_by(|a, b| {
        b.log_upper_mass
            .total_cmp(&a.log_upper_mass)
            .then_with(|| a.page.cmp(&b.page))
    });

    // Suffix log-partitions keep each promotion O(1) after the sort.
    // No omitted K vectors or omitted QK logits are consulted.
    let mut omitted_suffix = vec![f64::NEG_INFINITY; risks.len() + 1];
    for i in (0..risks.len()).rev() {
        omitted_suffix[i] =
            log_add_exp(risks[i].log_upper_mass, omitted_suffix[i + 1]);
    }

    let max_score = selected_eligible_logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut log_retained_lower = max_score
        + selected_eligible_logits
            .iter()
            .map(|score| (*score - max_score).exp())
            .sum::<f64>()
            .ln();

    let mut repaired = initial_page_mask.to_vec();
    let mut promoted = Vec::new();
    let mut selected_keys = initial.selected_eligible_keys;
    let mut omitted_keys = initial.omitted_eligible_keys;

    for (index, risk) in risks.iter().take(max_promotions).enumerate() {
        repaired[risk.page / 64] |= 1_u64 << (risk.page % 64);
        promoted.push(risk.page);
        selected_keys += risk.eligible_keys;
        omitted_keys -= risk.eligible_keys;
        log_retained_lower = log_add_exp(log_retained_lower, risk.log_lower_mass);
        let maximum_dropped_mass_bound = mass_bound(
            omitted_suffix[index + 1],
            log_retained_lower,
            omitted_keys,
        );
        let maximum_output_error_bound =
            output_bound(maximum_dropped_mass_bound, value_abs_bound);
        let certificate = RaeNormMassCertificate {
            page_count: pages.len(),
            selected_pages: initial.selected_pages + promoted.len(),
            selected_eligible_keys: selected_keys,
            omitted_eligible_keys: omitted_keys,
            maximum_dropped_mass: maximum_dropped_mass_bound,
            maximum_output_error: maximum_output_error_bound,
        };
        if certificate.within_budgets(maximum_dropped_mass, maximum_output_error)? {
            return Ok(RaeHybridPlan::BoundedSparse {
                selected_page_mask: repaired,
                promoted_pages: promoted,
                certificate,
            });
        }
    }

    Ok(RaeHybridPlan::DenseFallback {
        reason: RaeHybridDenseReason::BoundNotMet,
        initial_certificate: initial,
        attempted_promotions: promoted.len(),
    })
}

fn log_add_exp(first: f64, second: f64) -> f64 {
    if first == f64::NEG_INFINITY {
        return second;
    }
    if second == f64::NEG_INFINITY {
        return first;
    }
    let high = first.max(second);
    high + (first.min(second) - high).exp().ln_1p()
}

fn round_positive_up(value: f64) -> f64 {
    if value == 0.0 || value == 1.0 {
        value
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

fn mass_bound(log_omitted: f64, log_retained: f64, omitted_keys: usize) -> f64 {
    if omitted_keys == 0 {
        return 0.0;
    }
    let mass = if log_omitted >= log_retained {
        1.0 / (1.0 + (log_retained - log_omitted).exp())
    } else {
        let ratio = (log_omitted - log_retained).exp();
        ratio / (1.0 + ratio)
    };
    // A finite omitted partition has positive mathematical mass even if
    // an exponential underflows to zero in f64.
    if mass == 0.0 {
        f64::from_bits(1)
    } else {
        round_positive_up(mass.min(1.0))
    }
}

fn output_bound(mass: f64, value_abs_bound: f64) -> f64 {
    let raw = 2.0 * mass * value_abs_bound;
    if raw == 0.0 && mass > 0.0 && value_abs_bound > 0.0 {
        f64::from_bits(1)
    } else {
        round_positive_up(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rae_numeric_mass::audit_causal_page_mass;

    fn versions() -> RaeBoundVersions {
        RaeBoundVersions {
            query_key_generation: 3,
            norms_key_generation: 3,
            query_value_generation: 4,
            value_bound_generation: 4,
            query_mask_generation: 5,
            counts_mask_generation: 5,
        }
    }

    fn pages() -> [RaeCausalPageBounds; 3] {
        [
            RaeCausalPageBounds { eligible_keys: 1, max_key_l2_norm: 8.0 },
            RaeCausalPageBounds { eligible_keys: 1, max_key_l2_norm: 8.0 },
            RaeCausalPageBounds { eligible_keys: 1, max_key_l2_norm: 0.0 },
        ]
    }

    #[test]
    fn promotes_highest_risk_page_without_its_qk_logit() {
        let result = plan_bounded_hybrid_attention(
            &[0b001], &pages(), &[8.0], 1.0, 1.0,
            versions(), 0.001, 0.001, 1, false,
        ).unwrap();
        let RaeHybridPlan::BoundedSparse {
            selected_page_mask, promoted_pages, certificate,
        } = result else { panic!("one promotion should be sufficient") };
        assert_eq!(selected_page_mask, vec![0b011]);
        assert_eq!(promoted_pages, vec![1]);
        assert_eq!(certificate.selected_eligible_keys, 2);
        assert_eq!(certificate.omitted_eligible_keys, 1);
        assert!(certificate.maximum_dropped_mass < 0.001);
        assert!(certificate.maximum_output_error < 0.001);

        // Independent dense QK oracle: selected pages have logits [8,-8],
        // omitted original page 2 has logit 0. Its true dropped mass must
        // remain below the conservative planner bound.
        let oracle = audit_causal_page_mass(
            &[8.0, -8.0, 0.0],
            &[true, true, true],
            &[0, 1, 2],
            3,
            &selected_page_mask,
            1.0,
        ).unwrap();
        assert!(oracle.softmax.dropped_mass() <= certificate.maximum_dropped_mass);
    }

    #[test]
    fn budget_exhaustion_forces_dense_not_unqualified_sparse() {
        let plan = plan_bounded_hybrid_attention(
            &[0b001], &pages(), &[8.0], 1.0, 1.0,
            versions(), 0.001, 0.001, 0, false,
        ).unwrap();
        assert!(matches!(
            plan,
            RaeHybridPlan::DenseFallback {
                reason: RaeHybridDenseReason::BoundNotMet,
                attempted_promotions: 0,
                ..
            }
        ));
    }

    #[test]
    fn invalidation_forces_dense_even_when_initial_certificate_passes() {
        let plan = plan_bounded_hybrid_attention(
            &[0b111], &pages(), &[8.0, -8.0, 0.0], 1.0, 1.0,
            versions(), 0.0, 0.0, 2, true,
        ).unwrap();
        assert!(matches!(
            plan,
            RaeHybridPlan::DenseFallback {
                reason: RaeHybridDenseReason::ContextInvalidated,
                attempted_promotions: 0,
                ..
            }
        ));
    }

    #[test]
    fn identical_risks_are_broken_by_original_page_id() {
        let equal = [
            RaeCausalPageBounds { eligible_keys: 1, max_key_l2_norm: 10.0 },
            RaeCausalPageBounds { eligible_keys: 1, max_key_l2_norm: 0.0 },
            RaeCausalPageBounds { eligible_keys: 1, max_key_l2_norm: 0.0 },
        ];
        let plan = plan_bounded_hybrid_attention(
            &[0b001], &equal, &[10.0], 1.0, 1.0,
            versions(), 0.00006, 0.00012, 1, false,
        ).unwrap();
        match plan {
            RaeHybridPlan::BoundedSparse { promoted_pages, .. } => {
                assert_eq!(promoted_pages, vec![1]);
            }
            other => panic!("expected bounded sparse repair, got {other:?}"),
        }
    }

    #[test]
    fn cross_word_page_promotions_preserve_original_ids() {
        let mut metadata = vec![
            RaeCausalPageBounds { eligible_keys: 0, max_key_l2_norm: 0.0 };
            65
        ];
        metadata[0].eligible_keys = 1;
        metadata[64] = RaeCausalPageBounds { eligible_keys: 1, max_key_l2_norm: 6.0 };
        let plan = plan_bounded_hybrid_attention(
            &[1, 0], &metadata, &[6.0], 1.0, 1.0,
            versions(), 0.0, 0.0, 1, false,
        ).unwrap();
        match plan {
            RaeHybridPlan::BoundedSparse { selected_page_mask, promoted_pages, certificate } => {
                assert_eq!(selected_page_mask, vec![1, 1]);
                assert_eq!(promoted_pages, vec![64]);
                assert_eq!(certificate.maximum_dropped_mass, 0.0);
            }
            other => panic!("expected cross-word repair, got {other:?}"),
        }
    }

    #[test]
    fn stale_generation_and_invalid_budgets_fail_closed() {
        let mut stale = versions();
        stale.norms_key_generation += 1;
        assert_eq!(
            plan_bounded_hybrid_attention(
                &[0b001], &pages(), &[8.0], 1.0, 1.0,
                stale, 0.001, 0.001, 1, false,
            ),
            Err(RaeCertError::StaleKeyNorms)
        );
        assert_eq!(
            plan_bounded_hybrid_attention(
                &[0b001], &pages(), &[8.0], 1.0, 1.0,
                versions(), -0.1, 0.001, 1, true,
            ),
            Err(RaeCertError::InvalidDroppedMassBudget)
        );
    }
}
