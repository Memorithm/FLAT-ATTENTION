//! RAE-8: executable scalar numerical attention on RAE-7 repaired page masks.
//!
//! This research-only executor links an event-proposed original page mask to
//! real Q*K scores, causal softmax, and V accumulation. It provides a complete
//! dense fallback and a deterministic dense comparison test. It does not
//! change FLAT's stable API or authorize a production/GPU sparse route.
//!
//! Only initially selected K vectors are read to construct the RAE-7 plan.
//! After repair, newly selected K vectors and selected V vectors are read.
//! All page identities/eligibility metadata are scanned on the host, so this
//! implementation does NOT claim net bandwidth, energy or latency savings.

use core::fmt;

use crate::rae_hybrid_repair::{
    plan_bounded_hybrid_attention, RaeHybridDenseReason, RaeHybridPlan,
};
use crate::rae_norm_certificate::{
    RaeBoundVersions, RaeCausalPageBounds, RaeCertError, RaeNormMassCertificate,
    RAE_CERT_MAX_ELIGIBLE_KEYS, RAE_CERT_MAX_PAGES,
};

pub const RAE8_MAX_DIMENSION: usize = 16_384;

#[derive(Debug, Clone, Copy)]
pub struct RaeNumericQuery<'a> {
    /// One query vector, evaluated against original key and value row order.
    pub query: &'a [f64],
    /// Row-major keys, with query.len() components per key.
    pub keys: &'a [f64],
    /// Row-major values, with value_width components per key.
    pub values: &'a [f64],
    pub value_width: usize,
    /// Original page ID for every key, including causally ineligible keys.
    pub key_pages: &'a [usize],
    /// Trusted causal, padding and length eligibility, BEFORE page routing.
    pub eligible: &'a [bool],
    /// The reference score is scale * sum(query[d] * key[d]); scale >= 0.
    pub scale: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RaeNumericPolicy<'a> {
    /// Proposals from event routing, NOT a relevance certificate.
    pub impulse_pages: &'a [u64],
    /// Authoritative per-page max K norm and exact CURRENT causal key counts.
    pub bounds: &'a [RaeCausalPageBounds],
    pub versions: RaeBoundVersions,
    /// Independently asserted upper bound on every eligible V component.
    pub value_abs_bound: f64,
    pub maximum_dropped_mass: f64,
    pub maximum_output_error: f64,
    pub maximum_promotions: usize,
    /// Independently signalled query/context invalidation, not inferred.
    pub context_invalidated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RaeNumericalDenseReason {
    ExternalInvalidation,
    InvalidImpulseMask,
    NoInitiallySelectedKeys,
    CausalCountMismatch,
    InvalidScaledQueryNorm,
    InvalidSelectedValueBound,
    PlannerRejected(RaeCertError),
    PlannerFallback(RaeHybridDenseReason),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RaeNumericalRoute {
    BoundedSparse {
        selected_page_mask: Vec<u64>,
        promoted_pages: Vec<usize>,
        certificate: RaeNormMassCertificate,
    },
    DenseFallback {
        reason: RaeNumericalDenseReason,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaeNumericExecution {
    pub output: Vec<f64>,
    pub route: RaeNumericalRoute,
    pub eligible_keys: usize,
    pub executed_keys: usize,
    /// Count of key rows used in QK score evaluations, not physical DRAM reads.
    pub key_rows_scored: usize,
    /// Count of V rows used in accumulation, not physical DRAM reads.
    pub value_rows_accumulated: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaeNumericExecutionError {
    InvalidDimensions,
    InvalidScale,
    NonFiniteQuery { dimension: usize },
    InvalidKeyPage { key: usize, page: usize },
    NoEligibleKeys,
    NonFiniteKey { key: usize, dimension: usize },
    NonFiniteScore { key: usize },
    NonFiniteValue { key: usize, dimension: usize },
    NumericalOverflow,
}

impl fmt::Display for RaeNumericExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RaeNumericExecutionError {}

/// Execute causal scalar attention using only the selected K/V rows after
/// RAE-7 page repair; force dense on invalidations, missing candidate keys,
/// stale/invalid certificates, count mismatches, or unsatisfied budgets.
///
/// Shape errors and non-finite numerics are errors, not fabricated outputs.
/// The caller is responsible for authoritative key/value/mask generation,
/// norm provenance, and omitted V bounds; this executor can check selected
/// V against the asserted bound, but cannot certify OMITTED V without reading it.
/// This is a standalone host research oracle, not the production attention API.
pub fn execute_rae_hybrid(
    query: RaeNumericQuery<'_>,
    policy: RaeNumericPolicy<'_>,
) -> Result<RaeNumericExecution, RaeNumericExecutionError> {
    validate_shape(query, policy.bounds.len())?;
    let n = query.key_pages.len();
    let mut eligible_count = 0_usize;
    let mut counts = vec![0_usize; policy.bounds.len()];
    for (index, (&page, &eligible)) in query.key_pages.iter().zip(query.eligible).enumerate() {
        if page >= policy.bounds.len() {
            return Err(RaeNumericExecutionError::InvalidKeyPage { key: index, page });
        }
        if eligible {
            eligible_count += 1;
            counts[page] += 1;
        }
    }
    if eligible_count == 0 {
        return Err(RaeNumericExecutionError::NoEligibleKeys);
    }
    // NaN is an internal not-yet-scored sentinel. Non-finite computed scores
    // are rejected before being cached, so every cached value is finite.
    let mut scores = vec![f64::NAN; n];
    let mut key_rows_scored = 0_usize;

    let mut fallback = if policy.context_invalidated {
        Some(RaeNumericalDenseReason::ExternalInvalidation)
    } else if !valid_mask(policy.impulse_pages, policy.bounds.len()) {
        Some(RaeNumericalDenseReason::InvalidImpulseMask)
    } else if counts
        .iter()
        .zip(policy.bounds)
        .any(|(&count, bound)| count != bound.eligible_keys)
    {
        Some(RaeNumericalDenseReason::CausalCountMismatch)
    } else {
        None
    };

    let mut sparse = None;
    if fallback.is_none() {
        let mut selected_logits = Vec::new();
        for (key, (&page, &eligible)) in query.key_pages.iter().zip(query.eligible).enumerate() {
            if eligible && contains(policy.impulse_pages, page) {
                let score = score_one(query, key)?;
                scores[key] = score;
                selected_logits.push(score);
                key_rows_scored += 1;
            }
        }
        if selected_logits.is_empty() {
            fallback = Some(RaeNumericalDenseReason::NoInitiallySelectedKeys);
        } else {
            let query_norm = query.query.iter().copied().fold(0.0_f64, f64::hypot);
            let scaled_norm = query_norm * query.scale;
            if !scaled_norm.is_finite() {
                fallback = Some(RaeNumericalDenseReason::InvalidScaledQueryNorm);
            } else {
                match plan_bounded_hybrid_attention(
                    policy.impulse_pages,
                    policy.bounds,
                    &selected_logits,
                    scaled_norm,
                    policy.value_abs_bound,
                    policy.versions,
                    policy.maximum_dropped_mass,
                    policy.maximum_output_error,
                    policy.maximum_promotions,
                    false,
                ) {
                    Ok(RaeHybridPlan::BoundedSparse {
                        selected_page_mask,
                        promoted_pages,
                        certificate,
                    }) => {
                        sparse = Some((selected_page_mask, promoted_pages, certificate));
                    }
                    Ok(RaeHybridPlan::DenseFallback { reason, .. }) => {
                        fallback = Some(RaeNumericalDenseReason::PlannerFallback(reason));
                    }
                    Err(error) => {
                        fallback = Some(RaeNumericalDenseReason::PlannerRejected(error));
                    }
                }
            }
        }
    }

    if let Some((mask, _, _)) = &sparse {
        // Catch a falsified bound on SELECTED V before emitting an uncertified
        // sparse result. Omitted V still requires upstream provenance.
        for (key, (&page, &eligible)) in query.key_pages.iter().zip(query.eligible).enumerate() {
            if eligible && contains(mask, page) {
                let base = key * query.value_width;
                for &v in &query.values[base..base + query.value_width] {
                    if !v.is_finite() {
                        return Err(RaeNumericExecutionError::NonFiniteValue {
                            key,
                            dimension: 0,
                        });
                    }
                    if v.abs() > policy.value_abs_bound {
                        fallback = Some(RaeNumericalDenseReason::InvalidSelectedValueBound);
                        break;
                    }
                }
                if fallback.is_some() {
                    break;
                }
            }
        }
    }

    let route = if let Some(reason) = fallback {
        RaeNumericalRoute::DenseFallback { reason }
    } else {
        let (selected_page_mask, promoted_pages, certificate) =
            sparse.expect("sparse planner succeeded when no fallback exists");
        RaeNumericalRoute::BoundedSparse {
            selected_page_mask,
            promoted_pages,
            certificate,
        }
    };
    let mask = match &route {
        RaeNumericalRoute::BoundedSparse { selected_page_mask, .. } => Some(selected_page_mask.as_slice()),
        RaeNumericalRoute::DenseFallback { .. } => None,
    };

    // The same score and softmax implementation is used for dense and sparse
    // execution. No full dense QK oracle is evaluated in the sparse hot path.
    let mut selected = Vec::new();
    for (key, (&page, &eligible)) in query.key_pages.iter().zip(query.eligible).enumerate() {
        if !eligible || mask.is_some_and(|m| !contains(m, page)) {
            continue;
        }
        let score = if scores[key].is_nan() {
            key_rows_scored += 1;
            score_one(query, key)?
        } else {
            scores[key]
        };
        selected.push((key, score));
    }
    if selected.is_empty() {
        return Err(RaeNumericExecutionError::NoEligibleKeys);
    }
    let max_score = selected
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut weight_sum = 0.0_f64;
    let mut output = vec![0.0_f64; query.value_width];
    for (key, score) in &selected {
        let weight = (*score - max_score).exp();
        weight_sum += weight;
        let base = key * query.value_width;
        for (dimension, (out, &value)) in output
            .iter_mut()
            .zip(&query.values[base..base + query.value_width])
            .enumerate()
        {
            if !value.is_finite() {
                return Err(RaeNumericExecutionError::NonFiniteValue {
                    key: *key,
                    dimension,
                });
            }
            *out += weight * value;
        }
    }
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(RaeNumericExecutionError::NumericalOverflow);
    }
    for component in &mut output {
        *component /= weight_sum;
        if !component.is_finite() {
            return Err(RaeNumericExecutionError::NumericalOverflow);
        }
    }
    Ok(RaeNumericExecution {
        output,
        route,
        eligible_keys: eligible_count,
        executed_keys: selected.len(),
        key_rows_scored,
        value_rows_accumulated: selected.len(),
    })
}

fn validate_shape(
    query: RaeNumericQuery<'_>,
    page_count: usize,
) -> Result<(), RaeNumericExecutionError> {
    let n = query.key_pages.len();
    let d = query.query.len();
    let v = query.value_width;
    if n == 0
        || n > RAE_CERT_MAX_ELIGIBLE_KEYS
        || d == 0
        || d > RAE8_MAX_DIMENSION
        || v == 0
        || v > RAE8_MAX_DIMENSION
        || page_count == 0
        || page_count > RAE_CERT_MAX_PAGES
        || query.eligible.len() != n
        || n.checked_mul(d) != Some(query.keys.len())
        || n.checked_mul(v) != Some(query.values.len())
    {
        return Err(RaeNumericExecutionError::InvalidDimensions);
    }
    if !query.scale.is_finite() || query.scale < 0.0 {
        return Err(RaeNumericExecutionError::InvalidScale);
    }
    for (dimension, &value) in query.query.iter().enumerate() {
        if !value.is_finite() {
            return Err(RaeNumericExecutionError::NonFiniteQuery { dimension });
        }
    }
    Ok(())
}

fn valid_mask(mask: &[u64], page_count: usize) -> bool {
    if mask.len() != page_count.div_ceil(64) {
        return false;
    }
    let bits = page_count % 64;
    bits == 0 || mask[mask.len() - 1] >> bits == 0
}

fn contains(mask: &[u64], page: usize) -> bool {
    mask[page / 64] & (1_u64 << (page % 64)) != 0
}

fn score_one(
    query: RaeNumericQuery<'_>,
    key: usize,
) -> Result<f64, RaeNumericExecutionError> {
    let start = key * query.query.len();
    let mut dot = 0.0_f64;
    for (dimension, (&q, &k)) in query
        .query
        .iter()
        .zip(&query.keys[start..start + query.query.len()])
        .enumerate()
    {
        if !k.is_finite() {
            return Err(RaeNumericExecutionError::NonFiniteKey { key, dimension });
        }
        dot += q * k;
    }
    let score = dot * query.scale;
    if !score.is_finite() {
        return Err(RaeNumericExecutionError::NonFiniteScore { key });
    }
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query<'a>(q: &'a [f64], k: &'a [f64], v: &'a [f64]) -> RaeNumericQuery<'a> {
        RaeNumericQuery {
            query: q,
            keys: k,
            values: v,
            value_width: 1,
            key_pages: &[0, 1, 2],
            eligible: &[true, true, true],
            scale: 1.0,
        }
    }
    fn versions() -> RaeBoundVersions {
        RaeBoundVersions {
            query_key_generation: 1,
            norms_key_generation: 1,
            query_value_generation: 2,
            value_bound_generation: 2,
            query_mask_generation: 3,
            counts_mask_generation: 3,
        }
    }
    fn policy<'a>(pages: &'a [RaeCausalPageBounds], impulse_pages: &'a [u64]) -> RaeNumericPolicy<'a> {
        RaeNumericPolicy {
            impulse_pages,
            bounds: pages,
            versions: versions(),
            value_abs_bound: 1.0,
            maximum_dropped_mass: 0.001,
            maximum_output_error: 0.001,
            maximum_promotions: 1,
            context_invalidated: false,
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
    fn repairs_impulses_and_computes_real_softmax_with_fewer_key_scores() {
        let q = query(&[1.0], &[8.0, -8.0, 0.0], &[1.0, -1.0, 0.5]);
        let metadata = pages();
        let repaired = execute_rae_hybrid(q, policy(&metadata, &[0b001])).unwrap();
        let mut dense_policy = policy(&metadata, &[0b001]);
        dense_policy.context_invalidated = true;
        let dense = execute_rae_hybrid(q, dense_policy).unwrap();
        assert!(matches!(repaired.route, RaeNumericalRoute::BoundedSparse { .. }));
        assert!(matches!(dense.route, RaeNumericalRoute::DenseFallback { .. }));
        assert_eq!(repaired.executed_keys, 2);
        assert_eq!(repaired.key_rows_scored, 2);
        assert_eq!(repaired.value_rows_accumulated, 2);
        assert_eq!(dense.executed_keys, 3);
        let cert = match repaired.route {
            RaeNumericalRoute::BoundedSparse { certificate, .. } => certificate,
            _ => unreachable!(),
        };
        assert!((repaired.output[0] - dense.output[0]).abs() <= cert.maximum_output_error);
    }

    #[test]
    fn missing_candidate_and_context_invalidation_execute_full_dense() {
        let q = query(&[1.0], &[8.0, -8.0, 0.0], &[1.0, -1.0, 0.5]);
        let metadata = pages();
        let empty = execute_rae_hybrid(q, policy(&metadata, &[0])).unwrap();
        assert!(matches!(empty.route, RaeNumericalRoute::DenseFallback {
            reason: RaeNumericalDenseReason::NoInitiallySelectedKeys,
        }));
        assert_eq!(empty.executed_keys, 3);
        let mut invalidated = policy(&metadata, &[0b111]);
        invalidated.context_invalidated = true;
        let second = execute_rae_hybrid(q, invalidated).unwrap();
        assert_eq!(empty.output, second.output);
    }

    #[test]
    fn stale_norms_and_wrong_causal_counts_never_enable_sparse() {
        let q = query(&[1.0], &[8.0, -8.0, 0.0], &[1.0, -1.0, 0.5]);
        let metadata = pages();
        let mut stale = policy(&metadata, &[1]);
        stale.versions.norms_key_generation = 99;
        let first = execute_rae_hybrid(q, stale).unwrap();
        assert!(matches!(first.route, RaeNumericalRoute::DenseFallback {
            reason: RaeNumericalDenseReason::PlannerRejected(RaeCertError::StaleKeyNorms),
        }));
        let mut corrupted = metadata;
        corrupted[1].eligible_keys = 2;
        let second = execute_rae_hybrid(q, policy(&corrupted, &[1])).unwrap();
        assert!(matches!(second.route, RaeNumericalRoute::DenseFallback {
            reason: RaeNumericalDenseReason::CausalCountMismatch,
        }));
        assert_eq!(first.output, second.output);
    }

    #[test]
    fn unqualified_value_bound_forces_dense() {
        let q = query(&[1.0], &[8.0, -8.0, 0.0], &[1.0, 5.0, 0.5]);
        let metadata = pages();
        let result = execute_rae_hybrid(q, policy(&metadata, &[1])).unwrap();
        assert!(matches!(result.route, RaeNumericalRoute::DenseFallback {
            reason: RaeNumericalDenseReason::InvalidSelectedValueBound,
        }));
        assert_eq!(result.executed_keys, 3);
    }

    #[test]
    fn no_future_key_is_scored_or_accumulated() {
        let mut q = query(&[1.0], &[8.0, -8.0, f64::NAN], &[1.0, -1.0, f64::NAN]);
        q.eligible = &[true, true, false];
        let mut metadata = pages();
        metadata[2].eligible_keys = 0;
        let outcome = execute_rae_hybrid(q, policy(&metadata, &[0b011])).unwrap();
        assert!(matches!(outcome.route, RaeNumericalRoute::BoundedSparse { .. }));
        assert_eq!(outcome.eligible_keys, 2);
        assert_eq!(outcome.executed_keys, 2);
    }

    #[test]
    fn malformed_impulse_mask_uses_dense_while_invalid_tensors_fail() {
        let q = query(&[1.0], &[8.0, -8.0, 0.0], &[1.0, -1.0, 0.5]);
        let metadata = pages();
        let dense = execute_rae_hybrid(q, policy(&metadata, &[0b1000])).unwrap();
        assert!(matches!(dense.route, RaeNumericalRoute::DenseFallback {
            reason: RaeNumericalDenseReason::InvalidImpulseMask,
        }));
        assert_eq!(dense.executed_keys, 3);
        let bad_query = RaeNumericQuery { keys: &[1.0], ..q };
        assert_eq!(
            execute_rae_hybrid(bad_query, policy(&metadata, &[1])),
            Err(RaeNumericExecutionError::InvalidDimensions)
        );
    }
}
