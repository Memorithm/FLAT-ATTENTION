//! RAE-6b: monotone per-page upper bounds maintained at K write time.
//!
//! This opt-in ledger never reads the other keys in a page when a single
//! key is inserted/replaced/deleted. The bound can become loose after
//! smaller replacements/removals, but it does not shrink and therefore
//! cannot underestimate an existing valid upper bound *if all writes
//! flow through this ledger and the initial key set is empty*.
//!
//! Research-only. Native K/V store ownership, causal eligibility counts,
//! vector precision and full runtime fallback remain external contracts.

use core::fmt;

use crate::rae_norm_certificate::{RaeCausalPageBounds, RAE_CERT_MAX_PAGES};

pub const RAE_CERT_MAX_DIMENSION: usize = 16_384;

#[derive(Debug, Clone, PartialEq)]
pub enum RaeNormLedgerError {
    InvalidDimensions,
    WrongKeyWidth { expected: usize, actual: usize },
    WrongCountWidth { expected: usize, actual: usize },
    InvalidPage { page: usize, count: usize },
    NonIncreasingGeneration { previous: u64, next: u64 },
    NonFiniteKeyComponent { index: usize },
    OverflowingKeyNorm,
}

impl fmt::Display for RaeNormLedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RaeNormLedgerError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaeNormWriteEvidence {
    pub generation: u64,
    pub page: usize,
    pub key_components_read: usize,
    pub current_page_upper_norm: f64,
    pub page_bound_increased: bool,
}

/// A reusable, bounded, model-owned-at-integration-time K-norm ledger.
///
/// Construct only against a truly EMPTY key store; initial zero norm
/// bounds would be unsafe if unknown K vectors already existed.
/// Each insertion or replacement processes only the written K vector
/// (O(embedding dimension)). Each deletion is O(1) and leaves the
/// upper bound conservatively unchanged.
///
/// A full-model adapter must invalidate/correct the ledger whenever
/// writes bypass the wrapper, the score scaling changes, or K precision
/// changes. Generations are sequence numbers managed by an authoritative
/// host, not cryptographic attestation.
#[derive(Debug, Clone, PartialEq)]
pub struct RaeMonotoneKeyNormLedger {
    embedding_dimension: usize,
    upper_norm_by_page: Vec<f64>,
    key_generation: u64,
}

impl RaeMonotoneKeyNormLedger {
    /// Construct an initially empty KV key store metadata owner.
    pub fn new_empty(
        page_count: usize,
        embedding_dimension: usize,
        baseline_key_generation: u64,
    ) -> Result<Self, RaeNormLedgerError> {
        if page_count == 0
            || page_count > RAE_CERT_MAX_PAGES
            || embedding_dimension == 0
            || embedding_dimension > RAE_CERT_MAX_DIMENSION
        {
            return Err(RaeNormLedgerError::InvalidDimensions);
        }
        Ok(Self {
            embedding_dimension,
            upper_norm_by_page: vec![0.0; page_count],
            key_generation: baseline_key_generation,
        })
    }

    #[must_use]
    pub fn page_count(&self) -> usize {
        self.upper_norm_by_page.len()
    }

    #[must_use]
    pub const fn key_generation(&self) -> u64 {
        self.key_generation
    }

    #[must_use]
    pub const fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    /// Incrementally process exactly one *new or replaced* K vector.
    /// If a removed/replaced prior vector had the previous page maximum,
    /// the old bound remains valid but may be loose.
    ///
    /// Strictly increasing generation must be assigned by the K-store
    /// owner after every mutation. All validation precedes mutation.
    pub fn observe_key_write(
        &mut self,
        page: usize,
        key: &[f64],
        next_key_generation: u64,
    ) -> Result<RaeNormWriteEvidence, RaeNormLedgerError> {
        self.validate_page_and_generation(page, next_key_generation)?;
        if key.len() != self.embedding_dimension {
            return Err(RaeNormLedgerError::WrongKeyWidth {
                expected: self.embedding_dimension,
                actual: key.len(),
            });
        }
        let norm = upper_l2_norm(key)?;
        let previous = self.upper_norm_by_page[page];
        self.upper_norm_by_page[page] = previous.max(norm);
        self.key_generation = next_key_generation;
        Ok(RaeNormWriteEvidence {
            generation: next_key_generation,
            page,
            key_components_read: key.len(),
            current_page_upper_norm: self.upper_norm_by_page[page],
            page_bound_increased: norm > previous,
        })
    }

    /// A deletion cannot make the previous maximum too SMALL. Advance the
    /// authoritative K version, but do not scan or shrink the page bound.
    pub fn observe_key_removal(
        &mut self,
        page: usize,
        next_key_generation: u64,
    ) -> Result<(), RaeNormLedgerError> {
        self.validate_page_and_generation(page, next_key_generation)?;
        self.key_generation = next_key_generation;
        Ok(())
    }

    /// Reconstitute the versioned norm metadata for THIS query using
    /// independent, query-specific eligible key counts. This visits one
    /// small metadata record/page, not individual skipped K vectors.
    ///
    /// The caller must stamp the returned count array with the current
    /// causal mask generation. Its correctness cannot be inferred here.
    pub fn bounds_for_counts(
        &self,
        eligible_key_counts: &[usize],
    ) -> Result<Vec<RaeCausalPageBounds>, RaeNormLedgerError> {
        if eligible_key_counts.len() != self.upper_norm_by_page.len() {
            return Err(RaeNormLedgerError::WrongCountWidth {
                expected: self.upper_norm_by_page.len(),
                actual: eligible_key_counts.len(),
            });
        }
        Ok(self
            .upper_norm_by_page
            .iter()
            .zip(eligible_key_counts)
            .map(|(&max_key_l2_norm, &eligible_keys)| RaeCausalPageBounds {
                eligible_keys,
                max_key_l2_norm,
            })
            .collect())
    }

    #[must_use]
    pub fn page_upper_norm(&self, page: usize) -> Option<f64> {
        self.upper_norm_by_page.get(page).copied()
    }

    fn validate_page_and_generation(
        &self,
        page: usize,
        next: u64,
    ) -> Result<(), RaeNormLedgerError> {
        if page >= self.upper_norm_by_page.len() {
            return Err(RaeNormLedgerError::InvalidPage {
                page,
                count: self.upper_norm_by_page.len(),
            });
        }
        if next <= self.key_generation {
            return Err(RaeNormLedgerError::NonIncreasingGeneration {
                previous: self.key_generation,
                next,
            });
        }
        Ok(())
    }
}

/// Stable scaling avoids overflowing the sum-of-squares before the
/// square root. One nextUp step accounts only for the FINAL f64 rounding,
/// not a formal interval proof of all upstream component arithmetic.
fn upper_l2_norm(key: &[f64]) -> Result<f64, RaeNormLedgerError> {
    let mut scale = 0.0_f64;
    let mut sumsq = 1.0_f64;
    for (index, &component) in key.iter().enumerate() {
        if !component.is_finite() {
            return Err(RaeNormLedgerError::NonFiniteKeyComponent { index });
        }
        let magnitude = component.abs();
        if magnitude == 0.0 {
            continue;
        }
        if magnitude > scale {
            let factor = scale / magnitude;
            sumsq = 1.0 + sumsq * factor * factor;
            scale = magnitude;
        } else {
            let factor = magnitude / scale;
            sumsq += factor * factor;
        }
    }
    let norm = scale * sumsq.sqrt();
    if !norm.is_finite() {
        return Err(RaeNormLedgerError::OverflowingKeyNorm);
    }
    Ok(if norm == 0.0 {
        0.0
    } else {
        f64::from_bits(norm.to_bits() + 1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rae_norm_certificate::{certify_omitted_page_mass, RaeBoundVersions};

    fn versions(generation: u64) -> RaeBoundVersions {
        RaeBoundVersions {
            query_key_generation: generation,
            norms_key_generation: generation,
            query_value_generation: generation,
            value_bound_generation: generation,
            query_mask_generation: 1,
            counts_mask_generation: 1,
        }
    }

    #[test]
    fn insert_write_and_overwrite_reuse_page_metadata_without_skipped_key_reads() {
        let mut ledger = RaeMonotoneKeyNormLedger::new_empty(2, 1, 0).unwrap();
        let first = ledger.observe_key_write(0, &[10.0], 1).unwrap();
        assert_eq!(first.key_components_read, 1);
        assert!(first.page_bound_increased);
        let other = ledger.observe_key_write(1, &[0.0], 2).unwrap();
        assert_eq!(other.key_components_read, 1);
        let bounds = ledger.bounds_for_counts(&[1, 1]).unwrap();
        let safe = certify_omitted_page_mass(
            &[1],
            &bounds,
            &[10.0],
            1.0,
            1.0,
            versions(ledger.key_generation()),
        )
        .unwrap();
        assert!(safe.within_budgets(0.00005, 0.0001).unwrap());

        // Page 1 is now rewritten with a strong key. This update reads
        // only the new key, raises page 1 bound, and rejects omission.
        let write = ledger.observe_key_write(1, &[20.0], 3).unwrap();
        assert_eq!(write.key_components_read, 1);
        assert!(write.page_bound_increased);
        let updated = ledger.bounds_for_counts(&[1, 1]).unwrap();
        let rejected = certify_omitted_page_mass(
            &[1],
            &updated,
            &[10.0],
            1.0,
            1.0,
            versions(ledger.key_generation()),
        )
        .unwrap();
        assert!(!rejected.within_budgets(0.00005, 0.0001).unwrap());

        // Smaller replacement and a deletion must never lower the
        // potentially still valid conservative maximum.
        let smaller = ledger.observe_key_write(1, &[0.5], 4).unwrap();
        assert!(!smaller.page_bound_increased);
        let before_remove = ledger.page_upper_norm(1);
        ledger.observe_key_removal(1, 5).unwrap();
        assert_eq!(ledger.page_upper_norm(1), before_remove);
        let empty = ledger.bounds_for_counts(&[1, 0]).unwrap();
        let full = certify_omitted_page_mass(
            &[1],
            &empty,
            &[10.0],
            1.0,
            1.0,
            versions(ledger.key_generation()),
        )
        .unwrap();
        assert_eq!(full.maximum_dropped_mass, 0.0);
    }

    #[test]
    fn negative_updates_and_invalid_dimensions_are_transactional() {
        let mut ledger = RaeMonotoneKeyNormLedger::new_empty(2, 2, 10).unwrap();
        let original = ledger.clone();
        assert_eq!(
            ledger.observe_key_write(2, &[0.0, 1.0], 11),
            Err(RaeNormLedgerError::InvalidPage { page: 2, count: 2 })
        );
        assert_eq!(
            ledger.observe_key_write(0, &[0.0, 1.0], 10),
            Err(RaeNormLedgerError::NonIncreasingGeneration {
                previous: 10,
                next: 10,
            })
        );
        assert_eq!(
            ledger.observe_key_write(0, &[1.0], 11),
            Err(RaeNormLedgerError::WrongKeyWidth {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            ledger.observe_key_write(0, &[f64::NAN, 1.0], 11),
            Err(RaeNormLedgerError::NonFiniteKeyComponent { index: 0 })
        );
        assert_eq!(
            ledger.observe_key_write(0, &[f64::MAX, f64::MAX], 11),
            Err(RaeNormLedgerError::OverflowingKeyNorm)
        );
        assert_eq!(ledger, original);
    }

    #[test]
    fn page_64_boundary_and_stable_l2_rescaling_are_preserved() {
        let mut ledger = RaeMonotoneKeyNormLedger::new_empty(65, 2, 0).unwrap();
        let write = ledger.observe_key_write(64, &[3.0, 4.0], 1).unwrap();
        assert!(write.current_page_upper_norm >= 5.0);
        assert!(write.current_page_upper_norm < 5.00000000000001);
        let huge = ledger.observe_key_write(0, &[1.0e200, 1.0e-200], 2).unwrap();
        assert!(huge.current_page_upper_norm.is_finite());
        assert!(huge.current_page_upper_norm >= 1.0e200);
        assert_eq!(ledger.bounds_for_counts(&[0; 65]).unwrap().len(), 65);
    }
}
