use core::fmt;

pub const BKV5_RANDOM_BASELINE_ALGORITHM: &str = "splitmix64-page-ranking-v1";
pub const BKV5_POSITIONAL_BASELINE_ALGORITHM: &str = "tail-window-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Bkv5MatchedDensityError {
    CandidateOutOfRange {
        logical_page: usize,
        total_pages: usize,
    },
    CandidateOrderOrDuplicate {
        previous: usize,
        current: usize,
    },
    PageIndexNotRepresentableAsU64 {
        logical_page: usize,
    },
}

impl fmt::Display for Bkv5MatchedDensityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CandidateOutOfRange {
                logical_page,
                total_pages,
            } => write!(
                f,
                "Boolean candidate page {logical_page} is outside total page count {total_pages}"
            ),
            Self::CandidateOrderOrDuplicate { previous, current } => write!(
                f,
                "Boolean candidate pages must be strictly increasing and duplicate-free: {previous} then {current}"
            ),
            Self::PageIndexNotRepresentableAsU64 { logical_page } => write!(
                f,
                "logical page {logical_page} cannot be represented by the v1 u64 ranking contract"
            ),
        }
    }
}

impl std::error::Error for Bkv5MatchedDensityError {}

/// Selection-only BKV-K5 controls matched to one Boolean candidate density.
///
/// This reference contract deliberately separates *which logical pages are
/// selected* from numerical K/V execution and traffic measurement. The random
/// control ranks every logical page with a versioned SplitMix64 function and
/// takes exactly the Boolean candidate count. The positional control takes an
/// equally sized tail window. Neither control is a performance claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bkv5MatchedDensityBaselines {
    total_pages: usize,
    boolean_selected_pages: Vec<usize>,
    seed: u64,
}

impl Bkv5MatchedDensityBaselines {
    pub fn new(
        total_pages: usize,
        boolean_selected_pages: Vec<usize>,
        seed: u64,
    ) -> Result<Self, Bkv5MatchedDensityError> {
        validate_candidate_pages(total_pages, &boolean_selected_pages)?;
        Ok(Self {
            total_pages,
            boolean_selected_pages,
            seed,
        })
    }

    #[must_use]
    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    #[must_use]
    pub fn boolean_selected_pages(&self) -> &[usize] {
        &self.boolean_selected_pages
    }

    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[must_use]
    pub fn selected_count(&self) -> usize {
        self.boolean_selected_pages.len()
    }

    /// Return the exact candidate-density numerator and denominator.
    ///
    /// An empty cache is represented as `0/1` rather than an undefined ratio.
    #[must_use]
    pub fn candidate_density_fraction(&self) -> (usize, usize) {
        if self.total_pages == 0 {
            (0, 1)
        } else {
            (self.selected_count(), self.total_pages)
        }
    }

    #[must_use]
    pub const fn random_algorithm(&self) -> &'static str {
        BKV5_RANDOM_BASELINE_ALGORITHM
    }

    #[must_use]
    pub const fn positional_algorithm(&self) -> &'static str {
        BKV5_POSITIONAL_BASELINE_ALGORITHM
    }

    pub fn random_matched_pages(&self) -> Result<Vec<usize>, Bkv5MatchedDensityError> {
        let count = self.selected_count();
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut ranked = Vec::with_capacity(self.total_pages);
        for logical_page in 0..self.total_pages {
            let page_u64 = u64::try_from(logical_page).map_err(|_| {
                Bkv5MatchedDensityError::PageIndexNotRepresentableAsU64 { logical_page }
            })?;
            ranked.push((splitmix64(self.seed ^ page_u64), logical_page));
        }
        ranked.sort_unstable();
        ranked.truncate(count);

        let mut selected = ranked
            .into_iter()
            .map(|(_, logical_page)| logical_page)
            .collect::<Vec<_>>();
        selected.sort_unstable();
        Ok(selected)
    }

    #[must_use]
    pub fn positional_matched_pages(&self) -> Vec<usize> {
        let count = self.selected_count();
        if count == 0 {
            return Vec::new();
        }
        (self.total_pages - count..self.total_pages).collect()
    }

    #[must_use]
    pub fn full_pages(&self) -> Vec<usize> {
        (0..self.total_pages).collect()
    }
}

fn validate_candidate_pages(
    total_pages: usize,
    selected_pages: &[usize],
) -> Result<(), Bkv5MatchedDensityError> {
    let mut previous = None;
    for &logical_page in selected_pages {
        if logical_page >= total_pages {
            return Err(Bkv5MatchedDensityError::CandidateOutOfRange {
                logical_page,
                total_pages,
            });
        }
        if let Some(previous) = previous {
            if logical_page <= previous {
                return Err(Bkv5MatchedDensityError::CandidateOrderOrDuplicate {
                    previous,
                    current: logical_page,
                });
            }
        }
        previous = Some(logical_page);
    }
    Ok(())
}

fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::{
        Bkv5MatchedDensityBaselines, Bkv5MatchedDensityError, BKV5_POSITIONAL_BASELINE_ALGORITHM,
        BKV5_RANDOM_BASELINE_ALGORITHM,
    };

    #[test]
    fn controls_match_boolean_density_exactly() {
        let plan = Bkv5MatchedDensityBaselines::new(10, vec![1, 4, 9], 0xB1C5).unwrap();

        assert_eq!(plan.selected_count(), 3);
        assert_eq!(plan.candidate_density_fraction(), (3, 10));
        assert_eq!(plan.random_matched_pages().unwrap(), vec![0, 2, 3]);
        assert_eq!(plan.positional_matched_pages(), vec![7, 8, 9]);
        assert_eq!(plan.full_pages(), (0..10).collect::<Vec<_>>());
        assert_eq!(plan.random_algorithm(), BKV5_RANDOM_BASELINE_ALGORITHM);
        assert_eq!(
            plan.positional_algorithm(),
            BKV5_POSITIONAL_BASELINE_ALGORITHM
        );
    }

    #[test]
    fn cross_repository_reference_vectors_are_stable() {
        let seed_7 = Bkv5MatchedDensityBaselines::new(32, vec![0, 4, 8, 12, 16, 20], 7).unwrap();
        let seed_8 = Bkv5MatchedDensityBaselines::new(32, vec![0, 4, 8, 12, 16, 20], 8).unwrap();

        assert_eq!(
            seed_7.random_matched_pages().unwrap(),
            vec![4, 12, 13, 18, 19, 21]
        );
        assert_eq!(
            seed_8.random_matched_pages().unwrap(),
            vec![2, 3, 11, 26, 28, 29]
        );
        assert_eq!(
            seed_7.positional_matched_pages(),
            vec![26, 27, 28, 29, 30, 31]
        );
    }

    #[test]
    fn empty_controls_are_explicit() {
        let empty_selection = Bkv5MatchedDensityBaselines::new(5, Vec::new(), 0).unwrap();
        let empty_cache = Bkv5MatchedDensityBaselines::new(0, Vec::new(), 0).unwrap();

        assert_eq!(empty_selection.candidate_density_fraction(), (0, 5));
        assert!(empty_selection.random_matched_pages().unwrap().is_empty());
        assert!(empty_selection.positional_matched_pages().is_empty());
        assert_eq!(empty_cache.candidate_density_fraction(), (0, 1));
        assert!(empty_cache.full_pages().is_empty());
    }

    #[test]
    fn malformed_candidate_sets_fail_closed() {
        assert_eq!(
            Bkv5MatchedDensityBaselines::new(3, vec![0, 0], 0),
            Err(Bkv5MatchedDensityError::CandidateOrderOrDuplicate {
                previous: 0,
                current: 0,
            })
        );
        assert_eq!(
            Bkv5MatchedDensityBaselines::new(3, vec![2, 1], 0),
            Err(Bkv5MatchedDensityError::CandidateOrderOrDuplicate {
                previous: 2,
                current: 1,
            })
        );
        assert_eq!(
            Bkv5MatchedDensityBaselines::new(3, vec![3], 0),
            Err(Bkv5MatchedDensityError::CandidateOutOfRange {
                logical_page: 3,
                total_pages: 3,
            })
        );
    }
}
