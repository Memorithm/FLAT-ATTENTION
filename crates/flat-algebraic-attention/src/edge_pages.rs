//! RAE-2: a research-only rising-front -> logical-page wake router.
//!
//! A *wake* is a control event, not a proof that a numerical attention block
//! is irrelevant or a guarantee that a physical page is resident. Persistent
//! logical latches remain set when impulses stop. FLAT's numerical reference,
//! causal mask, and K/V payload remain authoritative.

use core::fmt;

use crate::rising_edge::{RisingEdgeDetector, RisingEdgeError, RisingEdgeEvidence};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageWakeError {
    ZeroPageCount,
    WrongFeatureMapLength { expected: usize, actual: usize },
    InvalidPage { feature: usize, page: usize, page_count: usize },
    WrongPageWordCount { expected: usize, actual: usize },
    RisingEdge(RisingEdgeError),
}

impl fmt::Display for PageWakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PageWakeError {}

impl From<RisingEdgeError> for PageWakeError {
    fn from(value: RisingEdgeError) -> Self {
        Self::RisingEdge(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageWakeEvidence {
    pub tick: u64,
    /// Number of newly active feature bits. Several may wake one page.
    pub rising_features: usize,
    /// Number of active feature bits, including those which stay high.
    pub active_features: usize,
    /// Number of falling feature bits. These do not issue positive wakes.
    pub falling_features: usize,
    /// Distinct pages receiving a wake, not individual memory accesses.
    pub woken_pages: usize,
    /// Distinct pages whose logical latch was previously unset.
    pub newly_latched_pages: usize,
}

/// Fixed feature -> page map with one persistent logical activation bit/page.
/// The caller must qualify any future MAA connection independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RisingEdgePageRouter {
    detector: RisingEdgeDetector,
    feature_pages: Vec<usize>,
    page_count: usize,
    fronts: Vec<u64>,
    latched_pages: Vec<u64>,
}

impl RisingEdgePageRouter {
    pub fn new(
        feature_width: usize,
        baseline_tick: u64,
        baseline: &[u64],
        page_count: usize,
        feature_pages: &[usize],
    ) -> Result<Self, PageWakeError> {
        let detector = RisingEdgeDetector::new(feature_width, baseline_tick, baseline)?;
        if page_count == 0 {
            return Err(PageWakeError::ZeroPageCount);
        }
        if feature_pages.len() != feature_width {
            return Err(PageWakeError::WrongFeatureMapLength {
                expected: feature_width,
                actual: feature_pages.len(),
            });
        }
        for (feature, page) in feature_pages.iter().copied().enumerate() {
            if page >= page_count {
                return Err(PageWakeError::InvalidPage {
                    feature,
                    page,
                    page_count,
                });
            }
        }
        let mut latched_pages = vec![0_u64; words_for(page_count)];
        for (word_index, word) in baseline.iter().copied().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let feature = word_index * 64 + remaining.trailing_zeros() as usize;
                let page = feature_pages[feature];
                latched_pages[page / 64] |= 1_u64 << (page % 64);
                remaining &= remaining - 1;
            }
        }
        Ok(Self {
            fronts: vec![0_u64; detector.word_count()],
            detector,
            feature_pages: feature_pages.to_vec(),
            page_count,
            latched_pages,
        })
    }

    #[must_use]
    pub const fn feature_width(&self) -> usize {
        self.detector.bit_width()
    }

    #[must_use]
    pub const fn page_count(&self) -> usize {
        self.page_count
    }

    #[must_use]
    pub fn page_word_count(&self) -> usize {
        words_for(self.page_count)
    }

    #[must_use]
    pub fn latched_pages(&self) -> &[u64] {
        &self.latched_pages
    }

    #[must_use]
    pub const fn last_tick(&self) -> u64 {
        self.detector.last_tick()
    }

    /// Wake pages with one or more 0 -> 1 feature transitions at this tick.
    /// Distinct pages are reported once. A persistent 1 does not re-wake.
    ///
    /// `wakes` and `new_latches` are disjoint caller-owned output buffers
    /// with page-word length. `new_latches` is a subset of `wakes`.
    /// Every error preserves the router and both output buffers.
    ///
    /// Detection still scans all packed feature words on every sample.
    /// Dispatching the sparse wake mapping visits only rising features.
    pub fn sample_into(
        &mut self,
        tick: u64,
        current: &[u64],
        wakes: &mut [u64],
        new_latches: &mut [u64],
    ) -> Result<PageWakeEvidence, PageWakeError> {
        let expected = self.page_word_count();
        if wakes.len() != expected {
            return Err(PageWakeError::WrongPageWordCount {
                expected,
                actual: wakes.len(),
            });
        }
        if new_latches.len() != expected {
            return Err(PageWakeError::WrongPageWordCount {
                expected,
                actual: new_latches.len(),
            });
        }

        // Falling counters are observational; they do not qualify a key drop.
        // A malformed sample is rejected by the detector before any mutation.
        let falling = current
            .iter()
            .zip(self.detector.previous())
            .map(|(next, previous)| (*previous & !*next).count_ones() as usize)
            .sum();

        let RisingEdgeEvidence {
            tick,
            rising_count,
            active_count,
        } = self.detector.sample_into(tick, current, &mut self.fronts)?;

        wakes.fill(0);
        new_latches.fill(0);
        for (word_index, word) in self.fronts.iter().copied().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let feature = word_index * 64 + remaining.trailing_zeros() as usize;
                let page = self.feature_pages[feature];
                let page_word = page / 64;
                let bit = 1_u64 << (page % 64);
                wakes[page_word] |= bit;
                if self.latched_pages[page_word] & bit == 0 {
                    new_latches[page_word] |= bit;
                    self.latched_pages[page_word] |= bit;
                }
                remaining &= remaining - 1;
            }
        }

        Ok(PageWakeEvidence {
            tick,
            rising_features: rising_count,
            active_features: active_count,
            falling_features: falling,
            woken_pages: count_set(wakes),
            newly_latched_pages: count_set(new_latches),
        })
    }
}

fn words_for(count: usize) -> usize {
    count / 64 + usize::from(count % 64 != 0)
}

fn count_set(words: &[u64]) -> usize {
    words.iter().map(|word| word.count_ones() as usize).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiple_fronts_coalesce_into_distinct_page_wakes() {
        let mut router = RisingEdgePageRouter::new(5, 0, &[0], 3, &[0, 0, 1, 2, 1]).unwrap();
        let mut wakes = [0];
        let mut added = [0];
        let first = router
            .sample_into(1, &[0b0_0111], &mut wakes, &mut added)
            .unwrap();
        assert_eq!(first.rising_features, 3);
        assert_eq!(first.woken_pages, 2);
        assert_eq!(first.newly_latched_pages, 2);
        assert_eq!(wakes, [0b011]);
        assert_eq!(added, [0b011]);

        let held = router
            .sample_into(2, &[0b0_0111], &mut wakes, &mut added)
            .unwrap();
        assert_eq!(held.active_features, 3);
        assert_eq!(held.rising_features, 0);
        assert_eq!(wakes, [0]);
        assert_eq!(added, [0]);
        assert_eq!(router.latched_pages(), &[0b011]);

        let fell = router
            .sample_into(3, &[0], &mut wakes, &mut added)
            .unwrap();
        assert_eq!(fell.falling_features, 3);
        assert_eq!(fell.woken_pages, 0);
        assert_eq!(router.latched_pages(), &[0b011]);

        let rearm = router
            .sample_into(4, &[0b0_0010], &mut wakes, &mut added)
            .unwrap();
        assert_eq!(rearm.rising_features, 1);
        assert_eq!(rearm.woken_pages, 1);
        assert_eq!(rearm.newly_latched_pages, 0);
        assert_eq!(wakes, [0b001]);
        assert_eq!(added, [0]);
    }

    #[test]
    fn initially_high_baseline_latches_without_synthetic_wake() {
        let mut router = RisingEdgePageRouter::new(2, 5, &[0b01], 3, &[2, 0]).unwrap();
        assert_eq!(router.latched_pages(), &[0b100]);
        let mut wakes = [9];
        let mut added = [9];
        let held = router
            .sample_into(6, &[0b01], &mut wakes, &mut added)
            .unwrap();
        assert_eq!(held.rising_features, 0);
        assert_eq!(wakes, [0]);
        assert_eq!(added, [0]);
    }

    #[test]
    fn bit_64_and_feature_63_can_wake_the_same_page() {
        let mut mapping = vec![0; 65];
        mapping[63] = 1;
        mapping[64] = 1;
        let mut router = RisingEdgePageRouter::new(65, 0, &[0, 0], 66, &mapping).unwrap();
        let mut wakes = [0_u64; 2];
        let mut added = [0_u64; 2];
        let observed = router
            .sample_into(1, &[1_u64 << 63, 1], &mut wakes, &mut added)
            .unwrap();
        assert_eq!(observed.rising_features, 2);
        assert_eq!(observed.woken_pages, 1);
        assert_eq!(wakes, [0b10, 0]);
        assert_eq!(added, [0b10, 0]);
    }

    #[test]
    fn invalid_sample_is_transactional() {
        let mut router = RisingEdgePageRouter::new(65, 4, &[1, 0], 2, &vec![0; 65]).unwrap();
        let before = router.clone();
        let mut wakes = [0xAB];
        let mut added = [0xCD];
        assert_eq!(
            router.sample_into(5, &[2, 2], &mut wakes, &mut added),
            Err(PageWakeError::RisingEdge(RisingEdgeError::NonZeroPadding))
        );
        assert_eq!(
            router.sample_into(4, &[2, 0], &mut wakes, &mut added),
            Err(PageWakeError::RisingEdge(
                RisingEdgeError::NonIncreasingTick {
                    previous: 4,
                    next: 4,
                }
            ))
        );
        assert_eq!(
            router.sample_into(5, &[2, 0], &mut [], &mut added),
            Err(PageWakeError::WrongPageWordCount {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(router, before);
        assert_eq!(wakes, [0xAB]);
        assert_eq!(added, [0xCD]);
    }

    #[test]
    fn malformed_mapping_fails_closed() {
        assert_eq!(
            RisingEdgePageRouter::new(1, 0, &[0], 0, &[0]),
            Err(PageWakeError::ZeroPageCount)
        );
        assert_eq!(
            RisingEdgePageRouter::new(1, 0, &[0], 2, &[]),
            Err(PageWakeError::WrongFeatureMapLength {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(
            RisingEdgePageRouter::new(1, 0, &[0], 2, &[2]),
            Err(PageWakeError::InvalidPage {
                feature: 0,
                page: 2,
                page_count: 2,
            })
        );
    }

    #[test]
    fn wake_is_not_a_relevance_or_physical_read_certificate() {
        let mut router = RisingEdgePageRouter::new(1, 0, &[0], 1, &[0]).unwrap();
        let mut wakes = [0];
        let mut added = [0];
        router
            .sample_into(1, &[1], &mut wakes, &mut added)
            .unwrap();
        // The source remains active. The page stays logically latched,
        // while no new wake is emitted. A *new query* may still need
        // numerical attention to this page: a zero wake is NOT a skip mask.
        router
            .sample_into(2, &[1], &mut wakes, &mut added)
            .unwrap();
        assert_eq!(wakes, [0]);
        assert_eq!(router.latched_pages(), &[1]);
    }
}
