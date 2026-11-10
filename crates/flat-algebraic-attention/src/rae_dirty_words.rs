//! RAE-5: event-driven rising fronts from externally supplied dirty words.
//!
//! Unlike RAE-1, the hot path does NOT scan every feature word per tick.
//! The input producer MUST provide every changed word, including falling-only
//! changes. This module cannot verify the completeness of an event stream
//! without a separate full-snapshot reconciliation.
//!
//! Logical page wakeups are NOT authority to skip numerical attention or KV.

use core::fmt;

/// One full packed u64 value for a word changed since the preceding tick.
/// Indices must be strictly ascending and unique within each tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyFeatureWord {
    pub index: usize,
    pub next: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirtyWordError {
    InvalidDimensions,
    WrongBaselineWordCount { expected: usize, actual: usize },
    WrongFeatureMapLength { expected: usize, actual: usize },
    InvalidPage { feature: usize, page: usize },
    NonZeroPadding { word: usize },
    NonIncreasingTick { previous: u64, next: u64 },
    InvalidWordIndex { index: usize },
    NonIncreasingWordIndex { previous: usize, next: usize },
}

impl fmt::Display for DirtyWordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for DirtyWordError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyWakeEvidence {
    pub tick: u64,
    /// Externally supplied words inspected, including unchanged supplied words.
    pub visited_words: usize,
    /// Supplied words whose actual packed value differs from the previous one.
    pub changed_words: usize,
    pub rising_features: usize,
    pub falling_features: usize,
    pub active_features: usize,
    pub woken_pages: usize,
    pub newly_latched_pages: usize,
}

/// Research-only sparse word-update router with persistent logical latches.
/// The state bitset and feature-to-page map are allocated once at creation.
/// A valid tick reads only the externally supplied packed feature words,
/// not an entire `feature_width` snapshot. This saves host-side logical
/// scanning *only when the producer already maintains exact dirty notices*.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirtyWordPageRouter {
    feature_width: usize,
    page_count: usize,
    feature_pages: Vec<usize>,
    previous: Vec<u64>,
    active_features: usize,
    latched_pages: Vec<u64>,
    last_seen_wake_tick: Vec<u64>,
    last_tick: u64,
}

const MAX_FEATURE_BITS: usize = 1_048_576;
const MAX_PAGES: usize = 65_536;

impl DirtyWordPageRouter {
    pub fn new(
        feature_width: usize,
        page_count: usize,
        baseline_tick: u64,
        baseline_words: &[u64],
        feature_pages: &[usize],
    ) -> Result<Self, DirtyWordError> {
        if feature_width == 0
            || feature_width > MAX_FEATURE_BITS
            || page_count == 0
            || page_count > MAX_PAGES
        {
            return Err(DirtyWordError::InvalidDimensions);
        }
        let count = feature_width.div_ceil(64);
        if baseline_words.len() != count {
            return Err(DirtyWordError::WrongBaselineWordCount {
                expected: count,
                actual: baseline_words.len(),
            });
        }
        check_padding(feature_width, count - 1, baseline_words[count - 1])?;
        if feature_pages.len() != feature_width {
            return Err(DirtyWordError::WrongFeatureMapLength {
                expected: feature_width,
                actual: feature_pages.len(),
            });
        }
        for (feature, &page) in feature_pages.iter().enumerate() {
            if page >= page_count {
                return Err(DirtyWordError::InvalidPage { feature, page });
            }
        }
        let mut latched_pages = vec![0_u64; page_count.div_ceil(64)];
        for (word_index, &word) in baseline_words.iter().enumerate() {
            for feature in indices_of(word, word_index) {
                let page = feature_pages[feature];
                latched_pages[page / 64] |= 1_u64 << (page % 64);
            }
        }
        Ok(Self {
            feature_width,
            page_count,
            feature_pages: feature_pages.to_vec(),
            previous: baseline_words.to_vec(),
            active_features: baseline_words
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum(),
            latched_pages,
            last_seen_wake_tick: vec![baseline_tick; page_count],
            last_tick: baseline_tick,
        })
    }

    #[must_use]
    pub const fn feature_width(&self) -> usize {
        self.feature_width
    }

    #[must_use]
    pub const fn page_count(&self) -> usize {
        self.page_count
    }

    #[must_use]
    pub const fn last_tick(&self) -> u64 {
        self.last_tick
    }

    #[must_use]
    pub fn word_count(&self) -> usize {
        self.previous.len()
    }

    #[must_use]
    pub const fn active_features(&self) -> usize {
        self.active_features
    }

    #[must_use]
    pub fn previous(&self) -> &[u64] {
        &self.previous
    }

    #[must_use]
    pub fn latched_pages(&self) -> &[u64] {
        &self.latched_pages
    }

    /// Process exactly the dirty words supplied by a trusted state producer.
    ///
    /// All shape, clock, ordering and padding validation happens before
    /// any state/output mutation. On error, router and both output vectors
    /// are unchanged. On success, output lists contain unique original page
    /// IDs in first-encounter order, and may allocate if caller capacity is
    /// insufficient. An empty dirty slice advances time without any scan.
    ///
    /// Absence of a dirty notice does NOT prove no context, mask or K/V change.
    pub fn apply_tick(
        &mut self,
        tick: u64,
        dirty: &[DirtyFeatureWord],
        wakes: &mut Vec<usize>,
        newly_latched: &mut Vec<usize>,
    ) -> Result<DirtyWakeEvidence, DirtyWordError> {
        if tick <= self.last_tick {
            return Err(DirtyWordError::NonIncreasingTick {
                previous: self.last_tick,
                next: tick,
            });
        }
        let mut last_index = None;
        for word in dirty {
            if word.index >= self.previous.len() {
                return Err(DirtyWordError::InvalidWordIndex { index: word.index });
            }
            if let Some(previous) = last_index {
                if word.index <= previous {
                    return Err(DirtyWordError::NonIncreasingWordIndex {
                        previous,
                        next: word.index,
                    });
                }
            }
            if word.index + 1 == self.previous.len() {
                check_padding(self.feature_width, word.index, word.next)?;
            }
            last_index = Some(word.index);
        }

        wakes.clear();
        newly_latched.clear();
        let mut changed_words = 0;
        let mut rising_features = 0;
        let mut falling_features = 0;
        for word in dirty {
            let old = self.previous[word.index];
            if word.next == old {
                continue;
            }
            changed_words += 1;
            let rising = word.next & !old;
            let falling = old & !word.next;
            rising_features += rising.count_ones() as usize;
            falling_features += falling.count_ones() as usize;
            self.active_features -= old.count_ones() as usize;
            self.active_features += word.next.count_ones() as usize;
            self.previous[word.index] = word.next;

            for feature in indices_of(rising, word.index) {
                let page = self.feature_pages[feature];
                if self.last_seen_wake_tick[page] == tick {
                    continue;
                }
                self.last_seen_wake_tick[page] = tick;
                wakes.push(page);
                let bit = 1_u64 << (page % 64);
                if self.latched_pages[page / 64] & bit == 0 {
                    self.latched_pages[page / 64] |= bit;
                    newly_latched.push(page);
                }
            }
        }
        self.last_tick = tick;
        Ok(DirtyWakeEvidence {
            tick,
            visited_words: dirty.len(),
            changed_words,
            rising_features,
            falling_features,
            active_features: self.active_features,
            woken_pages: wakes.len(),
            newly_latched_pages: newly_latched.len(),
        })
    }

    /// Explicit full-reference reconciliation. Returns the first mismatched
    /// original packed-word index; this scan is O(total feature words).
    ///
    /// An incomplete dirty stream can silently diverge until a reference
    /// snapshot is checked. It is the producer's responsibility to schedule
    /// these snapshots according to its own state-version/fault policy.
    pub fn first_snapshot_mismatch(
        &self,
        authoritative: &[u64],
    ) -> Result<Option<usize>, DirtyWordError> {
        if authoritative.len() != self.previous.len() {
            return Err(DirtyWordError::WrongBaselineWordCount {
                expected: self.previous.len(),
                actual: authoritative.len(),
            });
        }
        let last = self.previous.len() - 1;
        check_padding(self.feature_width, last, authoritative[last])?;
        Ok(self
            .previous
            .iter()
            .zip(authoritative)
            .position(|(stored, exact)| stored != exact))
    }
}

fn check_padding(width: usize, word_index: usize, word: u64) -> Result<(), DirtyWordError> {
    let remainder = width % 64;
    if remainder != 0 && word_index + 1 == width.div_ceil(64) && word >> remainder != 0 {
        return Err(DirtyWordError::NonZeroPadding { word: word_index });
    }
    Ok(())
}

fn indices_of(word: u64, word_index: usize) -> impl Iterator<Item = usize> {
    let mut rest = word;
    core::iter::from_fn(move || {
        if rest == 0 {
            return None;
        }
        let bit = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        Some(word_index * 64 + bit)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge_pages::RisingEdgePageRouter;

    #[test]
    fn matches_full_rea2_across_multiword_rearm_and_aliased_pages() {
        let mut map = vec![0; 65];
        map[0] = 2;
        map[63] = 1;
        map[64] = 1;
        let mut delta = DirtyWordPageRouter::new(65, 3, 7, &[1, 0], &map).unwrap();
        let mut full = RisingEdgePageRouter::new(65, 7, &[1, 0], 3, &map).unwrap();
        let mut previous = [1_u64, 0_u64];
        let mut wakes = Vec::new();
        let mut newly = Vec::new();
        let mut full_wakes = [0_u64];
        let mut full_newly = [0_u64];
        let samples = [
            (8_u64, [1 | (1_u64 << 63), 1_u64]),
            (9, [1 | (1_u64 << 63), 1]),
            (10, [1, 0]),
            (11, [1 | (1_u64 << 63), 1]),
        ];
        for (tick, state) in samples {
            let dirty = state
                .iter()
                .zip(previous)
                .enumerate()
                .filter_map(|(index, (&next, old))| {
                    (next != old).then_some(DirtyFeatureWord { index, next })
                })
                .collect::<Vec<_>>();
            let observed = delta
                .apply_tick(tick, &dirty, &mut wakes, &mut newly)
                .unwrap();
            let reference = full
                .sample_into(tick, &state, &mut full_wakes, &mut full_newly)
                .unwrap();

            let to_mask = |pages: &[usize]| {
                pages
                    .iter()
                    .fold(0_u64, |mask, page| mask | (1_u64 << *page))
            };
            assert_eq!(to_mask(&wakes), full_wakes[0]);
            assert_eq!(to_mask(&newly), full_newly[0]);
            assert_eq!(delta.latched_pages(), full.latched_pages());
            assert_eq!(observed.rising_features, reference.rising_features);
            assert_eq!(observed.falling_features, reference.falling_features);
            assert_eq!(observed.active_features, reference.active_features);
            assert_eq!(delta.first_snapshot_mismatch(&state).unwrap(), None);
            previous = state;
        }
        assert_eq!(delta.latched_pages(), &[0b110]);
    }

    #[test]
    fn empty_delta_advances_without_reading_any_feature_words() {
        let mut router = DirtyWordPageRouter::new(64, 1, 0, &[0b11], &vec![0; 64]).unwrap();
        let mut wakes = vec![42];
        let mut newly = vec![99];
        let evidence = router.apply_tick(1, &[], &mut wakes, &mut newly).unwrap();
        assert_eq!(evidence.visited_words, 0);
        assert_eq!(evidence.changed_words, 0);
        assert_eq!(evidence.rising_features, 0);
        assert_eq!(evidence.active_features, 2);
        assert!(wakes.is_empty());
        assert!(newly.is_empty());
        assert_eq!(router.latched_pages(), &[1]);
    }

    #[test]
    fn invalid_deltas_preserve_every_state_and_output() {
        let mut router = DirtyWordPageRouter::new(65, 2, 3, &[0, 0], &vec![0; 65]).unwrap();
        let before = router.clone();
        let mut wakes = vec![5];
        let mut newly = vec![6];
        assert_eq!(
            router.apply_tick(3, &[], &mut wakes, &mut newly),
            Err(DirtyWordError::NonIncreasingTick {
                previous: 3,
                next: 3,
            })
        );
        assert_eq!(
            router.apply_tick(
                4,
                &[
                    DirtyFeatureWord { index: 1, next: 1 },
                    DirtyFeatureWord { index: 1, next: 0 },
                ],
                &mut wakes,
                &mut newly,
            ),
            Err(DirtyWordError::NonIncreasingWordIndex {
                previous: 1,
                next: 1,
            })
        );
        assert_eq!(
            router.apply_tick(
                4,
                &[DirtyFeatureWord { index: 1, next: 2 }],
                &mut wakes,
                &mut newly,
            ),
            Err(DirtyWordError::NonZeroPadding { word: 1 })
        );
        assert_eq!(
            router.apply_tick(
                4,
                &[DirtyFeatureWord { index: 2, next: 0 }],
                &mut wakes,
                &mut newly,
            ),
            Err(DirtyWordError::InvalidWordIndex { index: 2 })
        );
        assert_eq!(router, before);
        assert_eq!(wakes, vec![5]);
        assert_eq!(newly, vec![6]);
    }

    #[test]
    fn missing_dirty_notice_is_detected_by_authoritative_checkpoint() {
        let mut router = DirtyWordPageRouter::new(128, 2, 0, &[0, 0], &vec![1; 128]).unwrap();
        let mut wakes = Vec::new();
        let mut newly = Vec::new();
        let record = router
            .apply_tick(
                1,
                &[DirtyFeatureWord { index: 0, next: 1 }],
                &mut wakes,
                &mut newly,
            )
            .unwrap();
        assert_eq!(record.woken_pages, 1);
        assert_eq!(router.first_snapshot_mismatch(&[1, 1]).unwrap(), Some(1));
        assert_eq!(router.last_tick(), 1);
        assert_eq!(router.previous(), &[1, 0]);
        router
            .apply_tick(
                2,
                &[DirtyFeatureWord { index: 1, next: 1 }],
                &mut wakes,
                &mut newly,
            )
            .unwrap();
        assert_eq!(router.first_snapshot_mismatch(&[1, 1]).unwrap(), None);
    }

    #[test]
    fn initially_high_page_does_not_wake_until_rearmed() {
        let mut router = DirtyWordPageRouter::new(1, 1, 10, &[1], &[0]).unwrap();
        let mut wakes = Vec::new();
        let mut newly = Vec::new();
        router.apply_tick(11, &[], &mut wakes, &mut newly).unwrap();
        assert!(wakes.is_empty());
        router
            .apply_tick(
                12,
                &[DirtyFeatureWord { index: 0, next: 0 }],
                &mut wakes,
                &mut newly,
            )
            .unwrap();
        assert!(wakes.is_empty());
        router
            .apply_tick(
                13,
                &[DirtyFeatureWord { index: 0, next: 1 }],
                &mut wakes,
                &mut newly,
            )
            .unwrap();
        assert_eq!(wakes, vec![0]);
        assert!(newly.is_empty());
    }
}
