//! RAE-1: deterministic, bit-packed detection of sampled rising fronts.
//!
//! Research-only control signal. A front is 0 -> 1; a sustained 1 is NOT a
//! second event. The caller owns the clock and initial baseline snapshot.
//! This is not an SNN, an attention approximation, or a production KV policy.

use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RisingEdgeError {
    ZeroWidth,
    WrongWordCount { expected: usize, actual: usize },
    NonZeroPadding,
    NonIncreasingTick { previous: u64, next: u64 },
    WrongOutputLength { expected: usize, actual: usize },
}

impl fmt::Display for RisingEdgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RisingEdgeError {}

/// Evidence from an explicitly sampled transition (not a physical timing measurement).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RisingEdgeEvidence {
    pub tick: u64,
    pub rising_count: usize,
    pub active_count: usize,
}

/// One bit per logical feature. Baseline is explicitly supplied at construction;
/// bits already high in the baseline do not emit an invented initial front.
/// Storage is one previous-state bitset; callers own the output bitset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RisingEdgeDetector {
    bit_width: usize,
    previous: Vec<u64>,
    last_tick: u64,
}

impl RisingEdgeDetector {
    pub fn new(
        bit_width: usize,
        baseline_tick: u64,
        baseline: &[u64],
    ) -> Result<Self, RisingEdgeError> {
        if bit_width == 0 {
            return Err(RisingEdgeError::ZeroWidth);
        }
        validate_words(bit_width, baseline)?;
        Ok(Self {
            bit_width,
            previous: baseline.to_vec(),
            last_tick: baseline_tick,
        })
    }

    #[must_use]
    pub const fn bit_width(&self) -> usize {
        self.bit_width
    }

    #[must_use]
    pub fn word_count(&self) -> usize {
        words_for(self.bit_width)
    }

    #[must_use]
    pub const fn last_tick(&self) -> u64 {
        self.last_tick
    }

    #[must_use]
    pub fn previous(&self) -> &[u64] {
        &self.previous
    }

    /// Emit all 0 -> 1 transitions from the last sampled state to `current`.
    /// Caller must supply a strictly increasing, logical tick. Any validation
    /// failure leaves the detector AND the output untouched.
    ///
    /// The scan is O(ceil(width / 64)) even when there are zero events:
    /// only downstream work over emitted fronts can be event-proportional.
    pub fn sample_into(
        &mut self,
        tick: u64,
        current: &[u64],
        output: &mut [u64],
    ) -> Result<RisingEdgeEvidence, RisingEdgeError> {
        if tick <= self.last_tick {
            return Err(RisingEdgeError::NonIncreasingTick {
                previous: self.last_tick,
                next: tick,
            });
        }
        validate_words(self.bit_width, current)?;
        if output.len() != self.previous.len() {
            return Err(RisingEdgeError::WrongOutputLength {
                expected: self.previous.len(),
                actual: output.len(),
            });
        }

        let mut rising_count = 0;
        let mut active_count = 0;
        for ((out, previous), next) in output
            .iter_mut()
            .zip(self.previous.iter_mut())
            .zip(current.iter().copied())
        {
            *out = next & !*previous;
            rising_count += (*out).count_ones() as usize;
            active_count += next.count_ones() as usize;
            *previous = next;
        }
        self.last_tick = tick;
        Ok(RisingEdgeEvidence {
            tick,
            rising_count,
            active_count,
        })
    }
}

fn words_for(bit_width: usize) -> usize {
    bit_width / 64 + usize::from(bit_width % 64 != 0)
}

fn validate_words(bit_width: usize, words: &[u64]) -> Result<(), RisingEdgeError> {
    let expected = words_for(bit_width);
    if words.len() != expected {
        return Err(RisingEdgeError::WrongWordCount {
            expected,
            actual: words.len(),
        });
    }
    let trailing = bit_width % 64;
    if trailing != 0 {
        let mask = (1_u64 << trailing) - 1;
        if words[expected - 1] & !mask != 0 {
            return Err(RisingEdgeError::NonZeroPadding);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_rise_does_not_repeat_until_fall_rearms() {
        let mut detector = RisingEdgeDetector::new(4, 0, &[0b0001]).unwrap();
        let mut fronts = [u64::MAX];
        let first = detector.sample_into(1, &[0b0011], &mut fronts).unwrap();
        assert_eq!(fronts, [0b0010]);
        assert_eq!(first.rising_count, 1);
        assert_eq!(first.active_count, 2);

        let held = detector.sample_into(2, &[0b0011], &mut fronts).unwrap();
        assert_eq!(held.rising_count, 0);
        assert_eq!(fronts, [0]);

        let fell = detector.sample_into(3, &[0b0001], &mut fronts).unwrap();
        assert_eq!(fell.rising_count, 0);
        assert_eq!(fronts, [0]);

        let rose_again = detector.sample_into(4, &[0b0011], &mut fronts).unwrap();
        assert_eq!(rose_again.rising_count, 1);
        assert_eq!(fronts, [0b0010]);
    }

    #[test]
    fn simultaneous_fronts_cross_packed_word_boundary() {
        let mut detector = RisingEdgeDetector::new(70, 100, &[0, 0]).unwrap();
        let mut fronts = [0, 0];
        let observed = detector
            .sample_into(101, &[(1_u64 << 63) | 1, 0b100001], &mut fronts)
            .unwrap();
        assert_eq!(fronts, [(1_u64 << 63) | 1, 0b100001]);
        assert_eq!(observed.rising_count, 4);
        assert_eq!(observed.active_count, 4);
        assert_eq!(detector.word_count(), 2);
    }

    #[test]
    fn invalid_tick_and_padding_leave_state_and_output_unchanged() {
        let mut detector = RisingEdgeDetector::new(65, 7, &[1, 0]).unwrap();
        let before = detector.clone();
        let mut output = [0xAA, 0xBB];
        assert_eq!(
            detector.sample_into(7, &[3, 0], &mut output),
            Err(RisingEdgeError::NonIncreasingTick {
                previous: 7,
                next: 7,
            })
        );
        assert_eq!(
            detector.sample_into(8, &[3, 2], &mut output),
            Err(RisingEdgeError::NonZeroPadding)
        );
        assert_eq!(
            detector.sample_into(8, &[3, 0], &mut output[..1]),
            Err(RisingEdgeError::WrongOutputLength {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(detector, before);
        assert_eq!(output, [0xAA, 0xBB]);
    }

    #[test]
    fn initial_high_baseline_is_not_a_new_event() {
        let mut detector = RisingEdgeDetector::new(64, 0, &[u64::MAX]).unwrap();
        let mut output = [u64::MAX];
        let evidence = detector.sample_into(1, &[u64::MAX], &mut output).unwrap();
        assert_eq!(evidence.rising_count, 0);
        assert_eq!(evidence.active_count, 64);
        assert_eq!(output, [0]);
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        assert_eq!(
            RisingEdgeDetector::new(0, 0, &[]),
            Err(RisingEdgeError::ZeroWidth)
        );
        assert_eq!(
            RisingEdgeDetector::new(65, 0, &[0]),
            Err(RisingEdgeError::WrongWordCount {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            RisingEdgeDetector::new(65, 0, &[0, 0b10]),
            Err(RisingEdgeError::NonZeroPadding)
        );
    }

    #[test]
    fn identical_quiescent_samples_emit_no_fronts() {
        let mut detector = RisingEdgeDetector::new(3, 0, &[0]).unwrap();
        let mut out = [1];
        for tick in 1..=4 {
            assert_eq!(
                detector.sample_into(tick, &[0], &mut out).unwrap().rising_count,
                0
            );
            assert_eq!(out, [0]);
        }
    }
}
