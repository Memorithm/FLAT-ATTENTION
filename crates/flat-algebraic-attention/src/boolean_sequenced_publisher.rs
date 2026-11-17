//! BPS-9: sequenced, source-bound front-window protocol for BPS-8.
//!
//! One committed frame corresponds to one strictly increasing, non-overlapping
//! watermark interval. Its source/epoch binding, gap-free sequence and previous
//! commitment are checked BEFORE advancing the Boolean recurrence.
//!
//! The 64-bit deterministic checksum is an *accidental corruption and continuity*
//! check, NOT a cryptographic digest, MAC, signature or proof of identity.
//! A malicious publisher can forge checksums and omit genuine rising fronts.
//! Authentication, replay protection across fresh instances, source completeness
//! and trusted checkpoint transfer require an external authenticated transport.
//!
//! BPS-9 also includes a contiguous sampled-word publisher for exact comparison
//! with BPS-3: sample coverage is checked, and rising masks honor held-high
//! inputs, falls, and re-arming across window boundaries. It costs O(T).
//!
//! Research-only; no production default route or energy claim.

use super::boolean_edge_memory::BooleanEdgeProgram;
use super::boolean_indexed_fronts::IndexedFront;
use super::boolean_watermarked_stream::{
    WatermarkedEdgeRuntime, WatermarkedError, WatermarkedWindow,
};

pub const FRONT_WIRE_VERSION: u16 = 1;

/// This binding must come from a separately authenticated source/session.
/// Changing either field *inside the same session* is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceEpoch {
    pub source_id: u64,
    pub epoch_id: u64,
}

/// An external trusted checkpoint supplies this anchor and the matching
/// recurrent memory and closing watermark. No self-authentication is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamAnchor {
    pub source: SourceEpoch,
    pub next_sequence: u64,
    pub prior_commitment: u64,
    pub closed_through: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameLimits {
    pub max_fronts: usize,
    pub max_queries: usize,
    /// Maximum number of newly closed ticks in any one frame.
    /// A zero limit prevents all progress.
    pub max_advance_ticks: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontFrameError {
    Quarantined,
    IncorrectVersion { received: u16 },
    WrongSource,
    WrongEpoch,
    WrongSequence { expected: u64, received: u64 },
    PreviousCommitmentMismatch,
    CommitmentMismatch,
    WrongWindowStart { expected: u64, received: u64 },
    NonIncreasingWatermark,
    WindowTooLarge,
    TooManyFronts,
    TooManyQueries,
    EmptySampleWindow,
    SampleWindowTooLarge,
    SequenceExhausted,
    TickOverflow,
    Core(WatermarkedError),
}

/// An owned, self-describing frame. Public fields are needed for wire
/// adapters and negative tests, but should never be trusted before admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontFrame {
    pub version: u16,
    pub source: SourceEpoch,
    pub sequence: u64,
    pub prior_commitment: u64,
    pub closed_from: u64,
    pub watermark: u64,
    pub fronts: Vec<IndexedFront>,
    pub commitment: u64,
}

fn fold_word(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}

/// Canonical, deterministic **non-cryptographic** 64-bit content checksum.
/// Includes *all* front ticks/masks and the previous frame's checksum.
/// Consumer query ticks are deliberately NOT part of publisher-signed data:
/// they are private consumer choices checked by BPS-8 independently.
#[must_use]
pub fn frame_commitment(frame: &FrontFrame) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    fold_word(&mut hash, 0x4250_5339_4652_4f4eu64); // "BPS9FRON"
    fold_word(&mut hash, u64::from(frame.version));
    fold_word(&mut hash, frame.source.source_id);
    fold_word(&mut hash, frame.source.epoch_id);
    fold_word(&mut hash, frame.sequence);
    fold_word(&mut hash, frame.prior_commitment);
    fold_word(&mut hash, frame.closed_from);
    fold_word(&mut hash, frame.watermark);
    fold_word(&mut hash, frame.fronts.len() as u64);
    for front in &frame.fronts {
        fold_word(&mut hash, front.tick);
        fold_word(&mut hash, front.rising);
    }
    hash
}

impl FrontFrame {
    /// Compose a claim from an external producer. This does NOT verify the
    /// producer delivered every genuine front or actually owns the source ID.
    #[must_use]
    pub fn claimed(
        anchor: StreamAnchor,
        watermark: u64,
        fronts: Vec<IndexedFront>,
    ) -> Self {
        let mut frame = Self {
            version: FRONT_WIRE_VERSION,
            source: anchor.source,
            sequence: anchor.next_sequence,
            prior_commitment: anchor.prior_commitment,
            closed_from: anchor.closed_through,
            watermark,
            fronts,
            commitment: 0,
        };
        frame.commitment = frame_commitment(&frame);
        frame
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcceptedFrontWindow {
    pub data: WatermarkedWindow,
    pub source: SourceEpoch,
    pub sequence: u64,
    pub commitment: u64,
}

/// Consumes only externally pre-indexed frames; state remains O(1), excluding
/// caller-owned frame arrays and the returned query outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequencedFrontRuntime {
    core: WatermarkedEdgeRuntime,
    source: SourceEpoch,
    next_sequence: u64,
    prior_commitment: u64,
    limits: FrameLimits,
    quarantined: bool,
}

impl SequencedFrontRuntime {
    #[must_use]
    pub const fn new(
        program: BooleanEdgeProgram,
        initial_state: u64,
        anchor: StreamAnchor,
        limits: FrameLimits,
    ) -> Self {
        Self {
            core: WatermarkedEdgeRuntime::new(
                program,
                anchor.closed_through,
                initial_state,
            ),
            source: anchor.source,
            next_sequence: anchor.next_sequence,
            prior_commitment: anchor.prior_commitment,
            limits,
            quarantined: false,
        }
    }

    #[must_use]
    pub const fn state(&self) -> u64 {
        self.core.state()
    }

    #[must_use]
    pub const fn closed_through(&self) -> u64 {
        self.core.closed_through()
    }

    #[must_use]
    pub const fn quarantined(&self) -> bool {
        self.quarantined
    }

    #[must_use]
    pub const fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    #[must_use]
    pub const fn prior_commitment(&self) -> u64 {
        self.prior_commitment
    }

    /// Any invalid or discontinuous frame quarantines this session; this
    /// deliberate fail-closed state change is the ONLY mutation on rejection.
    /// Recovery requires a fresh instance and a separately trusted checkpoint.
    /// It must not silently reset from arbitrary data supplied by this source.
    pub fn ingest(
        &mut self,
        frame: &FrontFrame,
        queries: &[u64],
    ) -> Result<AcceptedFrontWindow, FrontFrameError> {
        if self.quarantined {
            return Err(FrontFrameError::Quarantined);
        }
        match self.admit_and_apply(frame, queries) {
            Ok(value) => Ok(value),
            Err(error) => {
                self.quarantined = true;
                Err(error)
            }
        }
    }

    fn admit_and_apply(
        &mut self,
        frame: &FrontFrame,
        queries: &[u64],
    ) -> Result<AcceptedFrontWindow, FrontFrameError> {
        if frame.version != FRONT_WIRE_VERSION {
            return Err(FrontFrameError::IncorrectVersion {
                received: frame.version,
            });
        }
        if frame.source.source_id != self.source.source_id {
            return Err(FrontFrameError::WrongSource);
        }
        if frame.source.epoch_id != self.source.epoch_id {
            return Err(FrontFrameError::WrongEpoch);
        }
        if frame.sequence != self.next_sequence {
            return Err(FrontFrameError::WrongSequence {
                expected: self.next_sequence,
                received: frame.sequence,
            });
        }
        if frame.prior_commitment != self.prior_commitment {
            return Err(FrontFrameError::PreviousCommitmentMismatch);
        }
        if frame.closed_from != self.core.closed_through() {
            return Err(FrontFrameError::WrongWindowStart {
                expected: self.core.closed_through(),
                received: frame.closed_from,
            });
        }
        if frame.watermark <= frame.closed_from {
            return Err(FrontFrameError::NonIncreasingWatermark);
        }
        if frame.watermark - frame.closed_from > self.limits.max_advance_ticks {
            return Err(FrontFrameError::WindowTooLarge);
        }
        if frame.fronts.len() > self.limits.max_fronts {
            return Err(FrontFrameError::TooManyFronts);
        }
        if queries.len() > self.limits.max_queries {
            return Err(FrontFrameError::TooManyQueries);
        }
        if frame_commitment(frame) != frame.commitment {
            return Err(FrontFrameError::CommitmentMismatch);
        }
        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(FrontFrameError::SequenceExhausted)?;
        // BPS-8 checks all front masks/ticks, query ordering and counter
        // capacity before changing state. No partial application occurs.
        let data = self
            .core
            .advance_window(frame.watermark, &frame.fronts, queries)
            .map_err(FrontFrameError::Core)?;
        self.next_sequence = next_sequence;
        self.prior_commitment = frame.commitment;
        Ok(AcceptedFrontWindow {
            data,
            source: frame.source,
            sequence: frame.sequence,
            commitment: frame.commitment,
        })
    }
}

/// Exact reference publisher for contiguous samples, across batch boundaries.
/// It does NOT prove that the *physical* producer delivered all real samples.
/// It reads every supplied sample: O(T), not O(E).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampledFrontPublisher {
    anchor: StreamAnchor,
    previous_word: u64,
}

impl SampledFrontPublisher {
    #[must_use]
    pub const fn new(anchor: StreamAnchor, baseline_sample: u64) -> Self {
        Self {
            anchor,
            previous_word: baseline_sample,
        }
    }

    #[must_use]
    pub const fn anchor(&self) -> StreamAnchor {
        self.anchor
    }

    /// Publish a complete consecutive sample interval with exact rising masks.
    /// Reject empty windows and overly long batches before changing state.
    /// No gaps are inferred to be zero and no falling/rearming is guessed.
    pub fn publish_contiguous_samples(
        &mut self,
        samples: &[u64],
        max_samples: usize,
    ) -> Result<FrontFrame, FrontFrameError> {
        if samples.is_empty() {
            return Err(FrontFrameError::EmptySampleWindow);
        }
        if samples.len() > max_samples {
            return Err(FrontFrameError::SampleWindowTooLarge);
        }
        let span = u64::try_from(samples.len()).map_err(|_| FrontFrameError::TickOverflow)?;
        let watermark = self
            .anchor
            .closed_through
            .checked_add(span)
            .ok_or(FrontFrameError::TickOverflow)?;
        let next_sequence = self
            .anchor
            .next_sequence
            .checked_add(1)
            .ok_or(FrontFrameError::SequenceExhausted)?;
        let mut previous_word = self.previous_word;
        let mut fronts = Vec::new();
        for (index, &word) in samples.iter().enumerate() {
            let rise = word & !previous_word;
            if rise != 0 {
                fronts.push(IndexedFront {
                    tick: self.anchor.closed_through + 1 + index as u64,
                    rising: rise,
                });
            }
            previous_word = word;
        }
        let frame = FrontFrame::claimed(self.anchor, watermark, fronts);
        self.previous_word = previous_word;
        self.anchor.closed_through = watermark;
        self.anchor.next_sequence = next_sequence;
        self.anchor.prior_commitment = frame.commitment;
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean_indexed_fronts::{
        certify_indexed_fronts_against_samples, sampled_indexed_fronts,
    };
    use crate::boolean_scan_lstm::UnaryGate;

    fn program(shift: u8, mask: u64) -> BooleanEdgeProgram {
        BooleanEdgeProgram::new(
            UnaryGate {
                when_false: u64::MAX,
                when_true: u64::MAX,
            },
            UnaryGate::default(),
            mask,
            shift,
        )
        .unwrap()
    }

    const ANCHOR: StreamAnchor = StreamAnchor {
        source: SourceEpoch {
            source_id: 17,
            epoch_id: 42,
        },
        next_sequence: 1,
        prior_commitment: 0xf3e3_9918_4e9c_0721,
        closed_through: 0,
    };
    const LIMITS: FrameLimits = FrameLimits {
        max_fronts: 256,
        max_queries: 256,
        max_advance_ticks: 512,
    };

    #[test]
    fn sample_publisher_and_consumer_match_dense_and_one_shot_across_boundaries() {
        let mut seed = 0x8fe3_c429_9a01_1132u64;
        for shift in [0, 1, 7, 63] {
            for baseline in [0u64, 1, u64::MAX] {
                let p = program(shift, u64::MAX);
                let initial = seed.rotate_right(9);
                let mut previous = baseline;
                let mut samples = Vec::new();
                for tick in 0..197 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    if tick % 7 == 0 {
                        previous = seed;
                    } else if tick % 3 == 0 {
                        previous = 0;
                    }
                    samples.push(previous);
                }
                let dense = p.forward_dense(&samples, baseline, initial);
                let mut source = SampledFrontPublisher::new(ANCHOR, baseline);
                let mut runtime = SequencedFrontRuntime::new(p, initial, ANCHOR, LIMITS);
                let mut previous_end = 0;
                let mut replies = Vec::new();
                for end in [1usize, 9, 15, 16, 67, 120, 197] {
                    let frame = source
                        .publish_contiguous_samples(&samples[previous_end..end], 512)
                        .unwrap();
                    let queries: Vec<u64> =
                        ((previous_end as u64 + 1)..=end as u64).collect();
                    let answer = runtime.ingest(&frame, &queries).unwrap();
                    replies.extend(answer.data.outputs);
                    assert_eq!(answer.data.closed_through, end as u64);
                    previous_end = end;
                }
                assert_eq!(replies, dense);
                let all_fronts = sampled_indexed_fronts(&samples, baseline);
                let indexed = p
                    .forward_indexed_queries(
                        &all_fronts,
                        &(1..=samples.len() as u64).collect::<Vec<_>>(),
                        samples.len() as u64,
                        initial,
                    )
                    .unwrap();
                assert_eq!(runtime.state(), indexed.final_state);
                assert_eq!(runtime.closed_through(), samples.len() as u64);
                assert_eq!(runtime.next_sequence(), 8);
            }
        }
    }

    #[test]
    fn source_contiguity_includes_fall_and_rearming_across_frames() {
        let mut publisher = SampledFrontPublisher::new(ANCHOR, 1);
        let a = publisher.publish_contiguous_samples(&[1, 1], 4).unwrap();
        let b = publisher.publish_contiguous_samples(&[0, 1, 1], 4).unwrap();
        assert!(a.fronts.is_empty());
        assert_eq!(b.fronts, vec![IndexedFront { tick: 4, rising: 1 }]);
        let mut runtime =
            SequencedFrontRuntime::new(program(1, u64::MAX), 1, ANCHOR, LIMITS);
        assert_eq!(runtime.ingest(&a, &[1, 2]).unwrap().data.outputs, vec![1, 1]);
        assert_eq!(
            runtime.ingest(&b, &[3, 4, 5]).unwrap().data.outputs,
            vec![1, 2, 2]
        );
    }

    #[test]
    fn corrupt_data_gap_replay_epoch_source_and_version_quarantine_without_core_mutation() {
        let variants = 9;
        for case in 0..variants {
            let mut runtime =
                SequencedFrontRuntime::new(program(7, u64::MAX), 1, ANCHOR, LIMITS);
            let valid = FrontFrame::claimed(
                ANCHOR,
                5,
                vec![IndexedFront { tick: 2, rising: 1 }],
            );
            let mut invalid = valid.clone();
            match case {
                0 => invalid.version += 1,
                1 => invalid.source.source_id += 1,
                2 => invalid.source.epoch_id += 1,
                3 => invalid.sequence += 1,
                4 => invalid.prior_commitment += 1,
                5 => invalid.closed_from = 1,
                6 => invalid.commitment ^= 1,
                7 => invalid.fronts[0].rising = 0,
                8 => invalid.fronts[0].tick = 9,
                _ => unreachable!(),
            }
            let before = runtime.clone();
            assert!(runtime.ingest(&invalid, &[3]).is_err());
            assert!(runtime.quarantined());
            assert_eq!(runtime.state(), before.state());
            assert_eq!(runtime.closed_through(), before.closed_through());
            assert_eq!(runtime.next_sequence(), before.next_sequence());
            assert_eq!(runtime.prior_commitment(), before.prior_commitment());
            assert_eq!(runtime.ingest(&valid, &[3]), Err(FrontFrameError::Quarantined));
        }
    }

    #[test]
    fn second_frame_gap_or_replay_is_rejected_after_successful_first_frame() {
        let mut runtime =
            SequencedFrontRuntime::new(program(1, u64::MAX), 1, ANCHOR, LIMITS);
        let first = FrontFrame::claimed(ANCHOR, 5, vec![]);
        runtime.ingest(&first, &[1, 5]).unwrap();
        let second_anchor = StreamAnchor {
            next_sequence: 3,
            prior_commitment: first.commitment,
            closed_through: 5,
            ..ANCHOR
        };
        let skipped = FrontFrame::claimed(second_anchor, 10, vec![]);
        assert!(matches!(
            runtime.ingest(&skipped, &[10]),
            Err(FrontFrameError::WrongSequence { expected: 2, received: 3 })
        ));
        assert_eq!(runtime.closed_through(), 5);
        assert_eq!(runtime.ingest(&first, &[1]), Err(FrontFrameError::Quarantined));
    }

    #[test]
    fn bounds_and_bad_queries_are_fail_closed() {
        let narrow = FrameLimits {
            max_fronts: 1,
            max_queries: 1,
            max_advance_ticks: 3,
        };
        for case in 0..4 {
            let mut runtime = SequencedFrontRuntime::new(
                program(1, u64::MAX),
                1,
                ANCHOR,
                narrow,
            );
            let mut frame = FrontFrame::claimed(
                ANCHOR,
                3,
                vec![IndexedFront { tick: 1, rising: 1 }],
            );
            let queries = match case {
                0 => {
                    frame.watermark = 5;
                    frame.commitment = frame_commitment(&frame);
                    vec![2]
                }
                1 => {
                    frame.fronts.push(IndexedFront { tick: 2, rising: 1 });
                    frame.commitment = frame_commitment(&frame);
                    vec![2]
                }
                2 => vec![1, 2],
                3 => vec![4],
                _ => unreachable!(),
            };
            let prior = runtime.clone();
            assert!(runtime.ingest(&frame, &queries).is_err());
            assert_eq!(runtime.state(), prior.state());
            assert_eq!(runtime.closed_through(), prior.closed_through());
            assert!(runtime.quarantined());
        }
    }

    #[test]
    fn publisher_rejects_errors_atomically() {
        let mut publisher = SampledFrontPublisher::new(ANCHOR, 0);
        let prior = publisher.clone();
        assert_eq!(
            publisher.publish_contiguous_samples(&[], 5),
            Err(FrontFrameError::EmptySampleWindow)
        );
        assert_eq!(publisher, prior);
        assert_eq!(
            publisher.publish_contiguous_samples(&[0, 1, 2], 2),
            Err(FrontFrameError::SampleWindowTooLarge)
        );
        assert_eq!(publisher, prior);
        let over = StreamAnchor {
            closed_through: u64::MAX,
            ..ANCHOR
        };
        let mut publisher = SampledFrontPublisher::new(over, 0);
        let prior = publisher.clone();
        assert_eq!(
            publisher.publish_contiguous_samples(&[1], 5),
            Err(FrontFrameError::TickOverflow)
        );
        assert_eq!(publisher, prior);
        let over_sequence = StreamAnchor {
            next_sequence: u64::MAX,
            ..ANCHOR
        };
        let mut publisher = SampledFrontPublisher::new(over_sequence, 0);
        assert_eq!(
            publisher.publish_contiguous_samples(&[1], 5),
            Err(FrontFrameError::SequenceExhausted)
        );
    }

    #[test]
    fn declared_zero_front_frame_can_lie_and_recompute_checksum() {
        let p = program(1, u64::MAX);
        let actual = [1u64, 0, 1];
        let missing = FrontFrame::claimed(ANCHOR, 3, vec![]);
        let mut runtime = SequencedFrontRuntime::new(p, 1, ANCHOR, LIMITS);
        let outputs = runtime.ingest(&missing, &[3]).unwrap().data.outputs;
        assert_ne!(outputs[0], p.forward_dense(&actual, 0, 1)[2]);
        assert!(certify_indexed_fronts_against_samples(
            &missing.fronts,
            &actual,
            0,
        )
        .is_err());
        // Even an internally consistent 64-bit checksum cannot prove
        // completeness of an external event stream.
    }

    #[test]
    fn nonzero_baseline_tick_and_zero_event_windows_work() {
        let anchor = StreamAnchor {
            closed_through: 1_000,
            ..ANCHOR
        };
        let frame = FrontFrame::claimed(anchor, 2_000, vec![]);
        let mut runtime = SequencedFrontRuntime::new(
            program(63, 0b1010),
            0b1111,
            anchor,
            FrameLimits {
                max_advance_ticks: 10_000,
                ..LIMITS
            },
        );
        let result = runtime.ingest(&frame, &[1_001, 2_000]).unwrap();
        assert_eq!(result.data.outputs, vec![0b1010; 2]);
        assert_eq!(result.data.final_state, 0b1111);
    }

    #[test]
    fn checksum_changes_with_every_bound_field_and_event() {
        let frame = FrontFrame::claimed(
            ANCHOR,
            5,
            vec![IndexedFront { tick: 2, rising: 1 }],
        );
        let base = frame.commitment;
        let mut changed = frame.clone();
        changed.fronts[0].tick = 3;
        assert_ne!(frame_commitment(&changed), base);
        changed = frame.clone();
        changed.fronts[0].rising = 2;
        assert_ne!(frame_commitment(&changed), base);
        changed = frame.clone();
        changed.source.epoch_id += 1;
        assert_ne!(frame_commitment(&changed), base);
        changed = frame.clone();
        changed.prior_commitment += 1;
        assert_ne!(frame_commitment(&changed), base);
    }
}
