//! BPS-8: incremental, watermarked execution of indexed rising fronts.
//!
//! The publisher must guarantee that an accepted watermark closes its entire
//! event interval: all real rising fronts at or before that tick were delivered.
//! A watermark is NOT evidence of event-source completeness or authenticity.
//!
//! Once an interval is closed, late events and queries cannot rewrite history.
//! Validation of every batch is transactional. No prior fronts are retained.
//! Apart from the caller-owned batch and returned query outputs, persistent
//! memory is O(1) for one packed 64-bit recurrent state.
//!
//! Only already-indexed complete fronts give O(E + Q) CPU operations across
//! submitted batches. A sampled-word producer still pays O(T) to find events.
//! No wall-time, energy or learning advantage is established by these counts.

use super::boolean_edge_memory::BooleanEdgeProgram;
use super::boolean_indexed_fronts::{
    validate_indexed_fronts, IndexedFront, IndexedFrontError,
};

/// Any unsuccessful batch leaves the entire runtime unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WatermarkedError {
    NonIncreasingWatermark { previous: u64, next: u64 },
    Front(IndexedFrontError),
    LateFront { closed_through: u64, tick: u64 },
    ZeroQueryTick,
    NonIncreasingQuery { previous: u64, next: u64 },
    QueryBeyondWatermark { tick: u64, watermark: u64 },
    LateQuery { closed_through: u64, tick: u64 },
    CountOverflow,
}

/// One complete window's requested output vector and post-window state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WatermarkedWindow {
    /// In precisely the same order as the caller's query ticks.
    pub outputs: Vec<u64>,
    /// Includes all events through the watermark, even after the last query.
    pub final_state: u64,
    pub closed_through: u64,
    pub applied_fronts: usize,
    pub served_queries: usize,
}

/// State carried between closed watermark intervals, without retaining
/// any prior input samples or front vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WatermarkedEdgeRuntime {
    program: BooleanEdgeProgram,
    state: u64,
    closed_through: u64,
    total_fronts: usize,
    total_queries: usize,
    total_windows: usize,
}

impl WatermarkedEdgeRuntime {
    /// The baseline tick is already closed. No event at or before it may
    /// subsequently be ingested. The caller supplies memory at that point,
    /// not a baseline raw input word.
    #[must_use]
    pub const fn new(
        program: BooleanEdgeProgram,
        baseline_tick: u64,
        initial_state: u64,
    ) -> Self {
        Self {
            program,
            state: initial_state,
            closed_through: baseline_tick,
            total_fronts: 0,
            total_queries: 0,
            total_windows: 0,
        }
    }

    #[must_use]
    pub const fn state(&self) -> u64 {
        self.state
    }

    #[must_use]
    pub const fn closed_through(&self) -> u64 {
        self.closed_through
    }

    #[must_use]
    pub const fn total_fronts(&self) -> usize {
        self.total_fronts
    }

    #[must_use]
    pub const fn total_queries(&self) -> usize {
        self.total_queries
    }

    #[must_use]
    pub const fn total_windows(&self) -> usize {
        self.total_windows
    }

    /// Advance through the entire closed interval
    /// `(previous_watermark, watermark]`.
    ///
    /// `fronts` contains EVERY actual front in this interval, in strictly
    /// increasing tick order; `queries` contains desired observation ticks
    /// in the same interval, also strictly increasing. No event after the
    /// watermark may be supplied, and no event can arrive after its window
    /// closes. A front at tick `t` is applied BEFORE a query at `t`.
    ///
    /// The event publisher, not this API, is responsible for the truth and
    /// completeness of its watermark. This method only checks the syntactic
    /// contract, with no claim of authenticated hardware or message source.
    ///
    /// Input validation and capacity accounting complete *before* mutation.
    /// A malformed later batch does not corrupt previously accepted state.
    ///
    /// O(E+Q) execution for this batch, O(Q) returned memory, O(1)
    /// persistent memory. This excludes upstream event detection and the
    /// memory needed to present this batch of fronts/queries.
    pub fn advance_window(
        &mut self,
        watermark: u64,
        fronts: &[IndexedFront],
        queries: &[u64],
    ) -> Result<WatermarkedWindow, WatermarkedError> {
        if watermark <= self.closed_through {
            return Err(WatermarkedError::NonIncreasingWatermark {
                previous: self.closed_through,
                next: watermark,
            });
        }

        validate_indexed_fronts(fronts, watermark).map_err(WatermarkedError::Front)?;
        if let Some(front) = fronts.first() {
            if front.tick <= self.closed_through {
                return Err(WatermarkedError::LateFront {
                    closed_through: self.closed_through,
                    tick: front.tick,
                });
            }
        }

        let mut previous_query = 0;
        for &tick in queries {
            if tick == 0 {
                return Err(WatermarkedError::ZeroQueryTick);
            }
            if tick <= previous_query {
                return Err(WatermarkedError::NonIncreasingQuery {
                    previous: previous_query,
                    next: tick,
                });
            }
            if tick > watermark {
                return Err(WatermarkedError::QueryBeyondWatermark { tick, watermark });
            }
            if tick <= self.closed_through {
                return Err(WatermarkedError::LateQuery {
                    closed_through: self.closed_through,
                    tick,
                });
            }
            previous_query = tick;
        }

        let next_fronts = self
            .total_fronts
            .checked_add(fronts.len())
            .ok_or(WatermarkedError::CountOverflow)?;
        let next_queries = self
            .total_queries
            .checked_add(queries.len())
            .ok_or(WatermarkedError::CountOverflow)?;
        let next_windows = self
            .total_windows
            .checked_add(1)
            .ok_or(WatermarkedError::CountOverflow)?;

        let mut state = self.state;
        let mut event_index = 0;
        let mut outputs = Vec::with_capacity(queries.len());
        for &tick in queries {
            while event_index < fronts.len() && fronts[event_index].tick <= tick {
                state = self.program.transition(fronts[event_index].rising).apply(state);
                event_index += 1;
            }
            outputs.push(state & self.program.readout_mask);
        }
        for front in &fronts[event_index..] {
            state = self.program.transition(front.rising).apply(state);
        }

        self.state = state;
        self.closed_through = watermark;
        self.total_fronts = next_fronts;
        self.total_queries = next_queries;
        self.total_windows = next_windows;

        Ok(WatermarkedWindow {
            outputs,
            final_state: state,
            closed_through: watermark,
            applied_fronts: fronts.len(),
            served_queries: queries.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean_indexed_fronts::{
        certify_indexed_fronts_against_samples, sampled_indexed_fronts,
    };
    use crate::boolean_scan_lstm::UnaryGate;

    fn rotating(shift: u8, mask: u64) -> BooleanEdgeProgram {
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

    fn oracle_windows(
        program: BooleanEdgeProgram,
        inputs: &[u64],
        baseline: u64,
        initial: u64,
        front_source: &[IndexedFront],
        query_ticks: &[u64],
        watermarks: &[u64],
    ) {
        certify_indexed_fronts_against_samples(front_source, inputs, baseline).unwrap();
        let dense = program.forward_dense(inputs, baseline, initial);
        let indexed = program
            .forward_indexed_queries(front_source, query_ticks, inputs.len() as u64, initial)
            .unwrap();
        let mut runtime = WatermarkedEdgeRuntime::new(program, 0, initial);
        let mut consumed_fronts = 0;
        let mut consumed_queries = 0;
        let mut observed = Vec::new();
        for &watermark in watermarks {
            let next_fronts = front_source.partition_point(|item| item.tick <= watermark);
            let next_queries = query_ticks.partition_point(|&tick| tick <= watermark);
            let window = runtime
                .advance_window(
                    watermark,
                    &front_source[consumed_fronts..next_fronts],
                    &query_ticks[consumed_queries..next_queries],
                )
                .unwrap();
            observed.extend(window.outputs);
            assert_eq!(window.closed_through, watermark);
            assert_eq!(window.final_state, runtime.state());
            consumed_fronts = next_fronts;
            consumed_queries = next_queries;
        }
        assert_eq!(observed, indexed.outputs);
        assert_eq!(runtime.state(), indexed.final_state);
        assert_eq!(runtime.closed_through(), *watermarks.last().unwrap());
        assert_eq!(runtime.total_fronts(), front_source.len());
        assert_eq!(runtime.total_queries(), query_ticks.len());
        assert_eq!(runtime.total_windows(), watermarks.len());
        for (&query, &got) in query_ticks.iter().zip(&observed) {
            assert_eq!(got, dense[query as usize - 1], "query={query}");
        }
    }

    #[test]
    fn dense_oracle_exact_across_many_watermark_partitions() {
        let mut seed = 0x3b68_62f1_01cb_7de5u64;
        for length in [1usize, 2, 3, 7, 63, 64, 65, 127, 257] {
            for shift in [0u8, 1, 7, 31, 63] {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let program = BooleanEdgeProgram::new(
                    UnaryGate {
                        when_false: seed,
                        when_true: seed.rotate_left(19),
                    },
                    UnaryGate {
                        when_false: seed.rotate_right(11),
                        when_true: !seed,
                    },
                    seed.rotate_left(5),
                    shift,
                )
                .unwrap();
                let initial = seed.rotate_right(13);
                let baseline = seed.rotate_left(23);
                let mut current = baseline;
                let mut inputs = Vec::with_capacity(length);
                for index in 0..length {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    if index % 5 == 0 {
                        current = seed;
                    } else if index % 3 == 0 {
                        current = 0;
                    }
                    inputs.push(current);
                }
                let fronts = sampled_indexed_fronts(&inputs, baseline);
                let queries: Vec<_> = (1..=length as u64)
                    .filter(|tick| tick % 3 != 0 || tick % 11 == 0)
                    .collect();
                let mut cuts = vec![1u64];
                for cut in [2u64, 3, 7, 63, 64, 66, 128, 256] {
                    if cut < length as u64 && cut > *cuts.last().unwrap() {
                        cuts.push(cut);
                    }
                }
                if *cuts.last().unwrap() != length as u64 {
                    cuts.push(length as u64);
                }
                oracle_windows(
                    program,
                    &inputs,
                    baseline,
                    initial,
                    &fronts,
                    &queries,
                    &cuts,
                );
            }
        }
    }

    #[test]
    fn watermark_at_event_tick_is_visible_in_same_tick_query() {
        let mut runtime = WatermarkedEdgeRuntime::new(rotating(1, u64::MAX), 0, 1);
        let first = runtime
            .advance_window(10, &[IndexedFront { tick: 10, rising: 1 }], &[1, 10])
            .unwrap();
        assert_eq!(first.outputs, vec![1, 2]);
        let next = runtime
            .advance_window(12, &[IndexedFront { tick: 12, rising: 1 }], &[11, 12])
            .unwrap();
        assert_eq!(next.outputs, vec![2, 4]);
        assert_eq!(runtime.state(), 4);
    }

    #[test]
    fn no_output_windows_and_large_gaps_do_not_revisit_samples() {
        let mut runtime = WatermarkedEdgeRuntime::new(rotating(63, 0b1010), 1_000, 0b1111);
        let first = runtime.advance_window(1_000_000, &[], &[]).unwrap();
        assert_eq!(first.final_state, 0b1111);
        assert!(first.outputs.is_empty());
        let second = runtime
            .advance_window(
                u64::MAX,
                &[IndexedFront {
                    tick: u64::MAX,
                    rising: 1,
                }],
                &[1_000_001, u64::MAX],
            )
            .unwrap();
        assert_eq!(second.outputs, vec![0b1010, 0b0010]);
        assert_eq!(runtime.total_windows(), 2);
        assert_eq!(runtime.total_fronts(), 1);
        assert_eq!(runtime.total_queries(), 2);
    }

    #[test]
    fn invalid_window_is_atomic_including_after_prior_progress() {
        let mut runtime = WatermarkedEdgeRuntime::new(rotating(1, u64::MAX), 0, 1);
        runtime
            .advance_window(5, &[IndexedFront { tick: 2, rising: 1 }], &[3])
            .unwrap();
        let prior = runtime.clone();
        let invalid = [
            (
                5,
                vec![],
                vec![],
                WatermarkedError::NonIncreasingWatermark {
                    previous: 5,
                    next: 5,
                },
            ),
            (
                10,
                vec![IndexedFront { tick: 2, rising: 1 }],
                vec![],
                WatermarkedError::LateFront {
                    closed_through: 5,
                    tick: 2,
                },
            ),
            (
                10,
                vec![IndexedFront { tick: 6, rising: 0 }],
                vec![],
                WatermarkedError::Front(IndexedFrontError::EmptyFront { tick: 6 }),
            ),
            (
                10,
                vec![IndexedFront { tick: 12, rising: 1 }],
                vec![],
                WatermarkedError::Front(IndexedFrontError::EventBeyondHorizon {
                    tick: 12,
                    horizon: 10,
                }),
            ),
            (
                10,
                vec![IndexedFront { tick: 6, rising: 1 }],
                vec![5],
                WatermarkedError::LateQuery {
                    closed_through: 5,
                    tick: 5,
                },
            ),
            (
                10,
                vec![],
                vec![7, 7],
                WatermarkedError::NonIncreasingQuery {
                    previous: 7,
                    next: 7,
                },
            ),
            (
                10,
                vec![],
                vec![11],
                WatermarkedError::QueryBeyondWatermark {
                    tick: 11,
                    watermark: 10,
                },
            ),
            (10, vec![], vec![0], WatermarkedError::ZeroQueryTick),
        ];
        for (watermark, fronts, queries, expected) in invalid {
            assert_eq!(
                runtime.advance_window(watermark, &fronts, &queries),
                Err(expected)
            );
            assert_eq!(runtime, prior);
        }
        assert_eq!(
            runtime.advance_window(
                10,
                &[IndexedFront { tick: 6, rising: 1 }],
                &[10]
            )
            .unwrap()
            .outputs,
            vec![4]
        );
    }

    #[test]
    fn malformed_batch_does_not_accept_partial_valid_prefix() {
        let mut runtime = WatermarkedEdgeRuntime::new(rotating(7, u64::MAX), 0, 1);
        let before = runtime.clone();
        assert_eq!(
            runtime.advance_window(
                100,
                &[
                    IndexedFront { tick: 2, rising: 1 },
                    IndexedFront { tick: 50, rising: 1 },
                    IndexedFront { tick: 50, rising: 3 },
                ],
                &[1, 4],
            ),
            Err(WatermarkedError::Front(
                IndexedFrontError::NonIncreasingEvent {
                    previous: 50,
                    next: 50,
                }
            ))
        );
        assert_eq!(runtime, before);
    }

    #[test]
    fn front_source_completeness_cannot_be_deduced_from_watermarks() {
        let program = rotating(1, u64::MAX);
        let input = [1u64, 0, 1];
        let complete = sampled_indexed_fronts(&input, 0);
        assert_eq!(complete.len(), 2);
        // The synthetic "source" lies by omitting the second front:
        // this runtime will (correctly) only process what it was given.
        let mut runtime = WatermarkedEdgeRuntime::new(program, 0, 1);
        let incomplete = [complete[0]];
        let outcome = runtime.advance_window(3, &incomplete, &[3]).unwrap();
        assert_ne!(outcome.outputs, vec![program.forward_dense(&input, 0, 1)[2]]);
        assert_eq!(
            certify_indexed_fronts_against_samples(&incomplete, &input, 0),
            Err(IndexedFrontError::SampledFrontMismatch { tick: 3 })
        );
    }

    #[test]
    fn overflow_is_rejected_before_mutating_state() {
        let mut runtime = WatermarkedEdgeRuntime::new(rotating(1, 1), 0, 1);
        runtime.total_fronts = usize::MAX;
        let prior = runtime.clone();
        assert_eq!(
            runtime.advance_window(1, &[IndexedFront { tick: 1, rising: 1 }], &[]),
            Err(WatermarkedError::CountOverflow)
        );
        assert_eq!(runtime, prior);
    }
}
