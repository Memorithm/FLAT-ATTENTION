//! BPS-7: direct indexed-front, sparse-query Boolean recurrence.
//!
//! When the *input source* already supplies a trustworthy, exact rising-front
//! stream, BPS-3's recurrence can advance in O(E + Q) packed-word work,
//! where E is the number of eventful ticks and Q the number of requested
//! observations. There is no scan over the unobserved timeline.
//!
//! This is a DIFFERENT input contract from BPS-3's sampled-word detector:
//! missing events cannot be inferred from timestamps. A caller-supplied front
//! feed can only be identified as equivalent to dense sampled inputs after
//! independent front-source verification. No speed, energy, or learned-model
//! quality claim follows merely from the operation count.

use super::boolean_edge_memory::BooleanEdgeProgram;

/// Rising bits at an absolute sampled tick, numbered starting at 1.
/// These are FRONT masks (0 -> 1), NOT raw sampled values or pulse counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedFront {
    pub tick: u64,
    pub rising: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexedFrontError {
    ZeroEventTick,
    ZeroQueryTick,
    EmptyFront { tick: u64 },
    EventBeyondHorizon { tick: u64, horizon: u64 },
    QueryBeyondHorizon { tick: u64, horizon: u64 },
    NonIncreasingEvent { previous: u64, next: u64 },
    NonIncreasingQuery { previous: u64, next: u64 },
    SampledFrontMismatch { tick: u64 },
}

/// Output only at the explicit requested query ticks, in query order.
/// The end state incorporates ALL supplied events through horizon even if
/// no output is requested after the last event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedQueryRun {
    pub outputs: Vec<u64>,
    pub final_state: u64,
    pub executed_eventful_steps: usize,
    pub served_queries: usize,
    /// The event-only execution never inspects dense input words.
    /// This excludes any upstream cost for constructing/validating the feed.
    pub sampled_input_words_inspected: usize,
}

/// Validate an externally declared front stream *without* pretending
/// to authenticate the source that produced it.
pub fn validate_indexed_fronts(
    fronts: &[IndexedFront],
    horizon: u64,
) -> Result<(), IndexedFrontError> {
    let mut previous_tick = 0;
    for front in fronts {
        if front.tick == 0 {
            return Err(IndexedFrontError::ZeroEventTick);
        }
        if front.tick <= previous_tick {
            return Err(IndexedFrontError::NonIncreasingEvent {
                previous: previous_tick,
                next: front.tick,
            });
        }
        if front.tick > horizon {
            return Err(IndexedFrontError::EventBeyondHorizon {
                tick: front.tick,
                horizon,
            });
        }
        if front.rising == 0 {
            return Err(IndexedFrontError::EmptyFront { tick: front.tick });
        }
        previous_tick = front.tick;
    }
    Ok(())
}

fn validate_queries(queries: &[u64], horizon: u64) -> Result<(), IndexedFrontError> {
    let mut previous_tick = 0;
    for &tick in queries {
        if tick == 0 {
            return Err(IndexedFrontError::ZeroQueryTick);
        }
        if tick <= previous_tick {
            return Err(IndexedFrontError::NonIncreasingQuery {
                previous: previous_tick,
                next: tick,
            });
        }
        if tick > horizon {
            return Err(IndexedFrontError::QueryBeyondHorizon { tick, horizon });
        }
        previous_tick = tick;
    }
    Ok(())
}

impl BooleanEdgeProgram {
    /// Consume a pre-indexed front feed and evaluate the BPS-3 recurrence
    /// **without accessing a dense input time series**.
    ///
    /// The caller guarantees that the feed contains every real rising-front
    /// tick with its exact bits; validation checks ordering/bounds, but cannot
    /// detect an omitted physical or sampled event. Use
    /// `certify_indexed_fronts_against_samples` for an independent sampled
    /// reference, at the explicit cost of O(T) verification.
    ///
    /// A front at tick t affects the result queried *at* t.
    /// All other ticks are exact identity transitions. Output is static
    /// `state & readout_mask` and does not depend on raw inputs.
    ///
    /// E = fronts.len(), Q = queries.len(): O(E+Q) execution and O(Q)
    /// output memory. Offline source validation and output-materialization
    /// to every sampled tick are excluded; not a physical speedup claim.
    pub fn forward_indexed_queries(
        self,
        fronts: &[IndexedFront],
        queries: &[u64],
        horizon: u64,
        initial_state: u64,
    ) -> Result<IndexedQueryRun, IndexedFrontError> {
        validate_indexed_fronts(fronts, horizon)?;
        validate_queries(queries, horizon)?;

        let mut state = initial_state;
        let mut event_index = 0;
        let mut outputs = Vec::with_capacity(queries.len());
        for &tick in queries {
            while event_index < fronts.len() && fronts[event_index].tick <= tick {
                state = self.transition(fronts[event_index].rising).apply(state);
                event_index += 1;
            }
            outputs.push(state & self.readout_mask);
        }
        // Include events after the final observed query in the final state.
        for front in &fronts[event_index..] {
            state = self.transition(front.rising).apply(state);
        }
        Ok(IndexedQueryRun {
            outputs,
            final_state: state,
            executed_eventful_steps: fronts.len(),
            served_queries: queries.len(),
            sampled_input_words_inspected: 0,
        })
    }
}

/// Create a reference front feed from every explicitly sampled input word.
/// This convenience function necessarily costs O(T) and O(E) memory.
#[must_use]
pub fn sampled_indexed_fronts(inputs: &[u64], baseline_input: u64) -> Vec<IndexedFront> {
    let mut previous = baseline_input;
    let mut events = Vec::new();
    for (index, &current) in inputs.iter().enumerate() {
        let rising = current & !previous;
        previous = current;
        if rising != 0 {
            let tick = u64::try_from(index)
                .ok()
                .and_then(|n| n.checked_add(1))
                .expect("sampled input length must fit a u64 tick");
            events.push(IndexedFront { tick, rising });
        }
    }
    events
}

/// Independently prove that a caller-supplied feed represents *all* fronts
/// of a particular sampled reference (including baseline and re-arming).
///
/// This never infers a fall, re-arming or quiet tick from missing timestamps.
/// It requires O(T+E) verification and is intended for qualification rather
/// than the event-only serving path. A successfully certified feed only
/// proves equivalence to the given sampled inputs, not the physical source.
pub fn certify_indexed_fronts_against_samples(
    fronts: &[IndexedFront],
    inputs: &[u64],
    baseline_input: u64,
) -> Result<(), IndexedFrontError> {
    let horizon = u64::try_from(inputs.len()).expect("sample count must fit u64");
    validate_indexed_fronts(fronts, horizon)?;
    let mut previous = baseline_input;
    let mut event_index = 0;
    for (index, &current) in inputs.iter().enumerate() {
        let rising = current & !previous;
        previous = current;
        let tick = u64::try_from(index)
            .ok()
            .and_then(|n| n.checked_add(1))
            .expect("sample count must fit u64");
        if rising != 0 {
            if fronts.get(event_index).copied() != Some(IndexedFront { tick, rising }) {
                return Err(IndexedFrontError::SampledFrontMismatch { tick });
            }
            event_index += 1;
        } else if fronts
            .get(event_index)
            .is_some_and(|front| front.tick == tick)
        {
            return Err(IndexedFrontError::SampledFrontMismatch { tick });
        }
    }
    if event_index != fronts.len() {
        // This path is unreachable for a well-ordered feed inside the horizon,
        // but fail closed if input validation semantics change in the future.
        return Err(IndexedFrontError::SampledFrontMismatch { tick: horizon });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean_scan_lstm::UnaryGate;

    fn memory(shift: u8, output_mask: u64) -> BooleanEdgeProgram {
        BooleanEdgeProgram::new(
            UnaryGate {
                when_false: u64::MAX,
                when_true: u64::MAX,
            },
            UnaryGate::default(),
            output_mask,
            shift,
        )
        .unwrap()
    }

    fn teach(inputs: &[u64], baseline: u64, initial: u64, shift: u8, mask: u64) {
        let program = memory(shift, mask);
        let fronts = sampled_indexed_fronts(inputs, baseline);
        certify_indexed_fronts_against_samples(&fronts, inputs, baseline).unwrap();
        let queries: Vec<u64> = (1..=inputs.len() as u64).collect();
        let run = program
            .forward_indexed_queries(&fronts, &queries, inputs.len() as u64, initial)
            .unwrap();
        assert_eq!(
            run.outputs,
            program.forward_dense(inputs, baseline, initial)
        );
        assert_eq!(
            run.outputs,
            program.forward_event_scan(inputs, baseline, initial)
        );
        let expected_end = fronts.iter().fold(initial, |state, front| {
            program.transition(front.rising).apply(state)
        });
        assert_eq!(run.final_state, expected_end);
        assert_eq!(run.executed_eventful_steps, fronts.len());
        assert_eq!(run.sampled_input_words_inspected, 0);
    }

    #[test]
    fn sampled_and_indexed_full_query_oracles_match_rearming_and_baselines() {
        for shift in [0, 1, 7, 63] {
            teach(&[], 0, 1, shift, u64::MAX);
            teach(&[1, 1, 0, 1, 1, 0, 1], 0, 1, shift, u64::MAX);
            teach(&[1, 1, 0, 1, 0, 1], 1, 0b101, shift, u64::MAX);
            teach(&[u64::MAX, 0, u64::MAX, 0], u64::MAX, 3, shift, 0b101);
        }
    }

    #[test]
    fn sampled_and_indexed_sparse_queries_match_when_fronts_between_queries() {
        let program = memory(7, u64::MAX);
        let input = [1u64, 1, 0, 1, 0, 1, 1, 0, 1];
        let fronts = sampled_indexed_fronts(&input, 0);
        let query_ticks = [1, 4, 7, 9];
        let run = program
            .forward_indexed_queries(&fronts, &query_ticks, 9, 1)
            .unwrap();
        let reference = program.forward_dense(&input, 0, 1);
        assert_eq!(
            run.outputs,
            query_ticks
                .iter()
                .map(|&t| reference[(t - 1) as usize])
                .collect::<Vec<_>>()
        );
        assert_eq!(run.final_state, *reference.last().unwrap());
        assert_eq!(run.served_queries, 4);
        assert_eq!(run.executed_eventful_steps, 4);
    }

    #[test]
    fn no_queries_still_evaluates_all_supplied_events_through_horizon() {
        let program = memory(1, u64::MAX);
        let fronts = [
            IndexedFront { tick: 1, rising: 1 },
            IndexedFront {
                tick: 1_000_000,
                rising: 1,
            },
        ];
        let run = program
            .forward_indexed_queries(&fronts, &[], 1_000_000, 1)
            .unwrap();
        assert!(run.outputs.is_empty());
        assert_eq!(run.final_state, 4);
        assert_eq!(run.executed_eventful_steps, 2);
        assert_eq!(run.sampled_input_words_inspected, 0);
    }

    #[test]
    fn zero_event_long_horizon_has_no_dense_inspection() {
        let program = memory(63, 0b1010);
        let run = program
            .forward_indexed_queries(&[], &[1, 4, 1_000_000], 1_000_000, 0b1111)
            .unwrap();
        assert_eq!(run.outputs, vec![0b1010; 3]);
        assert_eq!(run.final_state, 0b1111);
        assert_eq!(run.executed_eventful_steps, 0);
        assert_eq!(run.sampled_input_words_inspected, 0);
    }

    #[test]
    fn certification_detects_missing_extra_or_incorrect_front_bits() {
        let inputs = [1u64, 1, 0, 1, 0, 0, 3];
        let certified = sampled_indexed_fronts(&inputs, 0);
        assert_eq!(certified.len(), 3);
        assert_eq!(
            certify_indexed_fronts_against_samples(&certified, &inputs, 0),
            Ok(())
        );
        let mut missing = certified.clone();
        missing.remove(1);
        assert_eq!(
            certify_indexed_fronts_against_samples(&missing, &inputs, 0),
            Err(IndexedFrontError::SampledFrontMismatch { tick: 4 })
        );
        let mut incorrect = certified.clone();
        incorrect[2].rising = 0b10;
        assert_eq!(
            certify_indexed_fronts_against_samples(&incorrect, &inputs, 0),
            Err(IndexedFrontError::SampledFrontMismatch { tick: 7 })
        );
        let extra = [
            IndexedFront { tick: 1, rising: 1 },
            IndexedFront { tick: 2, rising: 2 },
        ];
        assert_eq!(
            certify_indexed_fronts_against_samples(&extra, &inputs, 0),
            Err(IndexedFrontError::SampledFrontMismatch { tick: 2 })
        );
    }

    #[test]
    fn malformed_event_ticks_query_ticks_and_fronts_fail_closed() {
        let program = memory(1, u64::MAX);
        let base = IndexedFront { tick: 2, rising: 1 };
        for (fronts, expected) in [
            (
                vec![IndexedFront { tick: 0, rising: 1 }],
                IndexedFrontError::ZeroEventTick,
            ),
            (
                vec![IndexedFront { tick: 2, rising: 0 }],
                IndexedFrontError::EmptyFront { tick: 2 },
            ),
            (
                vec![base, base],
                IndexedFrontError::NonIncreasingEvent {
                    previous: 2,
                    next: 2,
                },
            ),
            (
                vec![IndexedFront { tick: 5, rising: 1 }],
                IndexedFrontError::EventBeyondHorizon {
                    tick: 5,
                    horizon: 4,
                },
            ),
        ] {
            assert_eq!(
                program.forward_indexed_queries(&fronts, &[], 4, 1),
                Err(expected)
            );
        }
        for (queries, expected) in [
            (vec![0], IndexedFrontError::ZeroQueryTick),
            (
                vec![1, 1],
                IndexedFrontError::NonIncreasingQuery {
                    previous: 1,
                    next: 1,
                },
            ),
            (
                vec![5],
                IndexedFrontError::QueryBeyondHorizon {
                    tick: 5,
                    horizon: 4,
                },
            ),
        ] {
            assert_eq!(
                program.forward_indexed_queries(&[base], &queries, 4, 1),
                Err(expected)
            );
        }
    }

    #[test]
    fn arbitrary_masks_dense_oracles_hold_on_many_deterministic_streams() {
        let mut seed = 0x8b58_1234_9980_eb11u64;
        for len in [0usize, 1, 2, 65, 127, 256] {
            for shift in [0, 1, 7, 31, 63] {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let program = BooleanEdgeProgram::new(
                    UnaryGate {
                        when_false: seed,
                        when_true: seed.rotate_left(19),
                    },
                    UnaryGate {
                        when_false: seed.rotate_right(9),
                        when_true: !seed,
                    },
                    seed.rotate_left(7),
                    shift,
                )
                .unwrap();
                let initial = seed.rotate_right(11);
                let baseline = seed.rotate_left(27);
                let mut current = baseline;
                let mut inputs = Vec::new();
                for tick in 0..len {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    if tick % 5 == 0 {
                        current = seed;
                    } else if tick % 3 == 0 {
                        current = 0;
                    }
                    inputs.push(current);
                }
                let dense = program.forward_dense(&inputs, baseline, initial);
                let fronts = sampled_indexed_fronts(&inputs, baseline);
                certify_indexed_fronts_against_samples(&fronts, &inputs, baseline).unwrap();
                let queries: Vec<_> = (1..=len as u64).collect();
                let event_run = program
                    .forward_indexed_queries(&fronts, &queries, len as u64, initial)
                    .unwrap();
                assert_eq!(event_run.outputs, dense, "len={len} shift={shift}");
                let reference_end = fronts.iter().fold(initial, |state, event| {
                    program.transition(event.rising).apply(state)
                });
                assert_eq!(event_run.final_state, reference_end);
            }
        }
    }
}
