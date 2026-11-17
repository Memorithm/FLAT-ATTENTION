//! BPS-3: event-gated recurrent Boolean memory driven by sampled rising edges.
//!
//! This is NOT bit-identical to BPS-2 for arbitrary raw inputs: only 0->1
//! sampled fronts enable a recurrent step. When no front occurs, the exact
//! transition is identity, including for nonzero recurrent rotations.
//! The 64-bit detector still samples every input tick: only downstream
//! recurrent transitions are skipped. No speed or joule claim is implied.

use super::boolean_rotating_memory::{
    scan_rotating_states, RotatingTransition, RotationError, ROTATING_LANES,
};
use super::boolean_scan_lstm::UnaryGate;
use super::rising_edge::{RisingEdgeDetector, RisingEdgeError};

/// Input-gated Boolean update, with a time-invariant readout mask.
///
/// Unlike BPS-2, the output is independent of the current raw input while
/// memory is idle; an input-dependent readout would invalidate skipping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BooleanEdgeProgram {
    pub forget: UnaryGate,
    pub write: UnaryGate,
    pub readout_mask: u64,
    shift: u8,
}

impl BooleanEdgeProgram {
    pub const fn new(
        forget: UnaryGate,
        write: UnaryGate,
        readout_mask: u64,
        shift: u8,
    ) -> Result<Self, RotationError> {
        if shift >= ROTATING_LANES {
            return Err(RotationError::InvalidShift { shift });
        }
        Ok(Self {
            forget,
            write,
            readout_mask,
            shift,
        })
    }

    #[must_use]
    pub const fn shift(self) -> u8 {
        self.shift
    }

    /// A zero-front tick is EXACT identity even if forget(0) would clear bits
    /// or the recurrent shift is nonzero. This defines BPS-3 semantics.
    #[must_use]
    pub fn transition(self, rising: u64) -> RotatingTransition {
        if rising == 0 {
            RotatingTransition::IDENTITY
        } else {
            RotatingTransition::new(
                self.forget.evaluate(rising),
                self.write.evaluate(rising),
                self.shift,
            )
            .expect("the program validates its recurrent shift")
        }
    }

    /// Dense oracle: evaluates and applies a transition on every sampled tick,
    /// including identity transitions on quiet ticks.
    #[must_use]
    pub fn forward_dense(
        self,
        inputs: &[u64],
        baseline_input: u64,
        initial_state: u64,
    ) -> Vec<u64> {
        let mut previous = baseline_input;
        let mut state = initial_state;
        inputs
            .iter()
            .map(|&current| {
                let rising = current & !previous;
                previous = current;
                state = self.transition(rising).apply(state);
                state & self.readout_mask
            })
            .collect()
    }

    /// Exact temporal scan of the same event-gated transition sequence.
    /// O(T log T) work, O(log T) dependency depth, O(T) memory.
    /// This does NOT skip identity transitions during scan construction.
    #[must_use]
    pub fn forward_scan(self, inputs: &[u64], baseline_input: u64, initial_state: u64) -> Vec<u64> {
        let mut previous = baseline_input;
        let transitions: Vec<RotatingTransition> = inputs
            .iter()
            .map(|&current| {
                let rising = current & !previous;
                previous = current;
                self.transition(rising)
            })
            .collect();
        scan_rotating_states(&transitions, initial_state)
            .into_iter()
            .map(|state| state & self.readout_mask)
            .collect()
    }

    /// Timestamped event-execution API. All observations must carry strictly
    /// increasing sample ticks; gaps are unsampled, not inferred as zeros.
    pub fn runtime(
        self,
        baseline_tick: u64,
        baseline_input: u64,
        initial_state: u64,
    ) -> Result<BooleanEdgeRuntime, RisingEdgeError> {
        BooleanEdgeRuntime::new(self, baseline_tick, baseline_input, initial_state)
    }

    /// Dense-sample input convenience path. Every sample is inspected for a
    /// front; only eventful samples execute the memory recurrence.
    pub fn forward_sparse(
        self,
        inputs: &[u64],
        baseline_input: u64,
        initial_state: u64,
    ) -> Result<EventRun, RisingEdgeError> {
        let mut runtime = self.runtime(0, baseline_input, initial_state)?;
        let mut outputs = Vec::with_capacity(inputs.len());
        for (index, &input) in inputs.iter().enumerate() {
            let tick = u64::try_from(index)
                .ok()
                .and_then(|value| value.checked_add(1))
                .expect("slice length cannot exceed representable sample ticks");
            outputs.push(runtime.sample(tick, input)?.output);
        }
        Ok(EventRun {
            outputs,
            activity: runtime.activity,
        })
    }
}

/// Cumulative per-sample work accounting. All values are logical counts,
/// not device execution traces or physical energy consumption.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EventActivity {
    pub sampled_steps: u128,
    pub eventful_steps: u128,
    pub skipped_identity_steps: u128,
    pub rising_bits: u128,
    pub evaluated_recurrent_steps: u128,
    pub state_flips: u128,
    pub output_flips: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeStep {
    pub tick: u64,
    pub rising: u64,
    pub output: u64,
    pub recurrent_transition_evaluated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventRun {
    pub outputs: Vec<u64>,
    pub activity: EventActivity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanEdgeRuntime {
    program: BooleanEdgeProgram,
    detector: RisingEdgeDetector,
    state: u64,
    activity: EventActivity,
}

impl BooleanEdgeRuntime {
    pub fn new(
        program: BooleanEdgeProgram,
        baseline_tick: u64,
        baseline_input: u64,
        initial_state: u64,
    ) -> Result<Self, RisingEdgeError> {
        Ok(Self {
            program,
            detector: RisingEdgeDetector::new(64, baseline_tick, &[baseline_input])?,
            state: initial_state,
            activity: EventActivity::default(),
        })
    }

    #[must_use]
    pub const fn state(&self) -> u64 {
        self.state
    }

    #[must_use]
    pub const fn activity(&self) -> EventActivity {
        self.activity
    }

    #[must_use]
    pub const fn last_tick(&self) -> u64 {
        self.detector.last_tick()
    }

    /// Invalid samples do not change detector, state, or work counters.
    pub fn sample(&mut self, tick: u64, current: u64) -> Result<EdgeStep, RisingEdgeError> {
        let mut rising = [0u64];
        self.detector.sample_into(tick, &[current], &mut rising)?;
        let fronts = rising[0];
        let prior_state = self.state;
        let prior_output = prior_state & self.program.readout_mask;
        let evaluated = fronts != 0;
        if evaluated {
            self.state = self.program.transition(fronts).apply(self.state);
            self.activity.eventful_steps += 1;
            self.activity.evaluated_recurrent_steps += 1;
        } else {
            // Deliberately do not calculate the rotate/forget/write path.
            self.activity.skipped_identity_steps += 1;
        }
        let output = self.state & self.program.readout_mask;
        self.activity.sampled_steps += 1;
        self.activity.rising_bits += u128::from(fronts.count_ones());
        self.activity.state_flips += u128::from((self.state ^ prior_state).count_ones());
        self.activity.output_flips += u128::from((output ^ prior_output).count_ones());
        Ok(EdgeStep {
            tick,
            rising: fronts,
            output,
            recurrent_transition_evaluated: evaluated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keep_all() -> UnaryGate {
        UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        }
    }

    fn zero_write() -> UnaryGate {
        UnaryGate::default()
    }

    fn keep_rotate(shift: u8) -> BooleanEdgeProgram {
        BooleanEdgeProgram::new(keep_all(), zero_write(), u64::MAX, shift).unwrap()
    }

    fn next_random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn held_high_baseline_never_invents_an_edge_or_rotates() {
        let program = keep_rotate(7);
        let inputs = [1u64; 16];
        let sparse = program.forward_sparse(&inputs, 1, 5).unwrap();
        assert_eq!(sparse.outputs, vec![5; inputs.len()]);
        assert_eq!(sparse.activity.sampled_steps, 16);
        assert_eq!(sparse.activity.skipped_identity_steps, 16);
        assert_eq!(sparse.activity.eventful_steps, 0);
        assert_eq!(sparse.activity.evaluated_recurrent_steps, 0);
        assert_eq!(sparse.activity.rising_bits, 0);
        assert_eq!(sparse.activity.state_flips, 0);
        assert_eq!(program.forward_dense(&inputs, 1, 5), sparse.outputs);
        assert_eq!(program.forward_scan(&inputs, 1, 5), sparse.outputs);
    }

    #[test]
    fn fall_rearms_and_second_rise_rotates_again_with_wraparound() {
        let program = keep_rotate(63);
        let inputs = [1u64, 1, 0, 1];
        let sparse = program.forward_sparse(&inputs, 0, 1).unwrap();
        let expected = vec![1u64 << 63, 1u64 << 63, 1u64 << 63, 1u64 << 62];
        assert_eq!(sparse.outputs, expected);
        assert_eq!(sparse.activity.rising_bits, 2);
        assert_eq!(sparse.activity.eventful_steps, 2);
        assert_eq!(sparse.activity.skipped_identity_steps, 2);
        assert_eq!(sparse.activity.state_flips, 4);
        assert_eq!(program.forward_dense(&inputs, 0, 1), expected);
        assert_eq!(program.forward_scan(&inputs, 0, 1), expected);
    }

    #[test]
    fn a_multi_bit_rise_is_one_recurrent_step_but_multiple_bit_events() {
        let program = keep_rotate(1);
        let inputs = [0b1011u64, 0b1011, 0b0011, 0b1011];
        let sparse = program.forward_sparse(&inputs, 0, 1).unwrap();
        assert_eq!(sparse.outputs, vec![2, 2, 2, 4]);
        assert_eq!(sparse.activity.eventful_steps, 2);
        assert_eq!(sparse.activity.rising_bits, 4);
        assert_eq!(sparse.activity.skipped_identity_steps, 2);
        assert_eq!(program.forward_dense(&inputs, 0, 1), sparse.outputs);
    }

    #[test]
    fn random_sparse_dense_scan_equivalence_with_arbitrary_gates_and_baselines() {
        let mut seed = 0xaef0_2244_6879_abceu64;
        for length in [0usize, 1, 2, 3, 5, 63, 64, 65, 127, 128, 257] {
            for shift in [0u8, 1, 7, 31, 63] {
                let program = BooleanEdgeProgram::new(
                    UnaryGate {
                        when_false: next_random(&mut seed),
                        when_true: next_random(&mut seed),
                    },
                    UnaryGate {
                        when_false: next_random(&mut seed),
                        when_true: next_random(&mut seed),
                    },
                    next_random(&mut seed),
                    shift,
                )
                .unwrap();
                let baseline = next_random(&mut seed);
                let initial = next_random(&mut seed);
                let inputs: Vec<_> = (0..length)
                    .map(|index| {
                        // Include quiet and falling ticks, not only iid samples.
                        if index % 3 == 0 {
                            0
                        } else if index % 3 == 1 {
                            u64::MAX
                        } else {
                            next_random(&mut seed)
                        }
                    })
                    .collect();
                let dense = program.forward_dense(&inputs, baseline, initial);
                let sparse = program.forward_sparse(&inputs, baseline, initial).unwrap();
                let scan = program.forward_scan(&inputs, baseline, initial);
                assert_eq!(dense, sparse.outputs, "length={length}, shift={shift}");
                assert_eq!(scan, dense, "length={length}, shift={shift}");
                assert_eq!(
                    sparse.activity.sampled_steps,
                    sparse.activity.eventful_steps + sparse.activity.skipped_identity_steps
                );
                assert_eq!(
                    sparse.activity.eventful_steps,
                    sparse.activity.evaluated_recurrent_steps
                );
            }
        }
    }

    #[test]
    fn invalid_tick_is_transactional_even_after_prior_events() {
        let program = keep_rotate(1);
        let mut runtime = program.runtime(100, 0, 1).unwrap();
        assert_eq!(runtime.sample(101, 1).unwrap().output, 2);
        let before = runtime.clone();
        assert_eq!(
            runtime.sample(101, 3),
            Err(RisingEdgeError::NonIncreasingTick {
                previous: 101,
                next: 101,
            })
        );
        assert_eq!(runtime, before);
        assert_eq!(
            runtime.sample(99, 0),
            Err(RisingEdgeError::NonIncreasingTick {
                previous: 101,
                next: 99,
            })
        );
        assert_eq!(runtime, before);
        assert_eq!(runtime.sample(104, 1).unwrap().rising, 0);
        assert_eq!(runtime.activity().eventful_steps, 1);
        assert_eq!(runtime.activity().skipped_identity_steps, 1);
    }

    #[test]
    fn no_front_is_identity_even_if_zero_input_gates_would_mutate_state() {
        let program = BooleanEdgeProgram::new(
            UnaryGate {
                when_false: 0,
                when_true: 0,
            },
            UnaryGate {
                when_false: u64::MAX,
                when_true: u64::MAX,
            },
            u64::MAX,
            1,
        )
        .unwrap();
        assert_eq!(program.transition(0), RotatingTransition::IDENTITY);
        assert_eq!(program.forward_dense(&[0, 0, 0], 0, 17), vec![17; 3]);
        assert_eq!(
            program.forward_sparse(&[0, 0, 0], 0, 17).unwrap().outputs,
            vec![17; 3]
        );
    }

    #[test]
    fn invalid_shift_is_rejected_before_runtime_construction() {
        assert_eq!(
            BooleanEdgeProgram::new(keep_all(), zero_write(), u64::MAX, 64),
            Err(RotationError::InvalidShift { shift: 64 })
        );
    }

    #[test]
    fn static_readout_mask_does_not_create_phantom_updates() {
        let program = BooleanEdgeProgram::new(keep_all(), zero_write(), 0b10, 1).unwrap();
        let sparse = program.forward_sparse(&[1, 1, 0, 1], 0, 1).unwrap();
        assert_eq!(sparse.outputs, vec![2, 2, 2, 0]);
        assert_eq!(sparse.activity.output_flips, 2);
        assert_eq!(program.forward_scan(&[1, 1, 0, 1], 0, 1), sparse.outputs);
    }
}
