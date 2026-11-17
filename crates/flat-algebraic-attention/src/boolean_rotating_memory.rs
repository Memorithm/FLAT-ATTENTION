//! BPS-2: scan-compatible recurrent cross-lane Boolean connectivity.
//!
//! A packed bit of memory is connected to exactly one predecessor bit via a
//! shared circular rotation. This is genuine state-to-state cross-lane
//! connectivity, unlike mixing only the current input. Its restricted
//! permutation topology preserves the exact associative scan contract.
//! It is not a classical LSTM, an arbitrary sparse matrix, or an energy claim.

use std::thread;

use super::boolean_scan_lstm::{
    validate_examples, BooleanGateProgram, BooleanTrainError, BooleanTrainingSequence,
};

/// Upper bound of the packed state word, shared with the BPS-1 program.
pub const ROTATING_LANES: u8 = 64;

/// Invalid recurrent permutation geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationError {
    InvalidShift { shift: u8 },
}

/// A Boolean memory transform of the form
/// (keep AND rotate_left(state, shift)) OR set.
///
/// Each output lane reads at most one previous-state lane. The shift is
/// represented explicitly so composition remains closed under rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotatingTransition {
    pub keep: u64,
    pub set: u64,
    shift: u8,
}

impl RotatingTransition {
    pub const IDENTITY: Self = Self {
        keep: u64::MAX,
        set: 0,
        shift: 0,
    };

    /// Validate the predecessor rotation. Shifts in 0..64 are supported.
    pub const fn new(keep: u64, set: u64, shift: u8) -> Result<Self, RotationError> {
        if shift >= ROTATING_LANES {
            return Err(RotationError::InvalidShift { shift });
        }
        Ok(Self { keep, set, shift })
    }

    #[must_use]
    pub const fn shift(self) -> u8 {
        self.shift
    }

    #[must_use]
    pub const fn apply(self, state: u64) -> u64 {
        (self.keep & state.rotate_left(self.shift as u32)) | self.set
    }

    /// Chronological composition: apply self, then next.
    ///
    /// rotate(a AND b, s) = rotate(a, s) AND rotate(b, s), so:
    /// keep = next.keep AND rotate(self.keep, next.shift)
    /// set  = next.set OR (next.keep AND rotate(self.set, next.shift))
    /// shift = (self.shift + next.shift) mod 64
    #[must_use]
    pub const fn then(self, next: Self) -> Self {
        let rotation = next.shift as u32;
        Self {
            keep: next.keep & self.keep.rotate_left(rotation),
            set: next.set | (next.keep & self.set.rotate_left(rotation)),
            shift: ((self.shift as u16 + next.shift as u16) % 64) as u8,
        }
    }
}

/// Inclusive associative scan of explicitly supplied rotating Boolean maps.
///
/// This shared BPS-2/BPS-3 core does not assume every timestep has an event.
/// Its identity transitions allow silent BPS-3 timesteps to retain exactly the
/// same state as an event-sparse implementation. This is an algorithmic
/// parallel-depth contract, not a speed or energy measurement.
#[must_use]
pub fn scan_rotating_states(transitions: &[RotatingTransition], initial_state: u64) -> Vec<u64> {
    let n = transitions.len();
    let mut prefixes = transitions.to_vec();
    let workers = thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let mut stride = 1usize;
    while stride < n {
        let mut next = vec![RotatingTransition::IDENTITY; n];
        if n >= 128 && workers > 1 {
            let chunk_size = n.div_ceil(workers);
            thread::scope(|scope| {
                for (chunk_index, chunk) in next.chunks_mut(chunk_size).enumerate() {
                    let previous = &prefixes;
                    scope.spawn(move || {
                        let start = chunk_index * chunk_size;
                        for (offset, result) in chunk.iter_mut().enumerate() {
                            let index = start + offset;
                            *result = if index < stride {
                                previous[index]
                            } else {
                                previous[index - stride].then(previous[index])
                            };
                        }
                    });
                }
            });
        } else {
            for index in 0..n {
                next[index] = if index < stride {
                    prefixes[index]
                } else {
                    prefixes[index - stride].then(prefixes[index])
                };
            }
        }
        prefixes = next;
        if stride > n / 2 {
            break;
        }
        stride *= 2;
    }
    prefixes
        .iter()
        .map(|transition| transition.apply(initial_state))
        .collect()
}

/// Research-only recurrent network with a shared trainable predecessor offset.
///
/// Gates are input-only (not state-dependent), permitting precomputation of
/// all per-time-step transitions before a temporal parallel scan. For a shift
/// other than zero, output lanes are connected across time, and the BPS-1
/// independent-lane gradient-free trainer MUST NOT be applied unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BooleanRotatingProgram {
    pub gates: BooleanGateProgram,
    recurrent_shift: u8,
}

impl BooleanRotatingProgram {
    pub const fn new(
        gates: BooleanGateProgram,
        recurrent_shift: u8,
    ) -> Result<Self, RotationError> {
        if recurrent_shift >= ROTATING_LANES {
            return Err(RotationError::InvalidShift {
                shift: recurrent_shift,
            });
        }
        Ok(Self {
            gates,
            recurrent_shift,
        })
    }

    #[must_use]
    pub const fn recurrent_shift(self) -> u8 {
        self.recurrent_shift
    }

    /// Return the proposed per-step rotating map without reading the state.
    #[must_use]
    pub const fn transition(self, input: u64) -> RotatingTransition {
        let gate = self.gates.transition(input);
        RotatingTransition {
            keep: gate.keep,
            set: gate.set,
            shift: self.recurrent_shift,
        }
    }

    /// Sequential reference, O(T) operations and O(1) auxiliary memory.
    #[must_use]
    pub fn forward_serial(&self, inputs: &[u64], initial_state: u64) -> Vec<u64> {
        let mut state = initial_state;
        inputs
            .iter()
            .copied()
            .map(|input| {
                state = self.transition(input).apply(state);
                state & self.gates.output.evaluate(input)
            })
            .collect()
    }

    /// Inclusive Hillis-Steele scan, exact for arbitrary step masks:
    /// O(T log T) work, O(log T) dependency depth and O(T) extra memory.
    ///
    /// Unlike a general recurrent layer, composed transitions stay in a
    /// constant-size (keep, set, shift) representation. Threading is
    /// opportunistic and may be slower than serial evaluation.
    #[must_use]
    pub fn forward_scan(&self, inputs: &[u64], initial_state: u64) -> Vec<u64> {
        let transitions: Vec<RotatingTransition> =
            inputs.iter().map(|&input| self.transition(input)).collect();
        scan_rotating_states(&transitions, initial_state)
            .into_iter()
            .zip(inputs.iter().copied())
            .map(|(state, input)| state & self.gates.output.evaluate(input))
            .collect()
    }

    /// Supervised Hamming loss over all labeled time steps and active lanes.
    pub fn hamming_loss(
        &self,
        examples: &[BooleanTrainingSequence<'_>],
        active_lanes: u64,
    ) -> Result<u64, BooleanTrainError> {
        validate_examples(examples, active_lanes)?;
        self.validated_loss(examples, active_lanes)
    }

    /// Exact bounded discrete search across all 64 possible recurrent shifts.
    ///
    /// Other gate parameters are held fixed. The original shift wins ties;
    /// otherwise the smallest shift with strictly lower training loss wins.
    /// Full enumeration is training work, NOT parallel training or a global
    /// optimum for joint connectivity and gate learning.
    pub fn fit_recurrent_shift(
        &mut self,
        examples: &[BooleanTrainingSequence<'_>],
        active_lanes: u64,
    ) -> Result<RotationFitReport, BooleanTrainError> {
        validate_examples(examples, active_lanes)?;
        let initial_loss = self.validated_loss(examples, active_lanes)?;
        let mut best_loss = initial_loss;
        let mut best_shift = self.recurrent_shift;
        for candidate_shift in 0..ROTATING_LANES {
            let candidate = Self {
                recurrent_shift: candidate_shift,
                ..*self
            };
            let loss = candidate.validated_loss(examples, active_lanes)?;
            if loss < best_loss {
                best_loss = loss;
                best_shift = candidate_shift;
            }
        }
        self.recurrent_shift = best_shift;
        Ok(RotationFitReport {
            initial_loss,
            final_loss: best_loss,
            selected_shift: best_shift,
            evaluated_shifts: ROTATING_LANES,
        })
    }

    /// Activity counters are logical event proxies, NOT joules or runtime.
    #[must_use]
    pub fn logical_activity(
        &self,
        inputs: &[u64],
        initial_state: u64,
        active_lanes: u64,
    ) -> RotatingActivity {
        let mut activity = RotatingActivity::default();
        let mut state = initial_state;
        let mut previous_output = 0;
        for &input in inputs {
            let transported = state.rotate_left(self.recurrent_shift as u32);
            let transition = self.transition(input);
            let next = transition.apply(state);
            let output = next & self.gates.output.evaluate(input);
            activity.active_predecessor_bits +=
                u128::from((transported & transition.keep & active_lanes).count_ones());
            activity.state_flips += u128::from(((next ^ state) & active_lanes).count_ones());
            activity.output_flips +=
                u128::from(((output ^ previous_output) & active_lanes).count_ones());
            state = next;
            previous_output = output;
        }
        activity
    }

    fn validated_loss(
        &self,
        examples: &[BooleanTrainingSequence<'_>],
        active_lanes: u64,
    ) -> Result<u64, BooleanTrainError> {
        let mut loss = 0u64;
        for example in examples {
            let mut state = example.initial_state;
            for (&input, &target) in example.inputs.iter().zip(example.targets.iter()) {
                state = self.transition(input).apply(state);
                let output = state & self.gates.output.evaluate(input);
                let errors = u64::from(((output ^ target) & active_lanes).count_ones());
                loss = loss
                    .checked_add(errors)
                    .ok_or(BooleanTrainError::LossOverflow)?;
            }
        }
        Ok(loss)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RotatingActivity {
    pub active_predecessor_bits: u128,
    pub state_flips: u128,
    pub output_flips: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotationFitReport {
    pub initial_loss: u64,
    pub final_loss: u64,
    pub selected_shift: u8,
    pub evaluated_shifts: u8,
}

#[cfg(test)]
mod tests {
    use super::super::boolean_scan_lstm::UnaryGate;
    use super::*;

    fn next_random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn keep_everything() -> BooleanGateProgram {
        BooleanGateProgram {
            forget: UnaryGate {
                when_false: u64::MAX,
                when_true: u64::MAX,
            },
            write: UnaryGate::default(),
            output: UnaryGate {
                when_false: u64::MAX,
                when_true: u64::MAX,
            },
        }
    }

    #[test]
    fn composing_rotated_maps_is_associative_and_ordered() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        for _ in 0..1024 {
            let a = RotatingTransition::new(
                next_random(&mut seed),
                next_random(&mut seed),
                (next_random(&mut seed) % 64) as u8,
            )
            .unwrap();
            let b = RotatingTransition::new(
                next_random(&mut seed),
                next_random(&mut seed),
                (next_random(&mut seed) % 64) as u8,
            )
            .unwrap();
            let c = RotatingTransition::new(
                next_random(&mut seed),
                next_random(&mut seed),
                (next_random(&mut seed) % 64) as u8,
            )
            .unwrap();
            let state = next_random(&mut seed);
            assert_eq!(a.then(b).then(c), a.then(b.then(c)));
            assert_eq!(
                a.then(b).then(c).apply(state),
                c.apply(b.apply(a.apply(state)))
            );
            assert_eq!(a.then(RotatingTransition::IDENTITY), a);
            assert_eq!(RotatingTransition::IDENTITY.then(a), a);
        }
    }

    #[test]
    fn shifted_memories_really_move_between_lanes() {
        let gates = keep_everything();
        let forward = BooleanRotatingProgram::new(gates, 1).unwrap();
        let backward = BooleanRotatingProgram::new(gates, 63).unwrap();
        assert_eq!(forward.forward_serial(&[0, 0, 0], 1), vec![2, 4, 8]);
        assert_eq!(
            backward.forward_serial(&[0, 0], 1),
            vec![1u64 << 63, 1u64 << 62]
        );
        assert_eq!(forward.forward_scan(&[0, 0, 0], 1), vec![2, 4, 8]);
    }

    #[test]
    fn zero_shift_recovers_existing_uncoupled_boolean_cell() {
        let mut seed = 0x1234_cafe_5678_3210u64;
        for _ in 0..32 {
            let gates = BooleanGateProgram {
                forget: UnaryGate {
                    when_false: next_random(&mut seed),
                    when_true: next_random(&mut seed),
                },
                write: UnaryGate {
                    when_false: next_random(&mut seed),
                    when_true: next_random(&mut seed),
                },
                output: UnaryGate {
                    when_false: next_random(&mut seed),
                    when_true: next_random(&mut seed),
                },
            };
            let initial = next_random(&mut seed);
            let inputs: Vec<u64> = (0..37).map(|_| next_random(&mut seed)).collect();
            let program = BooleanRotatingProgram::new(gates, 0).unwrap();
            assert_eq!(
                program.forward_serial(&inputs, initial),
                gates.forward_serial(&inputs, initial)
            );
            assert_eq!(
                program.forward_scan(&inputs, initial),
                gates.forward_scan(&inputs, initial)
            );
        }
    }

    #[test]
    fn temporal_scan_matches_serial_for_shifted_random_masks_and_all_lengths() {
        let mut seed = 0xfeed_beef_7123_4567u64;
        for length in [0usize, 1, 2, 3, 5, 31, 64, 65, 127, 128, 257] {
            let inputs: Vec<u64> = (0..length).map(|_| next_random(&mut seed)).collect();
            for shift in [0, 1, 7, 31, 63] {
                let gates = BooleanGateProgram {
                    forget: UnaryGate {
                        when_false: next_random(&mut seed),
                        when_true: next_random(&mut seed),
                    },
                    write: UnaryGate {
                        when_false: next_random(&mut seed),
                        when_true: next_random(&mut seed),
                    },
                    output: UnaryGate {
                        when_false: next_random(&mut seed),
                        when_true: next_random(&mut seed),
                    },
                };
                let program = BooleanRotatingProgram::new(gates, shift).unwrap();
                let initial = next_random(&mut seed);
                assert_eq!(
                    program.forward_scan(&inputs, initial),
                    program.forward_serial(&inputs, initial),
                    "length={length}, shift={shift}"
                );
            }
        }
    }

    #[test]
    fn exact_discrete_search_learns_cross_lane_memory_permutation() {
        let teacher = BooleanRotatingProgram::new(keep_everything(), 7).unwrap();
        let inputs = [0u64; 5];
        let initial_state = 1u64;
        let targets = teacher.forward_serial(&inputs, initial_state);
        let dataset = [BooleanTrainingSequence {
            inputs: &inputs,
            targets: &targets,
            initial_state,
        }];
        let mut model = BooleanRotatingProgram::new(keep_everything(), 0).unwrap();
        let report = model.fit_recurrent_shift(&dataset, u64::MAX).unwrap();
        assert!(report.initial_loss > 0);
        assert_eq!(report.final_loss, 0);
        assert_eq!(report.selected_shift, 7);
        assert_eq!(report.evaluated_shifts, 64);
        assert_eq!(model.forward_scan(&inputs, initial_state), targets);
    }

    #[test]
    fn invalid_geometry_and_training_sequences_are_rejected() {
        assert_eq!(
            RotatingTransition::new(0, 0, 64),
            Err(RotationError::InvalidShift { shift: 64 })
        );
        assert_eq!(
            BooleanRotatingProgram::new(keep_everything(), 64),
            Err(RotationError::InvalidShift { shift: 64 })
        );
        let mut program = BooleanRotatingProgram::new(keep_everything(), 1).unwrap();
        assert_eq!(
            program.fit_recurrent_shift(&[], u64::MAX),
            Err(BooleanTrainError::EmptyDataset)
        );
        let bad = BooleanTrainingSequence {
            inputs: &[0, 1],
            targets: &[0],
            initial_state: 0,
        };
        assert_eq!(
            program.hamming_loss(&[bad], 1),
            Err(BooleanTrainError::LengthMismatch { example_index: 0 })
        );
    }

    #[test]
    fn event_activity_is_an_explicit_non_energy_account() {
        let model = BooleanRotatingProgram::new(keep_everything(), 1).unwrap();
        let events = model.logical_activity(&[0, 0, 0], 1, u64::MAX);
        assert_eq!(events.active_predecessor_bits, 3);
        assert_eq!(events.state_flips, 6);
        assert_eq!(events.output_flips, 5);
    }
}
