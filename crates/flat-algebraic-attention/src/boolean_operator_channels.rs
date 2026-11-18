//! BPS-15: one recurrent predecessor connection per Boolean operator.
//!
//! Four independently routed channels perform retention (AND), excitation
//! (OR), inhibition (AND NOT), and toggling (XOR). Every channel uses a
//! trainable circular source offset and an input-dependent lane mask.
//! BPS-15 is a CPU-only, bit-packed research model, not a classical LSTM.
//! More fan-in adds work: throughput gains require physical measurement.

use super::boolean_rotating_memory::{RotationError, ROTATING_LANES};
use super::boolean_scan_lstm::{
    validate_examples, BooleanTrainError, BooleanTrainingSequence, UnaryGate,
};

/// Recurrent source selection, in deterministic operation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorChannel {
    Retain,
    Excite,
    Inhibit,
    Toggle,
}

impl OperatorChannel {
    pub const ALL: [Self; 4] = [Self::Retain, Self::Excite, Self::Inhibit, Self::Toggle];

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Retain => 0,
            Self::Excite => 1,
            Self::Inhibit => 2,
            Self::Toggle => 3,
        }
    }
}

/// Four incoming edges, plus direct write and output gates.
///
/// Every gate owns 64 input-dependent Boolean truth-table bits per value.
/// A gate with all zero bits is disabled. The four shift offsets (0..63)
/// are shared across destination lanes; per-lane arbitrary edges are NOT
/// implemented in this slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BooleanOperatorProgram {
    pub retain: UnaryGate,
    pub excite: UnaryGate,
    pub inhibit: UnaryGate,
    pub toggle: UnaryGate,
    pub write: UnaryGate,
    pub output: UnaryGate,
    shifts: [u8; 4],
}

impl BooleanOperatorProgram {
    pub const fn new(
        masks: [UnaryGate; 4],
        write: UnaryGate,
        output: UnaryGate,
        shifts: [u8; 4],
    ) -> Result<Self, RotationError> {
        let mut index = 0usize;
        while index < shifts.len() {
            if shifts[index] >= ROTATING_LANES {
                return Err(RotationError::InvalidShift {
                    shift: shifts[index],
                });
            }
            index += 1;
        }
        Ok(Self {
            retain: masks[0],
            excite: masks[1],
            inhibit: masks[2],
            toggle: masks[3],
            write,
            output,
            shifts,
        })
    }

    #[must_use]
    pub const fn shifts(self) -> [u8; 4] {
        self.shifts
    }

    pub const fn with_channel_shift(
        mut self,
        channel: OperatorChannel,
        shift: u8,
    ) -> Result<Self, RotationError> {
        if shift >= ROTATING_LANES {
            return Err(RotationError::InvalidShift { shift });
        }
        self.shifts[channel.index()] = shift;
        Ok(self)
    }

    #[must_use]
    pub const fn masks(self, input: u64) -> [u64; 4] {
        [
            self.retain.evaluate(input),
            self.excite.evaluate(input),
            self.inhibit.evaluate(input),
            self.toggle.evaluate(input),
        ]
    }

    /// Branchless reference: evaluate all four routed predecessors.
    ///
    /// All sources read the SAME old state. The order matters:
    ///   base = retain AND-source OR excite OR-source OR direct write
    ///   state' = (base AND NOT inhibit-source) XOR toggle-source
    ///   output = state' AND output(input)
    ///
    /// The channel names describe their algebraic roles, not physical
    /// hardware execution units. Each operation acts over 64 packed bits.
    #[must_use]
    pub const fn step_dense(self, input: u64, state: u64) -> (u64, u64) {
        let masks = self.masks(input);
        let retained = state.rotate_left(self.shifts[0] as u32) & masks[0];
        let excited = state.rotate_left(self.shifts[1] as u32) & masks[1];
        let inhibited = state.rotate_left(self.shifts[2] as u32) & masks[2];
        let toggled = state.rotate_left(self.shifts[3] as u32) & masks[3];
        let base = retained | excited | self.write.evaluate(input);
        let next = (base & !inhibited) ^ toggled;
        (next, next & self.output.evaluate(input))
    }

    /// Skip the source rotation of any channel whose input-gated word is zero.
    ///
    /// This computes the SAME state as `step_dense`. Branch overhead can
    /// make it slower on dense inputs; this is an option, not a speed claim.
    #[must_use]
    pub fn step_sparse(self, input: u64, state: u64) -> (u64, u64) {
        let masks = self.masks(input);
        let mut contributions = [0u64; 4];
        for index in 0..4 {
            if masks[index] != 0 {
                contributions[index] = state.rotate_left(self.shifts[index] as u32) & masks[index];
            }
        }
        let base = contributions[0] | contributions[1] | self.write.evaluate(input);
        let next = (base & !contributions[2]) ^ contributions[3];
        (next, next & self.output.evaluate(input))
    }

    /// All output timesteps, sequential O(T) and O(1) auxiliary state.
    #[must_use]
    pub fn forward_dense(self, inputs: &[u64], initial: u64) -> Vec<u64> {
        let mut state = initial;
        inputs
            .iter()
            .map(|&input| {
                let (next, output) = self.step_dense(input, state);
                state = next;
                output
            })
            .collect()
    }

    /// Exact same recurrence, with conditional channel activation.
    #[must_use]
    pub fn forward_sparse(self, inputs: &[u64], initial: u64) -> Vec<u64> {
        let mut state = initial;
        inputs
            .iter()
            .map(|&input| {
                let (next, output) = self.step_sparse(input, state);
                state = next;
                output
            })
            .collect()
    }

    /// Logical activity proxies; counts are NOT CPU retired instructions,
    /// memory reads, wall-clock time or physical joules.
    #[must_use]
    pub fn logical_activity(self, inputs: &[u64]) -> OperatorActivity {
        let mut enabled = [0u128; 4];
        for &input in inputs {
            let masks = self.masks(input);
            for (index, &mask) in masks.iter().enumerate() {
                if mask != 0 {
                    enabled[index] += 1;
                }
            }
        }
        let sampled = inputs.len() as u128;
        OperatorActivity {
            sampled,
            enabled_rotations: enabled,
            skipped_rotations: enabled.map(|count| sampled - count),
        }
    }

    /// Whole-trajectory training Hamming loss over selected output bits.
    pub fn hamming_loss(
        self,
        examples: &[BooleanTrainingSequence<'_>],
        active_output_bits: u64,
    ) -> Result<u64, BooleanTrainError> {
        validate_examples(examples, active_output_bits)?;
        self.loss_validated(examples, active_output_bits)
    }

    fn loss_validated(
        self,
        examples: &[BooleanTrainingSequence<'_>],
        active_output_bits: u64,
    ) -> Result<u64, BooleanTrainError> {
        let mut loss = 0u64;
        for example in examples {
            let mut state = example.initial_state;
            for (&input, &target) in example.inputs.iter().zip(example.targets.iter()) {
                let (next, output) = self.step_dense(input, state);
                state = next;
                let errors = u64::from(((output ^ target) & active_output_bits).count_ones());
                loss = loss
                    .checked_add(errors)
                    .ok_or(BooleanTrainError::LossOverflow)?;
            }
        }
        Ok(loss)
    }

    /// Bounded coordinate search of the 4 operation-specific source offsets.
    ///
    /// Only the TRAINING trajectories select improvements. A held-out set
    /// is evaluated before and after but is never used for selection, ties
    /// or early stopping. Gate masks remain fixed. No BPTT or global
    /// optimality claim; a plateau may require multi-coordinate moves.
    ///
    /// All mutations are committed only after validating the full result,
    /// so errors leave the original program unmodified.
    pub fn fit_shifts(
        &mut self,
        training: &[BooleanTrainingSequence<'_>],
        validation: &[BooleanTrainingSequence<'_>],
        active_output_bits: u64,
        policy: OperatorTrainPolicy,
    ) -> Result<OperatorFitReport, OperatorFitError> {
        validate_examples(training, active_output_bits).map_err(OperatorFitError::Training)?;
        validate_examples(validation, active_output_bits).map_err(OperatorFitError::Validation)?;
        if policy.max_epochs == 0 {
            return Err(OperatorFitError::Training(BooleanTrainError::ZeroEpochs));
        }
        if policy.max_candidates == 0 {
            return Err(OperatorFitError::ZeroCandidateBudget);
        }

        let initial_training = self
            .loss_validated(training, active_output_bits)
            .map_err(OperatorFitError::Training)?;
        let initial_validation = self
            .loss_validated(validation, active_output_bits)
            .map_err(OperatorFitError::Validation)?;
        let mut current = *self;
        let mut loss = initial_training;
        let mut candidates = 0usize;
        let mut accepted = 0usize;
        let mut epochs = 0usize;
        let mut exhausted = false;

        'search: for _ in 0..policy.max_epochs {
            if loss == 0 {
                break;
            }
            epochs += 1;
            let mut changed = false;
            for channel in OperatorChannel::ALL {
                let index = channel.index();
                for shift in 0..ROTATING_LANES {
                    if shift == current.shifts[index] {
                        continue;
                    }
                    if candidates == policy.max_candidates {
                        exhausted = true;
                        break 'search;
                    }
                    let mut candidate = current;
                    candidate.shifts[index] = shift;
                    let candidate_loss = candidate
                        .loss_validated(training, active_output_bits)
                        .map_err(OperatorFitError::Training)?;
                    candidates += 1;
                    if candidate_loss < loss {
                        current = candidate;
                        loss = candidate_loss;
                        accepted += 1;
                        changed = true;
                    }
                    if loss == 0 {
                        break 'search;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let final_validation = current
            .loss_validated(validation, active_output_bits)
            .map_err(OperatorFitError::Validation)?;
        let report = OperatorFitReport {
            train_loss_before: initial_training,
            train_loss_after: loss,
            validation_loss_before: initial_validation,
            validation_loss_after: final_validation,
            evaluated_candidates: candidates,
            accepted_changes: accepted,
            epochs_started: epochs,
            budget_exhausted: exhausted,
            shifts: current.shifts,
        };
        *self = current;
        Ok(report)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperatorActivity {
    pub sampled: u128,
    pub enabled_rotations: [u128; 4],
    pub skipped_rotations: [u128; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorTrainPolicy {
    pub max_epochs: usize,
    pub max_candidates: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorFitError {
    Training(BooleanTrainError),
    Validation(BooleanTrainError),
    ZeroCandidateBudget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorFitReport {
    pub train_loss_before: u64,
    pub train_loss_after: u64,
    pub validation_loss_before: u64,
    pub validation_loss_after: u64,
    pub evaluated_candidates: usize,
    pub accepted_changes: usize,
    pub epochs_started: usize,
    pub budget_exhausted: bool,
    pub shifts: [u8; 4],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean_rotating_memory::BooleanRotatingProgram;
    use crate::boolean_scan_lstm::BooleanGateProgram;

    fn always() -> UnaryGate {
        UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        }
    }

    fn never() -> UnaryGate {
        UnaryGate::default()
    }

    fn model(masks: [UnaryGate; 4], shifts: [u8; 4]) -> BooleanOperatorProgram {
        BooleanOperatorProgram::new(masks, never(), always(), shifts).unwrap()
    }

    fn next_random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn four_operation_channels_have_distinct_causal_effects() {
        let retained = model([always(), never(), never(), never()], [1, 0, 0, 0]);
        assert_eq!(retained.step_dense(0, 1).0, 2);
        let excited = model([never(), always(), never(), never()], [0, 3, 0, 0]);
        assert_eq!(excited.step_dense(0, 1).0, 8);
        let inhibited = model([always(), never(), always(), never()], [1, 0, 1, 0]);
        assert_eq!(inhibited.step_dense(0, 1).0, 0);
        let toggled = model([always(), never(), never(), always()], [1, 0, 0, 3]);
        assert_eq!(toggled.step_dense(0, 1).0, 2 ^ 8);
        let full = model([always(), always(), always(), always()], [1, 3, 1, 7]);
        assert_eq!(full.step_dense(0, 1).0, 8 ^ 128);
    }

    #[test]
    fn retain_only_recovers_existing_bps2_rotation() {
        let mut seed = 0x2ff9_442a_a10d_b755u64;
        for shift in [0, 1, 7, 31, 63] {
            let forget = UnaryGate {
                when_false: next_random(&mut seed),
                when_true: next_random(&mut seed),
            };
            let write = UnaryGate {
                when_false: next_random(&mut seed),
                when_true: next_random(&mut seed),
            };
            let output = UnaryGate {
                when_false: next_random(&mut seed),
                when_true: next_random(&mut seed),
            };
            let program = BooleanOperatorProgram::new(
                [forget, never(), never(), never()],
                write,
                output,
                [shift, 0, 0, 0],
            )
            .unwrap();
            let baseline = BooleanRotatingProgram::new(
                BooleanGateProgram {
                    forget,
                    write,
                    output,
                },
                shift,
            )
            .unwrap();
            let inputs: Vec<u64> = (0..129).map(|_| next_random(&mut seed)).collect();
            let initial = next_random(&mut seed);
            assert_eq!(
                program.forward_dense(&inputs, initial),
                baseline.forward_serial(&inputs, initial)
            );
        }
    }

    #[test]
    fn scalar_per_lane_oracle_matches_packed_four_channel_logic() {
        let mut seed = 0xcb62_835d_908e_77b3u64;
        for _ in 0..128 {
            let shifts = [
                (next_random(&mut seed) % 64) as u8,
                (next_random(&mut seed) % 64) as u8,
                (next_random(&mut seed) % 64) as u8,
                (next_random(&mut seed) % 64) as u8,
            ];
            let mut masks = [never(); 4];
            for mask in &mut masks {
                *mask = UnaryGate {
                    when_false: next_random(&mut seed),
                    when_true: next_random(&mut seed),
                };
            }
            let write = UnaryGate {
                when_false: next_random(&mut seed),
                when_true: next_random(&mut seed),
            };
            let output = UnaryGate {
                when_false: next_random(&mut seed),
                when_true: next_random(&mut seed),
            };
            let p = BooleanOperatorProgram::new(masks, write, output, shifts).unwrap();
            let state = next_random(&mut seed);
            let input = next_random(&mut seed);
            let gate_masks = p.masks(input);
            let mut expected = 0u64;
            for lane in 0..64u32 {
                let mut read = [0u64; 4];
                for index in 0..4 {
                    let predecessor = (lane + 64 - u32::from(shifts[index])) % 64;
                    read[index] = ((state >> predecessor) & 1) & ((gate_masks[index] >> lane) & 1);
                }
                let written = (write.evaluate(input) >> lane) & 1;
                let bit = (((read[0] | read[1] | written) & !read[2]) ^ read[3]) & 1;
                expected |= bit << lane;
            }
            assert_eq!(p.step_dense(input, state).0, expected);
            assert_eq!(p.step_sparse(input, state).0, expected);
            assert_eq!(
                p.step_dense(input, state).1,
                expected & output.evaluate(input)
            );
        }
    }

    #[test]
    fn sparse_and_dense_execution_match_across_mixed_input_sequences() {
        let mut seed = 0x10a8_f42c_7d19_e8b3u64;
        for length in [0usize, 1, 2, 3, 65, 129, 2049] {
            let gates = [
                UnaryGate {
                    when_false: 0,
                    when_true: u64::MAX,
                },
                UnaryGate {
                    when_false: u64::MAX,
                    when_true: 0,
                },
                never(),
                UnaryGate {
                    when_false: next_random(&mut seed),
                    when_true: next_random(&mut seed),
                },
            ];
            let program =
                BooleanOperatorProgram::new(gates, never(), always(), [1, 3, 5, 7]).unwrap();
            let inputs: Vec<u64> = (0..length)
                .map(|index| {
                    if index % 17 == 0 {
                        next_random(&mut seed)
                    } else if index % 7 == 0 {
                        u64::MAX
                    } else {
                        0
                    }
                })
                .collect();
            let initial = next_random(&mut seed);
            assert_eq!(
                program.forward_sparse(&inputs, initial),
                program.forward_dense(&inputs, initial)
            );
            let activity = program.logical_activity(&inputs);
            assert_eq!(activity.sampled, length as u128);
            assert_eq!(activity.enabled_rotations[2], 0);
            assert_eq!(activity.skipped_rotations[2], length as u128);
            for channel in 0..4 {
                assert_eq!(
                    activity.enabled_rotations[channel] + activity.skipped_rotations[channel],
                    length as u128
                );
            }
        }
    }

    #[test]
    fn learns_operation_specific_retention_and_excitation_routes() {
        let active = model([always(), always(), never(), never()], [1, 3, 0, 0]);
        let training_inputs = [0];
        let training_targets = active.forward_dense(&training_inputs, 1);
        let validation_inputs = [0, 0];
        let validation_targets = active.forward_dense(&validation_inputs, 2);
        let train = [BooleanTrainingSequence {
            inputs: &training_inputs,
            targets: &training_targets,
            initial_state: 1,
        }];
        let validation = [BooleanTrainingSequence {
            inputs: &validation_inputs,
            targets: &validation_targets,
            initial_state: 2,
        }];
        let mut learner = model([always(), always(), never(), never()], [0, 0, 0, 0]);
        let report = learner
            .fit_shifts(
                &train,
                &validation,
                u64::MAX,
                OperatorTrainPolicy {
                    max_epochs: 3,
                    max_candidates: 256,
                },
            )
            .unwrap();
        assert!(report.train_loss_before > 0);
        assert_eq!(report.train_loss_after, 0);
        assert_eq!(learner.shifts(), [1, 3, 0, 0]);
        assert_eq!(report.validation_loss_after, 0);
        assert!(report.evaluated_candidates <= 256);
    }

    #[test]
    fn invalid_configuration_and_training_do_not_mutate_program() {
        let p = model([always(), never(), never(), never()], [0, 0, 0, 0]);
        assert_eq!(
            p.with_channel_shift(OperatorChannel::Inhibit, 64),
            Err(RotationError::InvalidShift { shift: 64 })
        );
        assert_eq!(
            BooleanOperatorProgram::new([never(); 4], never(), always(), [0, 0, 64, 0]),
            Err(RotationError::InvalidShift { shift: 64 })
        );
        let inputs = [0u64];
        let targets = [2u64];
        let train = [BooleanTrainingSequence {
            inputs: &inputs,
            targets: &targets,
            initial_state: 1,
        }];
        let mut learner = p;
        assert_eq!(
            learner.fit_shifts(
                &train,
                &train,
                0,
                OperatorTrainPolicy {
                    max_epochs: 1,
                    max_candidates: 2,
                }
            ),
            Err(OperatorFitError::Training(
                BooleanTrainError::EmptyActiveLanes
            ))
        );
        assert_eq!(learner, p);
        assert_eq!(
            learner.fit_shifts(
                &train,
                &train,
                1,
                OperatorTrainPolicy {
                    max_epochs: 1,
                    max_candidates: 0,
                }
            ),
            Err(OperatorFitError::ZeroCandidateBudget)
        );
        assert_eq!(learner, p);
    }

    #[test]
    fn held_out_targets_never_influence_training_or_budget() {
        let teacher = model([always(), always(), never(), never()], [1, 3, 0, 0]);
        let inputs = [0u64, 0, 0];
        let training_targets = teacher.forward_dense(&inputs, 1);
        let other_targets = [u64::MAX; 3];
        let training = [BooleanTrainingSequence {
            inputs: &inputs,
            targets: &training_targets,
            initial_state: 1,
        }];
        let validation_a = training;
        let validation_b = [BooleanTrainingSequence {
            inputs: &inputs,
            targets: &other_targets,
            initial_state: 1,
        }];
        let mut a = model([always(), always(), never(), never()], [0, 0, 0, 0]);
        let mut b = a;
        let policy = OperatorTrainPolicy {
            max_epochs: 2,
            max_candidates: 1,
        };
        let result_a = a
            .fit_shifts(&training, &validation_a, u64::MAX, policy)
            .unwrap();
        let result_b = b
            .fit_shifts(&training, &validation_b, u64::MAX, policy)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(result_a.evaluated_candidates, 1);
        assert_eq!(result_b.evaluated_candidates, 1);
        assert_eq!(result_a.train_loss_after, result_b.train_loss_after);
        assert!(result_a.budget_exhausted && result_b.budget_exhausted);
    }
}
