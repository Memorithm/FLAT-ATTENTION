//! BPS-14: two recurrent incoming edges per packed Boolean neuron.
//!
//! Every output lane receives two predecessor lanes selected by trainable
//! circular offsets. This is a *different model* from the one-parent BPS-2
//! recurrence. OR and XOR are exact integer-valued alternatives; no floating
//! point, surrogate gradients, or neural-quality claims are involved.
//!
//! Because multi-parent composition does not generally stay in the compact
//! BPS-13 (keep, set, rotation) algebra, only the sequential oracle is
//! provided here. Never route this through BPS-13's one-parent scan.

use super::boolean_rotating_memory::{RotationError, ROTATING_LANES};
use super::boolean_scan_lstm::{
    validate_examples, BooleanGateProgram, BooleanTrainError, BooleanTrainingSequence,
};

/// Per-neuron aggregation of two recurrent predecessor bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoTapMix {
    Or,
    Xor,
}

impl TwoTapMix {
    #[must_use]
    pub const fn combine(self, first: u64, second: u64) -> u64 {
        match self {
            Self::Or => first | second,
            Self::Xor => first ^ second,
        }
    }

    #[must_use]
    const fn alternate(self) -> Self {
        match self {
            Self::Or => Self::Xor,
            Self::Xor => Self::Or,
        }
    }
}

/// Bounded fan-in 2 with shared, trainable predecessor offsets.
///
/// If the preceding state is c, the current input is x, and the shifts
/// are a and b, then:
///
///   aggregate = MIX(rotate_left(c,a), rotate_left(c,b))
///   c_next    = (forget(x) & aggregate) | write(x)
///   h_next    = c_next & output(x)
///
/// All gates are input-only and packed lane-wise; the *connections* are
/// state-dependent through two independent incoming recurrent edges.
/// This is not a classical LSTM and has no parallel-scan guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BooleanTwoTapProgram {
    pub gates: BooleanGateProgram,
    primary_shift: u8,
    secondary_shift: u8,
    mix: TwoTapMix,
}

impl BooleanTwoTapProgram {
    pub const fn new(
        gates: BooleanGateProgram,
        primary_shift: u8,
        secondary_shift: u8,
        mix: TwoTapMix,
    ) -> Result<Self, RotationError> {
        if primary_shift >= ROTATING_LANES {
            return Err(RotationError::InvalidShift {
                shift: primary_shift,
            });
        }
        if secondary_shift >= ROTATING_LANES {
            return Err(RotationError::InvalidShift {
                shift: secondary_shift,
            });
        }
        Ok(Self {
            gates,
            primary_shift,
            secondary_shift,
            mix,
        })
    }

    #[must_use]
    pub const fn shifts(self) -> (u8, u8) {
        (self.primary_shift, self.secondary_shift)
    }

    #[must_use]
    pub const fn mix(self) -> TwoTapMix {
        self.mix
    }

    /// Exact synchronous update: both incoming edges read the OLD state.
    #[must_use]
    pub const fn step(self, input: u64, previous_state: u64) -> (u64, u64) {
        let first = previous_state.rotate_left(self.primary_shift as u32);
        let second = previous_state.rotate_left(self.secondary_shift as u32);
        let coupled = self.mix.combine(first, second);
        let next = (self.gates.forget.evaluate(input) & coupled)
            | self.gates.write.evaluate(input);
        (next, next & self.gates.output.evaluate(input))
    }

    /// O(T) execution with O(1) auxiliary state, excluding returned outputs.
    #[must_use]
    pub fn forward_serial(self, inputs: &[u64], initial_state: u64) -> Vec<u64> {
        let mut state = initial_state;
        inputs
            .iter()
            .map(|&input| {
                let (next, output) = self.step(input, state);
                state = next;
                output
            })
            .collect()
    }

    /// Supervised Hamming loss on selected *output* lanes; hidden source
    /// lanes remain available to the recurrent topology.
    pub fn hamming_loss(
        self,
        examples: &[BooleanTrainingSequence<'_>],
        active_output_lanes: u64,
    ) -> Result<u64, BooleanTrainError> {
        validate_examples(examples, active_output_lanes)?;
        self.loss_validated(examples, active_output_lanes)
    }

    fn loss_validated(
        self,
        examples: &[BooleanTrainingSequence<'_>],
        active_output_lanes: u64,
    ) -> Result<u64, BooleanTrainError> {
        let mut loss = 0u64;
        for example in examples {
            let mut state = example.initial_state;
            for (&input, &target) in example.inputs.iter().zip(example.targets.iter()) {
                let (next, output) = self.step(input, state);
                state = next;
                let mistakes = u64::from(
                    ((output ^ target) & active_output_lanes).count_ones(),
                );
                loss = loss
                    .checked_add(mistakes)
                    .ok_or(BooleanTrainError::LossOverflow)?;
            }
        }
        Ok(loss)
    }

    /// Deterministic, budgeted coordinate descent over both recurrent offsets
    /// and the OR/XOR mixing rule. Boolean gate tables are held fixed.
    ///
    /// Candidate models are evaluated on the FULL recurrent training
    /// trajectories, not on independent output lanes. Training uses strict
    /// improvement only; ties retain the existing topology. Validation is
    /// reported before and after but NEVER used for candidate selection.
    /// Invalid data or loss overflow cannot mutate the original model.
    ///
    /// This is a baseline, not BPTT and not globally optimal.
    pub fn fit_topology(
        &mut self,
        training: &[BooleanTrainingSequence<'_>],
        validation: &[BooleanTrainingSequence<'_>],
        active_output_lanes: u64,
        policy: TwoTapTrainPolicy,
    ) -> Result<TwoTapFitReport, TwoTapFitError> {
        validate_examples(training, active_output_lanes)
            .map_err(TwoTapFitError::Training)?;
        validate_examples(validation, active_output_lanes)
            .map_err(TwoTapFitError::Validation)?;
        if policy.max_epochs == 0 {
            return Err(TwoTapFitError::Training(BooleanTrainError::ZeroEpochs));
        }
        if policy.max_candidates == 0 {
            return Err(TwoTapFitError::ZeroCandidateBudget);
        }

        let mut current = *self;
        let train_before = current
            .loss_validated(training, active_output_lanes)
            .map_err(TwoTapFitError::Training)?;
        let validation_before = current
            .loss_validated(validation, active_output_lanes)
            .map_err(TwoTapFitError::Validation)?;
        let mut current_loss = train_before;
        let mut candidates = 0usize;
        let mut accepted = 0usize;
        let mut epochs = 0usize;
        let mut exhausted = false;

        'search: for _ in 0..policy.max_epochs {
            if current_loss == 0 {
                break;
            }
            epochs += 1;
            let mut improved = false;

            // Explore both incoming edges against the full current topology.
            for edge in 0..2 {
                for shift in 0..ROTATING_LANES {
                    let old_shift = if edge == 0 {
                        current.primary_shift
                    } else {
                        current.secondary_shift
                    };
                    if shift == old_shift {
                        continue;
                    }
                    if candidates == policy.max_candidates {
                        exhausted = true;
                        break 'search;
                    }
                    let mut proposal = current;
                    if edge == 0 {
                        proposal.primary_shift = shift;
                    } else {
                        proposal.secondary_shift = shift;
                    }
                    let proposal_loss = proposal
                        .loss_validated(training, active_output_lanes)
                        .map_err(TwoTapFitError::Training)?;
                    candidates += 1;
                    if proposal_loss < current_loss {
                        current = proposal;
                        current_loss = proposal_loss;
                        accepted += 1;
                        improved = true;
                    }
                    if current_loss == 0 {
                        break 'search;
                    }
                }
            }

            // The OR/XOR choice itself is a trainable topology parameter.
            if candidates == policy.max_candidates {
                exhausted = true;
                break 'search;
            }
            let mut proposal = current;
            proposal.mix = current.mix.alternate();
            let proposal_loss = proposal
                .loss_validated(training, active_output_lanes)
                .map_err(TwoTapFitError::Training)?;
            candidates += 1;
            if proposal_loss < current_loss {
                current = proposal;
                current_loss = proposal_loss;
                accepted += 1;
                improved = true;
            }
            if !improved {
                break;
            }
        }

        let validation_after = current
            .loss_validated(validation, active_output_lanes)
            .map_err(TwoTapFitError::Validation)?;
        let report = TwoTapFitReport {
            train_loss_before: train_before,
            train_loss_after: current_loss,
            validation_loss_before: validation_before,
            validation_loss_after: validation_after,
            evaluated_candidates: candidates,
            accepted_changes: accepted,
            epochs_started: epochs,
            budget_exhausted: exhausted,
            selected_shifts: current.shifts(),
            selected_mix: current.mix,
        };
        *self = current;
        Ok(report)
    }
}

/// Hard caps on search work, expressed as whole-trajectory evaluations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoTapTrainPolicy {
    pub max_epochs: usize,
    pub max_candidates: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoTapFitError {
    Training(BooleanTrainError),
    Validation(BooleanTrainError),
    ZeroCandidateBudget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoTapFitReport {
    pub train_loss_before: u64,
    pub train_loss_after: u64,
    pub validation_loss_before: u64,
    pub validation_loss_after: u64,
    pub evaluated_candidates: usize,
    pub accepted_changes: usize,
    pub epochs_started: usize,
    pub budget_exhausted: bool,
    pub selected_shifts: (u8, u8),
    pub selected_mix: TwoTapMix,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean_rotating_memory::BooleanRotatingProgram;
    use crate::boolean_scan_lstm::UnaryGate;

    fn keep() -> BooleanGateProgram {
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

    fn example<'a>(
        inputs: &'a [u64],
        targets: &'a [u64],
        initial_state: u64,
    ) -> BooleanTrainingSequence<'a> {
        BooleanTrainingSequence {
            inputs,
            targets,
            initial_state,
        }
    }

    fn random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn two_distinct_incoming_edges_create_new_recurrent_states() {
        let model = BooleanTwoTapProgram::new(keep(), 1, 3, TwoTapMix::Or).unwrap();
        assert_eq!(model.forward_serial(&[0, 0], 1), vec![0b1010, 0b1010100]);
        let one_parent = BooleanRotatingProgram::new(keep(), 1).unwrap();
        assert_eq!(one_parent.forward_serial(&[0, 0], 1), vec![2, 4]);
    }

    #[test]
    fn duplicated_or_edge_is_exactly_the_existing_one_parent_model() {
        let mut seed = 0x9e41_831b_7de0_2255u64;
        for shift in [0, 1, 3, 31, 63] {
            let gates = BooleanGateProgram {
                forget: UnaryGate {
                    when_false: random(&mut seed),
                    when_true: random(&mut seed),
                },
                write: UnaryGate {
                    when_false: random(&mut seed),
                    when_true: random(&mut seed),
                },
                output: UnaryGate {
                    when_false: random(&mut seed),
                    when_true: random(&mut seed),
                },
            };
            let model = BooleanTwoTapProgram::new(gates, shift, shift, TwoTapMix::Or).unwrap();
            let one_parent = BooleanRotatingProgram::new(gates, shift).unwrap();
            let inputs: Vec<u64> = (0..129).map(|_| random(&mut seed)).collect();
            let initial = random(&mut seed);
            assert_eq!(
                model.forward_serial(&inputs, initial),
                one_parent.forward_serial(&inputs, initial),
                "shift={shift}"
            );
        }
    }

    #[test]
    fn bitwise_per_neuron_oracle_matches_both_aggregation_modes() {
        let mut seed = 0xc248_1201_c00f_7071u64;
        for _ in 0..256 {
            let first_shift = (random(&mut seed) % 64) as u8;
            let second_shift = (random(&mut seed) % 64) as u8;
            let state = random(&mut seed);
            let input = random(&mut seed);
            let gates = BooleanGateProgram {
                forget: UnaryGate {
                    when_false: random(&mut seed),
                    when_true: random(&mut seed),
                },
                write: UnaryGate {
                    when_false: random(&mut seed),
                    when_true: random(&mut seed),
                },
                output: UnaryGate {
                    when_false: random(&mut seed),
                    when_true: random(&mut seed),
                },
            };
            for mix in [TwoTapMix::Or, TwoTapMix::Xor] {
                let model = BooleanTwoTapProgram::new(
                    gates, first_shift, second_shift, mix,
                ).unwrap();
                let mut scalar_next = 0u64;
                for lane in 0..64u32 {
                    let from_first = (lane + 64 - u32::from(first_shift)) % 64;
                    let from_second = (lane + 64 - u32::from(second_shift)) % 64;
                    let a = (state >> from_first) & 1;
                    let b = (state >> from_second) & 1;
                    let mixed = match mix {
                        TwoTapMix::Or => a | b,
                        TwoTapMix::Xor => a ^ b,
                    };
                    let forget = (gates.forget.evaluate(input) >> lane) & 1;
                    let write = (gates.write.evaluate(input) >> lane) & 1;
                    scalar_next |= ((forget & mixed) | write) << lane;
                }
                let (state_after, output) = model.step(input, state);
                assert_eq!(state_after, scalar_next);
                assert_eq!(output, scalar_next & gates.output.evaluate(input));
            }
        }
    }

    #[test]
    fn learns_two_distinct_predecessors_with_bounded_full_trajectory_search() {
        let teacher = BooleanTwoTapProgram::new(keep(), 1, 3, TwoTapMix::Or).unwrap();
        let inputs = [0];
        let targets = teacher.forward_serial(&inputs, 1);
        let validation_inputs = [0, 0];
        let validation_targets = teacher.forward_serial(&validation_inputs, 2);
        let train = [example(&inputs, &targets, 1)];
        let holdout = [example(&validation_inputs, &validation_targets, 2)];
        let mut learner = BooleanTwoTapProgram::new(keep(), 0, 0, TwoTapMix::Or).unwrap();
        let report = learner
            .fit_topology(
                &train,
                &holdout,
                u64::MAX,
                TwoTapTrainPolicy {
                    max_epochs: 3,
                    max_candidates: 256,
                },
            )
            .unwrap();
        assert!(report.train_loss_before > 0);
        assert_eq!(report.train_loss_after, 0);
        assert_eq!(report.selected_shifts, (1, 3));
        assert_eq!(report.selected_mix, TwoTapMix::Or);
        assert_eq!(report.validation_loss_after, 0);
        assert!(report.accepted_changes >= 2);
        assert!(report.evaluated_candidates <= 256);
        assert_eq!(
            learner.forward_serial(&validation_inputs, 2),
            validation_targets
        );
    }

    #[test]
    fn xor_is_trainable_and_equal_offsets_cancel() {
        let teacher = BooleanTwoTapProgram::new(keep(), 1, 3, TwoTapMix::Xor).unwrap();
        let xor_double = BooleanTwoTapProgram::new(keep(), 5, 5, TwoTapMix::Xor).unwrap();
        assert_eq!(xor_double.forward_serial(&[0, 0], 1), vec![0, 0]);
        let inputs = [0, 0];
        let targets = teacher.forward_serial(&inputs, 1);
        let train = [example(&inputs, &targets, 1)];
        let holdout = [example(&inputs, &targets, 1)];
        let mut learner = BooleanTwoTapProgram::new(keep(), 1, 3, TwoTapMix::Or).unwrap();
        let report = learner
            .fit_topology(
                &train,
                &holdout,
                u64::MAX,
                TwoTapTrainPolicy {
                    max_epochs: 2,
                    max_candidates: 256,
                },
            )
            .unwrap();
        assert_eq!(report.train_loss_after, 0);
        assert_eq!(learner.mix(), TwoTapMix::Xor);
    }

    #[test]
    fn invalid_configuration_and_training_are_atomic() {
        assert_eq!(
            BooleanTwoTapProgram::new(keep(), 64, 1, TwoTapMix::Or),
            Err(RotationError::InvalidShift { shift: 64 })
        );
        assert_eq!(
            BooleanTwoTapProgram::new(keep(), 1, 64, TwoTapMix::Or),
            Err(RotationError::InvalidShift { shift: 64 })
        );
        let inputs = [0];
        let targets = [0];
        let train = [example(&inputs, &targets, 1)];
        let mut learner = BooleanTwoTapProgram::new(keep(), 0, 0, TwoTapMix::Or).unwrap();
        let original = learner;
        assert_eq!(
            learner.fit_topology(
                &train, &train, 0,
                TwoTapTrainPolicy { max_epochs: 1, max_candidates: 3 }
            ),
            Err(TwoTapFitError::Training(BooleanTrainError::EmptyActiveLanes))
        );
        assert_eq!(learner, original);
        assert_eq!(
            learner.fit_topology(
                &train, &train, 1,
                TwoTapTrainPolicy { max_epochs: 1, max_candidates: 0 }
            ),
            Err(TwoTapFitError::ZeroCandidateBudget)
        );
        assert_eq!(learner, original);
        assert_eq!(
            learner.fit_topology(
                &train, &train, 1,
                TwoTapTrainPolicy { max_epochs: 0, max_candidates: 3 }
            ),
            Err(TwoTapFitError::Training(BooleanTrainError::ZeroEpochs))
        );
        assert_eq!(learner, original);
        let bad_targets: [u64; 0] = [];
        let malformed = [example(&inputs, &bad_targets, 1)];
        assert_eq!(
            learner.fit_topology(
                &train, &malformed, 1,
                TwoTapTrainPolicy { max_epochs: 1, max_candidates: 3 }
            ),
            Err(TwoTapFitError::Validation(BooleanTrainError::LengthMismatch { example_index: 0 }))
        );
        assert_eq!(learner, original);
    }

    #[test]
    fn evaluation_budget_limits_work_without_changing_held_out_selection() {
        let inputs = [0, 0, 0];
        let desired = [2, 4, 8];
        let holdout_a = [example(&inputs, &desired, 1)];
        let holdout_b_targets = [u64::MAX; 3];
        let holdout_b = [example(&inputs, &holdout_b_targets, 1)];
        let mut a = BooleanTwoTapProgram::new(keep(), 0, 0, TwoTapMix::Or).unwrap();
        let mut b = a;
        let policy = TwoTapTrainPolicy {
            max_epochs: 3,
            max_candidates: 1,
        };
        let a_report = a.fit_topology(&holdout_a, &holdout_a, u64::MAX, policy).unwrap();
        let b_report = b.fit_topology(&holdout_a, &holdout_b, u64::MAX, policy).unwrap();
        assert_eq!(a, b);
        assert_eq!(a_report.evaluated_candidates, 1);
        assert_eq!(b_report.evaluated_candidates, 1);
        assert_eq!(a_report.train_loss_after, b_report.train_loss_after);
        assert!(a_report.budget_exhausted);
        assert!(b_report.budget_exhausted);
    }
}
