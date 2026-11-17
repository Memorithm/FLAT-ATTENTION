//! BPS-4: globally coupled, discrete joint learning of Boolean gates and topology.
//!
//! Both BPS-2 and BPS-3 are supported. Unlike the independent BPS-1 bit-plane
//! optimizer, *every* candidate is evaluated against the whole recurrent
//! trajectory before a mutation can be accepted. A bit outside the supervised
//! output mask can therefore be trained as an upstream recurrent contributor.
//!
//! This is deterministic strict-improvement coordinate search, not BPTT,
//! a global optimum, a conventional LSTM, or evidence of energy savings.

use super::boolean_edge_memory::BooleanEdgeProgram;
use super::boolean_rotating_memory::{BooleanRotatingProgram, ROTATING_LANES};
use super::boolean_scan_lstm::BooleanTrainError;

/// One labeled recurrent trajectory. Independent trajectories reset state.
/// The explicit input baseline matters only to the rising-edge BPS-3 model.
#[derive(Clone, Copy, Debug)]
pub struct CoupledExample<'a> {
    pub inputs: &'a [u64],
    pub targets: &'a [u64],
    pub baseline_input: u64,
    pub initial_state: u64,
}

/// Training policy, bounded in candidate forward evaluations across all rounds.
/// Baseline and final held-out evaluations do not consume candidate budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointTrainPolicy {
    pub max_epochs: usize,
    pub max_candidate_evaluations: usize,
}

/// Distinguishes topology and gate optimization in the two recurrent models.
/// These models have different quiet-tick and readout semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoupledModel {
    Rotating(BooleanRotatingProgram),
    RisingEdge(BooleanEdgeProgram),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointTrainError {
    InvalidTraining(BooleanTrainError),
    InvalidValidation(BooleanTrainError),
    ZeroCandidateBudget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointTrainReport {
    pub train_loss_before: u64,
    pub train_loss_after: u64,
    pub validation_loss_before: u64,
    pub validation_loss_after: u64,
    pub rotation_before: u8,
    pub rotation_after: u8,
    pub accepted_rotation_changes: usize,
    pub accepted_gate_flips: usize,
    pub candidates_evaluated: usize,
    pub epochs_started: usize,
    pub budget_exhausted: bool,
}

/// Explicitly evaluate a held-out dataset without modifying any parameters.
impl CoupledModel {
    #[must_use]
    pub const fn rotation(self) -> u8 {
        match self {
            Self::Rotating(program) => program.recurrent_shift(),
            Self::RisingEdge(program) => program.shift(),
        }
    }

    /// Hamming errors count only the selected *output* bits. Candidate gate
    /// updates still explore all 64 state lanes because hidden upstream bits
    /// may later rotate into observed output positions.
    pub fn hamming_loss(
        self,
        examples: &[CoupledExample<'_>],
        active_output_lanes: u64,
    ) -> Result<u64, BooleanTrainError> {
        validate_cases(examples, active_output_lanes)?;
        self.validated_loss(examples, active_output_lanes)
    }

    /// Coordinate-descent search with a strict, global training-loss gate.
    ///
    /// Each round first tests all alternative shared recurrent rotations,
    /// then toggles each Boolean gate truth-table bit individually. Mutations
    /// are accepted only if the **whole** training-set Hamming loss decreases.
    /// Gate and rotation decisions are interleaved across rounds.
    ///
    /// The held-out validation set is evaluated before and after training,
    /// never consulted for parameter selection, early stopping or ties.
    /// This makes worsening generalization observable rather than hidden.
    ///
    /// The receiver is unchanged on invalid data or an evaluation error.
    pub fn fit_joint(
        &mut self,
        train: &[CoupledExample<'_>],
        validation: &[CoupledExample<'_>],
        active_output_lanes: u64,
        policy: JointTrainPolicy,
    ) -> Result<JointTrainReport, JointTrainError> {
        validate_cases(train, active_output_lanes)
            .map_err(JointTrainError::InvalidTraining)?;
        validate_cases(validation, active_output_lanes)
            .map_err(JointTrainError::InvalidValidation)?;
        if policy.max_epochs == 0 {
            return Err(JointTrainError::InvalidTraining(
                BooleanTrainError::ZeroEpochs,
            ));
        }
        if policy.max_candidate_evaluations == 0 {
            return Err(JointTrainError::ZeroCandidateBudget);
        }

        let mut current = *self;
        let initial_train = current
            .validated_loss(train, active_output_lanes)
            .map_err(JointTrainError::InvalidTraining)?;
        let initial_validation = current
            .validated_loss(validation, active_output_lanes)
            .map_err(JointTrainError::InvalidValidation)?;
        let mut loss = initial_train;
        let mut accepted_rotations = 0;
        let mut accepted_gates = 0;
        let mut evaluated = 0;
        let mut epochs = 0;
        let mut budget_exhausted = false;

        'rounds: for _ in 0..policy.max_epochs {
            if loss == 0 {
                break;
            }
            epochs += 1;
            let mut improved = false;

            // Whole-word recurrent rotations couple previously independent
            // state lanes; any improvement must be checked globally.
            for shift in 0..ROTATING_LANES {
                if shift == current.rotation() {
                    continue;
                }
                if evaluated == policy.max_candidate_evaluations {
                    budget_exhausted = true;
                    break 'rounds;
                }
                let candidate = current.with_rotation(shift);
                let candidate_loss = candidate
                    .validated_loss(train, active_output_lanes)
                    .map_err(JointTrainError::InvalidTraining)?;
                evaluated += 1;
                if candidate_loss < loss {
                    current = candidate;
                    loss = candidate_loss;
                    accepted_rotations += 1;
                    improved = true;
                }
                if loss == 0 {
                    break 'rounds;
                }
            }

            for parameter in 0..current.gate_parameter_count() {
                // Iterate ALL lanes, including those outside supervised
                // outputs: rotation can make them upstream dependencies.
                for lane in 0..64 {
                    if evaluated == policy.max_candidate_evaluations {
                        budget_exhausted = true;
                        break 'rounds;
                    }
                    let mut candidate = current;
                    candidate.toggle_gate_bit(parameter, 1u64 << lane);
                    let candidate_loss = candidate
                        .validated_loss(train, active_output_lanes)
                        .map_err(JointTrainError::InvalidTraining)?;
                    evaluated += 1;
                    if candidate_loss < loss {
                        current = candidate;
                        loss = candidate_loss;
                        accepted_gates += 1;
                        improved = true;
                    }
                    if loss == 0 {
                        break 'rounds;
                    }
                }
            }
            if !improved {
                break;
            }
        }

        let final_validation = current
            .validated_loss(validation, active_output_lanes)
            .map_err(JointTrainError::InvalidValidation)?;
        let report = JointTrainReport {
            train_loss_before: initial_train,
            train_loss_after: loss,
            validation_loss_before: initial_validation,
            validation_loss_after: final_validation,
            rotation_before: self.rotation(),
            rotation_after: current.rotation(),
            accepted_rotation_changes: accepted_rotations,
            accepted_gate_flips: accepted_gates,
            candidates_evaluated: evaluated,
            epochs_started: epochs,
            budget_exhausted,
        };
        *self = current;
        Ok(report)
    }

    fn with_rotation(self, shift: u8) -> Self {
        match self {
            Self::Rotating(program) => Self::Rotating(
                BooleanRotatingProgram::new(program.gates, shift)
                    .expect("rotation search always uses a valid offset"),
            ),
            Self::RisingEdge(program) => Self::RisingEdge(
                BooleanEdgeProgram::new(
                    program.forget,
                    program.write,
                    program.readout_mask,
                    shift,
                )
                .expect("rotation search always uses a valid offset"),
            ),
        }
    }

    fn gate_parameter_count(self) -> usize {
        match self {
            Self::Rotating(_) => 6,
            Self::RisingEdge(_) => 5,
        }
    }

    fn toggle_gate_bit(&mut self, parameter: usize, mask: u64) {
        match self {
            Self::Rotating(program) => match parameter {
                0 => program.gates.forget.when_false ^= mask,
                1 => program.gates.forget.when_true ^= mask,
                2 => program.gates.write.when_false ^= mask,
                3 => program.gates.write.when_true ^= mask,
                4 => program.gates.output.when_false ^= mask,
                5 => program.gates.output.when_true ^= mask,
                _ => unreachable!("exactly six BPS-2 gate masks"),
            },
            Self::RisingEdge(program) => match parameter {
                0 => program.forget.when_false ^= mask,
                1 => program.forget.when_true ^= mask,
                2 => program.write.when_false ^= mask,
                3 => program.write.when_true ^= mask,
                4 => program.readout_mask ^= mask,
                _ => unreachable!("exactly five BPS-3 gate masks"),
            },
        }
    }

    fn validated_loss(
        self,
        examples: &[CoupledExample<'_>],
        active_output_lanes: u64,
    ) -> Result<u64, BooleanTrainError> {
        let mut loss = 0u64;
        for example in examples {
            let mut state = example.initial_state;
            let mut previous_input = example.baseline_input;
            for (&input, &target) in example.inputs.iter().zip(example.targets.iter()) {
                let output = match self {
                    Self::Rotating(program) => {
                        state = program.transition(input).apply(state);
                        state & program.gates.output.evaluate(input)
                    }
                    Self::RisingEdge(program) => {
                        let rising = input & !previous_input;
                        previous_input = input;
                        state = program.transition(rising).apply(state);
                        state & program.readout_mask
                    }
                };
                loss = loss
                    .checked_add(u64::from(
                        ((output ^ target) & active_output_lanes).count_ones(),
                    ))
                    .ok_or(BooleanTrainError::LossOverflow)?;
            }
        }
        Ok(loss)
    }
}

fn validate_cases(
    cases: &[CoupledExample<'_>],
    active_output_lanes: u64,
) -> Result<(), BooleanTrainError> {
    if cases.is_empty() {
        return Err(BooleanTrainError::EmptyDataset);
    }
    if active_output_lanes == 0 {
        return Err(BooleanTrainError::EmptyActiveLanes);
    }
    for (index, example) in cases.iter().enumerate() {
        if example.inputs.is_empty() {
            return Err(BooleanTrainError::EmptySequence {
                example_index: index,
            });
        }
        if example.inputs.len() != example.targets.len() {
            return Err(BooleanTrainError::LengthMismatch {
                example_index: index,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::boolean_scan_lstm::{BooleanGateProgram, UnaryGate};

    fn all_true() -> UnaryGate {
        UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        }
    }

    fn pure_rotation(shift: u8) -> BooleanEdgeProgram {
        BooleanEdgeProgram::new(all_true(), UnaryGate::default(), u64::MAX, shift)
            .unwrap()
    }

    fn policy() -> JointTrainPolicy {
        JointTrainPolicy {
            max_epochs: 3,
            max_candidate_evaluations: 1024,
        }
    }

    fn edge_case<'a>(
        inputs: &'a [u64],
        targets: &'a [u64],
        initial_state: u64,
    ) -> CoupledExample<'a> {
        CoupledExample {
            inputs,
            targets,
            initial_state,
            baseline_input: 0,
        }
    }

    #[test]
    fn bps3_jointly_learns_rotation_and_spurious_write_gate() {
        let teacher = pure_rotation(7);
        let inputs = [1u64, 0, 1, 0, 1, 0, 1];
        let expected = teacher.forward_dense(&inputs, 0, 1);
        let validation_inputs = [0u64, 1, 0, 0, 1, 0, 1, 0];
        let expected_validation = teacher.forward_dense(&validation_inputs, 0, 1u64 << 3);
        let train = [edge_case(&inputs, &expected, 1)];
        let holdout = [edge_case(
            &validation_inputs,
            &expected_validation,
            1u64 << 3,
        )];

        let mut faulty = pure_rotation(0);
        faulty.write.when_true = 1;
        let mut candidate = CoupledModel::RisingEdge(faulty);
        let report = candidate.fit_joint(&train, &holdout, u64::MAX, policy()).unwrap();
        assert!(report.train_loss_before > 0);
        assert_eq!(report.train_loss_after, 0);
        assert_eq!(report.validation_loss_after, 0);
        assert_eq!(report.rotation_after, 7);
        assert!(report.accepted_rotation_changes > 0);
        assert!(report.accepted_gate_flips > 0);
        if let CoupledModel::RisingEdge(learned) = candidate {
            assert_eq!(learned.forward_dense(&inputs, 0, 1), expected);
            assert_eq!(
                learned.forward_event_scan(&validation_inputs, 0, 1u64 << 3),
                expected_validation
            );
        } else {
            panic!("joint learner changed model family");
        }
    }

    #[test]
    fn upstream_lane_outside_output_mask_can_be_trained() {
        let program = BooleanRotatingProgram::new(
            BooleanGateProgram {
                forget: all_true(),
                write: UnaryGate::default(),
                output: all_true(),
            },
            1,
        )
        .unwrap();
        let inputs = [0u64, 0];
        // Only output bit 3 is supervised. The useful write is on bit 2 at
        // t=0, which then rotates into bit 3 at t=1.
        let targets = [0u64, 1u64 << 3];
        let data = [CoupledExample {
            inputs: &inputs,
            targets: &targets,
            baseline_input: 0,
            initial_state: 0,
        }];
        let mut model = CoupledModel::Rotating(program);
        let report = model.fit_joint(&data, &data, 1u64 << 3, policy()).unwrap();
        assert_eq!(report.train_loss_before, 1);
        assert_eq!(report.train_loss_after, 0);
        assert!(report.accepted_gate_flips > 0);
        if let CoupledModel::Rotating(learned) = model {
            assert_ne!(learned.gates.write.when_false & (1u64 << 2), 0);
            assert_eq!(learned.forward_serial(&inputs, 0)[1] & (1u64 << 3), 1u64 << 3);
        } else {
            panic!("joint learner changed model family");
        }
    }

    #[test]
    fn heldout_targets_are_scored_not_used_for_selection() {
        let teacher = pure_rotation(5);
        let inputs = [1u64, 0, 1, 0, 1];
        let targets = teacher.forward_dense(&inputs, 0, 1);
        let holdout_inputs = [1u64, 0, 1];
        let good = teacher.forward_dense(&holdout_inputs, 0, 1u64 << 2);
        let bad: Vec<u64> = good.iter().map(|x| !x).collect();
        let train = [edge_case(&inputs, &targets, 1)];
        let val_good = [edge_case(&holdout_inputs, &good, 1u64 << 2)];
        let val_bad = [edge_case(&holdout_inputs, &bad, 1u64 << 2)];
        let mut one = CoupledModel::RisingEdge(pure_rotation(0));
        let mut two = one;
        let report_one = one.fit_joint(&train, &val_good, u64::MAX, policy()).unwrap();
        let report_two = two.fit_joint(&train, &val_bad, u64::MAX, policy()).unwrap();
        assert_eq!(one, two);
        assert_eq!(report_one.train_loss_after, report_two.train_loss_after);
        assert_ne!(report_one.validation_loss_after, report_two.validation_loss_after);
    }

    #[test]
    fn candidate_budget_stops_without_misreporting_convergence() {
        let teacher = pure_rotation(7);
        let inputs = [1u64, 0, 1, 0, 1];
        let targets = teacher.forward_dense(&inputs, 0, 1);
        let train = [edge_case(&inputs, &targets, 1)];
        let mut model = CoupledModel::RisingEdge(pure_rotation(0));
        let report = model.fit_joint(
            &train,
            &train,
            u64::MAX,
            JointTrainPolicy {
                max_epochs: 5,
                max_candidate_evaluations: 1,
            },
        )
        .unwrap();
        assert_eq!(report.candidates_evaluated, 1);
        assert!(report.budget_exhausted);
        assert!(report.train_loss_after <= report.train_loss_before);
    }

    #[test]
    fn empty_and_invalid_datasets_do_not_mutate_model() {
        let data = [edge_case(&[1, 0], &[0], 1)];
        let valid = [edge_case(&[1], &[1], 0)];
        let original = CoupledModel::RisingEdge(pure_rotation(4));
        let mut model = original;
        assert_eq!(
            model.fit_joint(&data, &valid, 1, policy()),
            Err(JointTrainError::InvalidTraining(
                BooleanTrainError::LengthMismatch { example_index: 0 }
            ))
        );
        assert_eq!(model, original);
        assert_eq!(
            model.fit_joint(&valid, &[], 1, policy()),
            Err(JointTrainError::InvalidValidation(
                BooleanTrainError::EmptyDataset
            ))
        );
        assert_eq!(model, original);
        assert_eq!(
            model.fit_joint(&valid, &valid, 1, JointTrainPolicy {
                max_epochs: 0,
                max_candidate_evaluations: 1,
            }),
            Err(JointTrainError::InvalidTraining(BooleanTrainError::ZeroEpochs))
        );
        assert_eq!(
            model.fit_joint(&valid, &valid, 1, JointTrainPolicy {
                max_epochs: 1,
                max_candidate_evaluations: 0,
            }),
            Err(JointTrainError::ZeroCandidateBudget)
        );
        assert_eq!(model, original);
    }

    #[test]
    fn bps2_loss_matches_original_oracle_and_is_monotone_under_training() {
        let gates = BooleanGateProgram {
            forget: all_true(),
            write: UnaryGate::default(),
            output: all_true(),
        };
        let teacher = BooleanRotatingProgram::new(gates, 3).unwrap();
        let inputs = [0u64, 0, 0, 0, 0];
        let expected = teacher.forward_serial(&inputs, 1);
        let data = [CoupledExample {
            inputs: &inputs,
            targets: &expected,
            baseline_input: u64::MAX,
            initial_state: 1,
        }];
        let mut model = CoupledModel::Rotating(
            BooleanRotatingProgram::new(gates, 0).unwrap(),
        );
        assert_eq!(
            model.hamming_loss(&data, u64::MAX),
            model_rotating_loss_oracle(&model, &data, u64::MAX)
        );
        let report = model.fit_joint(&data, &data, u64::MAX, policy()).unwrap();
        assert_eq!(report.train_loss_after, 0);
        assert_eq!(report.rotation_after, 3);
        assert!(report.accepted_rotation_changes > 0);
    }

    fn model_rotating_loss_oracle(
        model: &CoupledModel,
        cases: &[CoupledExample<'_>],
        mask: u64,
    ) -> Result<u64, BooleanTrainError> {
        match *model {
            CoupledModel::Rotating(program) => {
                let mut sum = 0u64;
                for example in cases {
                    let outputs = program.forward_serial(example.inputs, example.initial_state);
                    for (&output, &target) in outputs.iter().zip(example.targets.iter()) {
                        sum += u64::from(((output ^ target) & mask).count_ones());
                    }
                }
                Ok(sum)
            }
            CoupledModel::RisingEdge(_) => unreachable!(),
        }
    }
}
