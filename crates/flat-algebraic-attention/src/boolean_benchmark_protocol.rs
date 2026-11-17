//! BPS-5: deterministic, bounded train/validation/test benchmark protocol.
//!
//! Teacher-generated trajectories test only whether the BPS-3 hypothesis
//! family can recover synthetic recurrent Boolean dynamics. This is NOT an
//! FP32 LSTM comparison, a natural-language benchmark, or physical energy
//! evidence. Test labels are only used after all models have been trained.

use super::boolean_coupled_learning::{
    CoupledExample, CoupledModel, JointTrainError, JointTrainPolicy, JointTrainReport,
};
use super::boolean_edge_memory::BooleanEdgeProgram;
use super::boolean_rotating_memory::ROTATING_LANES;
use super::boolean_scan_lstm::BooleanTrainError;
use super::rising_edge::RisingEdgeError;

/// Explicit, inspectable workload controls. No runtime RNG or external data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bps5Protocol {
    pub seed: u64,
    pub train_episodes: usize,
    pub validation_episodes: usize,
    pub test_episodes: usize,
    pub steps_per_episode: usize,
    /// Chance that a sampled input word is high; NOT event-front density.
    pub input_occupancy_percent: u8,
    /// Restrict input pulses to one bit, or choose a random bit per pulse.
    pub pulse_lane: Option<u8>,
    pub training: JointTrainPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bps5Error {
    EmptySplit,
    TooManyEpisodes,
    InvalidSequenceLength,
    InvalidOccupancy,
    InvalidPulseLane,
    Training(JointTrainError),
    InvalidCorpus(BooleanTrainError),
    InvalidRuntime(RisingEdgeError),
    OracleMismatch,
    ResourceOverflow,
}

/// Additive, dimensionless resource accounting. A zero item is identity.
/// No field represents joules, physical energy, instructions or CPU cycles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogicalResources {
    pub sampled_words: u128,
    pub eventful_recurrent_steps: u128,
    pub emitted_output_words: u128,
    pub state_bit_flips: u128,
}

impl LogicalResources {
    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            sampled_words: self.sampled_words.checked_add(other.sampled_words)?,
            eventful_recurrent_steps: self
                .eventful_recurrent_steps
                .checked_add(other.eventful_recurrent_steps)?,
            emitted_output_words: self
                .emitted_output_words
                .checked_add(other.emitted_output_words)?,
            state_bit_flips: self.state_bit_flips.checked_add(other.state_bit_flips)?,
        })
    }

    /// An explicitly *dimensionless* cost projection for future TDI
    /// sensitivity analyses. Weights are user hypotheses, NOT calibrated
    /// physical energy coefficients; do not label the result pJ or joules.
    pub fn weighted_proxy(self, weights: LogicalCostWeights) -> Option<u128> {
        self.sampled_words
            .checked_mul(weights.sampled_word)?
            .checked_add(
                self.eventful_recurrent_steps
                    .checked_mul(weights.recurrent_step)?,
            )?
            .checked_add(
                self.emitted_output_words
                    .checked_mul(weights.emitted_word)?,
            )?
            .checked_add(self.state_bit_flips.checked_mul(weights.state_flip)?)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicalCostWeights {
    pub sampled_word: u128,
    pub recurrent_step: u128,
    pub emitted_word: u128,
    pub state_flip: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bps5Episode {
    pub inputs: Vec<u64>,
    pub targets: Vec<u64>,
    pub baseline_input: u64,
    pub initial_state: u64,
}

impl Bps5Episode {
    fn as_example(&self) -> CoupledExample<'_> {
        CoupledExample {
            inputs: &self.inputs,
            targets: &self.targets,
            baseline_input: self.baseline_input,
            initial_state: self.initial_state,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bps5Dataset {
    pub train: Vec<Bps5Episode>,
    pub validation: Vec<Bps5Episode>,
    pub test: Vec<Bps5Episode>,
}

impl Bps5Dataset {
    fn examples(cases: &[Bps5Episode]) -> Vec<CoupledExample<'_>> {
        cases.iter().map(Bps5Episode::as_example).collect()
    }

    /// Exact baseline/labels provenance: all episodes are generated from the
    /// supplied frozen teacher and the fixed PRNG algorithm; initial-state
    /// source lanes are disjoint across train/validation/test by construction.
    pub fn generate(
        protocol: Bps5Protocol,
        teacher: BooleanEdgeProgram,
    ) -> Result<Self, Bps5Error> {
        if protocol.train_episodes == 0
            || protocol.validation_episodes == 0
            || protocol.test_episodes == 0
        {
            return Err(Bps5Error::EmptySplit);
        }
        let total = protocol
            .train_episodes
            .checked_add(protocol.validation_episodes)
            .and_then(|n| n.checked_add(protocol.test_episodes))
            .ok_or(Bps5Error::TooManyEpisodes)?;
        if total > 64 {
            return Err(Bps5Error::TooManyEpisodes);
        }
        if !(1..=4096).contains(&protocol.steps_per_episode) {
            return Err(Bps5Error::InvalidSequenceLength);
        }
        if protocol.input_occupancy_percent > 100 {
            return Err(Bps5Error::InvalidOccupancy);
        }
        if protocol.pulse_lane.is_some_and(|lane| lane >= 64) {
            return Err(Bps5Error::InvalidPulseLane);
        }

        let mut rng = XorShift64::new(protocol.seed);
        let mut episodes = Vec::with_capacity(total);
        for index in 0..total {
            // Distinct single-bit initial memories guarantee pairwise distinct
            // trajectories even at 0% occupancy; splits cannot duplicate.
            let initial_state = 1u64 << index;
            let baseline_input = if rng.next() & 3 == 0 {
                let lane = protocol.pulse_lane.unwrap_or((rng.next() % 64) as u8);
                1u64 << lane
            } else {
                0
            };
            let mut inputs = Vec::with_capacity(protocol.steps_per_episode);
            for _ in 0..protocol.steps_per_episode {
                if rng.next() % 100 < u64::from(protocol.input_occupancy_percent) {
                    let lane = protocol.pulse_lane.unwrap_or((rng.next() % 64) as u8);
                    inputs.push(1u64 << lane);
                } else {
                    inputs.push(0);
                }
            }
            let targets = teacher.forward_dense(&inputs, baseline_input, initial_state);
            episodes.push(Bps5Episode {
                inputs,
                targets,
                baseline_input,
                initial_state,
            });
        }
        // Draining in forward order preserves the separate PRNG provenance
        // and does not re-derive inputs from test labels.
        let test = episodes.split_off(protocol.train_episodes + protocol.validation_episodes);
        let validation = episodes.split_off(protocol.train_episodes);
        Ok(Self {
            train: episodes,
            validation,
            test,
        })
    }
}

/// Deterministic baseline scores. Each uses the same supervised output mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bps5Score {
    pub train_hamming: u64,
    pub validation_hamming: u64,
    pub test_hamming: u64,
    pub test_bit_opportunities: u64,
    pub rotation: u8,
    pub training_candidate_evaluations: usize,
    pub test_resources: LogicalResources,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bps5Comparison {
    pub input_occupancy_percent: u8,
    pub train_episodes: usize,
    pub validation_episodes: usize,
    pub test_episodes: usize,
    pub sample_steps_in_test: usize,
    pub actual_eventful_steps_in_test: u128,
    pub baseline_untrained: Bps5Score,
    pub rotation_only: Bps5Score,
    pub jointly_trained: Bps5Score,
    pub joint_training: JointTrainReport,
}

/// Compare **like-for-like BPS-3 models** under fixed gates, different
/// recurrent shifts, and globally coupled gate/shift training.
/// Neither validation nor test labels are consulted during training.
/// The test corpus remains untouched until all parameters are frozen.
pub fn compare_bps3(
    protocol: Bps5Protocol,
    teacher: BooleanEdgeProgram,
    student: BooleanEdgeProgram,
) -> Result<Bps5Comparison, Bps5Error> {
    let dataset = Bps5Dataset::generate(protocol, teacher)?;
    let train = Bps5Dataset::examples(&dataset.train);
    let validation = Bps5Dataset::examples(&dataset.validation);
    let test = Bps5Dataset::examples(&dataset.test);

    let original = CoupledModel::RisingEdge(student);
    // All parameter selection uses training examples exclusively.
    let mut rotation_model = original;
    let mut rotation_loss = rotation_model
        .hamming_loss(&train, u64::MAX)
        .map_err(Bps5Error::InvalidCorpus)?;
    let mut rotation_evaluations = 0;
    for shift in 0..ROTATING_LANES {
        if shift == student.shift() {
            continue;
        }
        let candidate_program =
            BooleanEdgeProgram::new(student.forget, student.write, student.readout_mask, shift)
                .expect("generated baseline shift is bounded");
        let candidate = CoupledModel::RisingEdge(candidate_program);
        let candidate_loss = candidate
            .hamming_loss(&train, u64::MAX)
            .map_err(Bps5Error::InvalidCorpus)?;
        rotation_evaluations += 1;
        if candidate_loss < rotation_loss {
            rotation_model = candidate;
            rotation_loss = candidate_loss;
        }
    }

    let mut joint_model = original;
    let joint_report = joint_model
        .fit_joint(&train, &validation, u64::MAX, protocol.training)
        .map_err(Bps5Error::Training)?;

    // Test labels are first examined here, strictly after training.
    let untrained = score(original, &train, &validation, &test, &dataset.test, 0)?;
    let rotation_only = score(
        rotation_model,
        &train,
        &validation,
        &test,
        &dataset.test,
        rotation_evaluations,
    )?;
    let jointly_trained = score(
        joint_model,
        &train,
        &validation,
        &test,
        &dataset.test,
        joint_report.candidates_evaluated,
    )?;
    let steps = dataset.test.len() * protocol.steps_per_episode;

    Ok(Bps5Comparison {
        input_occupancy_percent: protocol.input_occupancy_percent,
        train_episodes: protocol.train_episodes,
        validation_episodes: protocol.validation_episodes,
        test_episodes: protocol.test_episodes,
        sample_steps_in_test: steps,
        actual_eventful_steps_in_test: untrained.test_resources.eventful_recurrent_steps,
        baseline_untrained: untrained,
        rotation_only,
        jointly_trained,
        joint_training: joint_report,
    })
}

fn score(
    model: CoupledModel,
    train: &[CoupledExample<'_>],
    validation: &[CoupledExample<'_>],
    test: &[CoupledExample<'_>],
    test_episodes: &[Bps5Episode],
    training_candidate_evaluations: usize,
) -> Result<Bps5Score, Bps5Error> {
    let train_hamming = model
        .hamming_loss(train, u64::MAX)
        .map_err(Bps5Error::InvalidCorpus)?;
    let validation_hamming = model
        .hamming_loss(validation, u64::MAX)
        .map_err(Bps5Error::InvalidCorpus)?;
    let test_hamming = model
        .hamming_loss(test, u64::MAX)
        .map_err(Bps5Error::InvalidCorpus)?;
    let CoupledModel::RisingEdge(program) = model else {
        unreachable!("BPS-5 compares only BPS-3 variants");
    };
    let mut resources = LogicalResources::default();
    for episode in test_episodes {
        let run = program
            .forward_sparse(
                &episode.inputs,
                episode.baseline_input,
                episode.initial_state,
            )
            .map_err(Bps5Error::InvalidRuntime)?;
        if run.outputs
            != program.forward_dense(
                &episode.inputs,
                episode.baseline_input,
                episode.initial_state,
            )
            || run.outputs
                != program.forward_event_scan(
                    &episode.inputs,
                    episode.baseline_input,
                    episode.initial_state,
                )
        {
            return Err(Bps5Error::OracleMismatch);
        }
        let activity = run.activity;
        resources = resources
            .checked_add(LogicalResources {
                sampled_words: activity.sampled_steps,
                eventful_recurrent_steps: activity.evaluated_recurrent_steps,
                emitted_output_words: activity.sampled_steps,
                state_bit_flips: activity.state_flips,
            })
            .ok_or(Bps5Error::ResourceOverflow)?;
    }
    let test_bit_opportunities = u64::try_from(
        resources
            .sampled_words
            .checked_mul(64)
            .ok_or(Bps5Error::ResourceOverflow)?,
    )
    .map_err(|_| Bps5Error::ResourceOverflow)?;
    Ok(Bps5Score {
        train_hamming,
        validation_hamming,
        test_hamming,
        test_bit_opportunities,
        rotation: model.rotation(),
        training_candidate_evaluations,
        test_resources: resources,
    })
}

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::super::boolean_scan_lstm::UnaryGate;
    use super::*;

    fn teacher(shift: u8) -> BooleanEdgeProgram {
        BooleanEdgeProgram::new(
            UnaryGate {
                when_false: u64::MAX,
                when_true: u64::MAX,
            },
            UnaryGate::default(),
            u64::MAX,
            shift,
        )
        .unwrap()
    }

    fn protocol(occupancy: u8) -> Bps5Protocol {
        Bps5Protocol {
            seed: 0x93a1_f00d_9876_5432,
            train_episodes: 8,
            validation_episodes: 4,
            test_episodes: 8,
            steps_per_episode: 24,
            input_occupancy_percent: occupancy,
            pulse_lane: Some(3),
            training: JointTrainPolicy {
                max_epochs: 2,
                max_candidate_evaluations: 1024,
            },
        }
    }

    #[test]
    fn independent_splits_and_generation_are_bit_reproducible() {
        let config = protocol(25);
        let first = Bps5Dataset::generate(config, teacher(7)).unwrap();
        let second = Bps5Dataset::generate(config, teacher(7)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.train.len(), 8);
        assert_eq!(first.validation.len(), 4);
        assert_eq!(first.test.len(), 8);

        let mut initial_states = Vec::new();
        for episode in first
            .train
            .iter()
            .chain(first.validation.iter())
            .chain(first.test.iter())
        {
            assert_eq!(episode.targets.len(), config.steps_per_episode);
            initial_states.push(episode.initial_state);
            assert_eq!(episode.initial_state.count_ones(), 1);
        }
        initial_states.sort_unstable();
        initial_states.dedup();
        assert_eq!(initial_states.len(), 20);
        assert_ne!(first.train[0].initial_state, first.test[0].initial_state);
    }

    #[test]
    fn teacher_has_zero_hamming_loss_on_all_unseen_sequences() {
        let program = teacher(7);
        let dataset = Bps5Dataset::generate(protocol(50), program).unwrap();
        let holdout = Bps5Dataset::examples(&dataset.test);
        assert_eq!(
            CoupledModel::RisingEdge(program).hamming_loss(&holdout, u64::MAX),
            Ok(0)
        );
        assert_ne!(
            dataset.train[0].initial_state,
            dataset.validation[0].initial_state
        );
    }

    #[test]
    fn zero_events_are_exactly_accounted_and_do_not_force_recurrence() {
        let config = protocol(0);
        let mut prog = teacher(0);
        prog.write.when_true = 1u64 << 1;
        let comparison = compare_bps3(config, teacher(7), prog).unwrap();
        // At 0% occupancy, all inputs are zero; any initial high baseline
        // can only FALL, never create a rise, so no event is possible.
        assert_eq!(comparison.actual_eventful_steps_in_test, 0);
        assert_eq!(
            comparison
                .baseline_untrained
                .test_resources
                .eventful_recurrent_steps,
            0
        );
        assert_eq!(comparison.baseline_untrained.test_hamming, 0);
        assert_eq!(comparison.rotation_only.test_hamming, 0);
        assert_eq!(comparison.jointly_trained.test_hamming, 0);
    }

    #[test]
    fn paired_train_val_test_metrics_are_bounded_and_exact() {
        let config = protocol(50);
        let initial = teacher(0);
        let result = compare_bps3(config, teacher(7), initial).unwrap();
        assert_eq!(result.sample_steps_in_test, 8 * 24);
        assert_eq!(
            result.baseline_untrained.test_bit_opportunities,
            8 * 24 * 64
        );
        assert_eq!(result.rotation_only.test_bit_opportunities, 8 * 24 * 64);
        assert_eq!(result.jointly_trained.test_bit_opportunities, 8 * 24 * 64);
        assert!(result.actual_eventful_steps_in_test > 0);
        assert!(result.actual_eventful_steps_in_test < 8 * 24);
        assert_eq!(result.rotation_only.rotation, 7);
        assert_eq!(result.rotation_only.test_hamming, 0);
        assert!(result.jointly_trained.train_hamming <= result.baseline_untrained.train_hamming);
        assert!(
            result.joint_training.candidates_evaluated <= config.training.max_candidate_evaluations
        );
        assert_eq!(result.rotation_only.training_candidate_evaluations, 63);
    }

    #[test]
    fn resource_algebra_is_associative_and_explicitly_dimensionless() {
        let a = LogicalResources {
            sampled_words: 2,
            eventful_recurrent_steps: 1,
            emitted_output_words: 2,
            state_bit_flips: 3,
        };
        let b = LogicalResources {
            sampled_words: 5,
            eventful_recurrent_steps: 0,
            emitted_output_words: 5,
            state_bit_flips: 1,
        };
        let c = LogicalResources {
            sampled_words: 7,
            eventful_recurrent_steps: 3,
            emitted_output_words: 7,
            state_bit_flips: 9,
        };
        let zero = LogicalResources::default();
        assert_eq!(a.checked_add(zero), Some(a));
        assert_eq!(
            a.checked_add(b).and_then(|v| v.checked_add(c)),
            b.checked_add(c).and_then(|v| a.checked_add(v))
        );
        assert_eq!(a.checked_add(b), b.checked_add(a));
        let weights = LogicalCostWeights {
            sampled_word: 4,
            recurrent_step: 10,
            emitted_word: 2,
            state_flip: 1,
        };
        assert_eq!(a.weighted_proxy(weights), Some(25));
        assert_eq!(
            a.checked_add(b).unwrap().weighted_proxy(weights),
            Some(a.weighted_proxy(weights).unwrap() + b.weighted_proxy(weights).unwrap())
        );
        assert_eq!(
            LogicalResources {
                sampled_words: u128::MAX,
                ..a
            }
            .checked_add(b),
            None
        );
        assert_eq!(
            LogicalResources {
                sampled_words: u128::MAX,
                ..a
            }
            .weighted_proxy(weights),
            None
        );
    }

    #[test]
    fn malformed_protocol_is_rejected_before_allocations() {
        let mut config = protocol(25);
        config.test_episodes = 0;
        assert_eq!(
            Bps5Dataset::generate(config, teacher(7)),
            Err(Bps5Error::EmptySplit)
        );
        config = protocol(25);
        config.train_episodes = 61;
        assert_eq!(
            Bps5Dataset::generate(config, teacher(7)),
            Err(Bps5Error::TooManyEpisodes)
        );
        config = protocol(25);
        config.steps_per_episode = 0;
        assert_eq!(
            Bps5Dataset::generate(config, teacher(7)),
            Err(Bps5Error::InvalidSequenceLength)
        );
        config = protocol(25);
        config.input_occupancy_percent = 101;
        assert_eq!(
            Bps5Dataset::generate(config, teacher(7)),
            Err(Bps5Error::InvalidOccupancy)
        );
        config = protocol(25);
        config.pulse_lane = Some(64);
        assert_eq!(
            Bps5Dataset::generate(config, teacher(7)),
            Err(Bps5Error::InvalidPulseLane)
        );
    }
}
