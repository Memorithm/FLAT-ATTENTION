//! BPS-16: operation-specific *per-neuron* recurrent connections.
//!
//! BPS-15 selected one common predecessor rotation per operation. Here all
//! 64 packed destination neurons choose their OWN source neuron for each of
//! retention, excitation, inhibition, and toggling. The compiler groups
//! equal shifts so packed word rotations can serve multiple destinations.
//! This is exact synchronous CPU-only Boolean memory; no classical LSTM
//! equivalence, parallel temporal scan or physical speedup is claimed.

use super::boolean_operator_channels::{BooleanOperatorProgram, OperatorChannel};
use super::boolean_scan_lstm::{
    validate_examples, BooleanTrainError, BooleanTrainingSequence, UnaryGate,
};

pub const NEURON_COUNT: usize = 64;
pub const OPERATOR_COUNT: usize = 4;

/// Reject invalid per-neuron source routing without truncation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeuronGraphError {
    InvalidShift {
        channel: OperatorChannel,
        destination: usize,
        shift: u8,
    },
}

/// A source shift of s means destination bit i reads predecessor (i-s)%64.
///
/// Recurrent routing uses 4 x 64 individually selected edges. The gate
/// truth tables remain bit-sliced and input-only. The compiled
/// destinations_by_shift bitmap is a partition of the output bits for
/// each operation (one predecessor per operation per destination).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanNeuronGraph {
    pub gates: [UnaryGate; OPERATOR_COUNT],
    pub write: UnaryGate,
    pub output: UnaryGate,
    shifts: [[u8; NEURON_COUNT]; OPERATOR_COUNT],
    destinations_by_shift: [[u64; NEURON_COUNT]; OPERATOR_COUNT],
    active_shifts: [u64; OPERATOR_COUNT],
    // Specialize any operation whose 64 destination neurons share one
    // predecessor offset, including mixed topologies where other operations
    // still need the heterogeneous grouped kernel.
    channel_uniform_shifts: [Option<u8>; OPERATOR_COUNT],
    // Compiled once when every operator has one shared shift. This directly
    // reuses the exact BPS-15 kernel instead of looping over grouped masks.
    uniform_program: Option<BooleanOperatorProgram>,
}

impl BooleanNeuronGraph {
    /// Build a validated, grouped routing table without unsafe code.
    pub fn new(
        gates: [UnaryGate; OPERATOR_COUNT],
        write: UnaryGate,
        output: UnaryGate,
        shifts: [[u8; NEURON_COUNT]; OPERATOR_COUNT],
    ) -> Result<Self, NeuronGraphError> {
        let mut destinations_by_shift = [[0u64; NEURON_COUNT]; OPERATOR_COUNT];
        let mut active_shifts = [0u64; OPERATOR_COUNT];
        for channel in OperatorChannel::ALL {
            let index = channel.index();
            for destination in 0..NEURON_COUNT {
                let shift = shifts[index][destination];
                if usize::from(shift) >= NEURON_COUNT {
                    return Err(NeuronGraphError::InvalidShift {
                        channel,
                        destination,
                        shift,
                    });
                }
                destinations_by_shift[index][usize::from(shift)] |= 1u64 << destination;
                active_shifts[index] |= 1u64 << shift;
            }
        }
        let channel_uniform_shifts = active_shifts.map(|bits| {
            if bits.count_ones() == 1 {
                Some(bits.trailing_zeros() as u8)
            } else {
                None
            }
        });
        let uniform_program = if channel_uniform_shifts.iter().all(Option::is_some) {
            let uniform_shifts = channel_uniform_shifts.map(|value| {
                value.expect("every channel was proven to be uniform")
            });
            Some(
                BooleanOperatorProgram::new(gates, write, output, uniform_shifts)
                    .expect("validated grouped shifts cannot be invalid"),
            )
        } else {
            None
        };
        Ok(Self {
            gates,
            write,
            output,
            shifts,
            destinations_by_shift,
            active_shifts,
            channel_uniform_shifts,
            uniform_program,
        })
    }

    /// Embed a BPS-15 operator program as 64 identical per-channel offsets.
    ///
    /// The resulting model is bit-exact with BPS-15 for every state/input.
    pub fn from_shared(program: BooleanOperatorProgram) -> Self {
        let gate_masks = [
            program.retain,
            program.excite,
            program.inhibit,
            program.toggle,
        ];
        let shared = program.shifts();
        let shifts = std::array::from_fn(|index| [shared[index]; NEURON_COUNT]);
        Self::new(gate_masks, program.write, program.output, shifts)
            .expect("BPS-15 validates all shared source shifts")
    }

    #[must_use]
    pub fn source_shift(&self, channel: OperatorChannel, destination: usize) -> Option<u8> {
        self.shifts.get(channel.index())?.get(destination).copied()
    }

    /// Return a new graph with one destination neuron rewired on one channel.
    pub fn with_edge(
        &self,
        channel: OperatorChannel,
        destination: usize,
        shift: u8,
    ) -> Result<Self, NeuronGraphError> {
        if destination >= NEURON_COUNT || usize::from(shift) >= NEURON_COUNT {
            return Err(NeuronGraphError::InvalidShift {
                channel,
                destination,
                shift,
            });
        }
        let mut shifts = self.shifts;
        shifts[channel.index()][destination] = shift;
        Self::new(self.gates, self.write, self.output, shifts)
    }

    /// Active unique source rotations per operation before input gating.
    #[must_use]
    pub fn unique_shift_counts(&self) -> [u32; OPERATOR_COUNT] {
        self.active_shifts.map(u64::count_ones)
    }

    /// Present only when all 64 destinations of each operator share one shift.
    ///
    /// Rewiring via `with_edge` invalidates this fast path whenever a
    /// channel acquires a second unique predecessor offset.
    #[must_use]
    pub fn compiled_uniform_shifts(&self) -> Option<[u8; OPERATOR_COUNT]> {
        self.uniform_program.map(|program| program.shifts())
    }

    /// The per-operator static specialization, independent of the other
    /// operators. `Some(shift)` means all 64 destinations of this operator
    /// share that source rotation; `None` means a heterogeneous source set.
    /// Rewiring rebuilds these values with the other compiled routing masks.
    #[must_use]
    pub fn compiled_channel_uniform_shifts(&self) -> [Option<u8>; OPERATOR_COUNT] {
        self.channel_uniform_shifts
    }

    /// Independent scalar oracle: visits all four channels for every neuron.
    ///
    /// All incoming edges read the immutable *previous* state. The operation
    /// order exactly matches BPS-15: (retain OR excite OR write)
    /// AND NOT inhibit, then XOR toggle, then output gate.
    #[must_use]
    pub fn step_scalar(&self, input: u64, previous: u64) -> (u64, u64) {
        let gate_masks = self.gates.map(|gate| gate.evaluate(input));
        let direct_write = self.write.evaluate(input);
        let mut state = 0u64;
        for lane in 0..NEURON_COUNT {
            let mut sources = [0u64; OPERATOR_COUNT];
            for channel in 0..OPERATOR_COUNT {
                let shift = usize::from(self.shifts[channel][lane]);
                let from = (lane + NEURON_COUNT - shift) % NEURON_COUNT;
                let predecessor = (previous >> from) & 1;
                let enabled = (gate_masks[channel] >> lane) & 1;
                sources[channel] = predecessor & enabled;
            }
            let written = (direct_write >> lane) & 1;
            let next_bit = ((sources[0] | sources[1] | written) & !sources[2]) ^ sources[3];
            state |= (next_bit & 1) << lane;
        }
        (state, state & self.output.evaluate(input))
    }

    /// Packed execution using a precompiled mask for each unique rotation.
    ///
    /// No heap allocation on each tick. With U distinct shifts per operation
    /// (U <= 64), at most sum(U) rotations are performed per timestep.
    /// Results are bit-exact with step_scalar, including arbitrary topology.
    #[must_use]
    pub fn step_grouped(&self, input: u64, previous: u64) -> (u64, u64) {
        if let Some(program) = self.uniform_program {
            // Same topology, gates and synchronous old state as BPS-15:
            // bypass the generic per-channel bitmask dispatch completely.
            program.step_dense(input, previous)
        } else {
            self.step_grouped_impl(input, previous, false)
        }
    }

    /// Skip all rotations of an operation when its input gate word is zero.
    ///
    /// Branch overhead is not guaranteed to outperform step_grouped.
    #[must_use]
    pub fn step_grouped_sparse(&self, input: u64, previous: u64) -> (u64, u64) {
        if let Some(program) = self.uniform_program {
            program.step_sparse(input, previous)
        } else {
            self.step_grouped_impl(input, previous, true)
        }
    }

    fn step_grouped_impl(&self, input: u64, previous: u64, skip_disabled: bool) -> (u64, u64) {
        let gate_masks = self.gates.map(|gate| gate.evaluate(input));
        let mut sourced = [0u64; OPERATOR_COUNT];
        for channel in 0..OPERATOR_COUNT {
            if skip_disabled && gate_masks[channel] == 0 {
                continue;
            }
            if let Some(shift) = self.channel_uniform_shifts[channel] {
                // This channel is uniform even if the other channels are
                // heterogeneous. Its unique source rotation can be applied
                // directly, without scanning shift masks or OR-reducing.
                sourced[channel] = previous.rotate_left(shift as u32) & gate_masks[channel];
                continue;
            }
            let mut enabled = self.active_shifts[channel];
            while enabled != 0 {
                let shift = enabled.trailing_zeros() as usize;
                enabled &= enabled - 1;
                let destinations = self.destinations_by_shift[channel][shift];
                if skip_disabled {
                    // A partially active per-neuron gate can disable a
                    // particular shift group, even when another group in
                    // this same operator still contributes.
                    let enabled_destinations = destinations & gate_masks[channel];
                    if enabled_destinations != 0 {
                        sourced[channel] |=
                            previous.rotate_left(shift as u32) & enabled_destinations;
                    }
                } else {
                    sourced[channel] |= previous.rotate_left(shift as u32) & destinations;
                }
            }
            if !skip_disabled {
                sourced[channel] &= gate_masks[channel];
            }
        }
        let base = sourced[0] | sourced[1] | self.write.evaluate(input);
        let next = (base & !sourced[2]) ^ sourced[3];
        (next, next & self.output.evaluate(input))
    }

    #[must_use]
    pub fn forward_scalar(&self, inputs: &[u64], initial: u64) -> Vec<u64> {
        let mut state = initial;
        inputs
            .iter()
            .map(|&input| {
                let (next, output) = self.step_scalar(input, state);
                state = next;
                output
            })
            .collect()
    }

    #[must_use]
    pub fn forward_grouped(&self, inputs: &[u64], initial: u64) -> Vec<u64> {
        let mut state = initial;
        inputs
            .iter()
            .map(|&input| {
                let (next, output) = self.step_grouped(input, state);
                state = next;
                output
            })
            .collect()
    }

    #[must_use]
    pub fn forward_grouped_sparse(&self, inputs: &[u64], initial: u64) -> Vec<u64> {
        let mut state = initial;
        inputs
            .iter()
            .map(|&input| {
                let (next, output) = self.step_grouped_sparse(input, state);
                state = next;
                output
            })
            .collect()
    }

    pub fn hamming_loss(
        &self,
        examples: &[BooleanTrainingSequence<'_>],
        active_output_bits: u64,
    ) -> Result<u64, BooleanTrainError> {
        validate_examples(examples, active_output_bits)?;
        self.loss_validated(examples, active_output_bits)
    }

    fn loss_validated(
        &self,
        examples: &[BooleanTrainingSequence<'_>],
        active_output_bits: u64,
    ) -> Result<u64, BooleanTrainError> {
        let mut loss = 0u64;
        for example in examples {
            let mut state = example.initial_state;
            for (&input, &target) in example.inputs.iter().zip(example.targets.iter()) {
                let (next, output) = self.step_grouped(input, state);
                state = next;
                let misses = u64::from(((output ^ target) & active_output_bits).count_ones());
                loss = loss
                    .checked_add(misses)
                    .ok_or(BooleanTrainError::LossOverflow)?;
            }
        }
        Ok(loss)
    }

    /// Bounded, whole-trajectory, strict-improvement training of individual
    /// source edges. Every accepted rewire changes only ONE channel/destination.
    /// Gates and direct read/write logic are fixed. On a plateau, learning
    /// stops; this is not globally optimal topology search.
    ///
    /// Validation is evaluated before/after only. Invalid input or loss
    /// overflow does not mutate the original graph.
    pub fn fit_edges(
        &mut self,
        training: &[BooleanTrainingSequence<'_>],
        validation: &[BooleanTrainingSequence<'_>],
        active_output_bits: u64,
        policy: NeuronTrainPolicy,
    ) -> Result<NeuronFitReport, NeuronFitError> {
        validate_examples(training, active_output_bits).map_err(NeuronFitError::Training)?;
        validate_examples(validation, active_output_bits).map_err(NeuronFitError::Validation)?;
        if policy.max_epochs == 0 {
            return Err(NeuronFitError::Training(BooleanTrainError::ZeroEpochs));
        }
        if policy.max_candidates == 0 {
            return Err(NeuronFitError::ZeroCandidateBudget);
        }

        let train_before = self
            .loss_validated(training, active_output_bits)
            .map_err(NeuronFitError::Training)?;
        let validation_before = self
            .loss_validated(validation, active_output_bits)
            .map_err(NeuronFitError::Validation)?;

        let mut current = self.clone();
        let mut loss = train_before;
        let mut epochs = 0usize;
        let mut evaluated = 0usize;
        let mut accepted = 0usize;
        let mut exhausted = false;
        'epochs: for _ in 0..policy.max_epochs {
            if loss == 0 {
                break;
            }
            epochs += 1;
            let mut changed = false;
            for channel in OperatorChannel::ALL {
                for destination in 0..NEURON_COUNT {
                    for shift in 0..NEURON_COUNT as u8 {
                        if shift == current.shifts[channel.index()][destination] {
                            continue;
                        }
                        if evaluated == policy.max_candidates {
                            exhausted = true;
                            break 'epochs;
                        }
                        let candidate = current
                            .with_edge(channel, destination, shift)
                            .expect("candidate source shift and destination are valid");
                        let candidate_loss = candidate
                            .loss_validated(training, active_output_bits)
                            .map_err(NeuronFitError::Training)?;
                        evaluated += 1;
                        if candidate_loss < loss {
                            current = candidate;
                            loss = candidate_loss;
                            accepted += 1;
                            changed = true;
                        }
                        if loss == 0 {
                            break 'epochs;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let validation_after = current
            .loss_validated(validation, active_output_bits)
            .map_err(NeuronFitError::Validation)?;
        let report = NeuronFitReport {
            train_before,
            train_after: loss,
            validation_before,
            validation_after,
            candidates_evaluated: evaluated,
            accepted_rewires: accepted,
            epochs_started: epochs,
            budget_exhausted: exhausted,
        };
        *self = current;
        Ok(report)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeuronTrainPolicy {
    pub max_epochs: usize,
    pub max_candidates: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeuronFitError {
    Training(BooleanTrainError),
    Validation(BooleanTrainError),
    ZeroCandidateBudget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeuronFitReport {
    pub train_before: u64,
    pub train_after: u64,
    pub validation_before: u64,
    pub validation_after: u64,
    pub candidates_evaluated: usize,
    pub accepted_rewires: usize,
    pub epochs_started: usize,
    pub budget_exhausted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn always() -> UnaryGate {
        UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        }
    }

    fn gate(seed: &mut u64) -> UnaryGate {
        UnaryGate {
            when_false: next_random(seed),
            when_true: next_random(seed),
        }
    }

    fn shared() -> BooleanOperatorProgram {
        BooleanOperatorProgram::new(
            [always(), always(), always(), always()],
            UnaryGate::default(),
            always(),
            [1, 3, 5, 7],
        )
        .unwrap()
    }

    fn sample<'a>(
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

    #[test]
    fn shared_source_topology_matches_bps15_for_all_modes() {
        let mut seed = 0x48cc_55bb_f01e_241du64;
        for _ in 0..12 {
            let p = BooleanOperatorProgram::new(
                [
                    gate(&mut seed),
                    gate(&mut seed),
                    gate(&mut seed),
                    gate(&mut seed),
                ],
                gate(&mut seed),
                gate(&mut seed),
                [1, 3, 17, 63],
            )
            .unwrap();
            let g = BooleanNeuronGraph::from_shared(p);
            assert_eq!(g.unique_shift_counts(), [1; OPERATOR_COUNT]);
            let inputs: Vec<u64> = (0..65).map(|_| next_random(&mut seed)).collect();
            let initial = next_random(&mut seed);
            assert_eq!(
                g.forward_scalar(&inputs, initial),
                p.forward_dense(&inputs, initial)
            );
            assert_eq!(
                g.forward_grouped(&inputs, initial),
                p.forward_dense(&inputs, initial)
            );
            assert_eq!(
                g.forward_grouped_sparse(&inputs, initial),
                p.forward_sparse(&inputs, initial)
            );
        }
    }

    #[test]
    fn randomized_per_neuron_routing_matches_independent_scalar_oracle() {
        let mut seed = 0x8172_fa43_9a52_2bcdu64;
        for _ in 0..48 {
            let shifts = std::array::from_fn(|_| {
                std::array::from_fn(|_| (next_random(&mut seed) % 64) as u8)
            });
            let p = BooleanNeuronGraph::new(
                [
                    gate(&mut seed),
                    gate(&mut seed),
                    gate(&mut seed),
                    gate(&mut seed),
                ],
                gate(&mut seed),
                gate(&mut seed),
                shifts,
            )
            .unwrap();
            let inputs: Vec<u64> = (0..31).map(|_| next_random(&mut seed)).collect();
            let initial = next_random(&mut seed);
            assert_eq!(
                p.forward_grouped(&inputs, initial),
                p.forward_scalar(&inputs, initial)
            );
            assert_eq!(
                p.forward_grouped_sparse(&inputs, initial),
                p.forward_scalar(&inputs, initial)
            );
        }
    }

    #[test]
    fn independent_connections_can_split_one_bit_into_two_distinct_destinations() {
        let retain = UnaryGate {
            when_false: (1u64 << 1) | (1u64 << 3),
            when_true: (1u64 << 1) | (1u64 << 3),
        };
        let mut shifts = [[0u8; NEURON_COUNT]; OPERATOR_COUNT];
        shifts[0][1] = 1;
        shifts[0][3] = 3;
        let p = BooleanNeuronGraph::new(
            [
                retain,
                UnaryGate::default(),
                UnaryGate::default(),
                UnaryGate::default(),
            ],
            UnaryGate::default(),
            always(),
            shifts,
        )
        .unwrap();
        assert_eq!(p.forward_grouped(&[0], 1), vec![0b1010]);
        assert_eq!(p.source_shift(OperatorChannel::Retain, 1), Some(1));
        assert_eq!(p.source_shift(OperatorChannel::Retain, 3), Some(3));
        assert_eq!(p.source_shift(OperatorChannel::Retain, 64), None);
        assert_eq!(p.unique_shift_counts()[0], 3);
    }

    #[test]
    fn per_neuron_source_offsets_are_learned_without_validation_leakage() {
        let retain = UnaryGate {
            when_false: (1u64 << 1) | (1u64 << 3),
            when_true: (1u64 << 1) | (1u64 << 3),
        };
        let base = BooleanNeuronGraph::new(
            [
                retain,
                UnaryGate::default(),
                UnaryGate::default(),
                UnaryGate::default(),
            ],
            UnaryGate::default(),
            always(),
            [[0u8; NEURON_COUNT]; OPERATOR_COUNT],
        )
        .unwrap();
        let teacher = base
            .with_edge(OperatorChannel::Retain, 1, 1)
            .unwrap()
            .with_edge(OperatorChannel::Retain, 3, 3)
            .unwrap();
        let training_inputs = [0];
        let training_targets = teacher.forward_scalar(&training_inputs, 1);
        let validation_inputs = [0, 0];
        let validation_targets = teacher.forward_scalar(&validation_inputs, u64::MAX);
        let train = [sample(&training_inputs, &training_targets, 1)];
        let heldout = [sample(&validation_inputs, &validation_targets, u64::MAX)];
        let mut learner = base;
        let report = learner
            .fit_edges(
                &train,
                &heldout,
                u64::MAX,
                NeuronTrainPolicy {
                    max_epochs: 1,
                    max_candidates: 300,
                },
            )
            .unwrap();
        assert!(report.train_before > 0);
        assert_eq!(report.train_after, 0);
        assert_eq!(report.validation_after, 0);
        assert_eq!(learner.source_shift(OperatorChannel::Retain, 1), Some(1));
        assert_eq!(learner.source_shift(OperatorChannel::Retain, 3), Some(3));
        assert!(report.candidates_evaluated <= 300);
    }

    #[test]
    fn invalid_edges_and_invalid_learning_are_atomic() {
        let g = BooleanNeuronGraph::from_shared(shared());
        assert_eq!(
            g.with_edge(OperatorChannel::Excite, 64, 2),
            Err(NeuronGraphError::InvalidShift {
                channel: OperatorChannel::Excite,
                destination: 64,
                shift: 2
            })
        );
        assert_eq!(
            g.with_edge(OperatorChannel::Toggle, 63, 64),
            Err(NeuronGraphError::InvalidShift {
                channel: OperatorChannel::Toggle,
                destination: 63,
                shift: 64
            })
        );
        let inputs = [0];
        let targets = [0];
        let train = [sample(&inputs, &targets, 0)];
        let mut learner = g.clone();
        assert_eq!(
            learner.fit_edges(
                &train,
                &train,
                0,
                NeuronTrainPolicy {
                    max_epochs: 1,
                    max_candidates: 2,
                }
            ),
            Err(NeuronFitError::Training(
                BooleanTrainError::EmptyActiveLanes
            ))
        );
        assert_eq!(learner, g);
        assert_eq!(
            learner.fit_edges(
                &train,
                &train,
                1,
                NeuronTrainPolicy {
                    max_epochs: 1,
                    max_candidates: 0,
                }
            ),
            Err(NeuronFitError::ZeroCandidateBudget)
        );
        assert_eq!(learner, g);
    }

    #[test]
    fn strict_budget_and_holdout_targets_do_not_change_model_selection() {
        let initial = BooleanNeuronGraph::from_shared(shared());
        let inputs = [0, 0];
        let targets = [1, 2];
        let other_targets = [u64::MAX, 0];
        let training = [sample(&inputs, &targets, 1)];
        let validation_a = [sample(&inputs, &targets, 1)];
        let validation_b = [sample(&inputs, &other_targets, 1)];
        let mut a = initial.clone();
        let mut b = initial;
        let policy = NeuronTrainPolicy {
            max_epochs: 2,
            max_candidates: 1,
        };
        let ar = a
            .fit_edges(&training, &validation_a, u64::MAX, policy)
            .unwrap();
        let br = b
            .fit_edges(&training, &validation_b, u64::MAX, policy)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(ar.candidates_evaluated, 1);
        assert_eq!(br.candidates_evaluated, 1);
        assert!(ar.budget_exhausted && br.budget_exhausted);
    }

    #[test]
    fn bps17_compiled_uniform_path_is_invalidated_and_restored_by_rewiring() {
        let model = shared();
        let original = BooleanNeuronGraph::from_shared(model);
        assert_eq!(original.compiled_uniform_shifts(), Some([1, 3, 5, 7]));
        let heterogeneous = original.with_edge(OperatorChannel::Retain, 7, 11).unwrap();
        assert_eq!(heterogeneous.compiled_uniform_shifts(), None);
        assert_eq!(heterogeneous.unique_shift_counts(), [2, 1, 1, 1]);
        let restored = heterogeneous
            .with_edge(OperatorChannel::Retain, 7, 1)
            .unwrap();
        assert_eq!(restored.compiled_uniform_shifts(), Some([1, 3, 5, 7]));
        assert_eq!(restored, original);

        let mut seed = 0xb175_221a_a102_aeffu64;
        for _ in 0..128 {
            let previous = next_random(&mut seed);
            let input = next_random(&mut seed);
            assert_eq!(
                original.step_grouped(input, previous),
                model.step_dense(input, previous)
            );
            assert_eq!(
                original.step_grouped_sparse(input, previous),
                model.step_sparse(input, previous)
            );
            assert_eq!(
                heterogeneous.step_grouped(input, previous),
                heterogeneous.step_scalar(input, previous)
            );
            assert_eq!(
                heterogeneous.step_grouped_sparse(input, previous),
                heterogeneous.step_scalar(input, previous)
            );
        }
    }

    #[test]
    fn bps17_partial_lane_gates_are_exact_with_nonuniform_group_short_circuit() {
        let mut shifts = [[0u8; NEURON_COUNT]; OPERATOR_COUNT];
        for destination in 0..NEURON_COUNT {
            shifts[0][destination] = (destination % 7) as u8;
            shifts[1][destination] = ((destination * 3) % 11) as u8;
            shifts[2][destination] = ((destination * 5) % 13) as u8;
            shifts[3][destination] = ((destination * 7) % 17) as u8;
        }
        let mut seed = 0x8bfe_943d_5225_a1e7u64;
        let program = BooleanNeuronGraph::new(
            [
                UnaryGate {
                    when_false: 0,
                    when_true: 0x0000_0000_0000_0001,
                },
                UnaryGate {
                    when_false: 0x4000_0000_0000_0000,
                    when_true: 0x0000_0000_0000_8000,
                },
                UnaryGate {
                    when_false: 0,
                    when_true: 0x0100_0000_0000_0000,
                },
                UnaryGate {
                    when_false: 0x0000_1000_0000_0000,
                    when_true: 0x0000_0010_0000_0000,
                },
            ],
            gate(&mut seed),
            gate(&mut seed),
            shifts,
        )
        .unwrap();
        assert_eq!(program.compiled_uniform_shifts(), None);
        for _ in 0..192 {
            let input = next_random(&mut seed);
            let previous = next_random(&mut seed);
            let oracle = program.step_scalar(input, previous);
            assert_eq!(program.step_grouped(input, previous), oracle);
            assert_eq!(program.step_grouped_sparse(input, previous), oracle);
        }
        let inputs: Vec<u64> = (0..137)
            .map(|index| match index % 4 {
                0 => 0,
                1 => u64::MAX,
                _ => next_random(&mut seed),
            })
            .collect();
        let initial = next_random(&mut seed);
        assert_eq!(
            program.forward_grouped_sparse(&inputs, initial),
            program.forward_scalar(&inputs, initial)
        );
    }
}
