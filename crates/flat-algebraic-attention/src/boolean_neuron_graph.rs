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
    // Exact Boolean constraints on input bits which make ALL four
    // recurrent channel gate words fully active. An impossible guard
    // stays None and never triggers the dense execution fast path.
    dense_gate_guard: Option<DenseGateGuard>,
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
            let uniform_shifts = channel_uniform_shifts
                .map(|value| value.expect("every channel was proven to be uniform"));
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
            dense_gate_guard: DenseGateGuard::compile(&gates),
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

    /// Report the exact input predicate (required-one bits, required-zero
    /// bits) which makes all four recurrent gate words fully active.
    /// Returns None if no input can satisfy all four Boolean gate tables.
    #[must_use]
    pub fn compiled_dense_gate_guard(&self) -> Option<(u64, u64)> {
        self.dense_gate_guard
            .map(|guard| (guard.required_one, guard.required_zero))
    }

    /// True precisely when all four channel masks evaluate to u64::MAX.
    /// The predicate is compiled once from the Boolean gate truth tables.
    #[must_use]
    pub fn dense_gate_eligible(&self, input: u64) -> bool {
        self.dense_gate_guard
            .is_some_and(|guard| guard.matches(input))
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

    /// Inspect which exact routing kernel will be selected for each
    /// operation, given the input's active destination bits and a budget
    /// of direct predecessor reads. This does not inspect model outputs
    /// or hidden source state; the decision is deterministic.
    #[must_use]
    pub fn adaptive_route_plan(
        &self,
        input: u64,
        policy: AdaptiveRoutingPolicy,
    ) -> [AdaptiveRoute; OPERATOR_COUNT] {
        let masks = self.gates.map(|gate| gate.evaluate(input));
        std::array::from_fn(|channel| self.choose_adaptive_route(channel, masks[channel], policy))
    }

    fn choose_adaptive_route(
        &self,
        channel: usize,
        enabled_destinations: u64,
        policy: AdaptiveRoutingPolicy,
    ) -> AdaptiveRoute {
        if enabled_destinations == 0 {
            return AdaptiveRoute::Dormant;
        }
        if self.channel_uniform_shifts[channel].is_some() {
            return AdaptiveRoute::Uniform;
        }
        let active_lanes = enabled_destinations.count_ones();
        let unique_shifts = self.active_shifts[channel].count_ones();
        // Only take the scalar destination path when the actual number of
        // active lanes is smaller than the compiled topology's number of
        // unique rotation groups. No timer, hidden-state read or heuristic
        // input prefetch is needed to make this conservative choice.
        if active_lanes <= u32::from(policy.max_direct_lanes[channel])
            && active_lanes < unique_shifts
        {
            AdaptiveRoute::DirectLanes
        } else {
            AdaptiveRoute::Grouped
        }
    }

    /// Exact alternative to grouped sparse execution. Each operation picks
    /// a deterministic source kernel based on its live input gate mask and
    /// compiled topology:
    ///
    /// - Dormant: no recurrent source read for a zero gate word.
    /// - Uniform: one rotated word for the shared source offset.
    /// - DirectLanes: only enabled destination bits are visited and each
    ///   reads its own predecessor bit from the previous state.
    /// - Grouped: visit enabled per-shift destination groups.
    ///
    /// All four channels read the SAME old state and preserve the exact
    /// retain/OR, excitation/OR, inhibition/AND NOT and toggle/XOR order.
    /// This is a CPU research policy, not a measured throughput claim.
    #[must_use]
    pub fn step_adaptive(
        &self,
        input: u64,
        previous: u64,
        policy: AdaptiveRoutingPolicy,
    ) -> (u64, u64) {
        if let Some(program) = self.uniform_program {
            // BPS-17's exact whole-model fast path remains the best
            // non-allocating dispatch for fully uniform connectivity.
            return program.step_sparse(input, previous);
        }
        let masks = self.gates.map(|gate| gate.evaluate(input));
        let mut sourced = [0u64; OPERATOR_COUNT];
        for channel in 0..OPERATOR_COUNT {
            match self.choose_adaptive_route(channel, masks[channel], policy) {
                AdaptiveRoute::Dormant => {}
                AdaptiveRoute::Uniform => {
                    let shift = self.channel_uniform_shifts[channel]
                        .expect("Uniform implies a validated one-shift channel");
                    sourced[channel] = previous.rotate_left(shift as u32) & masks[channel];
                }
                AdaptiveRoute::DirectLanes => {
                    // Each destination chooses exactly one source per
                    // operation. A bit is read only when its gate is active.
                    let mut lanes = masks[channel];
                    while lanes != 0 {
                        let destination = lanes.trailing_zeros() as usize;
                        lanes &= lanes - 1;
                        let shift = usize::from(self.shifts[channel][destination]);
                        let source = (destination + NEURON_COUNT - shift) % NEURON_COUNT;
                        sourced[channel] |= ((previous >> source) & 1) << destination;
                    }
                }
                AdaptiveRoute::Grouped => {
                    let mut shifts = self.active_shifts[channel];
                    while shifts != 0 {
                        let shift = shifts.trailing_zeros() as usize;
                        shifts &= shifts - 1;
                        let enabled = self.destinations_by_shift[channel][shift] & masks[channel];
                        if enabled != 0 {
                            sourced[channel] |= previous.rotate_left(shift as u32) & enabled;
                        }
                    }
                }
            }
        }
        let base = sourced[0] | sourced[1] | self.write.evaluate(input);
        let next = (base & !sourced[2]) ^ sourced[3];
        (next, next & self.output.evaluate(input))
    }

    /// BPS-20 measured-density hybrid. If the compiled truth-table
    /// predicate proves that ALL four recurrent masks contain 64 enabled
    /// lanes, the BPS-18 dense grouped kernel is selected directly.
    /// Otherwise the exact BPS-19 sparse, topology-dependent algorithm
    /// is used. The branch has no uncertain arithmetic or output lookahead.
    ///
    /// The dense and adaptive paths both consume the same old recurrent
    /// state and produce identical next state and output bit patterns.
    #[must_use]
    pub fn step_activity_hybrid(
        &self,
        input: u64,
        previous: u64,
        policy: AdaptiveRoutingPolicy,
    ) -> (u64, u64) {
        if self.dense_gate_eligible(input) {
            self.step_grouped(input, previous)
        } else {
            self.step_adaptive(input, previous, policy)
        }
    }

    /// Multi-step exact hybrid recurrence: dense word inputs are selected
    /// through the compiled guard; sparse inputs retain BPS-19 routing.
    #[must_use]
    pub fn forward_activity_hybrid(
        &self,
        inputs: &[u64],
        initial: u64,
        policy: AdaptiveRoutingPolicy,
    ) -> Vec<u64> {
        let mut state = initial;
        inputs
            .iter()
            .map(|&input| {
                let (next, output) = self.step_activity_hybrid(input, state, policy);
                state = next;
                output
            })
            .collect()
    }

    /// Exact synchronous recurrence with O(T) steps and O(1) auxiliary
    /// recurrent state, excluding the dense returned output vector.
    #[must_use]
    pub fn forward_adaptive(
        &self,
        inputs: &[u64],
        initial: u64,
        policy: AdaptiveRoutingPolicy,
    ) -> Vec<u64> {
        let mut state = initial;
        inputs
            .iter()
            .map(|&input| {
                let (next, output) = self.step_adaptive(input, state, policy);
                state = next;
                output
            })
            .collect()
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

/// Compile the exact condition under which every recurrent gate word
/// contains 64 active lanes. For each input lane, all four gate tables
/// must admit at least one input value; incompatible requirements across
/// channels make this guard impossible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DenseGateGuard {
    required_one: u64,
    required_zero: u64,
}

impl DenseGateGuard {
    fn compile(gates: &[UnaryGate; OPERATOR_COUNT]) -> Option<Self> {
        let mut required_one = 0u64;
        let mut required_zero = 0u64;
        for gate in gates {
            if (gate.when_false | gate.when_true) != u64::MAX {
                return None;
            }
            required_one |= gate.when_true & !gate.when_false;
            required_zero |= gate.when_false & !gate.when_true;
        }
        if (required_one & required_zero) != 0 {
            return None;
        }
        Some(Self {
            required_one,
            required_zero,
        })
    }

    #[must_use]
    fn matches(self, input: u64) -> bool {
        (input & self.required_one) == self.required_one && (input & self.required_zero) == 0
    }
}

/// The exact kernel selected for one operation at one timestep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveRoute {
    Dormant,
    Uniform,
    DirectLanes,
    Grouped,
}

/// Upper bound on direct predecessor bit reads per operation and timestep.
///
/// The path is selected only if the active destination lane count is
/// strictly smaller than the count of distinct compiled shift groups.
/// A zero threshold disables the direct-lane path. Values above 64
/// have the same functional effect as 64; they are not invalid because
/// this is only a work cap, not an index into the neuron graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveRoutingPolicy {
    pub max_direct_lanes: [u8; OPERATOR_COUNT],
}

impl Default for AdaptiveRoutingPolicy {
    fn default() -> Self {
        Self {
            max_direct_lanes: [4; OPERATOR_COUNT],
        }
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

    #[test]
    fn bps18_one_channel_rewire_keeps_other_three_compiled_uniform() {
        let original = BooleanNeuronGraph::from_shared(shared());
        assert_eq!(
            original.compiled_channel_uniform_shifts(),
            [Some(1), Some(3), Some(5), Some(7)]
        );

        let hetero = original.with_edge(OperatorChannel::Retain, 7, 11).unwrap();
        assert_eq!(hetero.compiled_uniform_shifts(), None);
        assert_eq!(
            hetero.compiled_channel_uniform_shifts(),
            [None, Some(3), Some(5), Some(7)]
        );
        assert_eq!(hetero.unique_shift_counts(), [2, 1, 1, 1]);

        let more_hetero = hetero.with_edge(OperatorChannel::Toggle, 9, 21).unwrap();
        assert_eq!(
            more_hetero.compiled_channel_uniform_shifts(),
            [None, Some(3), Some(5), None]
        );
        let restored = more_hetero
            .with_edge(OperatorChannel::Retain, 7, 1)
            .unwrap();
        assert_eq!(
            restored.compiled_channel_uniform_shifts(),
            [Some(1), Some(3), Some(5), None]
        );
        let all_restored = restored.with_edge(OperatorChannel::Toggle, 9, 7).unwrap();
        assert_eq!(all_restored, original);
        assert_eq!(all_restored.compiled_uniform_shifts(), Some([1, 3, 5, 7]));

        let mut seed = 0x31aa_6574_f41c_10ddu64;
        for program in [&original, &hetero, &more_hetero, &restored] {
            for _ in 0..32 {
                let input = next_random(&mut seed);
                let previous = next_random(&mut seed);
                let oracle = program.step_scalar(input, previous);
                assert_eq!(program.step_grouped(input, previous), oracle);
                assert_eq!(program.step_grouped_sparse(input, previous), oracle);
            }
        }
    }

    #[test]
    fn bps18_randomized_mixed_channel_specialization_matches_scalar() {
        let mut seed = 0xba10_b418_983c_22d1u64;
        for _ in 0..48 {
            // Deliberately keep three operator channels uniform. Assign
            // genuinely independent predecessor choices to the fourth.
            let shifts = std::array::from_fn(|index| {
                std::array::from_fn(|destination| {
                    if index == 2 {
                        ((destination * 11) % 64) as u8
                    } else {
                        (next_random(&mut seed) % 64) as u8
                    }
                })
            });
            let mut shared_shifts = shifts;
            for channel in [0, 1, 3] {
                let shift = shifts[channel][0];
                shared_shifts[channel] = [shift; NEURON_COUNT];
            }
            let program = BooleanNeuronGraph::new(
                [
                    gate(&mut seed),
                    gate(&mut seed),
                    gate(&mut seed),
                    gate(&mut seed),
                ],
                gate(&mut seed),
                gate(&mut seed),
                shared_shifts,
            )
            .unwrap();
            let compiled = program.compiled_channel_uniform_shifts();
            assert!(compiled[0].is_some());
            assert!(compiled[1].is_some());
            assert_eq!(compiled[2], None);
            assert!(compiled[3].is_some());
            let inputs: Vec<u64> = (0..77).map(|_| next_random(&mut seed)).collect();
            let initial = next_random(&mut seed);
            assert_eq!(
                program.forward_grouped(&inputs, initial),
                program.forward_scalar(&inputs, initial)
            );
            assert_eq!(
                program.forward_grouped_sparse(&inputs, initial),
                program.forward_scalar(&inputs, initial)
            );
        }
    }

    #[test]
    fn bps19_adaptive_plan_selects_direct_lanes_only_when_bounded_and_useful() {
        let mut shifts = [[0u8; NEURON_COUNT]; OPERATOR_COUNT];
        for destination in 0..NEURON_COUNT {
            shifts[0][destination] = destination as u8;
        }
        shifts[1] = [3; NEURON_COUNT];
        shifts[2] = [5; NEURON_COUNT];
        shifts[3] = [7; NEURON_COUNT];
        let selective = UnaryGate {
            when_false: (1u64 << 1) | (1u64 << 61),
            when_true: u64::MAX,
        };
        let input_switch = UnaryGate {
            when_false: 0,
            when_true: u64::MAX,
        };
        let program = BooleanNeuronGraph::new(
            [
                selective,
                input_switch,
                UnaryGate::default(),
                UnaryGate::default(),
            ],
            UnaryGate::default(),
            always(),
            shifts,
        )
        .unwrap();
        assert_eq!(program.unique_shift_counts(), [64, 1, 1, 1]);
        assert_eq!(program.compiled_uniform_shifts(), None);
        let policy = AdaptiveRoutingPolicy::default();
        assert_eq!(
            program.adaptive_route_plan(0, policy),
            [
                AdaptiveRoute::DirectLanes,
                AdaptiveRoute::Dormant,
                AdaptiveRoute::Dormant,
                AdaptiveRoute::Dormant,
            ]
        );
        assert_eq!(
            program.adaptive_route_plan(u64::MAX, policy),
            [
                AdaptiveRoute::Grouped,
                AdaptiveRoute::Uniform,
                AdaptiveRoute::Dormant,
                AdaptiveRoute::Dormant,
            ]
        );
        let no_scalar = AdaptiveRoutingPolicy {
            max_direct_lanes: [0; OPERATOR_COUNT],
        };
        assert_eq!(
            program.adaptive_route_plan(0, no_scalar)[0],
            AdaptiveRoute::Grouped
        );
        let only_one = AdaptiveRoutingPolicy {
            max_direct_lanes: [1; OPERATOR_COUNT],
        };
        assert_eq!(
            program.adaptive_route_plan(0, only_one)[0],
            AdaptiveRoute::Grouped
        );
        let generous = AdaptiveRoutingPolicy {
            max_direct_lanes: [64; OPERATOR_COUNT],
        };
        assert_eq!(
            program.adaptive_route_plan(0, generous)[0],
            AdaptiveRoute::DirectLanes
        );
        assert_eq!(
            program.adaptive_route_plan(u64::MAX, generous)[0],
            AdaptiveRoute::Grouped
        );
        let mut seed = 0x80ad_352a_1937_64b1u64;
        for input in [0, u64::MAX] {
            for _ in 0..16 {
                let previous = next_random(&mut seed);
                for current_policy in [policy, no_scalar, only_one, generous] {
                    assert_eq!(
                        program.step_adaptive(input, previous, current_policy),
                        program.step_scalar(input, previous)
                    );
                }
            }
        }
    }

    #[test]
    fn bps19_random_per_neuron_graph_and_gate_masks_are_bit_exact() {
        let mut seed = 0x91f6_f942_13b4_a55au64;
        let policies = [
            AdaptiveRoutingPolicy {
                max_direct_lanes: [0; OPERATOR_COUNT],
            },
            AdaptiveRoutingPolicy::default(),
            AdaptiveRoutingPolicy {
                max_direct_lanes: [1, 2, 16, 64],
            },
            AdaptiveRoutingPolicy {
                max_direct_lanes: [u8::MAX; OPERATOR_COUNT],
            },
        ];
        for topology in 0..32 {
            let shifts = std::array::from_fn(|channel| {
                std::array::from_fn(|destination| {
                    if topology % 4 == 0 && channel != 2 {
                        (channel * 7) as u8
                    } else if topology % 4 == 1 && channel == 0 && destination > 0 {
                        3
                    } else {
                        (next_random(&mut seed) % 64) as u8
                    }
                })
            });
            let gates = [
                gate(&mut seed),
                gate(&mut seed),
                gate(&mut seed),
                gate(&mut seed),
            ];
            let program =
                BooleanNeuronGraph::new(gates, gate(&mut seed), gate(&mut seed), shifts).unwrap();
            let inputs: Vec<u64> = (0..73)
                .map(|index| match index % 11 {
                    0 => 0,
                    1 => u64::MAX,
                    2 => 1,
                    3 => 1 << 63,
                    _ => next_random(&mut seed),
                })
                .collect();
            let initial = next_random(&mut seed);
            let expected = program.forward_scalar(&inputs, initial);
            for policy in policies {
                assert_eq!(
                    program.forward_adaptive(&inputs, initial, policy),
                    expected,
                    "topology={topology} policy={policy:?}"
                );
            }
            assert_eq!(
                program.forward_adaptive(&[], initial, AdaptiveRoutingPolicy::default()),
                Vec::<u64>::new()
            );
        }
    }

    #[test]
    fn bps19_rewiring_preserves_adaptive_plan_and_full_trajectory() {
        let mut seed = 0x61f9_3c73_a087_14bdu64;
        let base = BooleanNeuronGraph::from_shared(
            BooleanOperatorProgram::new(
                [
                    gate(&mut seed),
                    gate(&mut seed),
                    gate(&mut seed),
                    gate(&mut seed),
                ],
                gate(&mut seed),
                always(),
                [1, 3, 5, 7],
            )
            .unwrap(),
        );
        let policy = AdaptiveRoutingPolicy::default();
        assert_eq!(base.compiled_uniform_shifts(), Some([1, 3, 5, 7]));
        let changed = base
            .with_edge(OperatorChannel::Retain, 11, 25)
            .unwrap()
            .with_edge(OperatorChannel::Retain, 41, 19)
            .unwrap();
        assert_eq!(changed.compiled_uniform_shifts(), None);
        assert_eq!(
            changed.compiled_channel_uniform_shifts(),
            [None, Some(3), Some(5), Some(7)]
        );
        let inputs: Vec<u64> = (0..257).map(|_| next_random(&mut seed)).collect();
        let state = next_random(&mut seed);
        assert_eq!(
            changed.forward_adaptive(&inputs, state, policy),
            changed.forward_scalar(&inputs, state)
        );
        let restored = changed
            .with_edge(OperatorChannel::Retain, 11, 1)
            .unwrap()
            .with_edge(OperatorChannel::Retain, 41, 1)
            .unwrap();
        assert_eq!(restored, base);
        assert_eq!(
            restored.forward_adaptive(&inputs, state, policy),
            base.forward_grouped_sparse(&inputs, state)
        );
    }

    #[test]
    fn bps20_compiles_exact_dense_guard_for_required_input_bits() {
        let always_on = always();
        let requires_one = UnaryGate {
            when_false: u64::MAX ^ 1,
            when_true: u64::MAX,
        };
        let requires_zero = UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX ^ 2,
        };
        let shifts = [[0u8; NEURON_COUNT]; OPERATOR_COUNT];
        let p = BooleanNeuronGraph::new(
            [requires_one, requires_zero, always_on, always_on],
            UnaryGate::default(),
            always_on,
            shifts,
        )
        .unwrap();
        assert_eq!(p.compiled_dense_gate_guard(), Some((1, 2)));
        assert!(p.dense_gate_eligible(1));
        assert!(p.dense_gate_eligible(u64::MAX ^ 2));
        assert!(!p.dense_gate_eligible(0));
        assert!(!p.dense_gate_eligible(3));
        let policy = AdaptiveRoutingPolicy::default();
        let mut seed = 0x1803_55cd_9273_4471u64;
        for input in [0, 1, 2, 3, u64::MAX, u64::MAX ^ 2] {
            for _ in 0..16 {
                let state = next_random(&mut seed);
                let expected = p.step_scalar(input, state);
                assert_eq!(p.step_activity_hybrid(input, state, policy), expected);
            }
        }
    }

    #[test]
    fn bps20_conflicting_or_impossible_gate_truth_tables_never_dispatch_dense() {
        let always_on = always();
        let requires_one = UnaryGate {
            when_false: u64::MAX ^ 1,
            when_true: u64::MAX,
        };
        let contradicts_one = UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX ^ 1,
        };
        let shifts = [[0u8; NEURON_COUNT]; OPERATOR_COUNT];
        let contradictory = BooleanNeuronGraph::new(
            [requires_one, contradicts_one, always_on, always_on],
            UnaryGate::default(),
            always_on,
            shifts,
        )
        .unwrap();
        assert_eq!(contradictory.compiled_dense_gate_guard(), None);
        let impossible = BooleanNeuronGraph::new(
            [UnaryGate::default(), always_on, always_on, always_on],
            UnaryGate::default(),
            always_on,
            shifts,
        )
        .unwrap();
        assert_eq!(impossible.compiled_dense_gate_guard(), None);
        for input in [0, 1, 3, u64::MAX] {
            assert!(!contradictory.dense_gate_eligible(input));
            assert!(!impossible.dense_gate_eligible(input));
        }
        assert_eq!(
            impossible.forward_activity_hybrid(
                &[0, 1, u64::MAX],
                u64::MAX,
                AdaptiveRoutingPolicy::default()
            ),
            impossible.forward_scalar(&[0, 1, u64::MAX], u64::MAX)
        );
    }

    #[test]
    fn bps20_random_truth_table_guard_matches_evaluated_channel_masks() {
        let mut seed = 0xad20_4b15_7649_18fau64;
        for trial in 0..64 {
            let mut gates = [always(); OPERATOR_COUNT];
            for gate_mask in &mut gates {
                *gate_mask = if trial % 2 == 0 {
                    gate(&mut seed)
                } else {
                    // Every output bit is reachable, but input constraints
                    // can conflict across independent operator channels.
                    let on = next_random(&mut seed);
                    UnaryGate {
                        when_false: on,
                        when_true: !on,
                    }
                };
            }
            let program = BooleanNeuronGraph::new(
                gates,
                UnaryGate::default(),
                always(),
                [[0u8; NEURON_COUNT]; OPERATOR_COUNT],
            )
            .unwrap();
            for input in [0, 1, u64::MAX, u64::MAX ^ 1, next_random(&mut seed)] {
                let expected = gates
                    .iter()
                    .all(|gate_mask| gate_mask.evaluate(input) == u64::MAX);
                assert_eq!(
                    program.dense_gate_eligible(input),
                    expected,
                    "trial={trial}"
                );
            }
        }
    }

    #[test]
    fn bps20_mixed_topologies_match_scalar_on_sparse_dense_and_transition_inputs() {
        let mut seed = 0xb220_95fa_1148_6decu64;
        let policies = [
            AdaptiveRoutingPolicy::default(),
            AdaptiveRoutingPolicy {
                max_direct_lanes: [0; OPERATOR_COUNT],
            },
            AdaptiveRoutingPolicy {
                max_direct_lanes: [8, 1, 32, 64],
            },
        ];
        for topology in 0..24 {
            let shifts = std::array::from_fn(|channel| {
                std::array::from_fn(|destination| {
                    if channel == 0 {
                        destination as u8
                    } else if topology % 2 == 0 {
                        (channel * 7) as u8
                    } else {
                        (next_random(&mut seed) % 64) as u8
                    }
                })
            });
            let enable = UnaryGate {
                when_false: 0x0000_0000_0000_0007,
                when_true: u64::MAX,
            };
            let inputs: Vec<u64> = (0..129)
                .map(|index| match index % 7 {
                    0 => 0,
                    1 => u64::MAX,
                    2 => 1,
                    _ => next_random(&mut seed),
                })
                .collect();
            let p = BooleanNeuronGraph::new(
                [enable; OPERATOR_COUNT],
                gate(&mut seed),
                gate(&mut seed),
                shifts,
            )
            .unwrap();
            assert!(p.dense_gate_eligible(u64::MAX));
            assert!(!p.dense_gate_eligible(0));
            let initial = next_random(&mut seed);
            let oracle = p.forward_scalar(&inputs, initial);
            for policy in policies {
                assert_eq!(
                    p.forward_activity_hybrid(&inputs, initial, policy),
                    oracle,
                    "topology={topology} policy={policy:?}"
                );
            }
            let changed = p
                .with_edge(OperatorChannel::Toggle, 17, (topology + 1) as u8)
                .unwrap();
            assert_eq!(
                changed.compiled_dense_gate_guard(),
                p.compiled_dense_gate_guard()
            );
            assert_eq!(
                changed.forward_activity_hybrid(&inputs, initial, policies[0]),
                changed.forward_scalar(&inputs, initial)
            );
        }
    }
}
