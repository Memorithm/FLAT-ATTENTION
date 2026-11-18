//! BPS-22: reproducible *individual-neuron* activation corpus and
//! input-only routing telemetry for the BPS-16..21 Boolean graph.
//!
//! Unlike whole-word 0/MAX benchmark inputs, these corpora have EXACT
//! per-sample Hamming weight and vary which of the 64 neuron lanes are set.
//! The input profile reports work proxies, never asserted physical speedups.
//! This module has no effect on the core recurrent transition.

use super::boolean_neuron_graph::{
    AdaptiveRoute, AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
};

/// The fixed length of a per-neuron activation histogram: weights 0..=64.
pub const WEIGHT_BUCKETS: usize = NEURON_COUNT + 1;

/// Reject impossible activation weights instead of silently clipping them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneMaskError {
    WeightTooLarge { requested: u8 },
}

/// Deterministic SplitMix64 source. Reproducible, not a security RNG.
///
/// Each sample selects unique destination lanes by a bounded partial
/// Fisher-Yates shuffle, with rejection sampling for the bounded indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactWeightMaskGenerator {
    state: u64,
}

impl ExactWeightMaskGenerator {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn random_word(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn bounded(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.random_word();
            if value >= threshold {
                return value % bound;
            }
        }
    }

    /// Return a mask with exactly `weight` activated bits, or reject
    /// weights above 64. Successive calls advance the deterministic PRNG
    /// for nontrivial weights, preserving full 64-lane candidate coverage.
    pub fn next_mask(&mut self, weight: u8) -> Result<u64, LaneMaskError> {
        if usize::from(weight) > NEURON_COUNT {
            return Err(LaneMaskError::WeightTooLarge { requested: weight });
        }
        if weight == 0 {
            return Ok(0);
        }
        if usize::from(weight) == NEURON_COUNT {
            return Ok(u64::MAX);
        }

        let mut lanes: [u8; NEURON_COUNT] = std::array::from_fn(|index| index as u8);
        let mut selected = 0u64;
        for index in 0..usize::from(weight) {
            let remaining = (NEURON_COUNT - index) as u64;
            let next = index + self.bounded(remaining) as usize;
            lanes.swap(index, next);
            selected |= 1u64 << lanes[index];
        }
        Ok(selected)
    }
}

/// Generate deterministic inputs with an exact number of active neuron
/// lanes per sample. The shape is validated before allocating.
pub fn exact_weight_corpus(
    seed: u64,
    weight: u8,
    sample_count: usize,
) -> Result<Vec<u64>, LaneMaskError> {
    if usize::from(weight) > NEURON_COUNT {
        return Err(LaneMaskError::WeightTooLarge { requested: weight });
    }
    let mut generator = ExactWeightMaskGenerator::new(seed);
    (0..sample_count)
        .map(|_| generator.next_mask(weight))
        .collect()
}

/// The *candidate* BPS-19 route counts for each operator, before BPS-20
/// may override the full step with its all-four-channel dense fallback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandidateRouteCounts {
    pub dormant: u64,
    pub uniform: u64,
    pub direct_lanes: u64,
    pub grouped: u64,
}

impl CandidateRouteCounts {
    fn count(&mut self, route: AdaptiveRoute) -> Result<(), ActivityProfileError> {
        let counter = match route {
            AdaptiveRoute::Dormant => &mut self.dormant,
            AdaptiveRoute::Uniform => &mut self.uniform,
            AdaptiveRoute::DirectLanes => &mut self.direct_lanes,
            AdaptiveRoute::Grouped => &mut self.grouped,
        };
        *counter = counter
            .checked_add(1)
            .ok_or(ActivityProfileError::CounterOverflow)?;
        Ok(())
    }

    #[must_use]
    pub fn total(self) -> Option<u64> {
        [self.dormant, self.uniform, self.direct_lanes, self.grouped]
            .into_iter()
            .try_fold(0u64, u64::checked_add)
    }
}

/// Bounded, exact count profile; not a physical throughput/energy monitor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeuronActivationProfile {
    pub samples: u64,
    pub input_one_bits: u64,
    pub gate_one_bits: [u64; OPERATOR_COUNT],
    pub input_weight_histogram: [u64; WEIGHT_BUCKETS],
    pub gate_weight_histograms: [[u64; WEIGHT_BUCKETS]; OPERATOR_COUNT],
    pub fully_active_ticks: [u64; OPERATOR_COUNT],
    pub dormant_ticks: [u64; OPERATOR_COUNT],
    pub global_dense_ticks: u64,
    /// Number of distinct predecessor offsets touched by active destination
    /// lanes, summed over all samples for each operator. This counts
    /// logical source groups, not executed rotations or elapsed time.
    pub touched_source_groups: [u64; OPERATOR_COUNT],
    pub candidate_routes: [CandidateRouteCounts; OPERATOR_COUNT],
}

impl Default for NeuronActivationProfile {
    fn default() -> Self {
        Self {
            samples: 0,
            input_one_bits: 0,
            gate_one_bits: [0; OPERATOR_COUNT],
            input_weight_histogram: [0; WEIGHT_BUCKETS],
            gate_weight_histograms: [[0; WEIGHT_BUCKETS]; OPERATOR_COUNT],
            fully_active_ticks: [0; OPERATOR_COUNT],
            dormant_ticks: [0; OPERATOR_COUNT],
            global_dense_ticks: 0,
            touched_source_groups: [0; OPERATOR_COUNT],
            candidate_routes: [CandidateRouteCounts::default(); OPERATOR_COUNT],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivityProfileError {
    CounterOverflow,
}

fn add(counter: &mut u64, increment: u64) -> Result<(), ActivityProfileError> {
    *counter = counter
        .checked_add(increment)
        .ok_or(ActivityProfileError::CounterOverflow)?;
    Ok(())
}

impl NeuronActivationProfile {
    /// Deterministically observe actual per-neuron gate activation and
    /// source-group diversity for a provided sequence, without running
    /// any model transition, mutating the graph or reading output targets.
    ///
    /// The route histogram describes candidate per-operator BPS-19
    /// decisions. The BPS-20 full-word dispatch is separately indicated
    /// by `global_dense_ticks`; physical executed-operation counts
    /// cannot be inferred by summing these two counters.
    pub fn observe(
        &mut self,
        graph: &BooleanNeuronGraph,
        input: u64,
        policy: AdaptiveRoutingPolicy,
    ) -> Result<(), ActivityProfileError> {
        add(&mut self.samples, 1)?;
        let input_ones = input.count_ones() as usize;
        add(&mut self.input_one_bits, input_ones as u64)?;
        add(&mut self.input_weight_histogram[input_ones], 1)?;

        if graph.dense_gate_eligible(input) {
            add(&mut self.global_dense_ticks, 1)?;
        }
        let routes = graph.adaptive_route_plan(input, policy);
        for (channel, &route) in routes.iter().enumerate() {
            let mask = graph.gates[channel].evaluate(input);
            let ones = mask.count_ones() as usize;
            add(&mut self.gate_one_bits[channel], ones as u64)?;
            add(&mut self.gate_weight_histograms[channel][ones], 1)?;
            if ones == 0 {
                add(&mut self.dormant_ticks[channel], 1)?;
            }
            if ones == NEURON_COUNT {
                add(&mut self.fully_active_ticks[channel], 1)?;
            }
            self.candidate_routes[channel].count(route)?;

            // One input gate bit may activate an independent destination
            // neuron with a different predecessor. Count each distinct
            // active predecessor rotation once per sample/channel.
            let mut enabled_destinations = mask;
            let mut enabled_shifts = 0u64;
            while enabled_destinations != 0 {
                let destination = enabled_destinations.trailing_zeros() as usize;
                enabled_destinations &= enabled_destinations - 1;
                let shift = graph
                    .source_shift(
                        super::boolean_operator_channels::OperatorChannel::ALL[channel],
                        destination,
                    )
                    .expect("validated graph has one shift for every destination");
                enabled_shifts |= 1u64 << shift;
            }
            add(
                &mut self.touched_source_groups[channel],
                enabled_shifts.count_ones() as u64,
            )?;
        }
        Ok(())
    }

    /// Profile full input-only sequences with exact lane-resolved histograms.
    /// The original graph, state and learning policy are never changed.
    pub fn from_inputs(
        graph: &BooleanNeuronGraph,
        inputs: &[u64],
        policy: AdaptiveRoutingPolicy,
    ) -> Result<Self, ActivityProfileError> {
        let mut profile = Self::default();
        for &word in inputs {
            profile.observe(graph, word, policy)?;
        }
        Ok(profile)
    }
}

#[cfg(test)]
mod tests {
    use super::super::boolean_scan_lstm::UnaryGate;
    use super::*;

    fn always() -> UnaryGate {
        UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        }
    }

    fn graph() -> BooleanNeuronGraph {
        let mut shifts = [[0u8; NEURON_COUNT]; OPERATOR_COUNT];
        for index in 0..NEURON_COUNT {
            shifts[0][index] = index as u8;
            shifts[1][index] = 3;
            shifts[2][index] = ((index * 5) % NEURON_COUNT) as u8;
            shifts[3][index] = ((index * 7) % NEURON_COUNT) as u8;
        }
        BooleanNeuronGraph::new(
            [
                UnaryGate {
                    when_false: 0,
                    when_true: u64::MAX,
                },
                always(),
                UnaryGate {
                    when_false: u64::MAX,
                    when_true: 0,
                },
                UnaryGate {
                    when_false: 0,
                    when_true: 0,
                },
            ],
            UnaryGate::default(),
            always(),
            shifts,
        )
        .unwrap()
    }

    #[test]
    fn bps22_all_exact_hamming_weights_and_seed_reproducibility() {
        for weight in 0..=NEURON_COUNT {
            let first = exact_weight_corpus(0, weight as u8, 71).unwrap();
            let second = exact_weight_corpus(0, weight as u8, 71).unwrap();
            assert_eq!(first, second);
            assert!(first.iter().all(|mask| mask.count_ones() == weight as u32));
            if (1..NEURON_COUNT).contains(&weight) {
                assert!(first.iter().any(|mask| *mask != first[0]));
            }
        }
        assert_ne!(
            exact_weight_corpus(0, 9, 64).unwrap(),
            exact_weight_corpus(1, 9, 64).unwrap()
        );
        assert_eq!(
            exact_weight_corpus(77, 65, 0).unwrap_err(),
            LaneMaskError::WeightTooLarge { requested: 65 }
        );
        assert!(exact_weight_corpus(8, 64, 0).unwrap().is_empty());
    }

    #[test]
    fn bps22_input_and_channel_histograms_are_independent_and_exact() {
        let graph = graph();
        let corpus = [0u64, 1, 3, u64::MAX];
        let profile =
            NeuronActivationProfile::from_inputs(&graph, &corpus, AdaptiveRoutingPolicy::default())
                .unwrap();
        assert_eq!(profile.samples, 4);
        assert_eq!(profile.input_one_bits, 67);
        assert_eq!(profile.input_weight_histogram[0], 1);
        assert_eq!(profile.input_weight_histogram[1], 1);
        assert_eq!(profile.input_weight_histogram[2], 1);
        assert_eq!(profile.input_weight_histogram[64], 1);
        assert_eq!(profile.gate_one_bits, [67, 256, 189, 0]);
        assert_eq!(profile.fully_active_ticks, [1, 4, 1, 0]);
        assert_eq!(profile.dormant_ticks, [1, 0, 1, 4]);
        assert_eq!(profile.global_dense_ticks, 0);
        assert_eq!(profile.touched_source_groups[0], 67);
        assert_eq!(profile.touched_source_groups[1], 4);
        assert_eq!(profile.touched_source_groups[3], 0);
        assert_eq!(
            profile.candidate_routes[3],
            CandidateRouteCounts {
                dormant: 4,
                ..CandidateRouteCounts::default()
            }
        );
        for route in &profile.candidate_routes {
            assert_eq!(route.total(), Some(4));
        }
    }

    #[test]
    fn bps22_mixed_lane_weights_stay_oracle_exact_across_all_paths() {
        let graph = graph();
        let policy = AdaptiveRoutingPolicy::default();
        let initial = 0xad55_1baa_fa94_10e3u64;
        for weight in [0, 1, 2, 4, 7, 16, 32, 48, 63, 64] {
            let inputs = exact_weight_corpus(0x4125_9aee_2bc4_1190, weight, 129).unwrap();
            let profile = NeuronActivationProfile::from_inputs(&graph, &inputs, policy).unwrap();
            assert_eq!(profile.samples, 129);
            assert_eq!(profile.input_one_bits, 129 * u64::from(weight));
            assert_eq!(profile.input_weight_histogram[usize::from(weight)], 129);
            let expected = graph.forward_scalar(&inputs, initial);
            assert_eq!(graph.forward_adaptive(&inputs, initial, policy), expected);
            assert_eq!(
                graph.forward_activity_hybrid(&inputs, initial, policy),
                expected
            );
            assert_eq!(graph.forward_grouped_sparse(&inputs, initial), expected);
        }
    }

    #[test]
    fn bps22_stats_across_incremental_chunks_equal_whole_sequence() {
        let graph = graph();
        let corpus = exact_weight_corpus(0xca1d_953b_7610_3444, 31, 209).unwrap();
        let policy = AdaptiveRoutingPolicy::default();
        let whole = NeuronActivationProfile::from_inputs(&graph, &corpus, policy).unwrap();
        let mut stream = NeuronActivationProfile::default();
        for chunk in corpus.chunks(17) {
            for &sample in chunk {
                stream.observe(&graph, sample, policy).unwrap();
            }
        }
        assert_eq!(stream, whole);
        let empty = NeuronActivationProfile::from_inputs(&graph, &[], policy).unwrap();
        assert_eq!(empty, NeuronActivationProfile::default());
    }
}
