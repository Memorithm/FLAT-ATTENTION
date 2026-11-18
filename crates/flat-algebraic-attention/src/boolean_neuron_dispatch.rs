//! BPS-23: exact, explicitly configurable kernel routing over the
//! number of active input neurons and uniform predecessor channels.
//!
//! BPS-22's Dell evidence observed different best kernels at different
//! Hamming weights and topologies. This module makes that choice
//! executable without silently promoting a one-host research policy.
//! All paths compute the same synchronous Boolean recurrence.

use super::boolean_neuron_graph::{
    AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
};

const UNIFORM_CHANNEL_BUCKETS: usize = OPERATOR_COUNT + 1;

/// An inclusive Hamming-weight range within the 64-bit neuron input word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseWeightWindow {
    pub first: u8,
    pub last: u8,
}

/// Reject invalid windows at construction, before any inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRoutingError {
    InvalidWindow {
        uniform_channels: usize,
        first: u8,
        last: u8,
    },
}

/// Exact kernel decision for a particular input word and validated graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRoutingDecision {
    /// Use the existing packed BPS-18 grouped-dense recurrence.
    GroupedDense,
    /// Preserve the BPS-20/21 per-operation sparse/dense adaptive hybrid.
    ExistingHybrid,
}

/// No implicit performance calibration: windows are supplied by the caller.
///
/// Entry `k` is the inclusive input Hamming-weight window for a graph
/// containing exactly `k` uniform predecessor channels, `k=0..=4`.
/// If absent, the graph retains its existing BPS-20/21 hybrid behavior.
/// The policy never looks at target labels, hidden states or future inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightRoutingPolicy {
    windows: [Option<DenseWeightWindow>; UNIFORM_CHANNEL_BUCKETS],
    adaptive: AdaptiveRoutingPolicy,
}

impl WeightRoutingPolicy {
    /// A no-op configuration: every sample takes the existing hybrid.
    pub const DISABLED: Self = Self {
        windows: [None; UNIFORM_CHANNEL_BUCKETS],
        adaptive: AdaptiveRoutingPolicy {
            max_direct_lanes: [4; OPERATOR_COUNT],
        },
    };

    /// Validate all five independently configured uniformity buckets.
    pub fn new(
        windows: [Option<DenseWeightWindow>; UNIFORM_CHANNEL_BUCKETS],
        adaptive: AdaptiveRoutingPolicy,
    ) -> Result<Self, WeightRoutingError> {
        for (uniform_channels, window) in windows.iter().enumerate() {
            if let Some(window) = window {
                if window.first > window.last || usize::from(window.last) > NEURON_COUNT {
                    return Err(WeightRoutingError::InvalidWindow {
                        uniform_channels,
                        first: window.first,
                        last: window.last,
                    });
                }
            }
        }
        Ok(Self { windows, adaptive })
    }

    #[must_use]
    pub const fn windows(self) -> [Option<DenseWeightWindow>; UNIFORM_CHANNEL_BUCKETS] {
        self.windows
    }

    #[must_use]
    pub const fn adaptive(self) -> AdaptiveRoutingPolicy {
        self.adaptive
    }

    #[must_use]
    pub fn decide(self, uniform_channels: usize, input: u64) -> WeightRoutingDecision {
        let Some(window) = self.windows.get(uniform_channels).copied().flatten() else {
            return WeightRoutingDecision::ExistingHybrid;
        };
        let weight = input.count_ones() as u8;
        if (window.first..=window.last).contains(&weight) {
            WeightRoutingDecision::GroupedDense
        } else {
            WeightRoutingDecision::ExistingHybrid
        }
    }
}

impl BooleanNeuronGraph {
    /// Count source-uniform operations, independent of the current gate
    /// activity. Rewiring rebuilds compiled topology in the original graph.
    #[must_use]
    pub fn uniform_operation_count(&self) -> usize {
        self.compiled_channel_uniform_shifts()
            .iter()
            .filter(|shift| shift.is_some())
            .count()
    }

    /// Inspect the dispatch before any state transition or output read.
    #[must_use]
    pub fn weight_routing_decision(
        &self,
        input: u64,
        policy: WeightRoutingPolicy,
    ) -> WeightRoutingDecision {
        policy.decide(self.uniform_operation_count(), input)
    }

    /// BPS-23 exact dispatcher. Both implementations use the SAME
    /// previous-state word for all four operation channels. Selection
    /// changes only computation strategy, never the result.
    #[must_use]
    pub fn step_weight_routed(
        &self,
        input: u64,
        previous: u64,
        policy: WeightRoutingPolicy,
    ) -> (u64, u64) {
        match self.weight_routing_decision(input, policy) {
            WeightRoutingDecision::GroupedDense => self.step_grouped(input, previous),
            WeightRoutingDecision::ExistingHybrid => {
                self.step_activity_hybrid(input, previous, policy.adaptive())
            }
        }
    }

    /// Exact synchronous trajectory over the caller-selected policy.
    #[must_use]
    pub fn forward_weight_routed(
        &self,
        inputs: &[u64],
        initial: u64,
        policy: WeightRoutingPolicy,
    ) -> Vec<u64> {
        let mut state = initial;
        inputs
            .iter()
            .map(|&input| {
                let (next, output) = self.step_weight_routed(input, state, policy);
                state = next;
                output
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::boolean_neuron_activity::exact_weight_corpus;
    use super::super::boolean_operator_channels::OperatorChannel;
    use super::super::boolean_scan_lstm::UnaryGate;
    use super::*;

    fn next_random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn gate(seed: &mut u64) -> UnaryGate {
        UnaryGate {
            when_false: next_random(seed),
            when_true: next_random(seed),
        }
    }

    fn graph(uniform: usize, seed: &mut u64) -> BooleanNeuronGraph {
        let shifts = std::array::from_fn(|channel| {
            std::array::from_fn(|destination| {
                if channel < uniform {
                    (channel * 9) as u8
                } else {
                    // Exactly 64 distinct predecessor offsets.
                    ((destination * (2 * channel + 1)) % NEURON_COUNT) as u8
                }
            })
        });
        BooleanNeuronGraph::new(
            [gate(seed), gate(seed), gate(seed), gate(seed)],
            gate(seed),
            gate(seed),
            shifts,
        )
        .unwrap()
    }

    fn experimental_windows() -> WeightRoutingPolicy {
        // An EXPLICIT research hypothesis selected from BPS-22 DEV
        // evidence, never the library default or a validated threshold.
        WeightRoutingPolicy::new(
            [
                Some(DenseWeightWindow { first: 4, last: 63 }),
                None,
                None,
                Some(DenseWeightWindow { first: 0, last: 64 }),
                None,
            ],
            AdaptiveRoutingPolicy::default(),
        )
        .unwrap()
    }

    #[test]
    fn bps23_invalid_weight_windows_fail_before_execution() {
        let invalid = WeightRoutingPolicy::new(
            [
                Some(DenseWeightWindow { first: 5, last: 4 }),
                None,
                None,
                None,
                None,
            ],
            AdaptiveRoutingPolicy::default(),
        );
        assert_eq!(
            invalid.unwrap_err(),
            WeightRoutingError::InvalidWindow {
                uniform_channels: 0,
                first: 5,
                last: 4
            }
        );
        let too_large = WeightRoutingPolicy::new(
            [
                None,
                None,
                Some(DenseWeightWindow { first: 0, last: 65 }),
                None,
                None,
            ],
            AdaptiveRoutingPolicy::default(),
        );
        assert_eq!(
            too_large.unwrap_err(),
            WeightRoutingError::InvalidWindow {
                uniform_channels: 2,
                first: 0,
                last: 65
            }
        );
        assert!(WeightRoutingPolicy::DISABLED
            .windows()
            .iter()
            .all(Option::is_none));
    }

    #[test]
    fn bps23_dispatch_is_topology_specific_and_weight_exact() {
        let mut seed = 0xc2ee_907f_1a21_d448u64;
        let policy = experimental_windows();
        for uniform in 0..=4 {
            let model = graph(uniform, &mut seed);
            assert_eq!(model.uniform_operation_count(), uniform);
            for weight in 0..=64u8 {
                let inputs = exact_weight_corpus(0x73a1_0510_u64, weight, 7).unwrap();
                let expected = if (uniform == 0 && (4..=63).contains(&weight)) || uniform == 3 {
                    WeightRoutingDecision::GroupedDense
                } else {
                    WeightRoutingDecision::ExistingHybrid
                };
                for input in inputs {
                    assert_eq!(
                        model.weight_routing_decision(input, policy),
                        expected,
                        "uniform={uniform} weight={weight}"
                    );
                    assert_eq!(
                        model.weight_routing_decision(input, WeightRoutingPolicy::DISABLED),
                        WeightRoutingDecision::ExistingHybrid
                    );
                }
            }
        }
    }

    #[test]
    fn bps23_rewire_changes_bucket_without_stale_cached_route() {
        let mut seed = 0x2389_744b_912a_507du64;
        let base = graph(3, &mut seed);
        let policy = experimental_windows();
        assert_eq!(base.uniform_operation_count(), 3);
        assert_eq!(
            base.weight_routing_decision(0, policy),
            WeightRoutingDecision::GroupedDense
        );
        let rewired = base.with_edge(OperatorChannel::Retain, 5, 17).unwrap();
        assert_eq!(rewired.uniform_operation_count(), 2);
        assert_eq!(
            rewired.weight_routing_decision(0, policy),
            WeightRoutingDecision::ExistingHybrid
        );
        let restored = rewired.with_edge(OperatorChannel::Retain, 5, 0).unwrap();
        assert_eq!(restored, base);
        let inputs = exact_weight_corpus(0x712c_9ba1_4f2b_14a3, 32, 61).unwrap();
        let initial = next_random(&mut seed);
        assert_eq!(
            rewired.forward_weight_routed(&inputs, initial, policy),
            rewired.forward_scalar(&inputs, initial)
        );
        assert_eq!(
            restored.forward_weight_routed(&inputs, initial, policy),
            base.forward_scalar(&inputs, initial)
        );
    }

    #[test]
    fn bps23_randomized_all_topologies_and_exact_weight_corpora_match_scalar() {
        let mut seed = 0x3322_b872_5a65_2181u64;
        let policies = [
            WeightRoutingPolicy::DISABLED,
            experimental_windows(),
            WeightRoutingPolicy::new(
                [Some(DenseWeightWindow { first: 0, last: 64 }); 5],
                AdaptiveRoutingPolicy {
                    max_direct_lanes: [0; OPERATOR_COUNT],
                },
            )
            .unwrap(),
        ];
        for uniform in 0..=4 {
            for _ in 0..6 {
                let model = graph(uniform, &mut seed);
                let initial = next_random(&mut seed);
                let mut inputs = Vec::new();
                for weight in [0, 1, 4, 16, 32, 48, 63, 64] {
                    inputs.extend(exact_weight_corpus(next_random(&mut seed), weight, 17).unwrap());
                }
                let oracle = model.forward_scalar(&inputs, initial);
                for policy in policies {
                    assert_eq!(
                        model.forward_weight_routed(&inputs, initial, policy),
                        oracle,
                        "uniform={uniform} policy={policy:?}"
                    );
                    assert_eq!(
                        model.forward_weight_routed(&[], initial, policy),
                        Vec::<u64>::new()
                    );
                }
            }
        }
    }
}
