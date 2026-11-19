//! BPS-26: explicit topology- and cost-aware direct-lane routing caps.
//!
//! Inputs to this policy are supplied by the caller, not invented or
//! automatically inferred from hardware. The computed thresholds only
//! select among already exact BPS-19/20/21 kernels.

use super::boolean_neuron_graph::{
    AdaptiveRoute, AdaptiveRoutingPolicy, BooleanNeuronGraph, OPERATOR_COUNT,
};
use super::boolean_neuron_stream::BooleanStreamError;

/// Fixed-point units for a caller-characterized route cost model.
///
/// Both unit costs must be nonzero. The setup costs may be zero. These are
/// estimates, not automatically measured cycle counts or calibrated timings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelRoutingCost {
    pub direct_lane_units: u32,
    pub grouped_shift_units: u32,
    pub direct_setup_units: u32,
    pub grouped_setup_units: u32,
}

/// Input validation is atomic: a malformed model never yields a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingCostError {
    ZeroDirectLaneCost { channel: usize },
    ZeroGroupedShiftCost { channel: usize },
}

/// Explicit input-only cost model for the four independent operations.
///
/// The caller should characterize the target machine on development data
/// before creating a plan. No timings, targets, labels or hidden states are
/// inspected here. Nothing is applied to the default library execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutingCostModel {
    channels: [ChannelRoutingCost; OPERATOR_COUNT],
    hard_caps: [u8; OPERATOR_COUNT],
}

impl RoutingCostModel {
    pub fn new(
        channels: [ChannelRoutingCost; OPERATOR_COUNT],
        hard_caps: [u8; OPERATOR_COUNT],
    ) -> Result<Self, RoutingCostError> {
        for (channel, cost) in channels.iter().enumerate() {
            if cost.direct_lane_units == 0 {
                return Err(RoutingCostError::ZeroDirectLaneCost { channel });
            }
            if cost.grouped_shift_units == 0 {
                return Err(RoutingCostError::ZeroGroupedShiftCost { channel });
            }
        }
        Ok(Self {
            channels,
            hard_caps,
        })
    }

    #[must_use]
    pub const fn costs(self) -> [ChannelRoutingCost; OPERATOR_COUNT] {
        self.channels
    }

    #[must_use]
    pub const fn hard_caps(self) -> [u8; OPERATOR_COUNT] {
        self.hard_caps
    }

    /// Compile one channel-specific cap from real graph shift diversity.
    ///
    /// For each lane count n, select direct predecessor reads only when
    /// (direct_setup + n * direct_lane) is strictly less than
    /// (grouped_setup + U * grouped_shift), U = distinct source rotations.
    /// The existing BPS-19 guard n < U is retained. Calculations use u64:
    /// each term has at most 64 * u32::MAX, so cannot overflow.
    #[must_use]
    pub fn thresholds(self, unique_shifts: [u32; OPERATOR_COUNT]) -> AdaptiveRoutingPolicy {
        let mut max_direct_lanes = [0u8; OPERATOR_COUNT];
        for channel in 0..OPERATOR_COUNT {
            let shifts = unique_shifts[channel].min(64);
            let cap = u32::from(self.hard_caps[channel]).min(shifts.saturating_sub(1));
            let cost = self.channels[channel];
            let grouped = u64::from(cost.grouped_setup_units)
                + u64::from(shifts) * u64::from(cost.grouped_shift_units);
            for lanes in 1..=cap {
                let direct = u64::from(cost.direct_setup_units)
                    + u64::from(lanes) * u64::from(cost.direct_lane_units);
                if direct >= grouped {
                    break;
                }
                max_direct_lanes[channel] = lanes as u8;
            }
        }
        AdaptiveRoutingPolicy { max_direct_lanes }
    }
}

/// The BPS-20 global dense-gate guard supersedes per-channel routing.
/// Otherwise this records all four BPS-19/BPS-21 operation choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostRoutingDecision {
    DenseGuard,
    Adaptive([AdaptiveRoute; OPERATOR_COUNT]),
}

/// Immutable compiled cost policy borrowed from the original neuron graph.
///
/// Rewiring returns a different graph and requires a freshly compiled plan.
pub struct PreparedCostRouter<'a> {
    graph: &'a BooleanNeuronGraph,
    adaptive: AdaptiveRoutingPolicy,
}

impl PreparedCostRouter<'_> {
    #[must_use]
    pub const fn adaptive_policy(&self) -> AdaptiveRoutingPolicy {
        self.adaptive
    }

    /// Exact gate-activity choice, independent of the previous state.
    #[must_use]
    pub fn decision(&self, input: u64) -> CostRoutingDecision {
        if self.graph.dense_gate_eligible(input) {
            CostRoutingDecision::DenseGuard
        } else {
            CostRoutingDecision::Adaptive(self.graph.adaptive_route_plan(input, self.adaptive))
        }
    }

    /// Exact synchronous transition using the existing BPS-20/21 executor.
    #[must_use]
    pub fn step(&self, input: u64, previous: u64) -> (u64, u64) {
        self.graph.step_activity_hybrid(input, previous, self.adaptive)
    }

    /// Process any chunk without allocating an output Vec. A size error
    /// leaves both the output buffer and the previous-state word untouched.
    pub fn scan_into(
        &self,
        inputs: &[u64],
        state: &mut u64,
        outputs: &mut [u64],
    ) -> Result<(), BooleanStreamError> {
        if inputs.len() != outputs.len() {
            return Err(BooleanStreamError::OutputLengthMismatch {
                inputs: inputs.len(),
                outputs: outputs.len(),
            });
        }
        for (&input, slot) in inputs.iter().zip(outputs.iter_mut()) {
            let (next, output) = self.step(input, *state);
            *state = next;
            *slot = output;
        }
        Ok(())
    }
}

impl BooleanNeuronGraph {
    /// BPS-26 opt-in route preparation. Uses each channel's actual unique
    /// predecessor count, not a single global topology approximation.
    #[must_use]
    pub fn prepare_cost_router(&self, model: RoutingCostModel) -> PreparedCostRouter<'_> {
        PreparedCostRouter {
            graph: self,
            adaptive: model.thresholds(self.unique_shift_counts()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::boolean_neuron_activity::exact_weight_corpus;
    use super::super::boolean_operator_channels::OperatorChannel;
    use super::super::boolean_scan_lstm::UnaryGate;
    use super::*;
    use super::super::boolean_neuron_graph::NEURON_COUNT;

    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn graph(uniform: usize, seed: &mut u64) -> BooleanNeuronGraph {
        let shifts = std::array::from_fn(|channel| {
            std::array::from_fn(|lane| {
                if channel < uniform {
                    (channel * 9) as u8
                } else {
                    ((lane * (2 * channel + 1)) % NEURON_COUNT) as u8
                }
            })
        });
        let gates = std::array::from_fn(|_| UnaryGate {
            when_false: next(seed),
            when_true: next(seed),
        });
        BooleanNeuronGraph::new(
            gates,
            UnaryGate {
                when_false: next(seed),
                when_true: next(seed),
            },
            UnaryGate {
                when_false: next(seed),
                when_true: next(seed),
            },
            shifts,
        )
        .unwrap()
    }

    fn model(direct: u32, grouped: u32, setup: u32) -> RoutingCostModel {
        let channels = std::array::from_fn(|channel| ChannelRoutingCost {
            direct_lane_units: direct + channel as u32,
            grouped_shift_units: grouped + channel as u32,
            direct_setup_units: setup,
            grouped_setup_units: channel as u32,
        });
        RoutingCostModel::new(channels, [64; OPERATOR_COUNT]).unwrap()
    }

    #[test]
    fn bps26_validate_costs_and_compile_per_channel_thresholds() {
        let invalid = [
            ChannelRoutingCost {
                direct_lane_units: 1,
                grouped_shift_units: 1,
                direct_setup_units: 0,
                grouped_setup_units: 0,
            };
            OPERATOR_COUNT
        ];
        let mut broken = invalid;
        broken[2].direct_lane_units = 0;
        assert_eq!(
            RoutingCostModel::new(broken, [4; OPERATOR_COUNT]),
            Err(RoutingCostError::ZeroDirectLaneCost { channel: 2 })
        );
        broken[2].direct_lane_units = 1;
        broken[1].grouped_shift_units = 0;
        assert_eq!(
            RoutingCostModel::new(broken, [4; OPERATOR_COUNT]),
            Err(RoutingCostError::ZeroGroupedShiftCost { channel: 1 })
        );
        let cheap = model(1, 5, 0);
        assert_eq!(
            cheap.thresholds([1, 2, 7, 64]).max_direct_lanes,
            [0, 1, 6, 63]
        );
        let expensive = model(1_000, 1, 10_000);
        assert_eq!(
            expensive.thresholds([1, 2, 7, 64]).max_direct_lanes,
            [0; OPERATOR_COUNT]
        );
        assert_eq!(
            RoutingCostModel::new(invalid, [0; OPERATOR_COUNT])
                .unwrap()
                .thresholds([64; OPERATOR_COUNT])
                .max_direct_lanes,
            [0; OPERATOR_COUNT]
        );
    }

    #[test]
    fn bps26_all_weights_topologies_and_cost_models_match_scalar_state() {
        let mut seed = 0x266f_4a39_821b_0d11u64;
        for uniform in 0..=OPERATOR_COUNT {
            let graph = graph(uniform, &mut seed);
            assert_eq!(graph.uniform_operation_count(), uniform);
            for model in [model(1, 5, 0), model(1_000, 1, 10_000), model(17, 4, 2)] {
                let prepared = graph.prepare_cost_router(model);
                let caps = prepared.adaptive_policy();
                for channel in 0..OPERATOR_COUNT {
                    assert!(
                        u32::from(caps.max_direct_lanes[channel])
                            < graph.unique_shift_counts()[channel]
                    );
                }
                for weight in 0..=64u8 {
                    let inputs = exact_weight_corpus(next(&mut seed), weight, 7).unwrap();
                    let initial = next(&mut seed);
                    let mut state = initial;
                    let mut outputs = vec![0u64; inputs.len()];
                    prepared.scan_into(&inputs, &mut state, &mut outputs).unwrap();
                    let mut scalar = initial;
                    let mut oracle = Vec::new();
                    for &input in &inputs {
                        let (next_state, output) = graph.step_scalar(input, scalar);
                        scalar = next_state;
                        oracle.push(output);
                        assert_eq!(
                            prepared.step(input, scalar),
                            graph.step_activity_hybrid(input, scalar, caps)
                        );
                        let expected = if graph.dense_gate_eligible(input) {
                            CostRoutingDecision::DenseGuard
                        } else {
                            CostRoutingDecision::Adaptive(
                                graph.adaptive_route_plan(input, caps)
                            )
                        };
                        assert_eq!(prepared.decision(input), expected);
                    }
                    assert_eq!(outputs, oracle, "uniform={uniform}, weight={weight}");
                    assert_eq!(state, scalar);
                }
            }
        }
    }

    #[test]
    fn bps26_chunk_rewiring_restoration_and_reject_are_exact() {
        let mut seed = 0x2626_711d_3fea_6c42u64;
        let original = graph(3, &mut seed);
        let rewired = original.with_edge(OperatorChannel::Retain, 5, 23).unwrap();
        let restored = rewired.with_edge(OperatorChannel::Retain, 5, 0).unwrap();
        assert_eq!(restored, original);
        let mut inputs = Vec::new();
        for weight in [0, 1, 4, 16, 32, 48, 63, 64] {
            inputs.extend(exact_weight_corpus(next(&mut seed), weight, 19).unwrap());
        }
        let initial = next(&mut seed);
        for graph in [&original, &rewired, &restored] {
            let prepared = graph.prepare_cost_router(model(2, 9, 0));
            let mut outputs = vec![0u64; inputs.len()];
            let mut state = initial;
            let ends = [0, 1, 1, 7, 31, 64, 129, inputs.len()];
            let mut start = 0;
            for &end in &ends {
                prepared.scan_into(
                    &inputs[start..end], &mut state, &mut outputs[start..end]
                ).unwrap();
                start = end;
            }
            assert_eq!(outputs, graph.forward_scalar(&inputs, initial));
            let mut oracle_state = initial;
            for &input in &inputs {
                oracle_state = graph.step_scalar(input, oracle_state).0;
            }
            assert_eq!(state, oracle_state);
            let previous = state;
            let mut bad_output = [0xfeedu64];
            assert_eq!(
                prepared.scan_into(&[1, 2], &mut state, &mut bad_output),
                Err(BooleanStreamError::OutputLengthMismatch {
                    inputs: 2,
                    outputs: 1
                })
            );
            assert_eq!(state, previous);
            assert_eq!(bad_output, [0xfeedu64]);
            prepared.scan_into(&[], &mut state, &mut []).unwrap();
            assert_eq!(state, previous);
        }
        assert_eq!(original.uniform_operation_count(), 3);
        assert_eq!(rewired.uniform_operation_count(), 2);
    }
}
