//! BPS-27: route by the source-shift groups actually enabled by each input.
//!
//! Unlike BPS-26's cost of all compiled groups, this opt-in research
//! executor computes an input-local estimate without reading recurrent state.

use super::boolean_neuron_cost_router::{CostRoutingDecision, RoutingCostModel};
use super::boolean_neuron_graph::{
    AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
};
use super::boolean_neuron_stream::BooleanStreamError;
use super::boolean_operator_channels::OperatorChannel;

/// BPS-27 prepared executor, including a bounded, immutable shift-mask table.
///
/// The graph is borrowed: a rewired graph must get its own prepared executor.
/// There are four channels and at most 64 distinct shifts per channel.
pub struct PreparedLiveCostRouter<'a> {
    graph: &'a BooleanNeuronGraph,
    model: RoutingCostModel,
    destinations_by_shift: [[u64; NEURON_COUNT]; OPERATOR_COUNT],
    active_shifts: [u64; OPERATOR_COUNT],
}

impl BooleanNeuronGraph {
    /// Compile exact destination masks for each source rotation once.
    /// This does not change the original BPS-26 policy or execution.
    #[must_use]
    pub fn prepare_live_cost_router(&self, model: RoutingCostModel) -> PreparedLiveCostRouter<'_> {
        let mut destinations_by_shift = [[0u64; NEURON_COUNT]; OPERATOR_COUNT];
        let mut active_shifts = [0u64; OPERATOR_COUNT];
        for channel in OperatorChannel::ALL {
            let index = channel.index();
            for destination in 0..NEURON_COUNT {
                let shift = usize::from(
                    self.source_shift(channel, destination)
                        .expect("validated graph has all 64 destinations"),
                );
                destinations_by_shift[index][shift] |= 1u64 << destination;
                active_shifts[index] |= 1u64 << shift;
            }
        }
        PreparedLiveCostRouter {
            graph: self,
            model,
            destinations_by_shift,
            active_shifts,
        }
    }
}

impl PreparedLiveCostRouter<'_> {
    /// Number of distinct compiled predecessor shifts actually touched
    /// by this input, independently for each recurrent operation.
    ///
    /// All mask intersections are input-only. Neither previous state,
    /// targets, nor outputs are read by this selector.
    #[must_use]
    pub fn live_shift_counts(&self, input: u64) -> [u32; OPERATOR_COUNT] {
        let mut touched = [0u32; OPERATOR_COUNT];
        for (channel, count) in touched.iter_mut().enumerate() {
            let enabled_destinations = self.graph.gates[channel].evaluate(input);
            if enabled_destinations == 0 {
                continue;
            }
            if enabled_destinations == u64::MAX {
                *count = self.active_shifts[channel].count_ones();
                continue;
            }
            let mut shifts = self.active_shifts[channel];
            while shifts != 0 {
                let shift = shifts.trailing_zeros() as usize;
                shifts &= shifts - 1;
                if self.destinations_by_shift[channel][shift] & enabled_destinations != 0 {
                    *count += 1;
                }
            }
        }
        touched
    }

    /// Compute exact BPS-19 direct-lane caps from the live gate masks and
    /// touched groups. Costs are caller-supplied estimates, never timings
    /// sampled from this run.
    ///
    /// The existing strict n < total_unique_shifts guard remains in force,
    /// even when a smaller touched-group count would suggest direct reads.
    #[must_use]
    pub fn adaptive_policy(&self, input: u64) -> AdaptiveRoutingPolicy {
        let costs = self.model.costs();
        let hard_caps = self.model.hard_caps();
        let mut caps = [0u8; OPERATOR_COUNT];
        for channel in 0..OPERATOR_COUNT {
            let gate = self.graph.gates[channel].evaluate(input);
            let lanes = gate.count_ones();
            let unique = self.active_shifts[channel].count_ones();
            if lanes == 0
                || lanes == NEURON_COUNT as u32
                || unique == 1
                || lanes >= unique
                || lanes > u32::from(hard_caps[channel])
            {
                continue;
            }
            let mut touched = 0u32;
            let mut shifts = self.active_shifts[channel];
            while shifts != 0 {
                let shift = shifts.trailing_zeros() as usize;
                shifts &= shifts - 1;
                if self.destinations_by_shift[channel][shift] & gate != 0 {
                    touched += 1;
                }
            }
            let cost = costs[channel];
            let direct = u64::from(cost.direct_setup_units)
                + u64::from(lanes) * u64::from(cost.direct_lane_units);
            let grouped = u64::from(cost.grouped_setup_units)
                + u64::from(touched) * u64::from(cost.grouped_shift_units);
            if direct < grouped {
                caps[channel] = lanes as u8;
            }
        }
        AdaptiveRoutingPolicy {
            max_direct_lanes: caps,
        }
    }

    /// Expose which exact original kernels would be selected for this input.
    #[must_use]
    pub fn decision(&self, input: u64) -> CostRoutingDecision {
        if self.graph.dense_gate_eligible(input) {
            CostRoutingDecision::DenseGuard
        } else {
            CostRoutingDecision::Adaptive(
                self.graph
                    .adaptive_route_plan(input, self.adaptive_policy(input)),
            )
        }
    }

    /// Execute one synchronous step using the original BPS-20/21 kernels.
    #[must_use]
    pub fn step(&self, input: u64, previous: u64) -> (u64, u64) {
        if self.graph.dense_gate_eligible(input) {
            // BPS-20 full-dense fast path: do not need to inspect live groups.
            self.graph.step_grouped(input, previous)
        } else {
            self.graph
                .step_adaptive(input, previous, self.adaptive_policy(input))
        }
    }

    /// Run an arbitrary chunk into caller-owned memory, keeping state.
    /// Length mismatch does not mutate state or output.
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

#[cfg(test)]
mod tests {
    use super::super::boolean_neuron_activity::exact_weight_corpus;
    use super::super::boolean_neuron_cost_router::ChannelRoutingCost;
    use super::super::boolean_neuron_graph::AdaptiveRoute;
    use super::super::boolean_scan_lstm::UnaryGate;
    use super::*;

    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn model(direct: u32, grouped: u32) -> RoutingCostModel {
        RoutingCostModel::new(
            [ChannelRoutingCost {
                direct_lane_units: direct,
                grouped_shift_units: grouped,
                direct_setup_units: 0,
                grouped_setup_units: 0,
            }; OPERATOR_COUNT],
            [64; OPERATOR_COUNT],
        )
        .unwrap()
    }

    fn graph(uniform: usize, seed: &mut u64) -> BooleanNeuronGraph {
        let shifts = std::array::from_fn(|channel| {
            std::array::from_fn(|destination| {
                if channel < uniform {
                    (channel * 9) as u8
                } else {
                    ((destination * (2 * channel + 1)) % NEURON_COUNT) as u8
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

    #[test]
    fn bps27_live_groups_change_kernel_choice_without_changing_results() {
        let shifts = std::array::from_fn(|_| std::array::from_fn(|destination| destination as u8));
        let gate = UnaryGate {
            when_false: 0,
            when_true: u64::MAX,
        };
        let graph =
            BooleanNeuronGraph::new([gate; OPERATOR_COUNT], UnaryGate::default(), gate, shifts)
                .unwrap();
        let candidate = model(2, 1);
        let static_router = graph.prepare_cost_router(candidate);
        let live_router = graph.prepare_live_cost_router(candidate);
        let input = 0x0000_0000_0000_0003u64;
        assert_eq!(live_router.live_shift_counts(input), [2; OPERATOR_COUNT]);
        assert_eq!(
            static_router.adaptive_policy().max_direct_lanes,
            [31; OPERATOR_COUNT]
        );
        assert_eq!(
            live_router.adaptive_policy(input).max_direct_lanes,
            [0; OPERATOR_COUNT]
        );
        assert_eq!(
            live_router.decision(input),
            CostRoutingDecision::Adaptive([AdaptiveRoute::Grouped; OPERATOR_COUNT])
        );
        for previous in [0, u64::MAX, 0x9aa1_24bb_bbc7_0220] {
            assert_eq!(
                live_router.step(input, previous),
                graph.step_scalar(input, previous)
            );
            assert_eq!(
                static_router.step(input, previous),
                graph.step_scalar(input, previous)
            );
        }
    }

    #[test]
    fn bps27_all_weights_topologies_and_costs_match_scalar_state_and_outputs() {
        let mut seed = 0x27ba_f201_a3dd_618eu64;
        for uniform in 0..=OPERATOR_COUNT {
            let graph = graph(uniform, &mut seed);
            for cost in [model(1, 8), model(8, 1), model(u32::MAX, u32::MAX)] {
                let router = graph.prepare_live_cost_router(cost);
                for weight in 0..=NEURON_COUNT as u8 {
                    let inputs = exact_weight_corpus(next(&mut seed), weight, 5).unwrap();
                    let initial = next(&mut seed);
                    let mut actual = vec![u64::MAX; inputs.len()];
                    let mut state = initial;
                    router.scan_into(&inputs, &mut state, &mut actual).unwrap();
                    let mut scalar = initial;
                    let mut expected = Vec::new();
                    for &input in &inputs {
                        let live = router.live_shift_counts(input);
                        for (channel, &groups) in live.iter().enumerate() {
                            assert!(groups <= graph.unique_shift_counts()[channel]);
                            assert!(groups <= graph.gates[channel].evaluate(input).count_ones());
                        }
                        let (next_state, output) = graph.step_scalar(input, scalar);
                        assert_eq!(router.step(input, scalar), (next_state, output));
                        scalar = next_state;
                        expected.push(output);
                    }
                    assert_eq!(actual, expected);
                    assert_eq!(state, scalar);
                }
            }
        }
    }

    #[test]
    fn bps27_rewiring_and_chunks_preserve_exactness() {
        let mut seed = 0x279a_291b_c0ff_e129u64;
        let original = graph(3, &mut seed);
        let rewired = original.with_edge(OperatorChannel::Retain, 5, 23).unwrap();
        let restored = rewired.with_edge(OperatorChannel::Retain, 5, 0).unwrap();
        assert_eq!(original, restored);
        let mut inputs = Vec::new();
        for weight in [0, 1, 4, 16, 32, 48, 63, 64] {
            inputs.extend(exact_weight_corpus(next(&mut seed), weight, 19).unwrap());
        }
        let initial = next(&mut seed);
        for graph in [&original, &rewired, &restored] {
            let router = graph.prepare_live_cost_router(model(2, 5));
            let mut state = initial;
            let mut output = vec![0; inputs.len()];
            let mut start = 0;
            for end in [0, 1, 1, 7, 31, 64, 129, inputs.len()] {
                router
                    .scan_into(&inputs[start..end], &mut state, &mut output[start..end])
                    .unwrap();
                start = end;
            }
            assert_eq!(output, graph.forward_scalar(&inputs, initial));
            let mut scalar = initial;
            for &input in &inputs {
                scalar = graph.step_scalar(input, scalar).0;
            }
            assert_eq!(state, scalar);
            let before = state;
            let mut wrong = [0xabab_u64];
            assert_eq!(
                router.scan_into(&[1, 2], &mut state, &mut wrong),
                Err(BooleanStreamError::OutputLengthMismatch {
                    inputs: 2,
                    outputs: 1,
                })
            );
            assert_eq!(state, before);
            assert_eq!(wrong, [0xabab_u64]);
        }
    }
}
