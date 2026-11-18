//! BPS-25: allocation-free chunked execution of the exact prepared Boolean router.
//!
//! The caller owns the recurrent state and the output buffer. No output
//! allocation, implicit reset, or graph reconfiguration occurs in this module.

use super::boolean_neuron_dispatch::PreparedWeightRouter;

/// Reject an output buffer mismatch before consuming any input or changing state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanStreamError {
    OutputLengthMismatch { inputs: usize, outputs: usize },
}

impl PreparedWeightRouter<'_> {
    /// Run one chunk into an existing output slice, retaining state for the next chunk.
    ///
    /// The output length must equal the input length. A mismatch fails
    /// atomically: neither the caller's state nor any output element changes.
    /// Once validated, every tick consumes the *previous* 64-neuron state
    /// and writes its output in input order. The kernel's routing choice
    /// remains the caller-configured, immutable BPS-24 plan.
    ///
    /// This method itself does not allocate; the caller controls storage.
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
        for (&input, output_slot) in inputs.iter().zip(outputs.iter_mut()) {
            let (next, output) = self.step(input, *state);
            *state = next;
            *output_slot = output;
        }
        Ok(())
    }

    /// Stream outputs directly into a caller-controlled sink without a Vec.
    ///
    /// State is advanced *before* each callback, so if a callback panics,
    /// the state retains the transition whose output was being delivered.
    /// Callbacks can use their own storage or transport; the router does not.
    pub fn scan_visit(&self, inputs: &[u64], state: &mut u64, mut visit: impl FnMut(u64)) {
        for &input in inputs {
            let (next, output) = self.step(input, *state);
            *state = next;
            visit(output);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::boolean_neuron_activity::exact_weight_corpus;
    use super::super::boolean_neuron_dispatch::{DenseWeightWindow, WeightRoutingPolicy};
    use super::super::boolean_neuron_graph::{
        AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
    };
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
            std::array::from_fn(|lane| {
                if channel < uniform {
                    (channel * 9) as u8
                } else {
                    ((lane * (2 * channel + 1)) % NEURON_COUNT) as u8
                }
            })
        });
        BooleanNeuronGraph::new(
            std::array::from_fn(|_| gate(seed)),
            gate(seed),
            gate(seed),
            shifts,
        )
        .unwrap()
    }

    fn policies() -> [WeightRoutingPolicy; 3] {
        [
            WeightRoutingPolicy::DISABLED,
            WeightRoutingPolicy::new(
                [Some(DenseWeightWindow { first: 0, last: 64 }); OPERATOR_COUNT + 1],
                AdaptiveRoutingPolicy::default(),
            )
            .unwrap(),
            WeightRoutingPolicy::new(
                [
                    Some(DenseWeightWindow { first: 4, last: 63 }),
                    None,
                    Some(DenseWeightWindow { first: 16, last: 48 }),
                    Some(DenseWeightWindow { first: 0, last: 64 }),
                    None,
                ],
                AdaptiveRoutingPolicy::default(),
            )
            .unwrap(),
        ]
    }

    fn oracle(graph: &BooleanNeuronGraph, inputs: &[u64], initial: u64) -> (Vec<u64>, u64) {
        let mut state = initial;
        let mut outputs = Vec::with_capacity(inputs.len());
        for &input in inputs {
            let (next, output) = graph.step_scalar(input, state);
            state = next;
            outputs.push(output);
        }
        (outputs, state)
    }

    #[test]
    fn bps25_mismatched_output_buffers_leave_state_and_outputs_unchanged() {
        let mut seed = 0x2540_238d_1910_7dbfu64;
        let graph = graph(2, &mut seed);
        let router = graph.prepare_weight_router(WeightRoutingPolicy::DISABLED);
        let inputs = [7u64, u64::MAX, 0];
        let mut state = 0xa30f_9b51_6637_12ddu64;
        let initial = state;
        let mut short = [0x1111u64, 0x2222];
        let before_short = short;
        assert_eq!(
            router.scan_into(&inputs, &mut state, &mut short),
            Err(BooleanStreamError::OutputLengthMismatch {
                inputs: 3,
                outputs: 2,
            })
        );
        assert_eq!(state, initial);
        assert_eq!(short, before_short);

        let mut long = [0x33u64; 4];
        let before_long = long;
        assert_eq!(
            router.scan_into(&inputs, &mut state, &mut long),
            Err(BooleanStreamError::OutputLengthMismatch {
                inputs: 3,
                outputs: 4,
            })
        );
        assert_eq!(state, initial);
        assert_eq!(long, before_long);
        assert_eq!(
            router.scan_into(&[], &mut state, &mut long),
            Err(BooleanStreamError::OutputLengthMismatch {
                inputs: 0,
                outputs: 4,
            })
        );
        assert_eq!(state, initial);
        assert_eq!(long, before_long);
    }

    #[test]
    fn bps25_all_65_weights_five_topologies_and_three_policies_match_scalar() {
        let mut seed = 0x2572_ea08_91cb_f44du64;
        for uniform in 0..=OPERATOR_COUNT {
            let graph = graph(uniform, &mut seed);
            assert_eq!(graph.uniform_operation_count(), uniform);
            for policy in policies() {
                let router = graph.prepare_weight_router(policy);
                for weight in 0..=NEURON_COUNT as u8 {
                    let inputs = exact_weight_corpus(next_random(&mut seed), weight, 5).unwrap();
                    let initial = next_random(&mut seed);
                    let (expected, final_state) = oracle(&graph, &inputs, initial);

                    let mut output = vec![0xdead_beef; inputs.len()];
                    let mut state = initial;
                    router.scan_into(&inputs, &mut state, &mut output).unwrap();
                    assert_eq!(output, expected, "weight={weight}, uniform={uniform}");
                    assert_eq!(state, final_state);
                    assert_eq!(router.forward(&inputs, initial), expected);

                    let mut visited = Vec::new();
                    let mut visited_state = initial;
                    router.scan_visit(&inputs, &mut visited_state, |word| visited.push(word));
                    assert_eq!(visited, expected);
                    assert_eq!(visited_state, final_state);
                }
            }
        }
    }

    #[test]
    fn bps25_irregular_chunks_and_rewiring_preserve_exact_state_trajectory() {
        let mut seed = 0x25a5_6741_8a36_d902u64;
        let original = graph(3, &mut seed);
        let altered = original
            .with_edge(OperatorChannel::Retain, 5, 23)
            .unwrap();
        let restored = altered
            .with_edge(OperatorChannel::Retain, 5, 0)
            .unwrap();
        assert_eq!(restored, original);

        let mut inputs = Vec::new();
        for weight in [0, 1, 4, 16, 32, 48, 63, 64] {
            inputs.extend(exact_weight_corpus(next_random(&mut seed), weight, 19).unwrap());
        }
        let initial = next_random(&mut seed);
        let ends = [0, 1, 1, 7, 31, 64, 129, inputs.len()];
        for model in [&original, &altered, &restored] {
            for policy in policies() {
                let router = model.prepare_weight_router(policy);
                let (expected, expected_state) = oracle(model, &inputs, initial);
                let mut output = vec![0u64; inputs.len()];
                let mut state = initial;
                let mut previous = 0;
                for &end in &ends {
                    router
                        .scan_into(
                            &inputs[previous..end],
                            &mut state,
                            &mut output[previous..end],
                        )
                        .unwrap();
                    previous = end;
                }
                assert_eq!(output, expected);
                assert_eq!(state, expected_state);

                let mut visited = Vec::new();
                let mut visited_state = initial;
                let mut previous = 0;
                for &end in &ends {
                    router.scan_visit(&inputs[previous..end], &mut visited_state, |word| {
                        visited.push(word)
                    });
                    previous = end;
                }
                assert_eq!(visited, expected);
                assert_eq!(visited_state, expected_state);
            }
        }
    }

    #[test]
    fn bps25_empty_chunk_is_state_and_output_noop() {
        let mut seed = 0x25ca_43ed_812b_b90fu64;
        let graph = graph(4, &mut seed);
        let router = graph.prepare_weight_router(WeightRoutingPolicy::DISABLED);
        let mut state = 0xdead_beef_abcd_0001;
        let mut visited = 0usize;
        router.scan_visit(&[], &mut state, |_| visited += 1);
        router.scan_into(&[], &mut state, &mut []).unwrap();
        assert_eq!(state, 0xdead_beef_abcd_0001);
        assert_eq!(visited, 0);
    }
}
