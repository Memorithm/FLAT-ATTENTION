//! BPS-25: allocation-free, chunked Boolean recurrence over a prepared plan.
use flat_algebraic_attention::boolean_neuron_activity::exact_weight_corpus;
use flat_algebraic_attention::boolean_neuron_dispatch::{DenseWeightWindow, WeightRoutingPolicy};
use flat_algebraic_attention::boolean_neuron_graph::{
    AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
};
use flat_algebraic_attention::boolean_neuron_stream::BooleanStreamError;
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn graph(uniform: usize) -> BooleanNeuronGraph {
    let shifts = std::array::from_fn(|channel| {
        std::array::from_fn(|lane| {
            if channel < uniform {
                (channel * 9) as u8
            } else {
                ((lane * (2 * channel + 1)) % NEURON_COUNT) as u8
            }
        })
    });
    let gates = std::array::from_fn(|channel| UnaryGate {
        when_false: (0x0110_0000_0000_0001u64).rotate_left((channel * 13) as u32),
        when_true: (!0x0100_0000_0000_0001u64).rotate_right((channel * 7) as u32),
    });
    BooleanNeuronGraph::new(
        gates,
        UnaryGate {
            when_false: 0x0001_0000_0000_0001,
            when_true: 0x0080_0000_0000_0080,
        },
        UnaryGate {
            when_false: u64::MAX,
            when_true: 0xfefe_fefe_fefe_fefe,
        },
        shifts,
    )
    .unwrap()
}

fn main() {
    let policy = WeightRoutingPolicy::new(
        [
            Some(DenseWeightWindow { first: 4, last: 63 }),
            None,
            None,
            Some(DenseWeightWindow { first: 0, last: 64 }),
            None,
        ],
        AdaptiveRoutingPolicy::default(),
    )
    .unwrap();
    for uniform in [0, 3, 4] {
        let graph = graph(uniform);
        let router = graph.prepare_weight_router(policy);
        let mut inputs = Vec::new();
        for weight in [0, 1, 4, 16, 32, 48, 63, 64] {
            inputs.extend(exact_weight_corpus(0x25bb_129a_e0d4_0f83, weight, 19).unwrap());
        }

        let initial = 0x91c2_20af_54d1_3d72u64;
        let mut output = vec![u64::MAX; inputs.len()];
        let mut state = initial;
        let mut start = 0;
        for &end in &[0, 1, 1, 7, 31, 64, 129, inputs.len()] {
            router
                .scan_into(&inputs[start..end], &mut state, &mut output[start..end])
                .unwrap();
            start = end;
        }
        assert_eq!(output, graph.forward_scalar(&inputs, initial));
        let mut scalar_state = initial;
        for &input in &inputs {
            scalar_state = graph.step_scalar(input, scalar_state).0;
        }
        assert_eq!(state, scalar_state);

        let mut seen = Vec::new();
        let mut callback_state = initial;
        router.scan_visit(&inputs, &mut callback_state, |word| seen.push(word));
        assert_eq!(seen, output);
        assert_eq!(callback_state, state);

        let mut one = [0x7777u64];
        let before = one;
        let mut rejected_state = state;
        assert_eq!(
            router.scan_into(&[1, 2], &mut rejected_state, &mut one),
            Err(BooleanStreamError::OutputLengthMismatch {
                inputs: 2,
                outputs: 1,
            })
        );
        assert_eq!(one, before);
        assert_eq!(rejected_state, state);
    }
    println!("BPS25_NO_OUTPUT_ALLOCATION_REQUIRED=true");
    println!("BPS25_CHUNK_STATE_EXACT=true");
    println!("BPS25_REJECT_MISMATCH_ATOMIC=true");
    println!("BPS25_ORACLE_EXACT=true");
    println!("BPS25_COMPLETE=true");
}
