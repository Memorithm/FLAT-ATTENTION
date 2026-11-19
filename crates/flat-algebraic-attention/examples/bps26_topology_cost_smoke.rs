//! BPS-26: opt-in per-channel cost and topology threshold smoke.
use flat_algebraic_attention::boolean_neuron_activity::exact_weight_corpus;
use flat_algebraic_attention::boolean_neuron_cost_router::{
    ChannelRoutingCost, RoutingCostModel,
};
use flat_algebraic_attention::boolean_neuron_graph::{
    BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
};
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
    // Deliberately illustrative costs, NOT physical Dell measurements.
    let costs = std::array::from_fn(|channel| ChannelRoutingCost {
        direct_lane_units: 2 + channel as u32,
        grouped_shift_units: 6 + channel as u32,
        direct_setup_units: 0,
        grouped_setup_units: 1,
    });
    let model = RoutingCostModel::new(costs, [64; OPERATOR_COUNT]).unwrap();
    for uniform in 0..=OPERATOR_COUNT {
        let graph = graph(uniform);
        let router = graph.prepare_cost_router(model);
        for weight in [0, 1, 4, 16, 32, 48, 63, 64] {
            let inputs = exact_weight_corpus(0x2600_b00f_528a_a129, weight, 23).unwrap();
            let initial = 0x197d_ba2c_0ff1_9237u64;
            let mut state = initial;
            let mut actual = vec![0u64; inputs.len()];
            router.scan_into(&inputs, &mut state, &mut actual).unwrap();
            assert_eq!(actual, graph.forward_scalar(&inputs, initial));
            let mut expected_state = initial;
            for &input in &inputs {
                expected_state = graph.step_scalar(input, expected_state).0;
            }
            assert_eq!(state, expected_state);
        }
    }
    println!("BPS26_CALLER_COSTS_NOT_AUTO_CALIBRATED=true");
    println!("BPS26_CHANNEL_TOPOLOGY_CAPS=true");
    println!("BPS26_SCALAR_OUTPUT_AND_STATE_EXACT=true");
    println!("BPS26_COMPLETE=true");
}
