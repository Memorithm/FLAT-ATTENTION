//! BPS-27 live-group dispatch: independent scalar output and state smoke.
use flat_algebraic_attention::boolean_neuron_activity::exact_weight_corpus;
use flat_algebraic_attention::boolean_neuron_cost_router::{ChannelRoutingCost, RoutingCostModel};
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
        when_false: 0x0000_0001_0000_0001u64.rotate_left((channel * 9) as u32),
        when_true: 0xfefe_fefe_fefe_fefeu64.rotate_right((channel * 5) as u32),
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
    // Costs are illustrative input parameters, not Dell measurements.
    let model = RoutingCostModel::new(
        [ChannelRoutingCost {
            direct_lane_units: 2,
            grouped_shift_units: 5,
            direct_setup_units: 0,
            grouped_setup_units: 1,
        }; OPERATOR_COUNT],
        [64; OPERATOR_COUNT],
    )
    .unwrap();
    for uniform in 0..=OPERATOR_COUNT {
        let graph = graph(uniform);
        let router = graph.prepare_live_cost_router(model);
        let mut inputs = Vec::new();
        for weight in [0, 1, 4, 16, 32, 48, 63, 64] {
            inputs.extend(exact_weight_corpus(0x27a0_9eff_28cb_1120, weight, 11).unwrap());
        }
        let initial = 0x28ba_98b2_01ac_e21fu64;
        let mut state = initial;
        let mut outputs = vec![0; inputs.len()];
        let mut previous = 0;
        for end in [0, 1, 1, 7, 23, 40, 77, inputs.len()] {
            router
                .scan_into(
                    &inputs[previous..end],
                    &mut state,
                    &mut outputs[previous..end],
                )
                .unwrap();
            previous = end;
        }
        assert_eq!(outputs, graph.forward_scalar(&inputs, initial));
        let mut oracle_state = initial;
        for &input in &inputs {
            oracle_state = graph.step_scalar(input, oracle_state).0;
        }
        assert_eq!(state, oracle_state);
    }
    println!("BPS27_LIVE_SHIFT_COUNTS=true");
    println!("BPS27_EXPLICIT_COSTS_NO_CALIBRATION=true");
    println!("BPS27_SCALAR_STATE_AND_OUTPUT_EXACT=true");
    println!("BPS27_COMPLETE=true");
}
