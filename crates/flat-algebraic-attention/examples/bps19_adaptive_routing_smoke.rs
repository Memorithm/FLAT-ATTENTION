//! BPS-19: adapt exact recurrent routing to live lane and graph sparsity.
use flat_algebraic_attention::boolean_neuron_graph::{
    AdaptiveRoute, AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
};
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn main() {
    let mut offsets = [[0u8; NEURON_COUNT]; OPERATOR_COUNT];
    for destination in 0..NEURON_COUNT {
        offsets[0][destination] = destination as u8;
    }
    offsets[1] = [3; NEURON_COUNT];
    offsets[2] = [5; NEURON_COUNT];
    offsets[3] = [7; NEURON_COUNT];
    let sparse = UnaryGate {
        when_false: (1 << 1) | (1 << 37),
        when_true: u64::MAX,
    };
    let switch = UnaryGate {
        when_false: 0,
        when_true: u64::MAX,
    };
    let output = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    let graph = BooleanNeuronGraph::new(
        [sparse, switch, UnaryGate::default(), UnaryGate::default()],
        UnaryGate::default(),
        output,
        offsets,
    )
    .unwrap();
    let policy = AdaptiveRoutingPolicy::default();
    assert_eq!(
        graph.adaptive_route_plan(0, policy)[0],
        AdaptiveRoute::DirectLanes
    );
    assert_eq!(
        graph.adaptive_route_plan(u64::MAX, policy)[0],
        AdaptiveRoute::Grouped
    );
    assert_eq!(
        graph.adaptive_route_plan(u64::MAX, policy)[1],
        AdaptiveRoute::Uniform
    );
    let inputs = [0, u64::MAX, 0, 0, u64::MAX, 0, u64::MAX, 0];
    let initial = 0x0101_2345_0000_0101u64;
    assert_eq!(
        graph.forward_adaptive(&inputs, initial, policy),
        graph.forward_scalar(&inputs, initial)
    );
    assert_eq!(
        graph.forward_adaptive(&inputs, initial, policy),
        graph.forward_grouped_sparse(&inputs, initial)
    );
    let disabled = AdaptiveRoutingPolicy {
        max_direct_lanes: [0; OPERATOR_COUNT],
    };
    assert_eq!(
        graph.forward_adaptive(&inputs, initial, disabled),
        graph.forward_scalar(&inputs, initial)
    );
    println!("BPS19_SCALAR_LANES_SELECTED=true");
    println!("BPS19_GROUPED_SELECTED=true");
    println!("BPS19_UNIFORM_SELECTED=true");
    println!("BPS19_EXACT_OUTPUT=true");
    println!("BPS19_COMPLETE=true");
}
