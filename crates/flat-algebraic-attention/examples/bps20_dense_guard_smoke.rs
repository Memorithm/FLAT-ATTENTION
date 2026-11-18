//! BPS-20: exact compiled Boolean guard for activity-dependent dense fallback.
use flat_algebraic_attention::boolean_neuron_graph::{
    AdaptiveRoute, AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
};
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn main() {
    let mut shifts = [[0u8; NEURON_COUNT]; OPERATOR_COUNT];
    for destination in 0..NEURON_COUNT {
        shifts[0][destination] = destination as u8;
    }
    shifts[1] = [3; NEURON_COUNT];
    shifts[2] = [5; NEURON_COUNT];
    shifts[3] = [7; NEURON_COUNT];

    let recurrent_gate = UnaryGate {
        when_false: 0x0000_0000_0000_0003,
        when_true: u64::MAX,
    };
    let output = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    let graph = BooleanNeuronGraph::new(
        [recurrent_gate; OPERATOR_COUNT],
        UnaryGate::default(),
        output,
        shifts,
    )
    .unwrap();
    assert_eq!(
        graph.compiled_dense_gate_guard(),
        Some((u64::MAX ^ 3, 0))
    );
    assert!(graph.dense_gate_eligible(u64::MAX));
    assert!(!graph.dense_gate_eligible(0));
    let policy = AdaptiveRoutingPolicy::default();
    assert_eq!(
        graph.adaptive_route_plan(0, policy)[0],
        AdaptiveRoute::DirectLanes
    );
    assert_eq!(
        graph.adaptive_route_plan(u64::MAX, policy)[0],
        AdaptiveRoute::Grouped
    );

    let inputs = [0, u64::MAX, 0, 0, u64::MAX, 0, u64::MAX, 0];
    let initial = 0x11ae_092f_cc81_350du64;
    assert_eq!(
        graph.forward_activity_hybrid(&inputs, initial, policy),
        graph.forward_scalar(&inputs, initial)
    );
    assert_eq!(
        graph.forward_activity_hybrid(&inputs, initial, policy),
        graph.forward_grouped_sparse(&inputs, initial)
    );
    assert_eq!(
        graph.step_activity_hybrid(u64::MAX, initial, policy),
        graph.step_grouped(u64::MAX, initial)
    );
    println!("BPS20_COMPILED_DENSE_GATE_GUARD=true");
    println!("BPS20_SPARSE_DIRECT_ROUTE=true");
    println!("BPS20_DENSE_GROUPED_ROUTE=true");
    println!("BPS20_EXACT_OUTPUT=true");
    println!("BPS20_COMPLETE=true");
}
