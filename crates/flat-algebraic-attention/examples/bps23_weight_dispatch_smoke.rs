//! BPS-23: explicit density/topology dispatcher validated against the
//! independent 64-neuron scalar oracle, no production policy changes.
use flat_algebraic_attention::boolean_neuron_activity::exact_weight_corpus;
use flat_algebraic_attention::boolean_neuron_dispatch::{
    DenseWeightWindow, WeightRoutingDecision, WeightRoutingPolicy,
};
use flat_algebraic_attention::boolean_neuron_graph::{
    AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
};
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn model(uniform: usize) -> BooleanNeuronGraph {
    let shifts = std::array::from_fn(|channel| {
        std::array::from_fn(|destination| {
            if channel < uniform {
                (channel * 9) as u8
            } else {
                ((destination * (2 * channel + 1)) % NEURON_COUNT) as u8
            }
        })
    });
    let gate = UnaryGate {
        when_false: 0x0000_0000_0000_001f,
        when_true: u64::MAX,
    };
    let all = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    BooleanNeuronGraph::new([gate; OPERATOR_COUNT], UnaryGate::default(), all, shifts).unwrap()
}

fn main() {
    // This candidate was selected using previous Dell development evidence.
    // It is NOT a measured independent speedup nor a default runtime setting.
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

    let zero_uniform = model(0);
    let three_uniform = model(3);
    assert_eq!(zero_uniform.uniform_operation_count(), 0);
    assert_eq!(three_uniform.uniform_operation_count(), 3);
    for weight in [0, 1, 4, 16, 32, 48, 63, 64] {
        let corpus = exact_weight_corpus(0xba95_1cdc_236d_c93a, weight, 91).unwrap();
        let initial = 0x912e_0bce_36a4_1825;
        for graph in [&zero_uniform, &three_uniform] {
            let expected = graph.forward_scalar(&corpus, initial);
            assert_eq!(
                graph.forward_weight_routed(&corpus, initial, policy),
                expected
            );
            assert_eq!(
                graph.forward_weight_routed(&corpus, initial, WeightRoutingPolicy::DISABLED),
                expected
            );
        }
        let expected_route = if (4..=63).contains(&weight) {
            WeightRoutingDecision::GroupedDense
        } else {
            WeightRoutingDecision::ExistingHybrid
        };
        assert_eq!(
            zero_uniform.weight_routing_decision(corpus[0], policy),
            expected_route
        );
        assert_eq!(
            three_uniform.weight_routing_decision(corpus[0], policy),
            WeightRoutingDecision::GroupedDense
        );
    }
    println!("BPS23_INPUT_WEIGHTS=8");
    println!("BPS23_TOPOLOGY_BUCKETS=5");
    println!("BPS23_DEFAULT_UNCHANGED=true");
    println!("BPS23_SCALAR_OUTPUT_EXACT=true");
    println!("BPS23_COMPLETE=true");
}
