//! BPS-24: exact compiled once-per-sequence Boolean routing plan.
use flat_algebraic_attention::boolean_neuron_activity::exact_weight_corpus;
use flat_algebraic_attention::boolean_neuron_dispatch::{
    DenseWeightWindow, WeightRoutingDecision, WeightRoutingPolicy,
};
use flat_algebraic_attention::boolean_neuron_graph::{
    AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
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
    let gate = UnaryGate {
        when_false: 0x0000_0000_0000_0003,
        when_true: u64::MAX,
    };
    let output = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    BooleanNeuronGraph::new([gate; OPERATOR_COUNT], UnaryGate::default(), output, shifts).unwrap()
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
        let prepared = graph.prepare_weight_router(policy);
        for weight in [0, 1, 4, 16, 32, 48, 63, 64] {
            let inputs = exact_weight_corpus(0xf1d2_c038_a573_922e, weight, 73).unwrap();
            let expected = graph.forward_scalar(&inputs, 0x772c_e0aa_9f11_38b5);
            assert_eq!(prepared.forward(&inputs, 0x772c_e0aa_9f11_38b5), expected);
            assert_eq!(
                graph.forward_weight_routed(&inputs, 0x772c_e0aa_9f11_38b5, policy),
                expected
            );
            for &input in &inputs {
                assert_eq!(
                    prepared.decision(input),
                    graph.weight_routing_decision(input, policy)
                );
            }
        }
    }
    let fully_dense = graph(3);
    assert_eq!(
        fully_dense.prepare_weight_router(policy).decision(0),
        WeightRoutingDecision::GroupedDense
    );
    let disabled = graph(0);
    assert_eq!(
        disabled
            .prepare_weight_router(WeightRoutingPolicy::DISABLED)
            .decision(u64::MAX),
        WeightRoutingDecision::ExistingHybrid
    );
    println!("BPS24_PREPARED_ONCE=true");
    println!("BPS24_NO_AUTOMATIC_CALIBRATION=true");
    println!("BPS24_ORACLE_EXACT=true");
    println!("BPS24_COMPLETE=true");
}
