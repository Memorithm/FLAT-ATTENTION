//! BPS-17: compile a shared four-operator graph into the exact BPS-15 kernel.
use flat_algebraic_attention::boolean_neuron_graph::BooleanNeuronGraph;
use flat_algebraic_attention::boolean_operator_channels::{
    BooleanOperatorProgram, OperatorChannel,
};
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn main() {
    let input_gate = UnaryGate {
        when_false: 0x0000_0000_0000_0001,
        when_true: 0x1111_1111_1111_1111,
    };
    let output_gate = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    let bps15 = BooleanOperatorProgram::new(
        [input_gate; 4],
        UnaryGate::default(),
        output_gate,
        [1, 3, 5, 7],
    )
    .unwrap();
    let shared = BooleanNeuronGraph::from_shared(bps15);
    assert_eq!(shared.compiled_uniform_shifts(), Some([1, 3, 5, 7]));

    let inputs = [0, 1, u64::MAX, 0x1234_5678_9abc_def0, 0, u64::MAX];
    let initial = 0xacc1_a5ef_138d_205cu64;
    assert_eq!(
        shared.forward_grouped(&inputs, initial),
        bps15.forward_dense(&inputs, initial)
    );
    assert_eq!(
        shared.forward_grouped_sparse(&inputs, initial),
        bps15.forward_sparse(&inputs, initial)
    );

    let individualized = shared.with_edge(OperatorChannel::Retain, 17, 9).unwrap();
    assert_eq!(individualized.compiled_uniform_shifts(), None);
    assert_eq!(
        individualized.forward_grouped_sparse(&inputs, initial),
        individualized.forward_scalar(&inputs, initial)
    );
    assert_eq!(
        individualized.forward_grouped(&inputs, initial),
        individualized.forward_scalar(&inputs, initial)
    );
    println!("BPS17_FASTPATH_UNIFORM=true");
    println!("BPS17_REWIRED_FALLBACK_EXACT=true");
    println!("BPS17_COMPLETE=true");
}
