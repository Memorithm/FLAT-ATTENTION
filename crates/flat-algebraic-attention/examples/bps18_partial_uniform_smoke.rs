//! BPS-18: selective compilation of operation-specific uniform routes.
use flat_algebraic_attention::boolean_neuron_graph::BooleanNeuronGraph;
use flat_algebraic_attention::boolean_operator_channels::{
    BooleanOperatorProgram, OperatorChannel,
};
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn main() {
    let gate = UnaryGate {
        when_false: 0x00ff_0010_0100_ff00,
        when_true: u64::MAX,
    };
    let output = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    let shared =
        BooleanOperatorProgram::new([gate; 4], UnaryGate::default(), output, [1, 3, 5, 7]).unwrap();
    let model = BooleanNeuronGraph::from_shared(shared);
    assert_eq!(model.compiled_uniform_shifts(), Some([1, 3, 5, 7]));
    let mixed = model.with_edge(OperatorChannel::Inhibit, 11, 12).unwrap();
    assert_eq!(mixed.compiled_uniform_shifts(), None);
    assert_eq!(
        mixed.compiled_channel_uniform_shifts(),
        [Some(1), Some(3), None, Some(7)]
    );
    let inputs = [0u64, 1, u64::MAX, 0x1234_9876_cdef_4567, 0, 1];
    let initial = 0x1111_0a0a_bbbb_dd99u64;
    assert_eq!(
        mixed.forward_grouped(&inputs, initial),
        mixed.forward_scalar(&inputs, initial)
    );
    assert_eq!(
        mixed.forward_grouped_sparse(&inputs, initial),
        mixed.forward_scalar(&inputs, initial)
    );
    let restored = mixed.with_edge(OperatorChannel::Inhibit, 11, 5).unwrap();
    assert_eq!(restored.compiled_uniform_shifts(), Some([1, 3, 5, 7]));
    assert_eq!(
        restored.forward_grouped(&inputs, initial),
        shared.forward_dense(&inputs, initial)
    );
    println!("BPS18_PARTIAL_UNIFORM_COUNT=3");
    println!("BPS18_REWIRED_FALLBACK_EXACT=true");
    println!("BPS18_RESTORED_SHARED_EXACT=true");
    println!("BPS18_COMPLETE=true");
}
