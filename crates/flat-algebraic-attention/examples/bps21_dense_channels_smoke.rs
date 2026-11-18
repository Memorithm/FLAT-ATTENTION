//! BPS-21: independently dense recurrent operators in an otherwise
//! heterogeneous 64-neuron graph, with exact scalar-output matching.
use flat_algebraic_attention::boolean_neuron_graph::{
    AdaptiveRoutingPolicy, BooleanNeuronGraph, NEURON_COUNT, OPERATOR_COUNT,
};
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn main() {
    let mut shifts = [[0u8; NEURON_COUNT]; OPERATOR_COUNT];
    for lane in 0..NEURON_COUNT {
        for (channel, offsets) in shifts.iter_mut().enumerate() {
            offsets[lane] = ((lane * (2 * channel + 1)) % NEURON_COUNT) as u8;
        }
    }
    let all = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    let gates = [
        all,
        UnaryGate {
            when_false: 1 << 7,
            when_true: u64::MAX,
        },
        UnaryGate {
            when_false: u64::MAX,
            when_true: 1 << 5,
        },
        UnaryGate {
            when_false: 0,
            when_true: u64::MAX,
        },
    ];
    let model = BooleanNeuronGraph::new(gates, UnaryGate::default(), all, shifts).unwrap();
    assert_eq!(model.compiled_dense_gate_guard(), None);
    assert_eq!(model.fully_active_channel_bits(0), 0b0101);
    assert_eq!(model.fully_active_channel_bits(u64::MAX), 0b1011);
    let inputs = [0, u64::MAX, 0x1256_14ac_1f01_ea42, 0, u64::MAX, 1];
    let initial = 0xa13f_98dc_66b0_55c7;
    let expected = model.forward_scalar(&inputs, initial);
    assert_eq!(
        model.forward_adaptive(&inputs, initial, AdaptiveRoutingPolicy::default()),
        expected
    );
    assert_eq!(
        model.forward_activity_hybrid(&inputs, initial, AdaptiveRoutingPolicy::default()),
        expected
    );
    println!("BPS21_MIXED_DENSE_CHANNELS=2");
    println!("BPS21_DENSE_GLOBAL_GUARD=false");
    println!("BPS21_EXACT_OUTPUT=true");
    println!("BPS21_COMPLETE=true");
}
