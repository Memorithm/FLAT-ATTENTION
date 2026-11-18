//! BPS-16: reproduce a two-edge learning task with per-neuron routing.
use flat_algebraic_attention::boolean_neuron_graph::{
    BooleanNeuronGraph, NeuronTrainPolicy, NEURON_COUNT, OPERATOR_COUNT,
};
use flat_algebraic_attention::boolean_operator_channels::OperatorChannel;
use flat_algebraic_attention::boolean_scan_lstm::{BooleanTrainingSequence, UnaryGate};

fn main() {
    let active = (1u64 << 1) | (1u64 << 3);
    let retention = UnaryGate {
        when_false: active,
        when_true: active,
    };
    let output = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    let initial = BooleanNeuronGraph::new(
        [retention, UnaryGate::default(), UnaryGate::default(), UnaryGate::default()],
        UnaryGate::default(),
        output,
        [[0u8; NEURON_COUNT]; OPERATOR_COUNT],
    )
    .unwrap();
    let teacher = initial
        .with_edge(OperatorChannel::Retain, 1, 1)
        .unwrap()
        .with_edge(OperatorChannel::Retain, 3, 3)
        .unwrap();
    let train_inputs = [0u64];
    let train_targets = teacher.forward_scalar(&train_inputs, 1);
    let heldout_inputs = [0u64, 0];
    let heldout_targets = teacher.forward_scalar(&heldout_inputs, 1 | (1 << 5));
    let train = [BooleanTrainingSequence {
        inputs: &train_inputs,
        targets: &train_targets,
        initial_state: 1,
    }];
    let heldout = [BooleanTrainingSequence {
        inputs: &heldout_inputs,
        targets: &heldout_targets,
        initial_state: 1 | (1 << 5),
    }];
    let mut learner = initial;
    let report = learner
        .fit_edges(
            &train,
            &heldout,
            u64::MAX,
            NeuronTrainPolicy {
                max_epochs: 1,
                max_candidates: 300,
            },
        )
        .unwrap();

    assert_eq!(report.train_after, 0);
    assert_eq!(report.validation_after, 0);
    assert_eq!(report.accepted_rewires, 2);
    assert_eq!(learner.source_shift(OperatorChannel::Retain, 1), Some(1));
    assert_eq!(learner.source_shift(OperatorChannel::Retain, 3), Some(3));
    assert_eq!(learner.forward_grouped_sparse(&heldout_inputs, 1 | (1 << 5)), heldout_targets);
    assert_eq!(learner.forward_grouped(&heldout_inputs, 1 | (1 << 5)), heldout_targets);
    println!("BPS16_INDEPENDENT_SOURCE_EDGES=2");
    println!("BPS16_LOSS_BEFORE={}", report.train_before);
    println!("BPS16_LOSS_AFTER={}", report.train_after);
    println!("BPS16_HELDOUT_LOSS_AFTER={}", report.validation_after);
    println!("BPS16_CANDIDATES={}", report.candidates_evaluated);
    println!("BPS16_COMPLETE=true");
}
