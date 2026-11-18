//! BPS-14 smoke: learn two recurrent inputs per Boolean neuron.
//! This checks exact discrete learning; it is NOT a speed, power or ML benchmark.

use flat_algebraic_attention::boolean_scan_lstm::{
    BooleanGateProgram, BooleanTrainingSequence, UnaryGate,
};
use flat_algebraic_attention::boolean_two_tap_memory::{
    BooleanTwoTapProgram, TwoTapMix, TwoTapTrainPolicy,
};

fn main() {
    let gates = BooleanGateProgram {
        forget: UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        },
        write: UnaryGate::default(),
        output: UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        },
    };
    let teacher = BooleanTwoTapProgram::new(gates, 1, 3, TwoTapMix::Or).unwrap();
    let train_inputs = [0u64];
    let train_targets = teacher.forward_serial(&train_inputs, 1);
    let heldout_inputs = [0u64, 0];
    let heldout_targets = teacher.forward_serial(&heldout_inputs, 2);
    let train = [BooleanTrainingSequence {
        inputs: &train_inputs,
        targets: &train_targets,
        initial_state: 1,
    }];
    let heldout = [BooleanTrainingSequence {
        inputs: &heldout_inputs,
        targets: &heldout_targets,
        initial_state: 2,
    }];
    let mut learner = BooleanTwoTapProgram::new(gates, 0, 0, TwoTapMix::Or).unwrap();
    let report = learner
        .fit_topology(
            &train,
            &heldout,
            u64::MAX,
            TwoTapTrainPolicy {
                max_epochs: 3,
                max_candidates: 256,
            },
        )
        .unwrap();
    assert_eq!(report.train_loss_after, 0);
    assert_eq!(learner.shifts(), (1, 3));
    assert_eq!(learner.mix(), TwoTapMix::Or);
    assert_eq!(learner.forward_serial(&heldout_inputs, 2), heldout_targets);
    assert!(report.evaluated_candidates <= 256);
    println!("bps14_two_tap=true");
    println!("bps14_train_initial_hamming={}", report.train_loss_before);
    println!("bps14_train_final_hamming={}", report.train_loss_after);
    println!(
        "bps14_holdout_final_hamming={}",
        report.validation_loss_after
    );
    println!("bps14_evaluated_candidates={}", report.evaluated_candidates);
    println!("bps14_result=PASS no_physical_speed_or_quality_claim");
}
