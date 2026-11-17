//! BPS-4 host-only reproduction of coupled gate/rotation learning.
//! This smoke is a correctness check, not a benchmark or energy measurement.

use flat_algebraic_attention::boolean_coupled_learning::{
    CoupledExample, CoupledModel, JointTrainPolicy,
};
use flat_algebraic_attention::boolean_edge_memory::BooleanEdgeProgram;
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn main() {
    let all_true = UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    };
    let teacher = BooleanEdgeProgram::new(all_true, UnaryGate::default(), u64::MAX, 7)
        .expect("valid teacher shift");

    let train_inputs = [1u64, 0, 1, 0, 1, 0, 1];
    let train_targets = teacher.forward_dense(&train_inputs, 0, 1);
    let validation_inputs = [0u64, 1, 0, 0, 1, 0, 1, 0];
    let validation_initial_state = 1u64 << 3;
    let validation_targets =
        teacher.forward_dense(&validation_inputs, 0, validation_initial_state);
    let train = [CoupledExample {
        inputs: &train_inputs,
        targets: &train_targets,
        baseline_input: 0,
        initial_state: 1,
    }];
    let validation = [CoupledExample {
        inputs: &validation_inputs,
        targets: &validation_targets,
        baseline_input: 0,
        initial_state: validation_initial_state,
    }];

    let mut initial = BooleanEdgeProgram::new(all_true, UnaryGate::default(), u64::MAX, 0)
        .expect("valid student shift");
    initial.write.when_true = 1;
    let mut model = CoupledModel::RisingEdge(initial);
    let report = model
        .fit_joint(
            &train,
            &validation,
            u64::MAX,
            JointTrainPolicy {
                max_epochs: 3,
                max_candidate_evaluations: 8_192,
            },
        )
        .expect("valid training and held-out datasets");

    assert!(report.train_loss_before > 0);
    assert_eq!(report.train_loss_after, 0);
    assert_eq!(report.validation_loss_after, 0);
    assert_eq!(report.rotation_after, 7);
    assert!(report.accepted_joint_moves > 0);
    assert!(report.accepted_gate_flips > 0);
    let CoupledModel::RisingEdge(learned) = model else {
        panic!("model family changed");
    };
    assert_eq!(
        learned.forward_event_scan(&validation_inputs, 0, validation_initial_state),
        validation_targets
    );

    println!("candidate=bps4_coupled_boolean_learning_v1");
    println!("initial_training_hamming_loss={}", report.train_loss_before);
    println!("final_training_hamming_loss={}", report.train_loss_after);
    println!(
        "held_out_hamming_loss_after={}",
        report.validation_loss_after
    );
    println!("learned_recurrent_shift={}", report.rotation_after);
    println!("accepted_gate_flips={}", report.accepted_gate_flips);
    println!("accepted_joint_moves={}", report.accepted_joint_moves);
    println!("candidate_evaluations={}", report.candidates_evaluated);
    println!("pair_candidates={}", report.pair_candidates_evaluated);
    println!("budget_exhausted={}", report.budget_exhausted);
    println!("energy_measurement=false");
    println!("speedup_claim=none");
}
