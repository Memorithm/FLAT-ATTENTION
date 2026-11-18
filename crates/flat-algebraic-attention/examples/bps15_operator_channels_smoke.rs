//! BPS-15: standalone correctness and optional exploratory CPU timings.
//!
//! The default mode performs deterministic bit-exact checks without timing.
//! Only --measure runs wall-clock timings. Do not construe CI timings as
//! physical performance or energy qualification.

use flat_algebraic_attention::boolean_operator_channels::{
    BooleanOperatorProgram, OperatorTrainPolicy,
};
use flat_algebraic_attention::boolean_scan_lstm::{BooleanTrainingSequence, UnaryGate};

fn always() -> UnaryGate {
    UnaryGate {
        when_false: u64::MAX,
        when_true: u64::MAX,
    }
}

fn main() {
    let never = UnaryGate::default();
    let masks = [always(), always(), never, never];
    let teacher = BooleanOperatorProgram::new(masks, never, always(), [1, 3, 0, 0]).unwrap();
    let training_inputs = [0u64];
    let training_targets = teacher.forward_dense(&training_inputs, 1);
    let holdout_inputs = [0u64, 0];
    let holdout_targets = teacher.forward_dense(&holdout_inputs, 2);
    let training = [BooleanTrainingSequence {
        inputs: &training_inputs,
        targets: &training_targets,
        initial_state: 1,
    }];
    let validation = [BooleanTrainingSequence {
        inputs: &holdout_inputs,
        targets: &holdout_targets,
        initial_state: 2,
    }];
    let mut learner = BooleanOperatorProgram::new(masks, never, always(), [0; 4]).unwrap();
    let report = learner
        .fit_shifts(
            &training,
            &validation,
            u64::MAX,
            OperatorTrainPolicy {
                max_epochs: 3,
                max_candidates: 256,
            },
        )
        .unwrap();
    assert_eq!(report.train_loss_after, 0);
    assert_eq!(report.validation_loss_after, 0);
    assert_eq!(report.shifts, [1, 3, 0, 0]);
    assert_eq!(learner.forward_sparse(&holdout_inputs, 2), holdout_targets);
    println!("bps15_result=PASS");
    println!("bps15_learned_operation_shifts={:?}", report.shifts);
    println!(
        "bps15_candidate_evaluations={}",
        report.evaluated_candidates
    );
    println!("bps15_holdout_hamming={}", report.validation_loss_after);

    // Do not run timing in CI; use the opt-in flag only with physical
    // host isolation and workload registration via Memorithm/RemoteOps.
    if std::env::args().any(|arg| arg == "--measure") {
        use std::hint::black_box;
        use std::time::Instant;
        let dynamic = BooleanOperatorProgram::new(
            [
                UnaryGate {
                    when_false: 0,
                    when_true: u64::MAX,
                },
                UnaryGate {
                    when_false: u64::MAX,
                    when_true: 0,
                },
                never,
                UnaryGate {
                    when_false: 0,
                    when_true: u64::MAX,
                },
            ],
            never,
            always(),
            [1, 3, 5, 7],
        )
        .unwrap();
        let inputs: Vec<u64> = (0..8192usize)
            .map(|i| if i % 13 == 0 { u64::MAX } else { 0 })
            .collect();
        let activity = dynamic.logical_activity(&inputs);
        for sparse in [false, true] {
            let start = Instant::now();
            for _ in 0..64 {
                let output = if sparse {
                    dynamic.forward_sparse(black_box(&inputs), black_box(1))
                } else {
                    dynamic.forward_dense(black_box(&inputs), black_box(1))
                };
                black_box(output);
            }
            let nanos = start.elapsed().as_nanos();
            println!(
                "bps15_mode={} total_ns={} samples={} skipped_rotations={:?}",
                if sparse { "sparse" } else { "dense" },
                nanos,
                64 * inputs.len(),
                activity.skipped_rotations
            );
        }
    }
}
