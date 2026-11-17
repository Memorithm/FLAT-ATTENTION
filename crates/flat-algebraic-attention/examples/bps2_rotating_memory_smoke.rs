//! Deterministic BPS-2 host smoke; not a speed, learning-quality, or energy benchmark.
use flat_algebraic_attention::boolean_rotating_memory::BooleanRotatingProgram;
use flat_algebraic_attention::boolean_scan_lstm::{
    BooleanGateProgram, BooleanTrainingSequence, UnaryGate,
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
    let inputs = [0u64; 257];
    let initial_state = 1u64;
    let teacher = BooleanRotatingProgram::new(gates, 7).expect("valid teacher shift");
    let targets = teacher.forward_serial(&inputs, initial_state);
    let data = [BooleanTrainingSequence {
        inputs: &inputs,
        targets: &targets,
        initial_state,
    }];

    let mut student = BooleanRotatingProgram::new(gates, 0).expect("valid initial shift");
    let report = student
        .fit_recurrent_shift(&data, u64::MAX)
        .expect("valid supervised dataset");
    assert_eq!(report.final_loss, 0, "shift must be learned");
    assert_eq!(report.selected_shift, 7, "teacher shift must be recovered");
    let serial = student.forward_serial(&inputs, initial_state);
    let scanned = student.forward_scan(&inputs, initial_state);
    assert_eq!(serial, scanned, "temporal scan must equal serial oracle");
    assert_eq!(serial, targets, "learned model must reproduce labels");

    let activity = student.logical_activity(&inputs, initial_state, u64::MAX);
    println!("candidate=bps2_recurrent_rotation_v1");
    println!("recurrent_fan_in=1");
    println!("state_lanes=64");
    println!("sequence_length={}", inputs.len());
    println!("initial_hamming_loss={}", report.initial_loss);
    println!("final_hamming_loss={}", report.final_loss);
    println!("learned_rotation={}", report.selected_shift);
    println!("evaluated_rotations={}", report.evaluated_shifts);
    println!("scan_equals_serial=true");
    println!("active_predecessor_bits={}", activity.active_predecessor_bits);
    println!("state_flips={}", activity.state_flips);
    println!("energy_measurement=false");
    println!("performance_claim=none");
}
