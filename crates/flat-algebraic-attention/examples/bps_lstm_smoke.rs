//! Deterministic host-only functional smoke for BPS-LSTM, not a benchmark.
use flat_algebraic_attention::boolean_scan_lstm::{
    BooleanGateProgram, BooleanTrainingSequence,
};

fn main() {
    let length = 4096usize;
    let mut seed = 0x1357_9bdf_2468_ace1u64;
    let mut inputs = Vec::with_capacity(length);
    for _ in 0..length {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        inputs.push(seed);
    }

    let mut model = BooleanGateProgram::default();
    let single_lane = [0u64, 1, 0, 1, 1, 0];
    let dataset = [BooleanTrainingSequence {
        inputs: &single_lane,
        targets: &single_lane,
        initial_state: 0,
    }];
    let report = model
        .fit_hamming(&dataset, 1, 4)
        .expect("valid Boolean supervised training task");
    assert_eq!(report.final_loss, 0, "toy task must be learned");

    let serial = model.forward_serial(&inputs, 0);
    let scanned = model.forward_scan(&inputs, 0);
    assert_eq!(serial, scanned, "scan must equal sequential reference");

    let activity = model.logical_activity(&inputs, 0, 1);
    println!("candidate=bps_lstm_boolean_scan_v0");
    println!("lanes_per_word=64");
    println!("sequence_length={length}");
    println!("scan_equals_serial=true");
    println!("initial_hamming_loss={}", report.initial_loss);
    println!("final_hamming_loss={}", report.final_loss);
    println!("accepted_boolean_flips={}", report.accepted_flips);
    println!("state_flips={}", activity.state_flips);
    println!("requested_write_bits={}", activity.requested_write_bits);
    println!("energy_measurement=false");
    println!("performance_claim=none");
}
