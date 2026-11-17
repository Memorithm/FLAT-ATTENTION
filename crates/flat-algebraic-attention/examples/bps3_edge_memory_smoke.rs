//! Deterministic host-only BPS-3 smoke. No timing, energy or speedup claims.
use flat_algebraic_attention::boolean_edge_memory::BooleanEdgeProgram;
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn main() {
    let program = BooleanEdgeProgram::new(
        UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        },
        UnaryGate::default(),
        u64::MAX,
        1,
    )
    .expect("valid recurrent rotation");

    let mut samples = vec![0u64; 1_024];
    samples[3] = 1;
    samples[4] = 1;
    samples[800] = 1;

    let dense = program.forward_dense(&samples, 0, 1);
    let scan = program.forward_scan(&samples, 0, 1);
    let event_scan = program.forward_event_scan(&samples, 0, 1);
    let sparse = program
        .forward_sparse(&samples, 0, 1)
        .expect("monotonic sampled sequence");

    assert_eq!(
        sparse.outputs, dense,
        "sparse must match dense semantic oracle"
    );
    assert_eq!(
        scan, dense,
        "temporal scan must match dense semantic oracle"
    );
    assert_eq!(event_scan, dense, "event-only scan must match dense oracle");
    assert_eq!(sparse.activity.sampled_steps, 1_024);
    assert_eq!(sparse.activity.eventful_steps, 2);
    assert_eq!(sparse.activity.skipped_identity_steps, 1_022);
    assert_eq!(sparse.activity.evaluated_recurrent_steps, 2);

    println!("candidate=bps3_event_gated_recurrence_v1");
    println!("samples={}", sparse.activity.sampled_steps);
    println!("eventful_steps={}", sparse.activity.eventful_steps);
    println!(
        "skipped_identity_steps={}",
        sparse.activity.skipped_identity_steps
    );
    println!(
        "recurrent_transitions_evaluated={}",
        sparse.activity.evaluated_recurrent_steps
    );
    println!("rising_bits={}", sparse.activity.rising_bits);
    println!("state_flips={}", sparse.activity.state_flips);
    println!("full_scan_equals_event_scan_equals_sparse_equals_dense=true");
    println!("physical_energy_measurement=false");
    println!("performance_claim=none");
}
