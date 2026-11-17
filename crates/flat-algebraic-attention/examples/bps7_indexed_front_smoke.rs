//! BPS-7 deterministic direct event-indexed evaluation smoke.
//! This tests exact outputs; it is not a wall-time or energy benchmark.
use flat_algebraic_attention::boolean_edge_memory::BooleanEdgeProgram;
use flat_algebraic_attention::boolean_indexed_fronts::{
    certify_indexed_fronts_against_samples, sampled_indexed_fronts,
};
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;

fn main() {
    let program = BooleanEdgeProgram::new(
        UnaryGate {
            when_false: u64::MAX,
            when_true: u64::MAX,
        },
        UnaryGate::default(),
        u64::MAX,
        7,
    )
    .expect("bounded rotation");
    let mut samples = vec![0u64; 4_096];
    for tick in (0..samples.len()).step_by(64) {
        samples[tick] = 1u64 << (tick / 64);
    }
    let events = sampled_indexed_fronts(&samples, 0);
    certify_indexed_fronts_against_samples(&events, &samples, 0)
        .expect("fronts agree with every sampled tick");
    let queries: Vec<u64> = (1..=4_096).step_by(32).collect();
    let result = program
        .forward_indexed_queries(&events, &queries, 4_096, 1)
        .expect("ordered bounded front and query events");
    let dense = program.forward_dense(&samples, 0, 1);
    for (&tick, &output) in queries.iter().zip(&result.outputs) {
        assert_eq!(output, dense[(tick - 1) as usize]);
    }
    let expected_last = events.iter().fold(1, |state, front| {
        program.transition(front.rising).apply(state)
    });
    assert_eq!(result.final_state, expected_last);
    println!(
        "BPS7_EVIDENCE source_sampled_steps={} indexed_events={} requested_queries={} event_recurrences={} dense_words_inspected_during_event_execution={} all_queries_match=true physical_energy_measured=false",
        samples.len(),
        events.len(),
        queries.len(),
        result.executed_eventful_steps,
        result.sampled_input_words_inspected,
    );
    println!(
        "BPS7_NOTE=The source detector and independent certification separately read all original samples; no end-to-end O(E+Q) claim applies to this sampled-source test"
    );
}
