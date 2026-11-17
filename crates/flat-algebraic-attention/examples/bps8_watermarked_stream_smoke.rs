//! BPS-8: bounded incremental event-window smoke, exact sampled oracle.
//! This verifies event-stream semantics only, not physical speed/energy.
use flat_algebraic_attention::boolean_edge_memory::BooleanEdgeProgram;
use flat_algebraic_attention::boolean_indexed_fronts::{
    certify_indexed_fronts_against_samples, sampled_indexed_fronts,
};
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;
use flat_algebraic_attention::boolean_watermarked_stream::WatermarkedEdgeRuntime;

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
    .expect("bounded recurrent shift");
    let mut sampled_words = vec![0u64; 4_096];
    for tick in (0..sampled_words.len()).step_by(64) {
        sampled_words[tick] = 1u64 << (tick / 64);
    }
    let events = sampled_indexed_fronts(&sampled_words, 0);
    certify_indexed_fronts_against_samples(&events, &sampled_words, 0)
        .expect("complete independent sampled front oracle");

    let queries: Vec<u64> = (1..=4_096).step_by(31).collect();
    let mut runtime = WatermarkedEdgeRuntime::new(program, 0, 1);
    let mut consumed_fronts = 0;
    let mut consumed_queries = 0;
    let mut outputs = Vec::new();

    for watermark in (256..=4_096).step_by(256) {
        let end_events = events.partition_point(|event| event.tick <= watermark);
        let end_queries = queries.partition_point(|&tick| tick <= watermark);
        let window = runtime
            .advance_window(
                watermark,
                &events[consumed_fronts..end_events],
                &queries[consumed_queries..end_queries],
            )
            .expect("strict non-overlapping event window");
        outputs.extend(window.outputs);
        consumed_fronts = end_events;
        consumed_queries = end_queries;
    }
    assert_eq!(runtime.total_fronts(), events.len());
    assert_eq!(runtime.total_queries(), queries.len());
    assert_eq!(runtime.total_windows(), 16);

    let dense = program.forward_dense(&sampled_words, 0, 1);
    for (&tick, &output) in queries.iter().zip(&outputs) {
        assert_eq!(output, dense[tick as usize - 1]);
    }
    let indexed = program
        .forward_indexed_queries(&events, &queries, 4_096, 1)
        .expect("indexed one-shot oracle");
    assert_eq!(outputs, indexed.outputs);
    assert_eq!(runtime.state(), indexed.final_state);

    println!(
        "BPS8_EVIDENCE sampled_source_steps={} eventful_steps={} watermark_windows={} sparse_queries={} samples_inspected_by_stream_runtime=0 exact_dense_and_indexed_oracles=true physical_energy_measured=false",
        sampled_words.len(),
        events.len(),
        runtime.total_windows(),
        queries.len(),
    );
    println!(
        "BPS8_CAUTION=Input front generation and independent certification inspected the sampled source separately; no end-to-end physical gain is claimed"
    );
}
