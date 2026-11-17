//! BPS-9 reproducible stream framing and downstream recurrence smoke.
//! This does NOT authenticate a remote publisher or measure physical energy.

use flat_algebraic_attention::boolean_edge_memory::BooleanEdgeProgram;
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;
use flat_algebraic_attention::boolean_sequenced_publisher::{
    FrameLimits, SampledFrontPublisher, SequencedFrontRuntime, SourceEpoch, StreamAnchor,
};

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
    .expect("valid rotation");
    let anchor = StreamAnchor {
        source: SourceEpoch {
            source_id: 111,
            epoch_id: 2026,
        },
        next_sequence: 1,
        prior_commitment: 0x4e55_52bb_9b34_22f1,
        closed_through: 0,
    };
    let limits = FrameLimits {
        max_fronts: 128,
        max_queries: 128,
        max_advance_ticks: 128,
    };
    let mut publisher = SampledFrontPublisher::new(anchor, 0);
    let mut consumer = SequencedFrontRuntime::new(program, 1, anchor, limits);
    let mut samples = vec![0u64; 512];
    for i in (0..samples.len()).step_by(19) {
        samples[i] = 1u64 << (i % 64);
    }
    let mut output = Vec::new();
    let mut fronts = 0usize;
    for (window_index, chunk) in samples.chunks(64).enumerate() {
        let frame = publisher.publish_contiguous_samples(chunk, 64).unwrap();
        fronts += frame.fronts.len();
        let start = window_index as u64 * 64 + 1;
        let queries: Vec<u64> = (start..start + 64).filter(|t| t % 3 == 0).collect();
        let receipt = consumer.ingest(&frame, &queries).unwrap();
        assert_eq!(receipt.sequence, (window_index + 1) as u64);
        assert_eq!(receipt.commitment, frame.commitment);
        output.extend(receipt.data.outputs);
    }
    let dense = program.forward_dense(&samples, 0, 1);
    let query_ticks: Vec<u64> = (1..=512).filter(|t| t % 3 == 0).collect();
    let expected: Vec<_> = query_ticks.iter().map(|&t| dense[t as usize - 1]).collect();
    assert_eq!(output, expected);
    assert_eq!(consumer.next_sequence(), 9);
    assert_eq!(consumer.closed_through(), 512);
    assert!(!consumer.quarantined());
    println!(
        "BPS9_SMOKE verified=true windows=8 samples={} fronts={} sparse_queries={} next_seq={} input_source_complete_relative_to_supplied_samples=true remote_source_authenticated=false physical_energy_measured=false",
        samples.len(),
        fronts,
        output.len(),
        consumer.next_sequence()
    );
}
