//! BPS-10 synthetic MAC + checkpoint + resumed BPS-8 stream.
//! Test-only deterministic shared key; NEVER deploy this key or treat the
//! derived in-process checkpoint head as a durable rollback authority.

use flat_algebraic_attention::boolean_authenticated_frames::{
    AuthenticatedFrontRuntime, AuthenticatedSampledPublisher, TrustedCheckpointHead,
};
use flat_algebraic_attention::boolean_edge_memory::BooleanEdgeProgram;
use flat_algebraic_attention::boolean_scan_lstm::UnaryGate;
use flat_algebraic_attention::boolean_sequenced_publisher::{
    FrameLimits, SourceEpoch, StreamAnchor,
};

fn main() {
    const DEMO_KEY: [u8; 32] = [0x19; 32]; // Synthetic example only
    let program = BooleanEdgeProgram::new(
        UnaryGate { when_false: u64::MAX, when_true: u64::MAX },
        UnaryGate::default(),
        u64::MAX,
        7,
    )
    .unwrap();
    let anchor = StreamAnchor {
        source: SourceEpoch { source_id: 42, epoch_id: 15 },
        next_sequence: 1,
        prior_commitment: 0xc971_4afe_a2b0_4118,
        closed_through: 0,
    };
    let limits = FrameLimits {
        max_fronts: 16,
        max_queries: 16,
        max_advance_ticks: 16,
    };
    let input = [1u64, 1, 0, 1, 0, 5, 0, 7, 7, 0, 7, 9];
    let mut publisher =
        AuthenticatedSampledPublisher::new(anchor, 0, 8, [0x73; 32]);
    let mut live = AuthenticatedFrontRuntime::new(
        program, 1, anchor, limits, 8, [0x73; 32],
    );
    let mut results = Vec::new();
    for (i, window) in input.chunks(4).enumerate() {
        let authenticated = publisher.publish(window, 16, &DEMO_KEY).unwrap();
        let start = i as u64 * 4 + 1;
        let queried: Vec<_> = (start..=start + 3).collect();
        let result = live.ingest(&authenticated, &queried, &DEMO_KEY).unwrap();
        results.extend(result.data.outputs);
        if i == 0 {
            let saved = live.checkpoint(&DEMO_KEY).unwrap();
            // In real deployment this must be fetched from a separate
            // atomically durable high-water authority. Local demo only.
            let trusted = TrustedCheckpointHead {
                source: saved.source,
                key_id: saved.key_id,
                checkpoint_mac: saved.mac,
            };
            live = AuthenticatedFrontRuntime::restore(
                &saved, &trusted, program, limits, 8, &DEMO_KEY,
            )
            .unwrap();
        }
    }
    assert_eq!(results, program.forward_dense(&input, 0, 1));
    assert_eq!(live.next_sequence(), 4);
    assert_eq!(live.closed_through(), input.len() as u64);
    assert!(!live.quarantined());
    println!(
        "BPS10_SMOKE dense_oracle_equal=true authenticated_frames=3 resumed_checkpoint=true artificial_trusted_head=true cryptographic_key_from_real_keystore=false physical_energy_measured=false"
    );
}
