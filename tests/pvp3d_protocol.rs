#[path = "../examples/support/pvp3d_protocol.rs"]
mod protocol;

use protocol::*;

fn generous_limits() -> Limits {
    Limits {
        buffer_bytes: 1 << 30,
        storage_binding_bytes: 1 << 30,
        workgroups: 65_535,
        invocations: 64,
        workgroup_x: 64,
        workgroup_storage_bytes: 128,
    }
}

#[test]
fn thirty_repetitions_balance_each_candidate_position_and_pair_order() {
    let mut positions = [[0; 3]; 3];
    let mut before = [[0; 3]; 3];
    for iteration in 0..30 {
        let order = candidate_order(iteration);
        for (position, candidate) in order.iter().enumerate() {
            positions[*candidate as usize][position] += 1;
            for later in &order[position + 1..] {
                before[*candidate as usize][*later as usize] += 1;
            }
        }
    }
    assert_eq!(positions, [[10; 3]; 3]);
    assert_eq!(before, [[0, 15, 15], [15, 0, 15], [15, 15, 0]]);
}

#[test]
fn tile8_dispatch_limit_rejects_the_whole_comparison_before_allocation() {
    let limits = generous_limits();
    assert_eq!(rejection(65_536, 512, limits), None);
    // vec4/fused2 fit, but tile8 needs 131072 workgroups.
    assert_eq!(
        rejection(65_536, 2048, limits),
        Some("cohort_dispatch_limit")
    );
    assert_eq!(
        rejection(
            8,
            257,
            Limits {
                workgroup_storage_bytes: 127,
                ..limits
            }
        ),
        Some("tile8_workgroup_storage_limit")
    );
    assert_eq!(
        rejection(
            8,
            257,
            Limits {
                storage_binding_bytes: 383,
                ..limits
            }
        ),
        Some("state_buffer_limit")
    );
    assert_eq!(
        rejection(
            8,
            257,
            Limits {
                invocations: 63,
                ..limits
            }
        ),
        Some("workgroup_size_limit")
    );
    assert_eq!(rejection(3, 128, limits), Some("invalid_geometry"));
    assert_eq!(
        rejection(8, usize::MAX, limits),
        Some("arithmetic_overflow")
    );
}

#[test]
fn frozen_grids_fallback_dispatches_and_evidence_helpers_are_explicit() {
    assert_eq!(K_GRID.len() * G_GRID.len(), 21);
    for (k, g) in SMOKE_GRID {
        assert_eq!(rejection(k, g, generous_limits()), None);
        for candidate in Candidate::ALL {
            assert!(!candidate.name().is_empty());
            assert!(candidate.dispatches(k) <= k.trailing_zeros());
        }
    }
    assert_eq!(Candidate::Tile8.dispatches(4), 2);
    assert_eq!(Candidate::Tile8.dispatches(8), 1);
    assert_eq!(Candidate::Fused2.dispatches(4), 1);
    assert_eq!(percentile_ns(&(1..=30).rev().collect::<Vec<_>>(), 50), 15);
    assert_eq!(percentile_ns(&(1..=30).collect::<Vec<_>>(), 95), 29);
    assert!(source_revision_valid(
        "ae9b26d7d28a4e518fbc96856da50f3abeaf92d1"
    ));
    assert!(!source_revision_valid("unknown"));
    assert!(!source_revision_valid(&"x".repeat(40)));
    // FNV-1a empty and four-NUL vectors; u32 0 is four little-endian NUL bytes.
    assert_eq!(checksum(&[]), 0xcbf2_9ce4_8422_2325);
    assert_eq!(checksum(&[0]), 0x4d25_767f_9dce_13f5);
}
