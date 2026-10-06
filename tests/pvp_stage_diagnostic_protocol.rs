#[path = "../examples/support/pvp_stage_diagnostic.rs"]
mod protocol;

use flat_attention::pvp_vec4::{pvp_subset_zeta_vec4_host_in_place, FlatPvpVec4LayoutV1};

#[test]
fn historical_transition_keeps_allocation_size_and_preserves_address_zero() {
    assert_eq!(protocol::ROUNDS, 12);
    let mut fingerprints = Vec::new();
    for (k, g) in protocol::GEOMETRIES {
        let layout = FlatPvpVec4LayoutV1::new(k, g).unwrap();
        assert_eq!(layout.storage_u32_words() * 4, 4 * 1024 * 1024);
        let original = protocol::fixture(layout);
        let mut transformed = original.clone();
        pvp_subset_zeta_vec4_host_in_place(&mut transformed).unwrap();
        let row_words = layout.vectors_per_address() * 4;
        assert_eq!(
            &transformed.words()[..row_words],
            &original.words()[..row_words]
        );
        fingerprints.push(original.words()[..4].to_vec());
    }
    assert_eq!(
        fingerprints[0],
        [2037321785, 4133997613, 4088336476, 624396622]
    );
    assert_eq!(
        fingerprints[1],
        [3549435859, 4230607782, 2314666888, 1497423403]
    );
    assert_ne!(fingerprints[0], fingerprints[1]);
}

#[test]
fn mismatch_report_checks_every_word_beyond_bounded_first_indices() {
    let expected = vec![11_u32; 100];
    let mut actual = expected.clone();
    for index in (0..100).step_by(3) {
        actual[index] = 99;
    }
    let report = protocol::differences(&actual, &expected);
    assert_eq!(report.count, 34);
    assert_eq!(report.first.len(), protocol::FIRST_MISMATCH_LIMIT);
    assert_eq!(report.first[0], (0, 99, 11));
    assert_eq!(report.first[7], (21, 99, 11));
    assert_eq!(protocol::differences(&expected, &expected).count, 0);
}
