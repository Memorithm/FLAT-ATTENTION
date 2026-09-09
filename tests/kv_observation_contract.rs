//! Integration coverage for the exported paged-KV observation contract.
//!
//! This test intentionally imports only the public `flat_attention::paged_kv`
//! surface. It therefore models an external consumer such as KVLab rather than
//! reaching into FLAT's source tree.

use flat_attention::paged_kv::{
    KvResidencyEvent, KvResidencyEventKind, PagedKvConfig, PagedKvObservation, PagedKvTable,
    PAGED_KV_OBSERVATION_SCHEMA_VERSION,
};

#[test]
fn public_observation_surface_preserves_geometry_topology_and_events() {
    let config = PagedKvConfig {
        page_size: 4,
        physical_pages: 3,
    };
    let mut table = PagedKvTable::new(config).unwrap();
    table.append(6).unwrap();

    let observation = PagedKvObservation::capture(&table).unwrap();
    assert_eq!(
        observation.schema_version(),
        PAGED_KV_OBSERVATION_SCHEMA_VERSION
    );
    assert_eq!(observation.config(), config);
    assert_eq!(observation.telemetry().live_tokens, 6);
    assert_eq!(observation.pages().len(), 2);
    assert_eq!(observation.pages()[0].logical_page(), 0);
    assert_eq!(observation.pages()[0].physical_page(), 0);
    assert_eq!(observation.pages()[0].logical_token_start(), 0);
    assert_eq!(observation.pages()[0].logical_token_end(), 4);
    assert_eq!(observation.pages()[1].logical_page(), 1);
    assert_eq!(observation.pages()[1].physical_page(), 1);
    assert_eq!(observation.pages()[1].logical_token_start(), 4);
    assert_eq!(observation.pages()[1].logical_token_end(), 6);

    let event = KvResidencyEvent::new(
        11,
        observation.pages()[1].physical_page(),
        observation.pages()[1].generation(),
        KvResidencyEventKind::Demote,
    );
    assert_eq!(event.sequence(), 11);
    assert_eq!(event.physical_page(), 1);
    assert_eq!(event.generation(), observation.telemetry().generation);
    assert_eq!(event.kind(), KvResidencyEventKind::Demote);
}

#[test]
fn public_topology_equality_does_not_claim_kv_content_identity() {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 4,
        physical_pages: 3,
    })
    .unwrap();
    table.append(6).unwrap();
    let before = PagedKvObservation::capture(&table).unwrap();

    // An external cache implementation may overwrite the truncated suffix with
    // different K/V bytes while the FLAT page-table topology returns to exactly
    // the same metadata state. Content/branch lineage must therefore be carried
    // separately by consumers that need it.
    table.truncate(4).unwrap();
    table.append(2).unwrap();
    let after = PagedKvObservation::capture(&table).unwrap();

    assert_eq!(before, after);
}
