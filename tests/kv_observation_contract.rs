//! Compile and regression-test the metadata-only paged-KV observation prototype.
//!
//! The prototype is intentionally not part of the stable reusable API yet. This
//! integration target keeps it executable and CI-qualified while its ownership
//! and lineage contracts are stabilized for later promotion.

pub use flat_attention::paged_kv;

#[path = "../src/kv_observation.rs"]
mod kv_observation;

#[test]
fn prototype_is_versioned_without_claiming_content_identity() {
    let mut table = paged_kv::PagedKvTable::new(paged_kv::PagedKvConfig {
        page_size: 4,
        physical_pages: 2,
    })
    .unwrap();
    table.append(5).unwrap();

    let observation = kv_observation::PagedKvObservation::capture(&table).unwrap();
    assert_eq!(observation.schema_version(), 1);
    assert_eq!(observation.pages().len(), 2);
}
