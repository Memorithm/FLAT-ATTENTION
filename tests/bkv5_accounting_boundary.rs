#![forbid(unsafe_code)]

use flat_attention::api::boolean_kv::{BooleanKvCache, PackedBooleanSignature};
use flat_attention::api::boolean_kv_paged_selection::{
    build_boolean_indexed_kv_selection, NumericalKvPageGeometry,
};
use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};

fn signature(byte: u8) -> PackedBooleanSignature {
    let bits = (0..8).map(|bit| byte & (1 << bit) != 0).collect::<Vec<_>>();
    PackedBooleanSignature::from_bools(&bits).expect("8-bit test signature")
}

#[test]
fn read_accounting_stays_distinct_from_resident_boolean_metadata() {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 4,
        physical_pages: 4,
    })
    .expect("valid paged KV geometry");
    table.append(10).expect("ten live numerical KV tokens");

    let mut cache = BooleanKvCache::new(8).expect("valid Boolean signature width");
    cache
        .append(signature(0b0000_0000), Some(signature(0b1111_1111)))
        .expect("page 0 Boolean metadata");
    cache
        .append(signature(0b0000_0001), Some(signature(0b1111_1110)))
        .expect("page 1 Boolean metadata");
    cache
        .append(signature(0b0000_0011), Some(signature(0b1111_1100)))
        .expect("page 2 Boolean metadata");

    let resident = cache
        .accounting()
        .expect("representable Boolean accounting");
    assert_eq!(resident.key_physical_bytes, 24);
    assert_eq!(resident.value_physical_bytes, 24);
    assert_eq!(resident.total_physical_bytes, 48);

    // The current CPU search examines every key signature before ranking and
    // applying the result limit. Optional Boolean value signatures are resident
    // metadata, but they are not read by the Hamming key search.
    let selection = build_boolean_indexed_kv_selection(
        &cache,
        &table,
        &signature(0),
        8,
        Some(1),
        NumericalKvPageGeometry {
            kv_heads: 2,
            head_dim: 8,
            scalar_bytes: 2,
        },
    )
    .expect("valid Boolean-indexed numerical KV selection");

    assert_eq!(selection.selected_page_count(), 1);
    assert_eq!(selection.boolean_pages_scanned, 3);
    assert_eq!(selection.boolean_key_bytes_read, 24);
    assert_eq!(selection.full_numerical_kv_bytes, 640);
    assert_eq!(selection.selected_numerical_kv_bytes, 256);
    assert_eq!(selection.avoided_numerical_kv_bytes, 384);
    assert_eq!(
        selection.numerical_bytes_avoided_per_boolean_byte(),
        Some(16.0)
    );

    // These quantities must not silently collapse into one metric: storage
    // residency includes the optional Boolean value signatures, while current
    // routing reads only the key-signature bytes.
    assert_eq!(
        resident.total_physical_bytes,
        2 * selection.boolean_key_bytes_read
    );
}
