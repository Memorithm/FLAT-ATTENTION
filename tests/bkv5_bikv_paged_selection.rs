#![forbid(unsafe_code)]

use flat_attention::api::boolean_kv::{BooleanKvCache, PackedBooleanSignature};
use flat_attention::api::boolean_kv_paged_selection::{
    build_boolean_indexed_kv_selection, NumericalKvPageGeometry,
};
use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};

fn signature(byte: u8) -> PackedBooleanSignature {
    let bits = (0..8).map(|bit| byte & (1 << bit) != 0).collect::<Vec<_>>();
    PackedBooleanSignature::from_bools(&bits).unwrap()
}

#[test]
fn bikv_selection_keeps_numerical_payload_authoritative() {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 4,
        physical_pages: 4,
    })
    .unwrap();
    table.append(10).unwrap();

    let mut cache = BooleanKvCache::new(8).unwrap();
    cache.append(signature(0b0000_0011), None).unwrap();
    cache.append(signature(0), None).unwrap();
    cache.append(signature(0b0000_0001), None).unwrap();

    let plan = build_boolean_indexed_kv_selection(
        &cache,
        &table,
        &signature(0),
        2,
        Some(2),
        NumericalKvPageGeometry {
            kv_heads: 2,
            head_dim: 8,
            scalar_bytes: 2,
        },
    )
    .unwrap();

    assert_eq!(plan.selected_page_ids(), vec![1, 2]);
    assert_eq!(plan.full_numerical_kv_bytes, 640);
    assert_eq!(plan.selected_numerical_kv_bytes, 384);
    assert_eq!(plan.avoided_numerical_kv_bytes, 256);
    assert_eq!(plan.boolean_key_bytes_read, 24);
    assert_eq!(plan.candidate_density(), 2.0 / 3.0);
    assert_eq!(
        plan.numerical_bytes_avoided_per_boolean_byte(),
        Some(256.0 / 24.0)
    );
}
