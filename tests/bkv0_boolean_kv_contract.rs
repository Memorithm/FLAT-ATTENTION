#[path = "../src/boolean_kv.rs"]
mod boolean_kv;

use boolean_kv::{
    BooleanKvCache, BooleanKvError, PackedBooleanSignature, BOOLEAN_KV_SCHEMA_VERSION,
};

fn bits(values: &[bool]) -> PackedBooleanSignature {
    PackedBooleanSignature::from_bools(values).unwrap()
}

#[test]
fn packing_is_canonical_across_word_boundary() {
    assert_eq!(BOOLEAN_KV_SCHEMA_VERSION, 1);
    let mut values = vec![false; 65];
    values[0] = true;
    values[63] = true;
    values[64] = true;
    let packed = bits(&values);
    assert_eq!(packed.words(), &[(1u64 << 63) | 1, 1]);
    assert_eq!(packed.logical_bits(), 65);
    assert_eq!(packed.physical_bits().unwrap(), 128);
    assert_eq!(packed.physical_bytes().unwrap(), 16);
}

#[test]
fn non_zero_tail_bits_fail_closed() {
    assert_eq!(
        PackedBooleanSignature::new(65, vec![0, 2]),
        Err(BooleanKvError::NonZeroTailBits)
    );
}

#[test]
fn hamming_and_xnor_are_exact() {
    let left = bits(&[false, true, false, true, true]);
    let right = bits(&[false, false, false, true, false]);
    assert_eq!(left.hamming_distance(&right).unwrap(), 2);
    assert_eq!(left.xnor_matches(&right).unwrap(), 3);
}

#[test]
fn cache_accounting_separates_logical_and_physical_storage() {
    let mut cache = BooleanKvCache::new(65).unwrap();
    assert_eq!(cache.signature_bits(), 65);
    assert_eq!(cache.len(), 0);
    let key = bits(&vec![false; 65]);
    let mut value_bits = vec![false; 65];
    value_bits[0] = true;
    let value = bits(&value_bits);
    cache.append(key.clone(), Some(value.clone())).unwrap();
    cache.append(value, None).unwrap();

    let accounting = cache.accounting().unwrap();
    assert_eq!(accounting.pages, 2);
    assert_eq!(accounting.signature_bits, 65);
    assert_eq!(accounting.key_logical_bits, 130);
    assert_eq!(accounting.value_logical_bits, 65);
    assert_eq!(accounting.key_physical_bytes, 32);
    assert_eq!(accounting.value_physical_bytes, 16);
    assert_eq!(accounting.total_physical_bytes, 48);
}

#[test]
fn reset_invalidates_generation_and_reuses_page_zero() {
    let mut cache = BooleanKvCache::new(4).unwrap();
    let old = cache
        .append(bits(&[true, false, false, false]), None)
        .unwrap()
        .clone();
    assert_eq!(old.logical_page, 0);
    assert_eq!(old.generation, 0);

    cache.reset().unwrap();
    assert_eq!(cache.generation(), 1);
    assert!(cache.is_empty());
    assert_eq!(cache.page(0), None);

    let new = cache
        .append(bits(&[false, true, false, false]), None)
        .unwrap();
    assert_eq!(new.logical_page, 0);
    assert_eq!(new.generation, 1);
}

#[test]
fn search_is_distance_then_page_deterministic() {
    let mut cache = BooleanKvCache::new(4).unwrap();
    cache
        .append(bits(&[false, false, false, false]), None)
        .unwrap();
    cache
        .append(bits(&[true, false, false, false]), None)
        .unwrap();
    cache
        .append(bits(&[false, true, false, false]), None)
        .unwrap();
    cache.append(bits(&[true, true, true, true]), None).unwrap();

    let matches = cache
        .search_hamming(&bits(&[false, false, false, false]), 1, None)
        .unwrap();
    let observed: Vec<_> = matches
        .iter()
        .map(|item| (item.logical_page, item.hamming_distance, item.xnor_matches))
        .collect();
    assert_eq!(observed, vec![(0, 0, 4), (1, 1, 3), (2, 1, 3)]);

    let limited = cache
        .search_hamming(&bits(&[false, false, false, false]), 4, Some(2))
        .unwrap();
    assert_eq!(
        limited
            .iter()
            .map(|item| item.logical_page)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
}

#[test]
fn incompatible_widths_and_invalid_thresholds_are_rejected() {
    let mut cache = BooleanKvCache::new(8).unwrap();
    assert_eq!(
        cache.append(bits(&[false; 7]), None),
        Err(BooleanKvError::SignatureWidthMismatch {
            expected: 8,
            actual: 7
        })
    );

    cache.append(bits(&[false; 8]), None).unwrap();
    assert_eq!(
        cache.search_hamming(&bits(&[false; 8]), 9, None),
        Err(BooleanKvError::InvalidMaxDistance {
            max_distance: 9,
            signature_bits: 8
        })
    );
    assert_eq!(
        cache.search_hamming(&bits(&[false; 8]), 1, Some(0)),
        Err(BooleanKvError::ZeroLimit)
    );
}
