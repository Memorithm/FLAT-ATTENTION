#[path = "../src/boolean_attention_mask.rs"]
mod boolean_attention_mask;

use boolean_attention_mask::{
    BooleanAttentionMask, BooleanAttentionMaskError, BOOLEAN_ATTENTION_MASK_SCHEMA_VERSION,
};

#[test]
fn admissions_are_bitpacked_canonically() {
    assert_eq!(BOOLEAN_ATTENTION_MASK_SCHEMA_VERSION, 1);
    let mut admitted = vec![false; 65];
    admitted[0] = true;
    admitted[63] = true;
    admitted[64] = true;

    let mask = BooleanAttentionMask::from_admissions(&admitted).unwrap();
    assert_eq!(mask.blocks(), 65);
    assert_eq!(mask.words(), &[(1u64 << 63) | 1, 1]);
    assert_eq!(mask.logical_bits(), 65);
    assert_eq!(mask.physical_bytes().unwrap(), 16);
    assert_eq!(mask.admitted_count(), 3);
    assert_eq!(mask.admitted_blocks(), vec![0, 63, 64]);
}

#[test]
fn admission_oracle_matches_declared_blocks() {
    let mask = BooleanAttentionMask::from_admissions(&[true, false, true, false]).unwrap();
    assert!(mask.is_admitted(0).unwrap());
    assert!(!mask.is_admitted(1).unwrap());
    assert!(mask.is_admitted(2).unwrap());
    assert!(!mask.is_admitted(3).unwrap());
}

#[test]
fn non_zero_tail_bits_fail_closed() {
    assert_eq!(
        BooleanAttentionMask::new(65, vec![0, 2]),
        Err(BooleanAttentionMaskError::NonZeroTailBits)
    );
}

#[test]
fn zero_blocks_and_wrong_storage_fail_closed() {
    assert_eq!(
        BooleanAttentionMask::from_admissions(&[]),
        Err(BooleanAttentionMaskError::ZeroBlocks)
    );
    assert_eq!(
        BooleanAttentionMask::new(65, vec![0]),
        Err(BooleanAttentionMaskError::WordCountMismatch {
            blocks: 65,
            expected_words: 2,
            actual_words: 1,
        })
    );
}

#[test]
fn out_of_range_queries_fail_closed() {
    let mask = BooleanAttentionMask::from_admissions(&[true, false]).unwrap();
    assert_eq!(
        mask.is_admitted(2),
        Err(BooleanAttentionMaskError::BlockOutOfRange {
            block: 2,
            blocks: 2,
        })
    );
}
