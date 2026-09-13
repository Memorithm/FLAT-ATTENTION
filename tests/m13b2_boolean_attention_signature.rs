#[path = "../src/boolean_attention_signature.rs"]
mod boolean_attention_signature;

use boolean_attention_signature::{
    BOOLEAN_ATTENTION_SIGNATURE_SCHEMA_VERSION, BooleanAttentionSignature,
    BooleanAttentionSignatureError, HammingAdmissionRule,
};

#[test]
fn xor_popcount_and_xnor_count_are_exact() {
    assert_eq!(BOOLEAN_ATTENTION_SIGNATURE_SCHEMA_VERSION, 1);
    let query = BooleanAttentionSignature::new(8, vec![0b1010_1100]).unwrap();
    let key = BooleanAttentionSignature::new(8, vec![0b1001_1110]).unwrap();

    assert_eq!(query.bits(), 8);
    assert_eq!(query.words(), &[0b1010_1100]);
    assert_eq!(query.hamming_distance(&key).unwrap(), 3);
    assert_eq!(query.xnor_match_count(&key).unwrap(), 5);
}

#[test]
fn hamming_rule_is_explicit_and_deterministic() {
    let query = BooleanAttentionSignature::new(8, vec![0b1010_1100]).unwrap();
    let near = BooleanAttentionSignature::new(8, vec![0b1010_1110]).unwrap();
    let far = BooleanAttentionSignature::new(8, vec![0b0101_0011]).unwrap();
    let rule = HammingAdmissionRule::new(1, 8).unwrap();

    assert_eq!(rule.bits(), 8);
    assert!(rule.admits(&query, &near).unwrap());
    assert!(!rule.admits(&query, &far).unwrap());
}

#[test]
fn hamming_rule_rejects_signatures_from_another_declared_width() {
    let rule = HammingAdmissionRule::new(1, 8).unwrap();
    let query = BooleanAttentionSignature::new(4, vec![0b1010]).unwrap();
    let key = BooleanAttentionSignature::new(4, vec![0b1011]).unwrap();

    assert_eq!(
        rule.admits(&query, &key),
        Err(BooleanAttentionSignatureError::RuleWidthMismatch {
            rule_bits: 8,
            signature_bits: 4,
        })
    );
}

#[test]
fn non_zero_tail_bits_fail_closed() {
    assert_eq!(
        BooleanAttentionSignature::new(65, vec![0, 2]),
        Err(BooleanAttentionSignatureError::NonZeroTailBits)
    );
}

#[test]
fn width_mismatch_and_invalid_threshold_fail_closed() {
    let query = BooleanAttentionSignature::new(8, vec![0]).unwrap();
    let key = BooleanAttentionSignature::new(7, vec![0]).unwrap();

    assert_eq!(
        query.hamming_distance(&key),
        Err(BooleanAttentionSignatureError::WidthMismatch {
            query_bits: 8,
            key_bits: 7,
        })
    );
    assert_eq!(
        HammingAdmissionRule::new(9, 8),
        Err(BooleanAttentionSignatureError::ThresholdOutOfRange {
            threshold: 9,
            bits: 8,
        })
    );
}
