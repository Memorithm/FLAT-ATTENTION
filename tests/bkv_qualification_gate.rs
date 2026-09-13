#[path = "../src/research_bkv_qualification.rs"]
mod research_bkv_qualification;

use research_bkv_qualification::{
    BikvAccountingInput, BikvLatencyInput, BikvPromotionDecision, BikvQualificationRecord,
    BIKV_QUALIFICATION_SCHEMA_VERSION,
};

#[test]
fn qualification_schema_and_fallback_contract_are_stable() {
    assert_eq!(BIKV_QUALIFICATION_SCHEMA_VERSION, 1);

    let record = BikvQualificationRecord::new(
        BikvAccountingInput {
            live_tokens: 256,
            selected_live_tokens: 128,
            mapped_pages: 4,
            selected_pages: 2,
            page_size: 64,
            kv_heads: 4,
            head_dim: 64,
            scalar_bytes: 4,
            boolean_index_bytes_read: 128,
        },
        BikvLatencyInput {
            signature_generation_ns: 100,
            boolean_search_ns: 200,
            synchronization_ns: 50,
            selected_attention_ns: 800,
            dense_attention_ns: 1_000,
        },
    )
    .unwrap();

    assert_eq!(record.total_bikv_latency_ns(), 1_150);
    assert_eq!(record.accounting().live_tokens, 256);
    assert_eq!(record.latency().dense_attention_ns, 1_000);
    assert_eq!(
        record.promotion_decision(true, true),
        BikvPromotionDecision::FallbackNoLatencyWin
    );
}
