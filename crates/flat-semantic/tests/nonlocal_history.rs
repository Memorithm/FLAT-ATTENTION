use flat_attention::api::research_nonlocal::{
    forward_reference_nonlocal_history, HistoryClassification, NonlocalAttentionConfig,
    NONLOCAL_ATTENTION_SEMANTIC_NAME, NONLOCAL_ATTENTION_SEMANTIC_REVISION,
};
use flat_attention::{AsymmetricGroupedAttentionShape, FlatAttentionConfig};
use flat_semantic::v1::{
    MaskSemantics, NonlocalHistorySoftmaxSemantic, SavedStateContract, SemanticFamily,
    StateSemantics, WeightSemantics,
};

fn deterministic_values(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| ((index as f32 * 0.173) + phase).sin() * 0.7)
        .collect()
}

fn shape() -> AsymmetricGroupedAttentionShape {
    AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        query_len: 3,
        kv_len: 6,
        head_dim: 8,
        query_position_offset: 2,
    }
}

#[test]
fn descriptor_matches_qualified_research_identity() {
    let semantic = NonlocalHistorySoftmaxSemantic::new(
        FlatAttentionConfig {
            causal: true,
            softmax_scale: Some(0.625),
        },
        NonlocalAttentionConfig::default(),
    )
    .unwrap();
    let descriptor = semantic.descriptor();

    assert_eq!(descriptor.id().family(), SemanticFamily::Experimental);
    assert_eq!(descriptor.id().name(), NONLOCAL_ATTENTION_SEMANTIC_NAME);
    assert_eq!(
        descriptor.id().revision(),
        NONLOCAL_ATTENTION_SEMANTIC_REVISION
    );
    assert_eq!(descriptor.mask(), MaskSemantics::Causal);
    assert_eq!(descriptor.state(), StateSemantics::Stateless);
    assert_eq!(descriptor.weights(), WeightSemantics::ProbabilitySimplex);
    assert_eq!(descriptor.saved_state(), SavedStateContract::LogSumExp);
}

#[test]
fn wrapper_is_bitwise_identical_to_direct_scalar_oracle() {
    let shape = shape();
    let q = deterministic_values(shape.q_tensor_len().unwrap(), 0.1);
    let k = deterministic_values(shape.kv_tensor_len().unwrap(), 0.7);
    let v = deterministic_values(shape.kv_tensor_len().unwrap(), 1.3);
    let attention = FlatAttentionConfig {
        causal: true,
        softmax_scale: Some(0.625),
    };
    let history = NonlocalAttentionConfig::default();

    let direct = forward_reference_nonlocal_history(&q, &k, &v, shape, attention, history).unwrap();
    let wrapped = NonlocalHistorySoftmaxSemantic::new(attention, history)
        .unwrap()
        .execute(&q, &k, &v, shape)
        .unwrap();

    assert_eq!(wrapped, direct);
    assert_eq!(wrapped.classification, HistoryClassification::Reference);
}

#[test]
fn noncausal_construction_fails_closed() {
    assert!(NonlocalHistorySoftmaxSemantic::new(
        FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(0.625),
        },
        NonlocalAttentionConfig::default(),
    )
    .is_err());
}
