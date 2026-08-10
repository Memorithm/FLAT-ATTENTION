use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric_biased, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, AttentionBias, FlatAttentionConfig,
};

#[test]
fn additive_bias_oracle_is_publicly_reachable() {
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 1,
        kv_heads: 1,
        query_len: 1,
        kv_len: 1,
        head_dim: 2,
        query_position_offset: 0,
    };
    let output = forward_reference_projection_grouped_rope_asymmetric_biased(
        &[1.0, 0.0],
        &[1.0, 0.0],
        &[2.0, 3.0],
        shape,
        FlatAttentionConfig::default(),
        AsymmetricRotaryEmbeddingConfig {
            theta: 10_000.0,
            query_position_offset: 0,
            kv_position_offset: 0,
        },
        AttentionBias::None,
    )
    .unwrap();
    assert_eq!(output.output, vec![2.0, 3.0]);
}
