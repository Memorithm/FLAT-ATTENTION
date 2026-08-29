use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, FlatAttentionConfig,
};
use flat_semantic::v1::NonlocalHistorySoftmaxSemantic;

fn deterministic_values(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| ((index as f32 * 0.137) + phase).cos() * 0.6)
        .collect()
}

#[test]
fn admitting_nonlocal_semantic_does_not_change_asymmetric_rope_reference_behavior() {
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        query_len: 2,
        kv_len: 7,
        head_dim: 8,
        query_position_offset: 5,
    };
    let q = deterministic_values(shape.q_tensor_len().unwrap(), 0.1);
    let k = deterministic_values(shape.kv_tensor_len().unwrap(), 0.7);
    let v = deterministic_values(shape.kv_tensor_len().unwrap(), 1.3);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: Some(0.5),
    };
    let rotary = AsymmetricRotaryEmbeddingConfig {
        theta: 10_000.0,
        query_position_offset: 105,
        kv_position_offset: 100,
    };

    let before = forward_reference_projection_grouped_rope_asymmetric(
        &q, &k, &v, shape, config, rotary,
    )
    .unwrap();

    let semantic = NonlocalHistorySoftmaxSemantic::new(
        config,
        flat_attention::api::research_nonlocal::NonlocalAttentionConfig::default(),
    )
    .unwrap();
    let descriptor = semantic.descriptor();
    assert_eq!(descriptor.id().name(), "nonlocal-history-softmax");

    let after = forward_reference_projection_grouped_rope_asymmetric(
        &q, &k, &v, shape, config, rotary,
    )
    .unwrap();
    assert_eq!(after, before);
}

#[test]
fn admitting_nonlocal_semantic_does_not_change_paged_kv_logical_order_or_generation() {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 3,
        physical_pages: 4,
    })
    .unwrap();
    table.append(8).unwrap();

    let before = (0..table.len())
        .map(|logical| table.address(logical).unwrap())
        .collect::<Vec<_>>();
    let generation_before = table.generation();

    let semantic = NonlocalHistorySoftmaxSemantic::new(
        FlatAttentionConfig {
            causal: true,
            softmax_scale: Some(0.5),
        },
        flat_attention::api::research_nonlocal::NonlocalAttentionConfig::default(),
    )
    .unwrap();
    let _ = semantic.descriptor();

    let after = (0..table.len())
        .map(|logical| table.address(logical).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(after, before);
    assert_eq!(table.generation(), generation_before);

    table.reset().unwrap();
    assert_eq!(table.generation(), generation_before + 1);
    assert!(table.is_empty());
    table.append(2).unwrap();
    assert_eq!(table.address(0).unwrap().physical_page, 0);
    assert_ne!(table.address(0).unwrap().generation, generation_before);
}
