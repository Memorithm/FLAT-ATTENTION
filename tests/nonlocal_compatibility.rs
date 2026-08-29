use flat_attention::{
    api::{
        research_nonlocal::{
            forward_reference_nonlocal_history, HistoryClassification, NonlocalAttentionConfig,
        },
        v1::AttentionConfig as ApiAttentionConfig,
    },
    forward_reference_grouped_asymmetric, forward_reference_grouped_rope,
    paged_kv::{PagedKvConfig, PagedKvTable},
    AsymmetricGroupedAttentionShape, FlatAttentionConfig, GroupedAttentionShape,
    RotaryEmbeddingConfig,
};

fn deterministic_values(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| ((index as f32 * 0.173) + phase).sin() * 0.7)
        .collect()
}

fn rotate_rows(
    input: &[f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Vec<f32> {
    let mut rotated = input.to_vec();
    let head_stride = seq_len * head_dim;

    for batch_index in 0..batch {
        for head in 0..heads {
            let head_base = (batch_index * heads + head) * head_stride;
            for position in 0..seq_len {
                let absolute_position = position_offset + position;
                let row_base = head_base + position * head_dim;
                for pair in 0..head_dim / 2 {
                    let dim = 2 * pair;
                    let exponent = -2.0 * pair as f32 / head_dim as f32;
                    let frequency = theta.powf(exponent);
                    let angle = absolute_position as f32 * frequency;
                    let (sin, cos) = angle.sin_cos();
                    let even = input[row_base + dim];
                    let odd = input[row_base + dim + 1];
                    rotated[row_base + dim] = even * cos - odd * sin;
                    rotated[row_base + dim + 1] = even * sin + odd * cos;
                }
            }
        }
    }

    rotated
}

fn scatter_to_paged_storage(logical: &[f32], table: &PagedKvTable, head_dim: usize) -> Vec<f32> {
    let config = table.config();
    let mut physical = vec![0.0_f32; config.capacity_tokens().unwrap() * head_dim];
    for logical_token in 0..table.len() {
        let address = table.address(logical_token).unwrap();
        let physical_token = address.physical_page * config.page_size + address.offset_in_page;
        let logical_base = logical_token * head_dim;
        let physical_base = physical_token * head_dim;
        physical[physical_base..physical_base + head_dim]
            .copy_from_slice(&logical[logical_base..logical_base + head_dim]);
    }
    physical
}

fn gather_from_paged_storage(physical: &[f32], table: &PagedKvTable, head_dim: usize) -> Vec<f32> {
    let config = table.config();
    let mut logical = vec![0.0_f32; table.len() * head_dim];
    for logical_token in 0..table.len() {
        let address = table.address(logical_token).unwrap();
        let physical_token = address.physical_page * config.page_size + address.offset_in_page;
        let logical_base = logical_token * head_dim;
        let physical_base = physical_token * head_dim;
        logical[logical_base..logical_base + head_dim]
            .copy_from_slice(&physical[physical_base..physical_base + head_dim]);
    }
    logical
}

#[test]
fn production_v1_default_remains_the_standard_attention_default() {
    let api_default = ApiAttentionConfig::default();
    assert_eq!(api_default.to_core_config(), FlatAttentionConfig::default());
    assert!(!api_default.causal);
    assert_eq!(api_default.softmax_scale, None);
}

#[test]
fn complete_history_preserves_native_gqa_mapping() {
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        query_len: 2,
        kv_len: 4,
        head_dim: 8,
        query_position_offset: 2,
    };
    let q = deterministic_values(shape.q_tensor_len().unwrap(), 0.15);
    let k = deterministic_values(shape.kv_tensor_len().unwrap(), 0.75);
    let v = deterministic_values(shape.kv_tensor_len().unwrap(), 1.35);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: Some(0.5),
    };

    let standard = forward_reference_grouped_asymmetric(&q, &k, &v, shape, config).unwrap();
    let research = forward_reference_nonlocal_history(
        &q,
        &k,
        &v,
        shape,
        config,
        NonlocalAttentionConfig::default(),
    )
    .unwrap();

    assert_eq!(research.attention, standard);
    assert_eq!(research.classification, HistoryClassification::Reference);
}

#[test]
fn future_kv_rows_are_mathematically_inert_under_causal_history() {
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 2,
        kv_heads: 1,
        query_len: 1,
        kv_len: 4,
        head_dim: 2,
        query_position_offset: 1,
    };
    let q = vec![0.2, -0.4, 0.5, 0.3];
    let k = vec![0.6, -0.2, -0.8, 0.3, 0.4, 0.9, -0.5, 0.7];
    let v = vec![1.0, -2.0, 0.5, 0.25, -0.75, 1.5, 2.0, -1.0];
    let mut poisoned_k = k.clone();
    let mut poisoned_v = v.clone();
    poisoned_k[4..].fill(1.0e20);
    poisoned_v[4..].fill(-1.0e20);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: Some(0.625),
    };

    let clean = forward_reference_nonlocal_history(
        &q,
        &k,
        &v,
        shape,
        config,
        NonlocalAttentionConfig::default(),
    )
    .unwrap();
    let poisoned = forward_reference_nonlocal_history(
        &q,
        &poisoned_k,
        &poisoned_v,
        shape,
        config,
        NonlocalAttentionConfig::default(),
    )
    .unwrap();

    assert_eq!(poisoned, clean);
}

#[test]
fn complete_history_composes_with_rope_without_rotating_v() {
    let grouped = GroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        seq_len: 5,
        head_dim: 8,
    };
    let q = deterministic_values(grouped.q_tensor_len().unwrap(), 0.2);
    let k = deterministic_values(grouped.kv_tensor_len().unwrap(), 0.8);
    let v = deterministic_values(grouped.kv_tensor_len().unwrap(), 1.4);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: Some(0.625),
    };
    let rotary = RotaryEmbeddingConfig {
        theta: 10_000.0,
        position_offset: 17,
    };

    let fused = forward_reference_grouped_rope(&q, &k, &v, grouped, config, rotary).unwrap();
    let rotated_q = rotate_rows(
        &q,
        grouped.batch,
        grouped.q_heads,
        grouped.seq_len,
        grouped.head_dim,
        rotary.theta,
        rotary.position_offset,
    );
    let rotated_k = rotate_rows(
        &k,
        grouped.batch,
        grouped.kv_heads,
        grouped.seq_len,
        grouped.head_dim,
        rotary.theta,
        rotary.position_offset,
    );
    let research = forward_reference_nonlocal_history(
        &rotated_q,
        &rotated_k,
        &v,
        AsymmetricGroupedAttentionShape::from(grouped),
        config,
        NonlocalAttentionConfig::default(),
    )
    .unwrap();

    assert_eq!(research.attention, fused);
}

#[test]
fn paged_kv_logical_order_and_generation_remain_authoritative() {
    let page_config = PagedKvConfig {
        page_size: 2,
        physical_pages: 3,
    };
    let mut table = PagedKvTable::new(page_config).unwrap();
    table.append(5).unwrap();

    let head_dim = 2;
    let logical_k = deterministic_values(table.len() * head_dim, 0.55);
    let logical_v = deterministic_values(table.len() * head_dim, 1.15);
    let physical_k = scatter_to_paged_storage(&logical_k, &table, head_dim);
    let physical_v = scatter_to_paged_storage(&logical_v, &table, head_dim);
    let gathered_k = gather_from_paged_storage(&physical_k, &table, head_dim);
    let gathered_v = gather_from_paged_storage(&physical_v, &table, head_dim);
    assert_eq!(gathered_k, logical_k);
    assert_eq!(gathered_v, logical_v);

    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 2,
        kv_heads: 1,
        query_len: 1,
        kv_len: table.len(),
        head_dim,
        query_position_offset: table.len() - 1,
    };
    let q = deterministic_values(shape.q_tensor_len().unwrap(), 0.05);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let direct = forward_reference_nonlocal_history(
        &q,
        &logical_k,
        &logical_v,
        shape,
        config,
        NonlocalAttentionConfig::default(),
    )
    .unwrap();
    let paged_order = forward_reference_nonlocal_history(
        &q,
        &gathered_k,
        &gathered_v,
        shape,
        config,
        NonlocalAttentionConfig::default(),
    )
    .unwrap();
    assert_eq!(paged_order, direct);

    let old_address = table.address(4).unwrap();
    table.reset().unwrap();
    assert!(table.address(0).is_none());
    table.append(5).unwrap();
    let new_address = table.address(4).unwrap();
    assert_eq!(new_address.physical_page, old_address.physical_page);
    assert_eq!(new_address.offset_in_page, old_address.offset_in_page);
    assert_ne!(new_address.generation, old_address.generation);
}
