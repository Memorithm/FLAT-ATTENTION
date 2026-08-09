use flat_attention::{
    forward_reference, forward_reference_grouped, AttentionShape, FlatAttentionConfig,
    GroupedAttentionShape,
};

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i as f32 * 0.037 + phase;
            x.sin() * 1.75 + (x * 0.41).cos() * 0.25
        })
        .collect()
}

fn expand_kv_to_query_heads(source: &[f32], shape: GroupedAttentionShape) -> Vec<f32> {
    let head_stride = shape.seq_len * shape.head_dim;
    let group_size = shape.q_heads / shape.kv_heads;
    let mut expanded = vec![0.0f32; shape.q_tensor_len().unwrap()];
    for batch in 0..shape.batch {
        for q_head in 0..shape.q_heads {
            let kv_head = q_head / group_size;
            let source_base = (batch * shape.kv_heads + kv_head) * head_stride;
            let target_base = (batch * shape.q_heads + q_head) * head_stride;
            expanded[target_base..target_base + head_stride]
                .copy_from_slice(&source[source_base..source_base + head_stride]);
        }
    }
    expanded
}

#[test]
fn grouped_oracle_matches_expanded_mha_oracle_bit_exactly() {
    for (q_heads, kv_heads) in [(8, 8), (8, 2), (8, 1)] {
        let shape = GroupedAttentionShape {
            batch: 2,
            q_heads,
            kv_heads,
            seq_len: 7,
            head_dim: 16,
        };
        let q = fixture(shape.q_tensor_len().unwrap(), 0.1);
        let k = fixture(shape.kv_tensor_len().unwrap(), 0.7);
        let v = fixture(shape.kv_tensor_len().unwrap(), 1.3);
        let expanded_k = expand_kv_to_query_heads(&k, shape);
        let expanded_v = expand_kv_to_query_heads(&v, shape);
        let mha_shape = AttentionShape {
            batch: shape.batch,
            heads: shape.q_heads,
            seq_len: shape.seq_len,
            head_dim: shape.head_dim,
        };

        for causal in [false, true] {
            for softmax_scale in [None, Some(0.19)] {
                let config = FlatAttentionConfig {
                    causal,
                    softmax_scale,
                };
                let grouped = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
                let expanded =
                    forward_reference(&q, &expanded_k, &expanded_v, mha_shape, config).unwrap();

                assert_eq!(
                    grouped
                        .output
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expanded
                        .output
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "O mismatch q_heads={q_heads} kv_heads={kv_heads} causal={causal}"
                );
                assert_eq!(
                    grouped
                        .lse
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expanded
                        .lse
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "LSE mismatch q_heads={q_heads} kv_heads={kv_heads} causal={causal}"
                );
            }
        }
    }
}

#[test]
fn mqa_storage_is_not_query_head_expanded() {
    let shape = GroupedAttentionShape {
        batch: 3,
        q_heads: 16,
        kv_heads: 1,
        seq_len: 129,
        head_dim: 64,
    };
    assert_eq!(shape.group_size().unwrap(), 16);
    assert_eq!(
        shape.q_tensor_len().unwrap(),
        16 * shape.kv_tensor_len().unwrap()
    );
}
