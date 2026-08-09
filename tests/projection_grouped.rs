use flat_attention::{
    forward_reference_grouped_rope, forward_reference_projection_grouped_rope, FlatAttentionConfig,
    GroupedAttentionShape, RotaryEmbeddingConfig,
};

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i as f32 * 0.023 + phase;
            x.sin() * 1.875 + (x * 0.61).cos() * 0.3125
        })
        .collect()
}

fn projection_to_head_major(
    source: &[f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let width = heads * head_dim;
    let head_stride = seq_len * head_dim;
    let mut out = vec![0.0f32; source.len()];
    for b in 0..batch {
        for position in 0..seq_len {
            for head in 0..heads {
                for dim in 0..head_dim {
                    let src = (b * seq_len + position) * width + head * head_dim + dim;
                    let dst = (b * heads + head) * head_stride + position * head_dim + dim;
                    out[dst] = source[src];
                }
            }
        }
    }
    out
}

fn head_major_to_projection(
    source: &[f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let width = heads * head_dim;
    let head_stride = seq_len * head_dim;
    let mut out = vec![0.0f32; source.len()];
    for b in 0..batch {
        for head in 0..heads {
            for position in 0..seq_len {
                for dim in 0..head_dim {
                    let src = (b * heads + head) * head_stride + position * head_dim + dim;
                    let dst = (b * seq_len + position) * width + head * head_dim + dim;
                    out[dst] = source[src];
                }
            }
        }
    }
    out
}

#[test]
fn projection_oracle_is_bit_exact_with_canonical_r1_after_layout_transform() {
    for shape in [
        GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 4,
            seq_len: 7,
            head_dim: 16,
        },
        GroupedAttentionShape {
            batch: 2,
            q_heads: 8,
            kv_heads: 2,
            seq_len: 9,
            head_dim: 64,
        },
        GroupedAttentionShape {
            batch: 1,
            q_heads: 8,
            kv_heads: 1,
            seq_len: 5,
            head_dim: 32,
        },
    ] {
        let q_projection = fixture(shape.q_tensor_len().unwrap(), 0.3);
        let k_projection = fixture(shape.kv_tensor_len().unwrap(), 0.9);
        let v_projection = fixture(shape.kv_tensor_len().unwrap(), 1.5);
        let q_head_major = projection_to_head_major(
            &q_projection,
            shape.batch,
            shape.q_heads,
            shape.seq_len,
            shape.head_dim,
        );
        let k_head_major = projection_to_head_major(
            &k_projection,
            shape.batch,
            shape.kv_heads,
            shape.seq_len,
            shape.head_dim,
        );
        let v_head_major = projection_to_head_major(
            &v_projection,
            shape.batch,
            shape.kv_heads,
            shape.seq_len,
            shape.head_dim,
        );

        for rotary in [
            RotaryEmbeddingConfig {
                theta: 10_000.0,
                position_offset: 0,
            },
            RotaryEmbeddingConfig {
                theta: 500_000.0,
                position_offset: 19,
            },
        ] {
            for causal in [false, true] {
                let config = FlatAttentionConfig {
                    causal,
                    softmax_scale: None,
                };
                let projection = forward_reference_projection_grouped_rope(
                    &q_projection,
                    &k_projection,
                    &v_projection,
                    shape,
                    config,
                    rotary,
                )
                .unwrap();
                let canonical = forward_reference_grouped_rope(
                    &q_head_major,
                    &k_head_major,
                    &v_head_major,
                    shape,
                    config,
                    rotary,
                )
                .unwrap();
                let canonical_projection = head_major_to_projection(
                    &canonical.output,
                    shape.batch,
                    shape.q_heads,
                    shape.seq_len,
                    shape.head_dim,
                );

                assert_eq!(
                    projection
                        .output
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    canonical_projection
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "O mismatch shape={shape:?} rotary={rotary:?} causal={causal}"
                );
                assert_eq!(
                    projection
                        .lse
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    canonical
                        .lse
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "LSE mismatch shape={shape:?} rotary={rotary:?} causal={causal}"
                );
            }
        }
    }
}
