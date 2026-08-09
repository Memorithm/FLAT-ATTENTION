use flat_attention::{
    forward_reference_grouped, forward_reference_grouped_rope, FlatAttentionConfig,
    GroupedAttentionShape, RotaryEmbeddingConfig,
};

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i as f32 * 0.029 + phase;
            x.sin() * 1.5 + (x * 0.37).cos() * 0.375
        })
        .collect()
}

fn materialize_head_local_rope(
    source: &[f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    rotary: RotaryEmbeddingConfig,
) -> Vec<f32> {
    let mut out = source.to_vec();
    let head_stride = seq_len * head_dim;
    for b in 0..batch {
        for head in 0..heads {
            let head_base = (b * heads + head) * head_stride;
            for position in 0..seq_len {
                let row_base = head_base + position * head_dim;
                let absolute_position = rotary.position_offset + position;
                for pair in 0..head_dim / 2 {
                    let dim = 2 * pair;
                    let exponent = -2.0 * pair as f32 / head_dim as f32;
                    let frequency = rotary.theta.powf(exponent);
                    let angle = absolute_position as f32 * frequency;
                    let (sin, cos) = angle.sin_cos();
                    let even = source[row_base + dim];
                    let odd = source[row_base + dim + 1];
                    out[row_base + dim] = even * cos - odd * sin;
                    out[row_base + dim + 1] = even * sin + odd * cos;
                }
            }
        }
    }
    out
}

#[test]
fn fused_rope_oracle_matches_materialized_rope_then_grouped_attention() {
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
        let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
        let k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
        let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);

        for rotary in [
            RotaryEmbeddingConfig {
                theta: 10_000.0,
                position_offset: 0,
            },
            RotaryEmbeddingConfig {
                theta: 500_000.0,
                position_offset: 37,
            },
        ] {
            let rotated_q = materialize_head_local_rope(
                &q,
                shape.batch,
                shape.q_heads,
                shape.seq_len,
                shape.head_dim,
                rotary,
            );
            let rotated_k = materialize_head_local_rope(
                &k,
                shape.batch,
                shape.kv_heads,
                shape.seq_len,
                shape.head_dim,
                rotary,
            );

            for causal in [false, true] {
                let config = FlatAttentionConfig {
                    causal,
                    softmax_scale: None,
                };
                let fused =
                    forward_reference_grouped_rope(&q, &k, &v, shape, config, rotary).unwrap();
                let materialized =
                    forward_reference_grouped(&rotated_q, &rotated_k, &v, shape, config).unwrap();

                assert_eq!(
                    fused
                        .output
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    materialized
                        .output
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "O mismatch shape={shape:?} rotary={rotary:?} causal={causal}"
                );
                assert_eq!(
                    fused
                        .lse
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    materialized
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
