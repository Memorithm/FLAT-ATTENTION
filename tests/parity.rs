use flat_attention::{forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionOutput};

fn naive_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
) -> FlatAttentionOutput {
    let scale = config.resolved_scale(shape.head_dim).unwrap();
    let mut output = vec![0.0; q.len()];
    let mut lse = vec![0.0; shape.batch * shape.heads * shape.seq_len];
    let head_stride = shape.seq_len * shape.head_dim;

    for batch in 0..shape.batch {
        for head in 0..shape.heads {
            let bh = batch * shape.heads + head;
            let base = bh * head_stride;
            for qi in 0..shape.seq_len {
                let q_base = base + qi * shape.head_dim;
                let keys = if config.causal { qi + 1 } else { shape.seq_len };
                let mut scores = Vec::with_capacity(keys);
                for kj in 0..keys {
                    let k_base = base + kj * shape.head_dim;
                    let mut dot = 0.0;
                    for d in 0..shape.head_dim {
                        dot += q[q_base + d] * k[k_base + d];
                    }
                    scores.push(dot * scale);
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0;
                for score in &mut scores {
                    *score = (*score - max).exp();
                    denom += *score;
                }
                for (kj, p_num) in scores.iter().copied().enumerate() {
                    let p = p_num / denom;
                    let v_base = base + kj * shape.head_dim;
                    for d in 0..shape.head_dim {
                        output[q_base + d] += p * v[v_base + d];
                    }
                }
                lse[bh * shape.seq_len + qi] = max + denom.ln();
            }
        }
    }

    FlatAttentionOutput { output, lse }
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let error = (a - b).abs();
        assert!(
            error <= tol,
            "index {index}: actual={a}, expected={b}, abs_error={error}, tol={tol}"
        );
    }
}

fn fixture(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len).map(|i| ((i as f32) * 0.173 - 0.7).sin()).collect();
    let k = (0..len).map(|i| ((i as f32) * 0.119 + 0.2).cos()).collect();
    let v = (0..len)
        .map(|i| ((i as f32) * 0.071 - 0.4).sin() * 1.7)
        .collect();
    (q, k, v)
}

#[test]
fn online_softmax_matches_naive_non_causal() {
    let shape = AttentionShape {
        batch: 2,
        heads: 3,
        seq_len: 7,
        head_dim: 5,
    };
    let (q, k, v) = fixture(shape);
    let config = FlatAttentionConfig {
        causal: false,
        softmax_scale: None,
    };
    let online = forward_reference(&q, &k, &v, shape, config).unwrap();
    let naive = naive_forward(&q, &k, &v, shape, config);
    assert_close(&online.output, &naive.output, 2.0e-6);
    assert_close(&online.lse, &naive.lse, 2.0e-6);
}

#[test]
fn online_softmax_matches_naive_causal() {
    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 9,
        head_dim: 8,
    };
    let (q, k, v) = fixture(shape);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: Some(0.25),
    };
    let online = forward_reference(&q, &k, &v, shape, config).unwrap();
    let naive = naive_forward(&q, &k, &v, shape, config);
    assert_close(&online.output, &naive.output, 2.0e-6);
    assert_close(&online.lse, &naive.lse, 2.0e-6);
}
