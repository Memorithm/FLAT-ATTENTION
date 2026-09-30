use flat_algebraic_attention::compact_preselection::compact_preselect;
use flat_attention::{AttentionShape, FlatAttentionConfig};

/// Independent full-sort oracle for the bounded-heap selector, not a benchmark.
#[test]
fn bounded_heap_matches_exhaustive_ranking_across_shapes_and_budgets() {
    for n in [1usize, 2, 7, 19] {
        for d in [1usize, 3, 17, 33] {
            let shape = AttentionShape {
                batch: 2,
                heads: 2,
                seq_len: n,
                head_dim: d,
            };
            let len = shape.tensor_len().unwrap();
            let q: Vec<f32> = (0..len)
                .map(|i| ((i * 7 % 11) as f32 - 5.0) / 8.0)
                .collect();
            let k: Vec<f32> = (0..len)
                .map(|i| ((i * 3 % 13) as f32 - 6.0) / 8.0)
                .collect();
            let coordinates: Vec<usize> = (0..d).step_by(2).collect();
            for causal in [false, true] {
                let config = FlatAttentionConfig {
                    causal,
                    softmax_scale: None,
                };
                let scale = f64::from(config.resolved_scale(d).unwrap());
                for budget in [0, 1, 2, n.saturating_sub(1), n, n + 3] {
                    let actual =
                        compact_preselect(&q, &k, shape, config, &coordinates, budget).unwrap();
                    let mut evaluated_pairs = 0usize;
                    for row in 0..shape.lse_len().unwrap() {
                        let eligible = if causal { row % n + 1 } else { n };
                        let query = &q[row * d..(row + 1) * d];
                        let head_start = row / n * n * d;
                        let mut scored: Vec<(usize, f64)> = (0..eligible)
                            .map(|key_position| {
                                let base = head_start + key_position * d;
                                let mut sum = 0.0f64;
                                for &coordinate in &coordinates {
                                    sum += f64::from(query[coordinate])
                                        * f64::from(k[base + coordinate]);
                                }
                                (key_position, sum * scale)
                            })
                            .collect();
                        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                        scored.truncate(budget.min(eligible));
                        let mut expected: Vec<usize> = scored.iter().map(|entry| entry.0).collect();
                        expected.sort_unstable();
                        assert_eq!(
                            actual.candidates.row(row).unwrap(),
                            expected.as_slice(),
                            "N={n}, D={d}, causal={causal}, budget={budget}, row={row}"
                        );
                        if budget != 0 {
                            evaluated_pairs += eligible;
                        }
                    }
                    assert_eq!(actual.counters.evaluated_pairs, evaluated_pairs);
                    assert_eq!(
                        actual.counters.evaluated_score_components,
                        evaluated_pairs * coordinates.len()
                    );
                }
            }
        }
    }
}
