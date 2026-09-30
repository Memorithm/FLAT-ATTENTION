use flat_algebraic_attention::compact_preselection::{compact_preselect, CompactPreselectionError};
use flat_algebraic_attention::softmax_mass::retained_softmax_mass;
use flat_attention::api::research_structural_routing::{
    forward_reference_structural_dense, forward_reference_structural_sparse, StructuralCandidateSet,
    StructuralRoutingError,
};
use flat_attention::{forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionError};

fn shape(seq_len: usize, head_dim: usize) -> AttentionShape {
    AttentionShape {
        batch: 1,
        heads: 1,
        seq_len,
        head_dim,
    }
}

fn config(causal: bool) -> FlatAttentionConfig {
    FlatAttentionConfig {
        causal,
        softmax_scale: Some(1.0),
    }
}

fn tensors(s: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = s.tensor_len().unwrap();
    let q = (0..len)
        .map(|i| ((i * 7 % 19) as f32 - 9.0) / 11.0)
        .collect();
    let k = (0..len)
        .map(|i| ((i * 5 % 23) as f32 - 11.0) / 13.0)
        .collect();
    let v = (0..len)
        .map(|i| ((i * 3 % 29) as f32 - 14.0) / 17.0)
        .collect();
    (q, k, v)
}

// Independent two-pass f64 oracle; it does not share FLAT's online update.
fn independent_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dim: usize,
    keys: &[usize],
    scale: f64,
) -> (Vec<f64>, f64) {
    let scores: Vec<f64> = keys
        .iter()
        .map(|&key| {
            q.iter()
                .zip(&k[key * dim..(key + 1) * dim])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
                * scale
        })
        .collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut out = vec![0.0; dim];
    for (&key, weight) in keys.iter().zip(weights) {
        for (value, &source) in out.iter_mut().zip(&v[key * dim..(key + 1) * dim]) {
            *value += weight * f64::from(source) / total;
        }
    }
    (out, max + total.ln())
}

#[test]
fn all_accept_preserves_dense_output_and_lse() {
    let s = shape(19, 17);
    let (q, k, v) = tensors(s);
    for causal in [false, true] {
        let cfg = config(causal);
        let screen = compact_preselect(&q, &k, s, cfg, &[0, 3, 16], usize::MAX).unwrap();
        let sparse =
            forward_reference_structural_sparse(&q, &k, &v, s, cfg, &screen.candidates).unwrap();
        let dense = forward_reference(&q, &k, &v, s, cfg).unwrap();
        assert_eq!(sparse.attention, dense);
        assert_eq!(sparse.counters.executed_pairs, screen.counters.selected_pairs);
    }
}

#[test]
fn sparse_output_matches_masked_and_independent_oracles_with_odd_multihead_geometry() {
    let s = AttentionShape {
        batch: 2,
        heads: 3,
        seq_len: 19,
        head_dim: 17,
    };
    let (q, k, v) = tensors(s);
    for causal in [false, true] {
        let cfg = config(causal);
        let screen = compact_preselect(&q, &k, s, cfg, &[0, 3, 16], 4).unwrap();
        let sparse =
            forward_reference_structural_sparse(&q, &k, &v, s, cfg, &screen.candidates).unwrap();
        let masked =
            forward_reference_structural_dense(&q, &k, &v, s, cfg, &screen.candidates).unwrap();
        assert_eq!(sparse, masked);
        for (row, query) in q.chunks_exact(s.head_dim).enumerate() {
            let head_base = row / s.seq_len * s.seq_len * s.head_dim;
            let keys = screen.candidates.row(row).unwrap();
            let (expected, lse) = independent_row(
                query,
                &k[head_base..],
                &v[head_base..],
                s.head_dim,
                keys,
                1.0,
            );
            for (&actual, expected) in sparse.attention.output
                [row * s.head_dim..(row + 1) * s.head_dim]
                .iter()
                .zip(expected)
            {
                assert!((f64::from(actual) - expected).abs() < 2.0e-5);
            }
            assert!((f64::from(sparse.attention.lse[row]) - lse).abs() < 2.0e-5);
            assert_eq!(
                keys.len(),
                if causal {
                    4.min(row % s.seq_len + 1)
                } else {
                    4
                }
            );
        }
    }
}

#[test]
fn causal_filter_precedes_budget_and_never_compacts_logical_positions() {
    let s = shape(7, 1);
    let q = vec![1.0; 7];
    let k: Vec<f32> = (0..7).map(|x| x as f32).collect();
    let result = compact_preselect(&q, &k, s, config(true), &[0], 1).unwrap();
    for row in 0..7 {
        assert_eq!(result.candidates.row(row).unwrap(), &[row]);
    }
    assert_eq!(result.counters.evaluated_pairs, 28);
    assert_eq!(result.counters.selected_pairs, 7);
}

#[test]
fn finite_future_changes_cannot_change_past_selection_or_output() {
    let s = shape(7, 3);
    let (q, mut k, mut v) = tensors(s);
    let cfg = config(true);
    let before = compact_preselect(&q, &k, s, cfg, &[0, 2], 2).unwrap();
    let out_before =
        forward_reference_structural_sparse(&q, &k, &v, s, cfg, &before.candidates).unwrap();
    k[18..].fill(100.0);
    v[18..].fill(-100.0);
    let after = compact_preselect(&q, &k, s, cfg, &[0, 2], 2).unwrap();
    let out_after =
        forward_reference_structural_sparse(&q, &k, &v, s, cfg, &after.candidates).unwrap();
    for row in 0..6 {
        assert_eq!(
            before.candidates.row(row).unwrap(),
            after.candidates.row(row).unwrap()
        );
    }
    assert_eq!(
        out_before.attention.output[..18],
        out_after.attention.output[..18]
    );
    assert_eq!(out_before.attention.lse[..6], out_after.attention.lse[..6]);
}

#[test]
fn equal_scores_prefer_lowest_original_key_ids() {
    let s = shape(19, 17);
    let q = vec![0.0; s.tensor_len().unwrap()];
    let k = vec![1.0; q.len()];
    let first = compact_preselect(&q, &k, s, config(false), &[0, 16], 3).unwrap();
    let second = compact_preselect(&q, &k, s, config(false), &[0, 16], 3).unwrap();
    assert_eq!(first, second);
    for row in 0..19 {
        assert_eq!(first.candidates.row(row).unwrap(), &[0, 1, 2]);
    }
}

#[test]
fn zero_budget_is_explicit_and_numerical_execution_rejects_empty_rows() {
    let s = shape(3, 2);
    let (q, k, v) = tensors(s);
    let result = compact_preselect(&q, &k, s, config(true), &[0], 0).unwrap();
    assert_eq!(result.counters.evaluated_pairs, 0);
    assert_eq!(result.counters.evaluated_score_components, 0);
    assert_eq!(result.counters.selected_pairs, 0);
    assert!(matches!(
        forward_reference_structural_sparse(&q, &k, &v, s, config(true), &result.candidates),
        Err(StructuralRoutingError::EmptyEffectiveCandidates { row: 0 })
    ));
}

#[test]
fn logical_projection_accounting_does_not_hide_retained_full_kv() {
    let s = shape(19, 17);
    let (q, k, _) = tensors(s);
    let result = compact_preselect(&q, &k, s, config(false), &[0, 3, 16], 2).unwrap();
    assert_eq!(result.counters.projected_key_payload_bytes, 19 * 3 * 4);
    assert_eq!(result.counters.evaluated_pairs, 19 * 19);
    assert_eq!(result.counters.evaluated_score_components, 19 * 19 * 3);
    assert_eq!(result.candidates.offsets().len(), 20);
    assert_eq!(result.candidates.key_positions().len(), 38);
    assert_eq!(k.len(), 19 * 17);
}

#[test]
fn invalid_projection_coordinates_are_rejected() {
    let s = shape(3, 3);
    let (q, k, _) = tensors(s);
    for coordinates in [vec![], vec![3], vec![1, 1], vec![2, 1]] {
        assert!(compact_preselect(&q, &k, s, config(false), &coordinates, 1).is_err());
    }
}

#[test]
fn malformed_shapes_lengths_and_scale_fail_closed() {
    let s = shape(3, 2);
    let (q, k, _) = tensors(s);
    assert!(compact_preselect(&q[..5], &k, s, config(false), &[0], 1).is_err());
    assert!(compact_preselect(&q, &k[..5], s, config(false), &[0], 1).is_err());
    assert!(compact_preselect(&[], &[], shape(0, 2), config(false), &[0], 1).is_err());
    assert!(compact_preselect(&[], &[], shape(usize::MAX, 2), config(false), &[0], 1).is_err());
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let cfg = FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(scale),
        };
        assert!(compact_preselect(&q, &k, s, cfg, &[0], 1).is_err());
    }
}

#[test]
fn nonfinite_unprojected_or_future_inputs_are_not_silently_ignored() {
    let s = shape(3, 2);
    let (q, k, _) = tensors(s);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad_k = k.clone();
        bad_k[5] = value;
        assert!(matches!(
            compact_preselect(&q, &bad_k, s, config(true), &[0], 0),
            Err(CompactPreselectionError::Attention(
                FlatAttentionError::NonFiniteInput {
                    tensor: "K",
                    index: 5
                }
            ))
        ));
        let mut bad_q = q.clone();
        bad_q[1] = value;
        assert!(compact_preselect(&bad_q, &k, s, config(true), &[0], 1).is_err());
    }
}

#[test]
fn finite_f32_extremes_do_not_overflow_selector_accumulation() {
    let s = shape(3, 17);
    let q = vec![f32::MAX; s.tensor_len().unwrap()];
    let k = q.clone();
    let result = compact_preselect(&q, &k, s, config(false), &[0, 16], 1).unwrap();
    for row in 0..3 {
        assert_eq!(result.candidates.row(row).unwrap(), &[0]);
    }
    // This qualifies only selector f64 accumulation, not f32 numerical attention.
}

#[test]
fn omitted_coordinate_can_drop_the_dominant_key_and_exact_rescoring_cannot_repair_it() {
    let s = shape(3, 2);
    let q = vec![1.0; 6];
    let k = vec![0.0, 20.0, 1.0, 0.0, 2.0, 0.0];
    let v = vec![10.0, 10.0, 0.0, 0.0, 0.0, 0.0];
    let compact = compact_preselect(&q, &k, s, config(false), &[0], 1).unwrap();
    let full = compact_preselect(&q, &k, s, config(false), &[0, 1], 1).unwrap();
    assert_eq!(compact.candidates.row(2).unwrap(), &[2]);
    assert_eq!(full.candidates.row(2).unwrap(), &[0]);
    let mass =
        retained_softmax_mass(&[20.0, 1.0, 2.0], compact.candidates.row(2).unwrap()).unwrap();
    assert!(mass.retained_mass() < 1.0e-7);
    let sparse =
        forward_reference_structural_sparse(&q, &k, &v, s, config(false), &compact.candidates)
            .unwrap();
    let dense = forward_reference(&q, &k, &v, s, config(false)).unwrap();
    assert!(dense.output[4] - sparse.attention.output[4] > 9.9);
}

fn random_rank(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

/// Frozen descriptive panel. No holdout, timing, model quality or promotion claim.
#[test]
fn emit_frozen_quality_panel() {
    let s = shape(5, 2);
    for adverse in [false, true] {
        let case = if adverse {
            "omitted_dominant_coordinate"
        } else {
            "aligned_coordinate"
        };
        let q = [1.0, if adverse { 1.0 } else { 0.0 }].repeat(5);
        let mut k = Vec::new();
        let mut v = Vec::new();
        for key in 0..5 {
            k.extend_from_slice(&[
                key as f32,
                if adverse && key == 0 { 20.0 } else { 0.0 },
            ]);
            v.extend_from_slice(&[key as f32, 4.0 - key as f32]);
        }
        let dense = forward_reference(&q, &k, &v, s, config(false)).unwrap();
        let compact = compact_preselect(&q, &k, s, config(false), &[0], 2).unwrap();
        let full = compact_preselect(&q, &k, s, config(false), &[0, 1], 2).unwrap();
        let mut random: Vec<usize> = (0..5).collect();
        random.sort_by_key(|&key| (random_rank(key as u64 ^ 0x43535031), key));
        random.truncate(2);
        let arms = [
            ("all_accept", StructuralCandidateSet::all(s).unwrap()),
            ("compact_top2", compact.candidates),
            ("full_coordinate_top2", full.candidates),
            (
                "recency_top2",
                StructuralCandidateSet::from_rows(s, vec![vec![3, 4]; 5]).unwrap(),
            ),
            (
                "matched_random_top2",
                StructuralCandidateSet::from_rows(s, vec![random; 5]).unwrap(),
            ),
        ];
        let scores: Vec<f64> = k
            .chunks_exact(2)
            .map(|key| {
                f64::from(q[0]) * f64::from(key[0]) + f64::from(q[1]) * f64::from(key[1])
            })
            .collect();
        for (arm, candidates) in arms {
            let selected = candidates.row(4).unwrap();
            let mass = retained_softmax_mass(&scores, selected).unwrap();
            let sparse =
                forward_reference_structural_sparse(&q, &k, &v, s, config(false), &candidates)
                    .unwrap();
            let error = dense.output[8..10]
                .iter()
                .zip(&sparse.attention.output[8..10])
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if arm == "all_accept" {
                assert_eq!(error, 0.0);
            }
            if adverse && arm == "compact_top2" {
                assert!(mass.retained_mass() < 1.0e-6);
            }
            println!("{{\"schema\":\"flat.compact-preselection-smoke/v1\",\"case\":\"{case}\",\"arm\":\"{arm}\",\"row\":4,\"selected\":{selected:?},\"retained_mass\":{},\"max_abs_output_error\":{error},\"executed_pairs\":{},\"timing_measured\":false,\"model_quality_measured\":false,\"promotion_authorized\":false}}", mass.retained_mass(), sparse.counters.executed_pairs);
        }
    }
}
