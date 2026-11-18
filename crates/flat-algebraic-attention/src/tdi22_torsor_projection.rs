//! Opt-in f32 bridge for TDI-22 torsor attention using the existing FLAT oracle.
//!
//! This is NOT TDI's f64 scientific evaluator and NOT a new FLAT kernel.
//! It implements the versioned TDI-22.0 sign convention at the boundary:
//! C = M(P) + P x R; M(Q) = C - Q x R.
//! The canonical FLAT six-channel dense oracle receives
//! Q' = (v + Q x omega, omega), K' = (R,C), and V' = (R,C).
//! Thus no new scoring or softmax semantics need to be introduced into FLAT.
//! Every input and derived value is checked for finite f32 arithmetic.
//! This research adapter cannot change public api::v1 or runtime dispatch.

use flat_attention::{forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionError};

/// Versioned non-production software bridge. Does not imply a TDI dependency.
pub const TDI22_FLAT_PROJECTION_V1: &str = "tdi22-origin-reduced-flat6-f32-v1";
/// Explicit development bound: dense reference costs O(N^2) score evaluations.
pub const MAX_TDI22_REFERENCE_TOKENS: usize = 4096;
const CHANNELS: usize = 6;

/// One TDI-22 torsor (wrench) with a declared reduction point P.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsorKey {
    pub resultant: [f32; 3],
    pub moment_at_reference: [f32; 3],
    pub reference: [f32; 3],
}

/// One TDI-22 twist and an independently declared evaluation point Q.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsorQuery {
    pub linear: [f32; 3],
    pub angular: [f32; 3],
    pub evaluation_point: [f32; 3],
}

/// The normalized resultant, moment transported to this row's Q, and FLAT LSE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsorOutputRow {
    pub resultant: [f32; 3],
    pub moment_at_query: [f32; 3],
    pub log_sum_exp: f32,
}

/// No hidden fallback for invalid geometry, shape or derived arithmetic.
#[derive(Debug)]
pub enum TorsorProjectionError {
    EmptySequence,
    TooManyTokens { count: usize, limit: usize },
    QueryKeyLengthMismatch { queries: usize, keys: usize },
    NonFiniteInput { index: usize, field: &'static str },
    NonFiniteDerived { index: usize, stage: &'static str },
    Flat(FlatAttentionError),
}

impl From<FlatAttentionError> for TorsorProjectionError {
    fn from(error: FlatAttentionError) -> Self {
        Self::Flat(error)
    }
}

fn finite3(a: [f32; 3], index: usize, field: &'static str) -> Result<(), TorsorProjectionError> {
    if a.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(TorsorProjectionError::NonFiniteInput { index, field })
    }
}

fn checked3(
    a: [f32; 3],
    index: usize,
    stage: &'static str,
) -> Result<[f32; 3], TorsorProjectionError> {
    if a.iter().all(|v| v.is_finite()) {
        Ok(a)
    } else {
        Err(TorsorProjectionError::NonFiniteDerived { index, stage })
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn plus(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn minus(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Reusable prepared (R,C) bank; one flattened f32 buffer serves as both K and V.
///
/// This bank is designed for a square reference batch. It deliberately cannot
/// masquerade as a qualified cross-attention, GPU or heterogeneous-head API.
#[derive(Clone, Debug)]
pub struct PreparedTorsorBank {
    flat_keys_and_values: Vec<f32>,
    len: usize,
}

impl PreparedTorsorBank {
    /// Validate and factorize all torsors once at their own reduction points.
    pub fn build(keys: &[TorsorKey]) -> Result<Self, TorsorProjectionError> {
        if keys.is_empty() {
            return Err(TorsorProjectionError::EmptySequence);
        }
        if keys.len() > MAX_TDI22_REFERENCE_TOKENS {
            return Err(TorsorProjectionError::TooManyTokens {
                count: keys.len(),
                limit: MAX_TDI22_REFERENCE_TOKENS,
            });
        }
        let mut flat = Vec::with_capacity(keys.len() * CHANNELS);
        for (i, key) in keys.iter().enumerate() {
            finite3(key.resultant, i, "key.resultant")?;
            finite3(key.moment_at_reference, i, "key.moment_at_reference")?;
            finite3(key.reference, i, "key.reference")?;
            let p_cross_r = checked3(cross(key.reference, key.resultant), i, "P x R")?;
            let c = checked3(
                plus(key.moment_at_reference, p_cross_r),
                i,
                "C = M(P) + P x R",
            )?;
            flat.extend_from_slice(&key.resultant);
            flat.extend_from_slice(&c);
        }
        Ok(Self {
            flat_keys_and_values: flat,
            len: keys.len(),
        })
    }

    /// Number of prepared tokens.
    pub fn len(&self) -> usize {
        self.len
    }

    /// This bank cannot be empty by construction.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Materialized f32 channels, K and V sharing the same physical slice.
    pub fn stored_scalar_count(&self) -> usize {
        self.flat_keys_and_values.len()
    }

    /// Compute all query rows with FLAT's dense scalar reference. Causality,
    /// online-softmax/LSE and scale are exclusively the native FLAT semantics.
    ///
    /// In exact arithmetic scores and normalized moments are equivalent to
    /// TDI-22 direct-Varignon pairing, but this f32 bridge does not promise
    /// bit-for-bit identity with TDI's independent f64 software reference.
    pub fn forward(
        &self,
        queries: &[TorsorQuery],
        config: FlatAttentionConfig,
    ) -> Result<Vec<TorsorOutputRow>, TorsorProjectionError> {
        if queries.len() != self.len {
            return Err(TorsorProjectionError::QueryKeyLengthMismatch {
                queries: queries.len(),
                keys: self.len,
            });
        }
        let mut flat_queries = Vec::with_capacity(self.flat_keys_and_values.len());
        for (i, query) in queries.iter().enumerate() {
            finite3(query.linear, i, "query.linear")?;
            finite3(query.angular, i, "query.angular")?;
            finite3(query.evaluation_point, i, "query.evaluation_point")?;
            let q_cross_omega =
                checked3(cross(query.evaluation_point, query.angular), i, "Q x omega")?;
            let shifted = checked3(plus(query.linear, q_cross_omega), i, "v + Q x omega")?;
            flat_queries.extend_from_slice(&shifted);
            flat_queries.extend_from_slice(&query.angular);
        }

        let dense = forward_reference(
            &flat_queries,
            &self.flat_keys_and_values,
            &self.flat_keys_and_values,
            AttentionShape {
                batch: 1,
                heads: 1,
                seq_len: self.len,
                head_dim: CHANNELS,
            },
            config,
        )?;

        dense
            .output
            .chunks_exact(CHANNELS)
            .zip(dense.lse)
            .zip(queries.iter())
            .enumerate()
            .map(|(i, ((row, lse), query))| {
                let resultant = checked3([row[0], row[1], row[2]], i, "output resultant")?;
                let c = checked3([row[3], row[4], row[5]], i, "output origin moment")?;
                let q_cross_r =
                    checked3(cross(query.evaluation_point, resultant), i, "output Q x R")?;
                let moment_at_query = checked3(minus(c, q_cross_r), i, "output M(Q)")?;
                if !lse.is_finite() {
                    return Err(TorsorProjectionError::NonFiniteDerived {
                        index: i,
                        stage: "output log-sum-exp",
                    });
                }
                Ok(TorsorOutputRow {
                    resultant,
                    moment_at_query,
                    log_sum_exp: lse,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: usize) -> TorsorKey {
        let t = i as f32;
        TorsorKey {
            resultant: [0.4 * t - 0.3, (0.25 * t).sin(), 0.3 * (0.7 * t).cos()],
            moment_at_reference: [0.11 * t, -0.25 + t * 0.04, (t * 0.3).sin()],
            reference: [t * 0.17, -t * 0.12, 0.5 + 0.03 * t],
        }
    }

    fn query(i: usize) -> TorsorQuery {
        let t = i as f32;
        TorsorQuery {
            linear: [(t * 0.4).cos(), 0.7, -0.3],
            angular: [-0.2, (t * 0.6).sin(), 0.4],
            evaluation_point: [0.8 + t * 0.03, -0.1 * t, 0.4],
        }
    }

    /// Independent dense oracle uses original P and direct Varignon transport
    /// per key and f64 arithmetic. It never invokes the (R,C) factorization.
    fn independent_direct_dense(
        keys: &[TorsorKey],
        queries: &[TorsorQuery],
        causal: bool,
        scale: f64,
    ) -> Vec<([f64; 3], [f64; 3], f64)> {
        fn f64v(x: [f32; 3]) -> [f64; 3] {
            [f64::from(x[0]), f64::from(x[1]), f64::from(x[2])]
        }
        fn xproduct(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        }
        fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }
        queries
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let v = f64v(q.linear);
                let omega = f64v(q.angular);
                let point = f64v(q.evaluation_point);
                let visible = if causal { i + 1 } else { keys.len() };
                let mut scores = Vec::with_capacity(visible);
                let mut moments = Vec::with_capacity(visible);
                for key in &keys[..visible] {
                    let r = f64v(key.resultant);
                    let p = f64v(key.reference);
                    let m = f64v(key.moment_at_reference);
                    let offset = xproduct([p[0] - point[0], p[1] - point[1], p[2] - point[2]], r);
                    let mq = [m[0] + offset[0], m[1] + offset[1], m[2] + offset[2]];
                    scores.push(scale * (dot(v, r) + dot(omega, mq)));
                    moments.push(mq);
                }
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let sum: f64 = weights.iter().sum();
                let mut out_r = [0.0_f64; 3];
                let mut out_m = [0.0_f64; 3];
                for (j, weight) in weights.iter().enumerate() {
                    for axis in 0..3 {
                        out_r[axis] += weight * f64::from(keys[j].resultant[axis]) / sum;
                        out_m[axis] += weight * moments[j][axis] / sum;
                    }
                }
                (out_r, out_m, maximum + sum.ln())
            })
            .collect()
    }

    fn assert_close(a: f32, b: f64) {
        let difference = (f64::from(a) - b).abs();
        let allowance = 4.0e-5 * (1.0 + b.abs());
        assert!(
            difference <= allowance,
            "{a} vs {b}: diff {difference}, limit {allowance}"
        );
    }

    #[test]
    fn flat_projection_matches_independent_direct_varignon_full_and_causal() {
        let keys: Vec<_> = (0..17).map(key).collect();
        let queries: Vec<_> = (0..keys.len()).map(query).collect();
        let bank = PreparedTorsorBank::build(&keys).unwrap();
        assert_eq!(bank.stored_scalar_count(), 6 * keys.len());
        assert_eq!(bank.len(), keys.len());
        assert!(!bank.is_empty());
        for causal in [false, true] {
            let cfg = FlatAttentionConfig {
                causal,
                softmax_scale: Some(0.45),
            };
            let actual = bank.forward(&queries, cfg).unwrap();
            let expected = independent_direct_dense(&keys, &queries, causal, 0.45_f32 as f64);
            assert_eq!(actual.len(), keys.len());
            for (a, (r, m, lse)) in actual.iter().zip(&expected) {
                for dim in 0..3 {
                    assert_close(a.resultant[dim], r[dim]);
                    assert_close(a.moment_at_query[dim], m[dim]);
                }
                assert_close(a.log_sum_exp, *lse);
            }
        }
    }

    #[test]
    fn source_reference_change_and_joint_origin_translation_are_invariant() {
        let keys: Vec<_> = (0..8).map(key).collect();
        let queries: Vec<_> = (0..8).map(query).collect();
        let cfg = FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(0.3),
        };
        let original = PreparedTorsorBank::build(&keys)
            .unwrap()
            .forward(&queries, cfg)
            .unwrap();

        let delta = [0.31, -0.2, 0.5];
        let common_shift = [2.0, -1.25, 0.7];
        let moved: Vec<_> = keys
            .iter()
            .map(|key| {
                let new_reference = plus(key.reference, delta);
                let moment_at_reference = plus(
                    key.moment_at_reference,
                    cross(minus(key.reference, new_reference), key.resultant),
                );
                TorsorKey {
                    resultant: key.resultant,
                    moment_at_reference,
                    reference: plus(new_reference, common_shift),
                }
            })
            .collect();
        let shifted_queries: Vec<_> = queries
            .iter()
            .map(|q| TorsorQuery {
                evaluation_point: plus(q.evaluation_point, common_shift),
                ..*q
            })
            .collect();
        let changed = PreparedTorsorBank::build(&moved)
            .unwrap()
            .forward(&shifted_queries, cfg)
            .unwrap();
        for (a, b) in original.iter().zip(changed) {
            for j in 0..3 {
                assert_close(a.resultant[j], f64::from(b.resultant[j]));
                assert_close(a.moment_at_query[j], f64::from(b.moment_at_query[j]));
            }
            assert_close(a.log_sum_exp, f64::from(b.log_sum_exp));
        }
    }

    #[test]
    fn masked_future_keys_do_not_affect_causal_prefix() {
        let keys: Vec<_> = (0..9).map(key).collect();
        let queries: Vec<_> = (0..9).map(query).collect();
        let cfg = FlatAttentionConfig {
            causal: true,
            softmax_scale: Some(0.4),
        };
        let reference = PreparedTorsorBank::build(&keys)
            .unwrap()
            .forward(&queries, cfg)
            .unwrap();
        let mut altered = keys;
        for future in altered.iter_mut().skip(4) {
            future.resultant = [100.0, -50.0, 20.0];
            future.moment_at_reference = [-10.0, 40.0, 70.0];
        }
        let observed = PreparedTorsorBank::build(&altered)
            .unwrap()
            .forward(&queries, cfg)
            .unwrap();
        assert_eq!(&reference[..4], &observed[..4]);
    }

    #[test]
    fn huge_finite_logits_are_stable_via_flat_online_softmax() {
        let keys = vec![
            TorsorKey {
                resultant: [0.0, 0.0, 0.0],
                moment_at_reference: [0.0; 3],
                reference: [0.0; 3],
            },
            TorsorKey {
                resultant: [1.0, 0.0, 0.0],
                moment_at_reference: [0.0; 3],
                reference: [0.0; 3],
            },
        ];
        let q = TorsorQuery {
            linear: [1.0e6, 0.0, 0.0],
            angular: [0.0; 3],
            evaluation_point: [0.0; 3],
        };
        let rows = PreparedTorsorBank::build(&keys)
            .unwrap()
            .forward(&[q, q], FlatAttentionConfig::default())
            .unwrap();
        assert_eq!(rows[1].resultant[0], 1.0);
        assert!(rows[1].log_sum_exp.is_finite());
    }

    #[test]
    fn invalid_inputs_and_nonfinite_derived_values_fail_closed() {
        assert!(matches!(
            PreparedTorsorBank::build(&[]),
            Err(TorsorProjectionError::EmptySequence)
        ));
        let bank = PreparedTorsorBank::build(&[key(0), key(1)]).unwrap();
        assert!(matches!(
            bank.forward(&[query(0)], FlatAttentionConfig::default()),
            Err(TorsorProjectionError::QueryKeyLengthMismatch { .. })
        ));
        let mut bad_key = key(0);
        bad_key.resultant[0] = f32::NAN;
        assert!(matches!(
            PreparedTorsorBank::build(&[bad_key]),
            Err(TorsorProjectionError::NonFiniteInput { .. })
        ));
        let overflow = TorsorKey {
            resultant: [1.0e30, 0.0, 0.0],
            moment_at_reference: [0.0; 3],
            reference: [0.0, 1.0e30, 0.0],
        };
        assert!(matches!(
            PreparedTorsorBank::build(&[overflow]),
            Err(TorsorProjectionError::NonFiniteDerived { .. })
        ));
        let mut bad_query = query(0);
        bad_query.angular[2] = f32::NAN;
        assert!(matches!(
            bank.forward(&[bad_query, query(1)], FlatAttentionConfig::default()),
            Err(TorsorProjectionError::NonFiniteInput { .. })
        ));
        let overflow_query = TorsorQuery {
            evaluation_point: [0.0, 1.0e30, 0.0],
            angular: [0.0, 0.0, 1.0e30],
            linear: [0.0; 3],
        };
        assert!(matches!(
            bank.forward(&[overflow_query, query(1)], FlatAttentionConfig::default(),),
            Err(TorsorProjectionError::NonFiniteDerived { .. })
        ));
        let invalid_scale = FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(0.0),
        };
        assert!(matches!(
            bank.forward(&[query(0), query(1)], invalid_scale),
            Err(TorsorProjectionError::Flat(
                FlatAttentionError::InvalidScale(_)
            ))
        ));
    }
}
