//! Research-only Q4 Boolean Ising states -> exact sparse FLAT attention candidates.
//! Causal eligibility is applied before creating the Ising local fields:
//! future-key spin interactions cannot change earlier query-row decisions.
//! Both positive and negative ACTIVE spins admit original numerical K/V;
//! dormant and structurally masked sites do not. No numerical speedup claim.

use crate::ising_quaternary::{BooleanQuaternary, Spin4Error};
use crate::ising_sparse_state::SparseIsingEdge;
use core::fmt;
use flat_attention::api::research_structural_routing::{
    StructuralCandidateSet, StructuralRoutingError,
};
use flat_attention::{AttentionShape, FlatAttentionConfig, FlatAttentionError};

/// Per-batch/head Ising inputs. Every plan has seq_len key vertices.
#[derive(Debug, Clone, Copy)]
pub struct QuaternaryHeadPlan<'a> {
    pub biases: &'a [i32],
    pub activation_costs: &'a [i32],
    pub eligible: &'a [u64],
    pub active: &'a [u64],
    pub polarity: &'a [u64],
    pub couplings: &'a [SparseIsingEdge],
}

/// Logical counters, not measured latency, bandwidth or saved K/V bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuaternaryRoutingCounters {
    pub controllers_built: usize,
    pub executed_sweeps: usize,
    pub initial_pairs: usize,
    pub selected_pairs: usize,
    pub accepted_transitions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuaternaryRoutingResult {
    pub candidates: StructuralCandidateSet,
    pub counters: QuaternaryRoutingCounters,
}

#[derive(Debug)]
pub enum QuaternaryRoutingError {
    Attention(FlatAttentionError),
    Structural(StructuralRoutingError),
    Spin(Spin4Error),
    HeadPlanCount { expected: usize, actual: usize },
    EmptyEffectiveCandidates { row: usize },
    CounterOverflow,
}
impl From<FlatAttentionError> for QuaternaryRoutingError {
    fn from(error: FlatAttentionError) -> Self {
        Self::Attention(error)
    }
}
impl From<StructuralRoutingError> for QuaternaryRoutingError {
    fn from(error: StructuralRoutingError) -> Self {
        Self::Structural(error)
    }
}
impl From<Spin4Error> for QuaternaryRoutingError {
    fn from(error: Spin4Error) -> Self {
        Self::Spin(error)
    }
}
impl fmt::Display for QuaternaryRoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for QuaternaryRoutingError {}

/// Generate an exact FLAT structural candidate set from row-scoped Q4 states.
///
/// A controller is constructed for each query row, after the causal mask is
/// applied to eligible key vertices. The same per-head initial state is never
/// mutated across query rows. Zero sweeps is the frozen initial-state control.
/// Empty rows are errors, not implicit dense fallback. This is O(N+E) working
/// storage per row, but the full returned candidate set can still be quadratic
/// for an all-accept experiment; no sparse-memory claim is made.
pub fn quaternary_structural_candidates(
    shape: AttentionShape,
    config: FlatAttentionConfig,
    plans: &[QuaternaryHeadPlan<'_>],
    max_sweeps_per_row: usize,
) -> Result<QuaternaryRoutingResult, QuaternaryRoutingError> {
    if shape.batch == 0 || shape.heads == 0 || shape.seq_len == 0 || shape.head_dim == 0 {
        return Err(FlatAttentionError::ZeroDimension.into());
    }
    config.resolved_scale(shape.head_dim)?;
    let heads = shape
        .batch
        .checked_mul(shape.heads)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    if heads != plans.len() {
        return Err(QuaternaryRoutingError::HeadPlanCount {
            expected: heads,
            actual: plans.len(),
        });
    }
    let query_rows = shape.lse_len()?;
    let words = shape.seq_len.div_ceil(64);
    let mut candidates = Vec::with_capacity(query_rows);
    let mut totals = QuaternaryRoutingCounters {
        controllers_built: 0,
        executed_sweeps: 0,
        initial_pairs: 0,
        selected_pairs: 0,
        accepted_transitions: 0,
    };

    for row in 0..query_rows {
        let plan = plans[row / shape.seq_len];
        let position = row % shape.seq_len;
        if plan.biases.len() != shape.seq_len
            || plan.activation_costs.len() != shape.seq_len
            || plan.eligible.len() != words
            || plan.active.len() != words
            || plan.polarity.len() != words
        {
            return Err(Spin4Error::Shape.into());
        }
        if shape.seq_len % 64 != 0 {
            let invalid = !((1_u64 << (shape.seq_len % 64)) - 1);
            if (plan.eligible[words - 1] | plan.active[words - 1] | plan.polarity[words - 1])
                & invalid
                != 0
            {
                return Err(Spin4Error::Padding.into());
            }
        }
        for (&a, &e) in plan.active.iter().zip(plan.eligible) {
            if a & !e != 0 {
                return Err(Spin4Error::MaskViolation.into());
            }
        }

        // Causal eligibility is computed BEFORE the interacting Ising fields.
        let mut eligible = plan.eligible.to_vec();
        if config.causal {
            for (word_index, word) in eligible.iter_mut().enumerate() {
                let start = word_index * 64;
                if start > position {
                    *word = 0;
                } else {
                    let remaining = position - start + 1;
                    if remaining < 64 {
                        *word &= (1_u64 << remaining) - 1;
                    }
                }
            }
        }
        let active: Vec<u64> = plan
            .active
            .iter()
            .zip(&eligible)
            .map(|(&a, &e)| a & e)
            .collect();
        let initial = active
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        totals.initial_pairs = totals
            .initial_pairs
            .checked_add(initial)
            .ok_or(QuaternaryRoutingError::CounterOverflow)?;
        let mut spins = BooleanQuaternary::new(
            plan.biases,
            plan.activation_costs,
            &eligible,
            &active,
            plan.polarity,
            plan.couplings,
        )?;
        totals.controllers_built += 1;
        for _ in 0..max_sweeps_per_row {
            let changes = spins.sweep();
            totals.executed_sweeps = totals
                .executed_sweeps
                .checked_add(1)
                .ok_or(QuaternaryRoutingError::CounterOverflow)?;
            totals.accepted_transitions = totals
                .accepted_transitions
                .checked_add(changes)
                .ok_or(QuaternaryRoutingError::CounterOverflow)?;
            if changes == 0 {
                break;
            }
        }
        let mut row_keys = Vec::new();
        for key_position in 0..shape.seq_len {
            if spins.state(key_position)?.is_some_and(|s| s.active()) {
                row_keys.push(key_position);
            }
        }
        if row_keys.is_empty() {
            return Err(QuaternaryRoutingError::EmptyEffectiveCandidates { row });
        }
        totals.selected_pairs = totals
            .selected_pairs
            .checked_add(row_keys.len())
            .ok_or(QuaternaryRoutingError::CounterOverflow)?;
        candidates.push(row_keys);
    }
    let candidates = StructuralCandidateSet::from_rows(shape, candidates)?;
    Ok(QuaternaryRoutingResult {
        candidates,
        counters: totals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_attention::api::research_structural_routing::{
        forward_reference_structural_dense, forward_reference_structural_sparse,
    };
    use flat_attention::forward_reference;

    fn shape(n: usize, d: usize) -> AttentionShape {
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: n,
            head_dim: d,
        }
    }
    fn cfg(causal: bool) -> FlatAttentionConfig {
        FlatAttentionConfig {
            causal,
            softmax_scale: Some(1.0),
        }
    }

    #[test]
    fn all_accept_matches_dense_exactly_multihead_and_causal() {
        let s = AttentionShape {
            batch: 2,
            heads: 2,
            seq_len: 5,
            head_dim: 3,
        };
        let b = [0; 5];
        let mask = [0b11111];
        let plan = QuaternaryHeadPlan {
            biases: &b,
            activation_costs: &b,
            eligible: &mask,
            active: &mask,
            polarity: &mask,
            couplings: &[],
        };
        let plans = [plan; 4];
        let len = s.tensor_len().unwrap();
        let q: Vec<f32> = (0..len)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) / 7.0)
            .collect();
        let k: Vec<f32> = (0..len)
            .map(|i| ((i * 3 % 17) as f32 - 8.0) / 9.0)
            .collect();
        let v: Vec<f32> = (0..len)
            .map(|i| ((i * 5 % 19) as f32 - 9.0) / 11.0)
            .collect();

        for causal in [false, true] {
            let config = cfg(causal);
            let route = quaternary_structural_candidates(s, config, &plans, 0).unwrap();
            let selected =
                forward_reference_structural_sparse(&q, &k, &v, s, config, &route.candidates)
                    .unwrap();
            let dense = forward_reference(&q, &k, &v, s, config).unwrap();
            assert_eq!(selected.attention, dense);
            assert_eq!(route.counters.controllers_built, 20);
            assert_eq!(route.counters.executed_sweeps, 0);
            for row in 0..20 {
                assert_eq!(
                    route.candidates.row(row).unwrap().len(),
                    if causal { row % 5 + 1 } else { 5 }
                );
            }
        }
    }

    #[test]
    fn negative_active_spins_admit_exact_numerical_keys() {
        let s = shape(5, 2);
        let zero = [0; 5];
        let plan = QuaternaryHeadPlan {
            biases: &zero,
            activation_costs: &zero,
            eligible: &[0b11111],
            active: &[0b01011],
            polarity: &[0b10101],
            couplings: &[],
        };
        let route = quaternary_structural_candidates(s, cfg(false), &[plan], 0).unwrap();
        for row in 0..5 {
            assert_eq!(route.candidates.row(row).unwrap(), &[0, 1, 3]);
        }
        let q = vec![0.25; 10];
        let k = vec![0.5; 10];
        let v: Vec<f32> = (0..10).map(|i| i as f32 / 4.0).collect();
        let sparse =
            forward_reference_structural_sparse(&q, &k, &v, s, cfg(false), &route.candidates)
                .unwrap();
        let masked =
            forward_reference_structural_dense(&q, &k, &v, s, cfg(false), &route.candidates)
                .unwrap();
        assert_eq!(sparse, masked);
        assert_eq!(route.counters.selected_pairs, 15);
    }

    #[test]
    fn energy_update_reduces_active_key_density() {
        let s = shape(3, 1);
        let plan = QuaternaryHeadPlan {
            biases: &[3, -2, 3],
            activation_costs: &[0, 3, 0],
            eligible: &[0b111],
            active: &[0b111],
            polarity: &[0b111],
            couplings: &[],
        };
        let route = quaternary_structural_candidates(s, cfg(false), &[plan], 2).unwrap();
        for row in 0..3 {
            assert_eq!(route.candidates.row(row).unwrap(), &[0, 2]);
        }
        assert_eq!(route.counters.initial_pairs, 9);
        assert_eq!(route.counters.selected_pairs, 6);
        assert_eq!(route.counters.accepted_transitions, 3);
    }

    #[test]
    fn future_keys_cannot_affect_past_via_couplings() {
        let s = shape(5, 1);
        let first = [10, 1, 1, 1, 10];
        let future_changed = [10, 1, 1, 1, -100];
        let edges = [SparseIsingEdge {
            left: 0,
            right: 4,
            coupling: -100,
        }];
        let base = QuaternaryHeadPlan {
            biases: &first,
            activation_costs: &[0; 5],
            eligible: &[0b11111],
            active: &[0b11111],
            polarity: &[0b11111],
            couplings: &edges,
        };
        let changed = QuaternaryHeadPlan {
            biases: &future_changed,
            ..base
        };
        let a = quaternary_structural_candidates(s, cfg(true), &[base], 2).unwrap();
        let b = quaternary_structural_candidates(s, cfg(true), &[changed], 2).unwrap();
        for row in 0..4 {
            assert_eq!(
                a.candidates.row(row).unwrap(),
                b.candidates.row(row).unwrap()
            );
            assert!(a.candidates.row(row).unwrap().iter().all(|&key| key <= row));
        }
    }

    #[test]
    fn invalid_geometry_masks_and_empty_rows_fail_closed() {
        let s = shape(3, 1);
        let base = QuaternaryHeadPlan {
            biases: &[0; 3],
            activation_costs: &[0; 3],
            eligible: &[0b111],
            active: &[0b111],
            polarity: &[0b111],
            couplings: &[],
        };
        assert!(matches!(
            quaternary_structural_candidates(s, cfg(false), &[], 0),
            Err(QuaternaryRoutingError::HeadPlanCount { .. })
        ));
        let empty = QuaternaryHeadPlan {
            active: &[0],
            ..base
        };
        assert!(matches!(
            quaternary_structural_candidates(s, cfg(true), &[empty], 0),
            Err(QuaternaryRoutingError::EmptyEffectiveCandidates { row: 0 })
        ));
        let bad = QuaternaryHeadPlan {
            active: &[0b1000],
            ..base
        };
        assert!(matches!(
            quaternary_structural_candidates(s, cfg(false), &[bad], 0),
            Err(QuaternaryRoutingError::Spin(Spin4Error::Padding))
        ));
        let masked = QuaternaryHeadPlan {
            eligible: &[0b001],
            active: &[0b011],
            ..base
        };
        assert!(matches!(
            quaternary_structural_candidates(s, cfg(false), &[masked], 0),
            Err(QuaternaryRoutingError::Spin(Spin4Error::MaskViolation))
        ));
    }
}
