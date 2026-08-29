//! Research-only structured-history attention semantics.
//!
//! This module is deliberately backend-neutral and scalar. It does not alter
//! the historical StandardSoftmax path, register a production kernel, or make
//! any performance claim. The first research revision keeps weighting and
//! scheduling intentionally conservative: identity weighting over every
//! retained true K/V position, with either complete history or an explicitly
//! approximate trailing window.

use core::fmt;

use crate::{
    validate_input, AsymmetricGroupedAttentionShape, FlatAttentionConfig, FlatAttentionError,
    FlatAttentionOutput,
};

/// Stable semantic slug for the first structured-history research rule.
pub const NONLOCAL_ATTENTION_SEMANTIC_NAME: &str = "nonlocal-history-softmax";
/// Stable semantic revision for [`NONLOCAL_ATTENTION_SEMANTIC_NAME`].
pub const NONLOCAL_ATTENTION_SEMANTIC_REVISION: u32 = 1;

/// Which causally visible K/V history is retained by the research semantic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HistoryMode {
    /// Retain every causally visible K/V position. This is the reference mode.
    Complete,
    /// Retain only the newest `max_tokens` causally visible positions.
    ///
    /// This is always classified as an approximation, even when a particular
    /// small input happens to fit entirely inside the window.
    Window { max_tokens: usize },
}

/// Which retained true logical positions participate in the first revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HistorySchedule {
    /// Visit every retained true logical position in ascending order.
    EveryToken,
}

/// Multiplicative history weighting used before softmax normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HistoryWeighting {
    /// Exact multiplicative identity. No position-dependent reweighting occurs.
    Identity,
}

/// Resource budget policy for retained history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HistoryBudgetPolicy {
    /// No semantic-side retained-token budget.
    Unlimited,
    /// Reject a query whose selected history would exceed the declared limit.
    ///
    /// The budget never truncates or changes the semantic rule implicitly.
    RejectAbove { max_retained_tokens: usize },
}

/// Typed research configuration kept outside the production `api::v1` config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NonlocalAttentionConfig {
    /// Reference or explicitly bounded history.
    pub history_mode: HistoryMode,
    /// Position-selection schedule over retained true logical positions.
    pub history_schedule: HistorySchedule,
    /// Multiplicative history weighting.
    pub history_weighting: HistoryWeighting,
    /// Explicit resource budget policy.
    pub history_budget_policy: HistoryBudgetPolicy,
}

impl Default for NonlocalAttentionConfig {
    fn default() -> Self {
        Self {
            history_mode: HistoryMode::Complete,
            history_schedule: HistorySchedule::EveryToken,
            history_weighting: HistoryWeighting::Identity,
            history_budget_policy: HistoryBudgetPolicy::Unlimited,
        }
    }
}

impl NonlocalAttentionConfig {
    /// Validate configuration values without touching Q/K/V data or a device.
    ///
    /// # Errors
    ///
    /// Zero-sized windows and zero-sized rejection budgets fail closed.
    pub fn validate(self) -> Result<(), NonlocalAttentionError> {
        if let HistoryMode::Window { max_tokens: 0 } = self.history_mode {
            return Err(NonlocalAttentionError::InvalidHistoryWindow);
        }
        if let HistoryBudgetPolicy::RejectAbove {
            max_retained_tokens: 0,
        } = self.history_budget_policy
        {
            return Err(NonlocalAttentionError::InvalidHistoryBudget);
        }
        Ok(())
    }

    /// Evidence classification implied by the configured semantic rule.
    #[must_use]
    pub const fn classification(self) -> HistoryClassification {
        match self.history_mode {
            HistoryMode::Complete => HistoryClassification::Reference,
            HistoryMode::Window { .. } => HistoryClassification::Approximation,
        }
    }
}

/// Evidence classification for one configured history rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistoryClassification {
    /// Complete causally visible history with no semantic reduction.
    Reference,
    /// Explicit bounded-history approximation.
    Approximation,
}

/// Scalar research result plus the semantic evidence classification.
#[derive(Debug, Clone, PartialEq)]
pub struct NonlocalAttentionOutput {
    /// Standard attention output/LSE layout for the selected history rule.
    pub attention: FlatAttentionOutput,
    /// Whether the configured rule is reference or approximate.
    pub classification: HistoryClassification,
}

/// Typed failures for the research-only structured-history oracle.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NonlocalAttentionError {
    /// Existing FLAT shape/input/numerical validation failed.
    Flat(FlatAttentionError),
    /// The first structured-history semantic is causal-only.
    NonCausalUnsupported,
    /// A bounded history window must retain at least one token.
    InvalidHistoryWindow,
    /// A rejecting history budget must allow at least one retained token.
    InvalidHistoryBudget,
    /// The declared budget cannot contain the selected history for a query.
    HistoryBudgetExceeded {
        /// Number of positions required by the semantic rule.
        required: usize,
        /// Caller-declared hard limit.
        limit: usize,
    },
}

impl From<FlatAttentionError> for NonlocalAttentionError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Flat(value)
    }
}

impl fmt::Display for NonlocalAttentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Flat(error) => write!(formatter, "FLAT validation failed: {error}"),
            Self::NonCausalUnsupported => {
                formatter.write_str("nonlocal-history-softmax revision 1 requires causal attention")
            }
            Self::InvalidHistoryWindow => {
                formatter.write_str("history window must retain at least one token")
            }
            Self::InvalidHistoryBudget => {
                formatter.write_str("history budget must allow at least one retained token")
            }
            Self::HistoryBudgetExceeded { required, limit } => write!(
                formatter,
                "selected history requires {required} tokens, exceeding explicit budget {limit}"
            ),
        }
    }
}

impl std::error::Error for NonlocalAttentionError {}

/// Deterministic scalar oracle for the first structured-history semantic.
///
/// Revision 1 is deliberately narrow:
///
/// - causal attention only;
/// - native GQA/MQA head mapping;
/// - true absolute query/key positions from [`AsymmetricGroupedAttentionShape`];
/// - identity history weighting;
/// - every retained position visited in ascending logical order;
/// - complete history or an explicit trailing-window approximation;
/// - online softmax with the same update order as FLAT's existing asymmetric
///   scalar oracle.
///
/// Complete-history/default research configuration is therefore bitwise
/// comparable to the existing causal asymmetric StandardSoftmax oracle. A
/// windowed configuration changes only which causally visible K/V rows are
/// admitted; it is always reported as an approximation.
///
/// # Errors
///
/// Fails closed on invalid shape/input values, non-causal requests, invalid
/// research configuration, or explicit budget exhaustion. No fallback to
/// StandardSoftmax or silent history truncation occurs.
pub fn forward_reference_nonlocal_history(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AsymmetricGroupedAttentionShape,
    attention_config: FlatAttentionConfig,
    history_config: NonlocalAttentionConfig,
) -> Result<NonlocalAttentionOutput, NonlocalAttentionError> {
    shape.validate()?;
    history_config.validate()?;
    if !attention_config.causal {
        return Err(NonlocalAttentionError::NonCausalUnsupported);
    }

    let q_tensor_len = shape.q_tensor_len()?;
    let kv_tensor_len = shape.kv_tensor_len()?;
    validate_input("Q", q, q_tensor_len)?;
    validate_input("K", k, kv_tensor_len)?;
    validate_input("V", v, kv_tensor_len)?;

    let scale = attention_config.resolved_scale(shape.head_dim)?;
    let group_size = shape.q_heads / shape.kv_heads;
    let q_head_stride = shape.query_len * shape.head_dim;
    let kv_head_stride = shape.kv_len * shape.head_dim;
    let mut output = vec![0.0_f32; q_tensor_len];
    let mut lse = vec![0.0_f32; shape.lse_len()?];

    for batch in 0..shape.batch {
        for q_head in 0..shape.q_heads {
            let kv_head = q_head / group_size;
            let q_bh = batch * shape.q_heads + q_head;
            let kv_bh = batch * shape.kv_heads + kv_head;
            let q_head_base = q_bh * q_head_stride;
            let kv_head_base = kv_bh * kv_head_stride;
            let lse_base = q_bh * shape.query_len;

            for query_pos in 0..shape.query_len {
                let absolute_query_pos = shape.query_position_offset + query_pos;
                let visible_end = absolute_query_pos.saturating_add(1).min(shape.kv_len);
                let history_start = match history_config.history_mode {
                    HistoryMode::Complete => 0,
                    HistoryMode::Window { max_tokens } => visible_end.saturating_sub(max_tokens),
                };
                let retained = visible_end - history_start;

                if let HistoryBudgetPolicy::RejectAbove {
                    max_retained_tokens,
                } = history_config.history_budget_policy
                {
                    if retained > max_retained_tokens {
                        return Err(NonlocalAttentionError::HistoryBudgetExceeded {
                            required: retained,
                            limit: max_retained_tokens,
                        });
                    }
                }

                let q_base = q_head_base + query_pos * shape.head_dim;
                let out_base = q_base;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0_f32;

                match history_config.history_schedule {
                    HistorySchedule::EveryToken => {
                        for key_pos in history_start..visible_end {
                            let kv_base = kv_head_base + key_pos * shape.head_dim;
                            let mut dot = 0.0_f32;
                            for dim in 0..shape.head_dim {
                                dot += q[q_base + dim] * k[kv_base + dim];
                            }

                            let score = match history_config.history_weighting {
                                HistoryWeighting::Identity => dot * scale,
                            };
                            let new_max = running_max.max(score);
                            let alpha = if running_max.is_infinite() {
                                0.0
                            } else {
                                (running_max - new_max).exp()
                            };
                            let probability_numerator = (score - new_max).exp();

                            for dim in 0..shape.head_dim {
                                output[out_base + dim] = output[out_base + dim] * alpha
                                    + probability_numerator * v[kv_base + dim];
                            }
                            running_sum = running_sum * alpha + probability_numerator;
                            running_max = new_max;
                        }
                    }
                }

                let inv_sum = running_sum.recip();
                for dim in 0..shape.head_dim {
                    output[out_base + dim] *= inv_sum;
                }
                lse[lse_base + query_pos] = running_max + running_sum.ln();
            }
        }
    }

    Ok(NonlocalAttentionOutput {
        attention: FlatAttentionOutput { output, lse },
        classification: history_config.classification(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_reference_grouped_asymmetric;

    fn deterministic_values(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index as f32 * 0.173) + phase).sin() * 0.7)
            .collect()
    }

    fn shape() -> AsymmetricGroupedAttentionShape {
        AsymmetricGroupedAttentionShape {
            batch: 2,
            q_heads: 4,
            kv_heads: 2,
            query_len: 3,
            kv_len: 6,
            head_dim: 8,
            query_position_offset: 2,
        }
    }

    #[test]
    fn complete_history_is_bitwise_identical_to_causal_asymmetric_oracle() {
        let shape = shape();
        let q = deterministic_values(shape.q_tensor_len().unwrap(), 0.1);
        let k = deterministic_values(shape.kv_tensor_len().unwrap(), 0.7);
        let v = deterministic_values(shape.kv_tensor_len().unwrap(), 1.3);
        let attention_config = FlatAttentionConfig {
            causal: true,
            softmax_scale: Some(0.625),
        };

        let expected =
            forward_reference_grouped_asymmetric(&q, &k, &v, shape, attention_config).unwrap();
        let actual = forward_reference_nonlocal_history(
            &q,
            &k,
            &v,
            shape,
            attention_config,
            NonlocalAttentionConfig::default(),
        )
        .unwrap();

        assert_eq!(actual.attention, expected);
        assert_eq!(actual.classification, HistoryClassification::Reference);
    }

    #[test]
    fn full_window_matches_reference_but_remains_classified_as_approximation() {
        let shape = shape();
        let q = deterministic_values(shape.q_tensor_len().unwrap(), 0.2);
        let k = deterministic_values(shape.kv_tensor_len().unwrap(), 0.8);
        let v = deterministic_values(shape.kv_tensor_len().unwrap(), 1.4);
        let attention_config = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };
        let reference = forward_reference_nonlocal_history(
            &q,
            &k,
            &v,
            shape,
            attention_config,
            NonlocalAttentionConfig::default(),
        )
        .unwrap();
        let bounded = forward_reference_nonlocal_history(
            &q,
            &k,
            &v,
            shape,
            attention_config,
            NonlocalAttentionConfig {
                history_mode: HistoryMode::Window {
                    max_tokens: shape.kv_len,
                },
                ..NonlocalAttentionConfig::default()
            },
        )
        .unwrap();

        assert_eq!(bounded.attention, reference.attention);
        assert_eq!(bounded.classification, HistoryClassification::Approximation);
    }

    #[test]
    fn bounded_history_changes_only_the_explicitly_retained_history() {
        let shape = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            query_len: 1,
            kv_len: 4,
            head_dim: 1,
            query_position_offset: 3,
        };
        let q = [0.0_f32];
        let k = [0.0_f32; 4];
        let v = [1.0_f32, 2.0, 3.0, 4.0];
        let attention_config = FlatAttentionConfig {
            causal: true,
            softmax_scale: Some(1.0),
        };

        let complete = forward_reference_nonlocal_history(
            &q,
            &k,
            &v,
            shape,
            attention_config,
            NonlocalAttentionConfig::default(),
        )
        .unwrap();
        let windowed = forward_reference_nonlocal_history(
            &q,
            &k,
            &v,
            shape,
            attention_config,
            NonlocalAttentionConfig {
                history_mode: HistoryMode::Window { max_tokens: 2 },
                ..NonlocalAttentionConfig::default()
            },
        )
        .unwrap();

        assert_eq!(complete.attention.output, vec![2.5]);
        assert_eq!(windowed.attention.output, vec![3.5]);
        assert_eq!(complete.attention.lse, vec![(4.0_f32).ln()]);
        assert_eq!(windowed.attention.lse, vec![(2.0_f32).ln()]);
    }

    #[test]
    fn budget_exhaustion_rejects_instead_of_truncating() {
        let shape = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            query_len: 1,
            kv_len: 4,
            head_dim: 1,
            query_position_offset: 3,
        };
        let q = [0.0_f32];
        let k = [0.0_f32; 4];
        let v = [1.0_f32; 4];
        let error = forward_reference_nonlocal_history(
            &q,
            &k,
            &v,
            shape,
            FlatAttentionConfig {
                causal: true,
                softmax_scale: Some(1.0),
            },
            NonlocalAttentionConfig {
                history_budget_policy: HistoryBudgetPolicy::RejectAbove {
                    max_retained_tokens: 3,
                },
                ..NonlocalAttentionConfig::default()
            },
        )
        .unwrap_err();

        assert_eq!(
            error,
            NonlocalAttentionError::HistoryBudgetExceeded {
                required: 4,
                limit: 3,
            }
        );
    }

    #[test]
    fn invalid_or_noncausal_configuration_fails_closed() {
        assert_eq!(
            NonlocalAttentionConfig {
                history_mode: HistoryMode::Window { max_tokens: 0 },
                ..NonlocalAttentionConfig::default()
            }
            .validate()
            .unwrap_err(),
            NonlocalAttentionError::InvalidHistoryWindow
        );
        assert_eq!(
            NonlocalAttentionConfig {
                history_budget_policy: HistoryBudgetPolicy::RejectAbove {
                    max_retained_tokens: 0,
                },
                ..NonlocalAttentionConfig::default()
            }
            .validate()
            .unwrap_err(),
            NonlocalAttentionError::InvalidHistoryBudget
        );

        let shape = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            query_len: 1,
            kv_len: 1,
            head_dim: 1,
            query_position_offset: 0,
        };
        let error = forward_reference_nonlocal_history(
            &[0.0],
            &[0.0],
            &[1.0],
            shape,
            FlatAttentionConfig {
                causal: false,
                softmax_scale: Some(1.0),
            },
            NonlocalAttentionConfig::default(),
        )
        .unwrap_err();
        assert_eq!(error, NonlocalAttentionError::NonCausalUnsupported);
    }
}
