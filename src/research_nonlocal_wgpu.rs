//! Research-only WGPU execution adapter for the qualified nonlocal semantic.
//!
//! This module adds no shader, kernel route, autotune registration, or
//! production fallback. It reuses the already-qualified portable grouped WGPU
//! implementation only for the exact subset of `nonlocal-history-softmax@1`
//! that is mathematically identical to equal-length causal grouped softmax:
//! complete history, every-token scheduling, identity weighting, no semantic
//! history budget, and zero absolute query offset.

use core::fmt;

use crate::{
    api::research_nonlocal::{
        HistoryBudgetPolicy, HistoryMode, HistorySchedule, HistoryWeighting,
        NonlocalAttentionConfig,
    },
    AsymmetricGroupedAttentionShape, FlatAttentionConfig, FlatAttentionOutput,
    GroupedAttentionShape, WgpuFlatAttentionError, WgpuGroupedAttention,
};

/// Typed rejection or backend failure for the first nonlocal WGPU candidate.
#[derive(Debug)]
#[non_exhaustive]
pub enum NonlocalWgpuCandidateError {
    /// Revision 1 of the research semantic is causal-only.
    NonCausalUnsupported,
    /// The first WGPU candidate covers reference/complete history only.
    HistoryModeUnsupported,
    /// Semantic-side history budgets are not implemented by this candidate.
    HistoryBudgetUnsupported,
    /// The reused grouped kernel requires equal query and K/V lengths.
    AsymmetricLengthUnsupported { query_len: usize, kv_len: usize },
    /// The reused grouped kernel has no absolute-query-offset parameter.
    QueryPositionOffsetUnsupported { query_position_offset: usize },
    /// Existing grouped-WGPU validation or execution failed.
    Wgpu(WgpuFlatAttentionError),
}

impl fmt::Display for NonlocalWgpuCandidateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonCausalUnsupported => {
                formatter.write_str("nonlocal WGPU candidate requires causal attention")
            }
            Self::HistoryModeUnsupported => {
                formatter.write_str("nonlocal WGPU candidate supports complete history only")
            }
            Self::HistoryBudgetUnsupported => formatter.write_str(
                "nonlocal WGPU candidate does not implement semantic history budgets",
            ),
            Self::AsymmetricLengthUnsupported { query_len, kv_len } => write!(
                formatter,
                "nonlocal WGPU candidate requires query_len == kv_len, got {query_len} and {kv_len}"
            ),
            Self::QueryPositionOffsetUnsupported {
                query_position_offset,
            } => write!(
                formatter,
                "nonlocal WGPU candidate requires query_position_offset == 0, got {query_position_offset}"
            ),
            Self::Wgpu(error) => write!(formatter, "nonlocal WGPU candidate failed: {error}"),
        }
    }
}

impl std::error::Error for NonlocalWgpuCandidateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Wgpu(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WgpuFlatAttentionError> for NonlocalWgpuCandidateError {
    fn from(value: WgpuFlatAttentionError) -> Self {
        Self::Wgpu(value)
    }
}

/// Validated geometry for the first research WGPU candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonlocalWgpuReferencePlan {
    grouped_shape: GroupedAttentionShape,
}

impl NonlocalWgpuReferencePlan {
    /// Validate exact representability by the existing grouped WGPU kernel.
    ///
    /// # Errors
    ///
    /// Returns a typed rejection for every unsupported semantic or geometry.
    pub fn new(
        shape: AsymmetricGroupedAttentionShape,
        attention: FlatAttentionConfig,
        history: NonlocalAttentionConfig,
    ) -> Result<Self, NonlocalWgpuCandidateError> {
        if !attention.causal {
            return Err(NonlocalWgpuCandidateError::NonCausalUnsupported);
        }
        if history.history_mode != HistoryMode::Complete
            || history.history_schedule != HistorySchedule::EveryToken
            || history.history_weighting != HistoryWeighting::Identity
        {
            return Err(NonlocalWgpuCandidateError::HistoryModeUnsupported);
        }
        if history.history_budget_policy != HistoryBudgetPolicy::Unlimited {
            return Err(NonlocalWgpuCandidateError::HistoryBudgetUnsupported);
        }
        if shape.query_len != shape.kv_len {
            return Err(NonlocalWgpuCandidateError::AsymmetricLengthUnsupported {
                query_len: shape.query_len,
                kv_len: shape.kv_len,
            });
        }
        if shape.query_position_offset != 0 {
            return Err(NonlocalWgpuCandidateError::QueryPositionOffsetUnsupported {
                query_position_offset: shape.query_position_offset,
            });
        }

        Ok(Self {
            grouped_shape: GroupedAttentionShape {
                batch: shape.batch,
                q_heads: shape.q_heads,
                kv_heads: shape.kv_heads,
                seq_len: shape.query_len,
                head_dim: shape.head_dim,
            },
        })
    }

    /// Equal-length grouped geometry delegated to WGPU.
    #[must_use]
    pub const fn grouped_shape(self) -> GroupedAttentionShape {
        self.grouped_shape
    }
}

/// Opt-in WGPU candidate for the reference subset of
/// `nonlocal-history-softmax@1`.
#[derive(Debug, Clone)]
pub struct WgpuNonlocalHistoryReferenceCandidate {
    grouped: WgpuGroupedAttention,
}

impl WgpuNonlocalHistoryReferenceCandidate {
    /// Create the existing portable grouped WGPU execution context.
    ///
    /// # Errors
    ///
    /// Propagates adapter/device/pipeline failures without fallback.
    pub fn new() -> Result<Self, NonlocalWgpuCandidateError> {
        Ok(Self {
            grouped: WgpuGroupedAttention::new()?,
        })
    }

    /// Name of the physical WGPU adapter used for qualification evidence.
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        self.grouped.adapter_name()
    }

    /// Execute one exact-reference research request on WGPU.
    ///
    /// Unsupported semantics fail before delegation. There is no semantic or
    /// backend fallback.
    ///
    /// # Errors
    ///
    /// Returns a typed semantic-support rejection or the existing WGPU error.
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: AsymmetricGroupedAttentionShape,
        attention: FlatAttentionConfig,
        history: NonlocalAttentionConfig,
    ) -> Result<FlatAttentionOutput, NonlocalWgpuCandidateError> {
        let plan = NonlocalWgpuReferencePlan::new(shape, attention, history)?;
        Ok(self
            .grouped
            .forward(q, k, v, plan.grouped_shape(), attention)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::research_nonlocal::forward_reference_nonlocal_history;

    const ATOL: f32 = 6.0e-5;
    const RTOL: f32 = 6.0e-4;

    fn shape() -> AsymmetricGroupedAttentionShape {
        AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            query_len: 7,
            kv_len: 7,
            head_dim: 32,
            query_position_offset: 0,
        }
    }

    fn attention() -> FlatAttentionConfig {
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        }
    }

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.021 + phase;
                x.sin() * 2.0 + (x * 0.73).cos() * 0.35
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = ATOL + RTOL * expected.abs();
            let error = (actual - expected).abs();
            assert!(
                actual.is_finite() && error <= tolerance,
                "index={index} actual={actual} expected={expected} error={error} tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn exact_reference_subset_maps_to_grouped_geometry() {
        let plan = NonlocalWgpuReferencePlan::new(
            shape(),
            attention(),
            NonlocalAttentionConfig::default(),
        )
        .unwrap();
        assert_eq!(
            plan.grouped_shape(),
            GroupedAttentionShape {
                batch: 1,
                q_heads: 4,
                kv_heads: 2,
                seq_len: 7,
                head_dim: 32,
            }
        );
    }

    #[test]
    fn unsupported_reductions_and_geometry_fail_before_dispatch() {
        let window = NonlocalAttentionConfig {
            history_mode: HistoryMode::Window { max_tokens: 3 },
            ..NonlocalAttentionConfig::default()
        };
        assert!(matches!(
            NonlocalWgpuReferencePlan::new(shape(), attention(), window),
            Err(NonlocalWgpuCandidateError::HistoryModeUnsupported)
        ));

        let budget = NonlocalAttentionConfig {
            history_budget_policy: HistoryBudgetPolicy::RejectAbove {
                max_retained_tokens: 4,
            },
            ..NonlocalAttentionConfig::default()
        };
        assert!(matches!(
            NonlocalWgpuReferencePlan::new(shape(), attention(), budget),
            Err(NonlocalWgpuCandidateError::HistoryBudgetUnsupported)
        ));

        let mut asymmetric = shape();
        asymmetric.query_len = 1;
        assert!(matches!(
            NonlocalWgpuReferencePlan::new(
                asymmetric,
                attention(),
                NonlocalAttentionConfig::default()
            ),
            Err(NonlocalWgpuCandidateError::AsymmetricLengthUnsupported { .. })
        ));

        let mut offset = shape();
        offset.query_position_offset = 1;
        assert!(matches!(
            NonlocalWgpuReferencePlan::new(offset, attention(), NonlocalAttentionConfig::default()),
            Err(NonlocalWgpuCandidateError::QueryPositionOffsetUnsupported { .. })
        ));

        assert!(matches!(
            NonlocalWgpuReferencePlan::new(
                shape(),
                FlatAttentionConfig::default(),
                NonlocalAttentionConfig::default()
            ),
            Err(NonlocalWgpuCandidateError::NonCausalUnsupported)
        ));
    }

    #[test]
    fn wgpu_candidate_matches_nonlocal_scalar_oracle() {
        let candidate = match WgpuNonlocalHistoryReferenceCandidate::new() {
            Ok(candidate) => candidate,
            Err(NonlocalWgpuCandidateError::Wgpu(WgpuFlatAttentionError::Unavailable))
                if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
            {
                eprintln!("WGPU adapter unavailable; optional nonlocal candidate test skipped");
                return;
            }
            Err(error) => panic!("nonlocal WGPU candidate creation failed: {error}"),
        };
        eprintln!("nonlocal WGPU adapter: {}", candidate.adapter_name());

        let shape = shape();
        let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
        let k = fixture(shape.kv_tensor_len().unwrap(), 0.9);
        let v = fixture(shape.kv_tensor_len().unwrap(), 1.7);
        let history = NonlocalAttentionConfig::default();
        let expected =
            forward_reference_nonlocal_history(&q, &k, &v, shape, attention(), history).unwrap();
        let actual = candidate
            .forward(&q, &k, &v, shape, attention(), history)
            .unwrap();

        assert_close(&actual.output, &expected.attention.output);
        assert_close(&actual.lse, &expected.attention.lse);
    }
}
