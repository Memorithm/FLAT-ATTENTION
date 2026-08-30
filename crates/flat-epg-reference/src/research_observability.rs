//! Research-only observability for the scalar EPG oracle.
//!
//! This module deliberately does not alter [`crate::forward_reference_grouped_epg`].
//! It provides a separate opt-in execution surface for mechanistic diagnostics
//! and controlled interventions. Diagnostics are accumulated from streaming
//! sufficient statistics; no dense score or probability matrix is materialized.

use crate::{rotation::epg_dot, EpgEmbeddingConfig, EpgError};
use flat_attention::{FlatAttentionConfig, FlatAttentionOutput, GroupedAttentionShape};

/// Stable semantic provenance carried by research traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResearchSemanticIdentity {
    /// Stable semantic family slug.
    pub family: &'static str,
    /// Stable semantic rule slug.
    pub name: &'static str,
    /// Semantic rule revision.
    pub revision: u32,
}

const STANDARD_SOFTMAX_IDENTITY: ResearchSemanticIdentity = ResearchSemanticIdentity {
    family: "standard-softmax",
    name: "standard-softmax",
    revision: 1,
};

/// One logical score/value interaction in the research oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContributionObservation {
    /// Batch index.
    pub batch: usize,
    /// Query-head index.
    pub query_head: usize,
    /// K/V-head index selected by GQA/MQA grouping.
    pub kv_head: usize,
    /// Query token position.
    pub query_position: usize,
    /// Key/value token position.
    pub key_position: usize,
    /// Pre-intervention scaled score produced by the EPG dot product.
    pub score: f32,
    /// Online-softmax maximum before this contribution is applied.
    pub running_max_before: f32,
    /// Online-softmax normalizer before this contribution is applied.
    pub running_sum_before: f32,
}

/// Streaming per-query controls computed without materializing an `N x N` row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryDiagnostics {
    /// Batch index.
    pub batch: usize,
    /// Query-head index.
    pub query_head: usize,
    /// K/V-head index selected by GQA/MQA grouping.
    pub kv_head: usize,
    /// Query token position.
    pub query_position: usize,
    /// Number of causally/admissibly visible contributions.
    pub visible_contributions: usize,
    /// Shannon entropy of the post-intervention softmax row.
    pub entropy: f32,
    /// Entropy divided by `ln(visible_contributions)` when at least two entries exist.
    pub normalized_entropy: f32,
    /// Largest post-intervention softmax weight.
    pub max_weight: f32,
    /// Sum of squared post-intervention softmax weights.
    pub l2_concentration: f32,
    /// `exp(entropy)`, an entropy-derived effective support control.
    pub effective_support: f32,
    /// Log-sum-exp produced by the row.
    pub lse: f32,
    /// L2 norm of the completed output row.
    pub output_l2: f32,
}

/// Bounded research event retained for replay/diagnostics.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ResearchEvent {
    /// One pre-intervention score/value interaction.
    Contribution(ContributionObservation),
    /// One completed query row with streaming diagnostics.
    QueryComplete(QueryDiagnostics),
}

/// Explicit intervention applied only by the observed research oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InterventionDecision {
    /// Preserve the reference score and value contribution.
    Keep,
    /// Replace the current scaled score by zero before softmax accumulation.
    ZeroScore,
    /// Keep the score/normalization contribution but replace the current V row by zero.
    ZeroValue,
    /// Apply both `ZeroScore` and `ZeroValue` to the current interaction.
    ZeroScoreAndValue,
}

/// Hook for controlled research interventions and downstream observations.
///
/// The production/reference EPG path never constructs or calls this trait.
pub trait ResearchObserver {
    /// Inspect one pre-intervention contribution and choose an explicit action.
    fn on_contribution(&mut self, _observation: &ContributionObservation) -> InterventionDecision {
        InterventionDecision::Keep
    }

    /// Observe one completed post-intervention query row.
    fn on_query_complete(&mut self, _diagnostics: &QueryDiagnostics) {}
}

/// Observer that preserves the reference computation exactly.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoIntervention;

impl ResearchObserver for NoIntervention {}

/// Bounded in-memory trace for research execution.
///
/// Once `max_events` is reached, execution continues and additional events are
/// counted in `dropped_events`; the trace never grows beyond the declared bound.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedResearchTrace {
    semantic: ResearchSemanticIdentity,
    max_events: usize,
    events: Vec<ResearchEvent>,
    dropped_events: usize,
}

impl BoundedResearchTrace {
    /// Construct a trace for the StandardSoftmax weighting rule used by EPG.
    #[must_use]
    pub fn new(max_events: usize) -> Self {
        Self {
            semantic: STANDARD_SOFTMAX_IDENTITY,
            max_events,
            events: Vec::with_capacity(max_events.min(1024)),
            dropped_events: 0,
        }
    }

    /// Semantic identity associated with the observed weighting rule.
    #[must_use]
    pub const fn semantic(&self) -> ResearchSemanticIdentity {
        self.semantic
    }

    /// Maximum retained event count.
    #[must_use]
    pub const fn max_events(&self) -> usize {
        self.max_events
    }

    /// Retained events in deterministic execution order.
    #[must_use]
    pub fn events(&self) -> &[ResearchEvent] {
        &self.events
    }

    /// Number of events omitted after the retention bound was reached.
    #[must_use]
    pub const fn dropped_events(&self) -> usize {
        self.dropped_events
    }

    fn record(&mut self, event: ResearchEvent) {
        if self.events.len() < self.max_events {
            self.events.push(event);
        } else {
            self.dropped_events = self.dropped_events.saturating_add(1);
        }
    }
}

fn validate_input(name: &'static str, data: &[f32], expected: usize) -> Result<(), EpgError> {
    if data.len() != expected {
        return Err(EpgError::LengthMismatch {
            tensor: name,
            actual: data.len(),
            expected,
        });
    }
    if let Some(index) = data.iter().position(|x| !x.is_finite()) {
        return Err(EpgError::NonFiniteInput {
            tensor: name,
            index,
        });
    }
    Ok(())
}

/// Deterministic scalar EPG oracle with bounded research observability.
///
/// This function is intentionally separate from
/// [`crate::forward_reference_grouped_epg`]. With [`NoIntervention`] it follows
/// the same score/value update order while additionally accumulating diagnostic
/// sufficient statistics. Other observers can apply explicit score/value
/// ablations without changing the production/reference route.
///
/// No dense score/probability matrix is created. Trace memory is bounded by the
/// caller-provided [`BoundedResearchTrace`].
///
/// # Errors
///
/// Propagates the same shape, input, geometry and numerical contract errors as
/// the uninstrumented scalar EPG oracle.
pub fn forward_reference_grouped_epg_observed<O: ResearchObserver>(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    epg: EpgEmbeddingConfig,
    observer: &mut O,
    trace: &mut BoundedResearchTrace,
) -> Result<FlatAttentionOutput, EpgError> {
    let group_size = shape.group_size()?;
    epg.validate(shape.head_dim, shape.seq_len)?;
    let q_tensor_len = shape.q_tensor_len()?;
    let kv_tensor_len = shape.kv_tensor_len()?;
    let lse_len = shape.lse_len()?;
    validate_input("Q", q, q_tensor_len)?;
    validate_input("K", k, kv_tensor_len)?;
    validate_input("V", v, kv_tensor_len)?;
    let scale = config.resolved_scale(shape.head_dim)?;

    let mut output = vec![0.0f32; q_tensor_len];
    let mut lse = vec![0.0f32; lse_len];
    let head_stride = shape.seq_len * shape.head_dim;

    for batch in 0..shape.batch {
        for q_head in 0..shape.q_heads {
            let kv_head = q_head / group_size;
            let q_bh = batch * shape.q_heads + q_head;
            let kv_bh = batch * shape.kv_heads + kv_head;
            let q_head_base = q_bh * head_stride;
            let kv_head_base = kv_bh * head_stride;
            let lse_base = q_bh * shape.seq_len;

            for query_pos in 0..shape.seq_len {
                let q_base = q_head_base + query_pos * shape.head_dim;
                let query_position = epg.resolve_position(query_pos)?;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;
                let mut weighted_score_sum = 0.0f32;
                let mut squared_numerator_sum = 0.0f32;
                let mut visible_contributions = 0usize;

                for key_pos in 0..shape.seq_len {
                    if config.causal && key_pos > query_pos {
                        break;
                    }
                    let kv_base = kv_head_base + key_pos * shape.head_dim;
                    let key_position = epg.resolve_position(key_pos)?;
                    let dot = epg_dot(
                        &q[q_base..q_base + shape.head_dim],
                        &k[kv_base..kv_base + shape.head_dim],
                        shape.head_dim,
                        query_position,
                        key_position,
                        epg,
                    )?;
                    let score = dot * scale;
                    let observation = ContributionObservation {
                        batch,
                        query_head: q_head,
                        kv_head,
                        query_position: query_pos,
                        key_position: key_pos,
                        score,
                        running_max_before: running_max,
                        running_sum_before: running_sum,
                    };
                    trace.record(ResearchEvent::Contribution(observation));
                    let intervention = observer.on_contribution(&observation);
                    let effective_score = match intervention {
                        InterventionDecision::Keep | InterventionDecision::ZeroValue => score,
                        InterventionDecision::ZeroScore
                        | InterventionDecision::ZeroScoreAndValue => 0.0,
                    };
                    let zero_value = matches!(
                        intervention,
                        InterventionDecision::ZeroValue | InterventionDecision::ZeroScoreAndValue
                    );

                    let new_max = running_max.max(effective_score);
                    let alpha = if running_max.is_infinite() {
                        0.0
                    } else {
                        (running_max - new_max).exp()
                    };
                    let numerator = (effective_score - new_max).exp();

                    for dim in 0..shape.head_dim {
                        let value = if zero_value { 0.0 } else { v[kv_base + dim] };
                        output[q_base + dim] = output[q_base + dim] * alpha + numerator * value;
                    }
                    running_sum = running_sum * alpha + numerator;
                    weighted_score_sum = weighted_score_sum * alpha + numerator * effective_score;
                    squared_numerator_sum =
                        squared_numerator_sum * alpha * alpha + numerator * numerator;
                    running_max = new_max;
                    visible_contributions = visible_contributions.saturating_add(1);
                }

                let inv_sum = running_sum.recip();
                let mut output_l2_squared = 0.0f32;
                for dim in 0..shape.head_dim {
                    output[q_base + dim] *= inv_sum;
                    output_l2_squared += output[q_base + dim] * output[q_base + dim];
                }
                let row_lse = running_max + running_sum.ln();
                lse[lse_base + query_pos] = row_lse;

                let expected_score = weighted_score_sum * inv_sum;
                let entropy = row_lse - expected_score;
                let normalized_entropy = if visible_contributions > 1 {
                    entropy / (visible_contributions as f32).ln()
                } else {
                    0.0
                };
                let diagnostics = QueryDiagnostics {
                    batch,
                    query_head: q_head,
                    kv_head,
                    query_position: query_pos,
                    visible_contributions,
                    entropy,
                    normalized_entropy,
                    max_weight: inv_sum,
                    l2_concentration: squared_numerator_sum * inv_sum * inv_sum,
                    effective_support: entropy.exp(),
                    lse: row_lse,
                    output_l2: output_l2_squared.sqrt(),
                };
                trace.record(ResearchEvent::QueryComplete(diagnostics));
                observer.on_query_complete(&diagnostics);
            }
        }
    }

    Ok(FlatAttentionOutput { output, lse })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::So4Geometry;

    fn fixture(shape: GroupedAttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_len = shape.q_tensor_len().unwrap();
        let kv_len = shape.kv_tensor_len().unwrap();
        let q = (0..q_len)
            .map(|i| ((i * 17 + 3) % 101) as f32 / 53.0 - 0.9)
            .collect();
        let k = (0..kv_len)
            .map(|i| ((i * 29 + 7) % 103) as f32 / 59.0 - 0.8)
            .collect();
        let v = (0..kv_len)
            .map(|i| ((i * 11 + 5) % 97) as f32 / 47.0 - 1.0)
            .collect();
        (q, k, v)
    }

    #[test]
    fn no_intervention_is_bitwise_identical_to_uninstrumented_oracle() {
        let shape = GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 5,
            head_dim: 8,
        };
        let (q, k, v) = fixture(shape);
        let config = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };
        let epg = EpgEmbeddingConfig::hybrid_so4(10_000.0, 3, 8, So4Geometry::Biplanar).unwrap();
        let expected =
            crate::forward_reference_grouped_epg(&q, &k, &v, shape, config, epg).unwrap();
        let mut observer = NoIntervention;
        let mut trace = BoundedResearchTrace::new(256);
        let actual = forward_reference_grouped_epg_observed(
            &q,
            &k,
            &v,
            shape,
            config,
            epg,
            &mut observer,
            &mut trace,
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(trace.semantic().name, "standard-softmax");
        assert_eq!(trace.semantic().revision, 1);
    }

    #[test]
    fn trace_retention_is_strictly_bounded() {
        let shape = GroupedAttentionShape {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            seq_len: 3,
            head_dim: 2,
        };
        let (q, k, v) = fixture(shape);
        let config = FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(1.0),
        };
        let epg = EpgEmbeddingConfig::so2(10_000.0, 0).unwrap();
        let mut observer = NoIntervention;
        let mut trace = BoundedResearchTrace::new(2);
        forward_reference_grouped_epg_observed(
            &q,
            &k,
            &v,
            shape,
            config,
            epg,
            &mut observer,
            &mut trace,
        )
        .unwrap();

        assert_eq!(trace.events().len(), 2);
        assert!(trace.dropped_events() > 0);
    }

    #[test]
    fn uniform_scores_produce_analytic_streaming_controls() {
        let shape = GroupedAttentionShape {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            seq_len: 2,
            head_dim: 2,
        };
        let q = vec![0.0; shape.q_tensor_len().unwrap()];
        let k = vec![0.0; shape.kv_tensor_len().unwrap()];
        let v = vec![1.0; shape.kv_tensor_len().unwrap()];
        let config = FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(1.0),
        };
        let epg = EpgEmbeddingConfig::so2(10_000.0, 0).unwrap();
        let mut observer = NoIntervention;
        let mut trace = BoundedResearchTrace::new(32);
        forward_reference_grouped_epg_observed(
            &q,
            &k,
            &v,
            shape,
            config,
            epg,
            &mut observer,
            &mut trace,
        )
        .unwrap();

        let diagnostics = trace
            .events()
            .iter()
            .find_map(|event| match event {
                ResearchEvent::QueryComplete(diagnostics) => Some(diagnostics),
                ResearchEvent::Contribution(_) => None,
            })
            .unwrap();
        let ln2 = 2.0f32.ln();
        assert!((diagnostics.entropy - ln2).abs() <= 1e-6);
        assert!((diagnostics.normalized_entropy - 1.0).abs() <= 1e-6);
        assert!((diagnostics.max_weight - 0.5).abs() <= 1e-6);
        assert!((diagnostics.l2_concentration - 0.5).abs() <= 1e-6);
        assert!((diagnostics.effective_support - 2.0).abs() <= 1e-6);
    }

    struct ZeroFirstValue;

    impl ResearchObserver for ZeroFirstValue {
        fn on_contribution(
            &mut self,
            observation: &ContributionObservation,
        ) -> InterventionDecision {
            if observation.key_position == 0 {
                InterventionDecision::ZeroValue
            } else {
                InterventionDecision::Keep
            }
        }
    }

    #[test]
    fn value_intervention_is_explicit_and_research_only() {
        let shape = GroupedAttentionShape {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            seq_len: 2,
            head_dim: 2,
        };
        let q = vec![0.0; shape.q_tensor_len().unwrap()];
        let k = vec![0.0; shape.kv_tensor_len().unwrap()];
        let v = vec![1.0, 1.0, 3.0, 3.0];
        let config = FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(1.0),
        };
        let epg = EpgEmbeddingConfig::so2(10_000.0, 0).unwrap();
        let baseline =
            crate::forward_reference_grouped_epg(&q, &k, &v, shape, config, epg).unwrap();
        let mut observer = ZeroFirstValue;
        let mut trace = BoundedResearchTrace::new(32);
        let intervened = forward_reference_grouped_epg_observed(
            &q,
            &k,
            &v,
            shape,
            config,
            epg,
            &mut observer,
            &mut trace,
        )
        .unwrap();

        assert_eq!(baseline.output, vec![2.0, 2.0, 2.0, 2.0]);
        assert_eq!(intervened.output, vec![1.5, 1.5, 1.5, 1.5]);
    }
}
