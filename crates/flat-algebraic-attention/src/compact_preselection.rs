//! Research-only coordinate-projected key screening before exact attention.
//!
//! This is a deterministic ablation baseline, not a learned projection, a KV
//! replacement, a CSA2 reproduction, or a quality/performance qualification.
//! Original numerical K/V remains authoritative. The returned canonical rows
//! feed FLAT's existing structural sparse online-softmax oracle unchanged.

use core::cmp::Ordering;
use core::fmt;
use std::collections::BinaryHeap;

use flat_attention::api::research_structural_routing::{
    StructuralCandidateSet, StructuralRoutingError,
};
use flat_attention::{AttentionShape, FlatAttentionConfig, FlatAttentionError};

/// Logical work and payload counts, not allocator or physical traffic evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactPreselectionCounters {
    /// Number of explicitly retained coordinates per key.
    pub projection_dimension: usize,
    /// Temporary projected K payload only; excludes full K/V and all metadata.
    pub projected_key_payload_bytes: usize,
    /// Query-key pairs actually scored by the selector.
    pub evaluated_pairs: usize,
    /// Scalar products used in those projected scores.
    pub evaluated_score_components: usize,
    /// Number of retained canonical query-key pairs.
    pub selected_pairs: usize,
}

/// A research candidate set and its logical selector accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactPreselection {
    pub candidates: StructuralCandidateSet,
    pub counters: CompactPreselectionCounters,
}

/// Invalid input is rejected, including non-finite unprojected coordinates.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CompactPreselectionError {
    Attention(FlatAttentionError),
    Routing(StructuralRoutingError),
    EmptyProjection,
    CoordinateOutOfRange {
        coordinate: usize,
        head_dim: usize,
    },
    CoordinatesNotStrictlyIncreasing,
    NonFiniteScore {
        row: usize,
        key_position: usize,
    },
    AccountingOverflow,
}

impl fmt::Display for CompactPreselectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attention(error) => write!(f, "invalid attention input: {error}"),
            Self::Routing(error) => write!(f, "invalid structural candidates: {error}"),
            Self::EmptyProjection => write!(f, "projection must retain at least one coordinate"),
            Self::CoordinateOutOfRange {
                coordinate,
                head_dim,
            } => write!(
                f,
                "projection coordinate {coordinate} is outside 0..{head_dim}"
            ),
            Self::CoordinatesNotStrictlyIncreasing => {
                write!(
                    f,
                    "projection coordinates must be unique and strictly increasing"
                )
            }
            Self::NonFiniteScore { row, key_position } => {
                write!(
                    f,
                    "non-finite projected score for row {row}, key {key_position}"
                )
            }
            Self::AccountingOverflow => write!(f, "selector accounting overflows usize"),
        }
    }
}

impl std::error::Error for CompactPreselectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Attention(error) => Some(error),
            Self::Routing(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FlatAttentionError> for CompactPreselectionError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Attention(value)
    }
}

impl From<StructuralRoutingError> for CompactPreselectionError {
    fn from(value: StructuralRoutingError) -> Self {
        Self::Routing(value)
    }
}

#[derive(Debug, Clone, Copy)]
struct ScoredKey {
    score: f64,
    key_position: usize,
}

// The heap exposes the worst retained key: lowest score, then highest key ID.
// total_cmp also keeps Eq/Ord consistent for signed zeros.
impl Ord for ScoredKey {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.key_position.cmp(&other.key_position))
    }
}

impl PartialOrd for ScoredKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ScoredKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ScoredKey {}

/// Screen keys with an explicit coordinate projection and a per-query budget.
///
/// Scores use f64 accumulation and the original head-dimension softmax scale,
/// not the projected dimension. Causality is applied BEFORE ranking, so future
/// keys cannot displace eligible keys. Equal scores prefer the smallest original
/// key position. Output rows preserve original positions in canonical CSR form.
///
/// A zero budget produces empty rows and evaluates no scores; the existing
/// numerical oracle rejects those rows rather than dividing by zero or silently
/// falling back. There is no hidden repair or policy for choosing coordinates.
///
/// Auxiliary storage is O(B*H*N*R + B*H*N*min(budget,N) + min(budget,N)),
/// including temporary projected K and returned candidates. No score matrix or
/// dense Boolean mask is constructed. At budget=N the candidate metadata itself
/// is quadratic; this all-accept control is not a sparse-memory claim.
///
/// Q/K are validated in full. Building and validating the projected snapshot
/// reads original K; projected payload bytes must NOT be reported as total KV
/// memory savings, avoided physical traffic, or an end-to-end improvement.
/// RoPE, position bias, cross-layer reuse, GPU execution and learned projections
/// are outside this initial host-only screen.
pub fn compact_preselect(
    q: &[f32],
    k: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
    coordinates: &[usize],
    budget: usize,
) -> Result<CompactPreselection, CompactPreselectionError> {
    if shape.batch == 0 || shape.heads == 0 || shape.seq_len == 0 || shape.head_dim == 0 {
        return Err(FlatAttentionError::ZeroDimension.into());
    }
    let tensor_len = shape.tensor_len()?;
    validate_tensor("Q", q, tensor_len)?;
    validate_tensor("K", k, tensor_len)?;
    let scale = f64::from(config.resolved_scale(shape.head_dim)?);
    if coordinates.is_empty() {
        return Err(CompactPreselectionError::EmptyProjection);
    }
    for &coordinate in coordinates {
        if coordinate >= shape.head_dim {
            return Err(CompactPreselectionError::CoordinateOutOfRange {
                coordinate,
                head_dim: shape.head_dim,
            });
        }
    }
    if coordinates.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(CompactPreselectionError::CoordinatesNotStrictlyIncreasing);
    }
    let projection_dimension = coordinates.len();
    let projected_len = shape
        .lse_len()?
        .checked_mul(projection_dimension)
        .ok_or(CompactPreselectionError::AccountingOverflow)?;
    let projected_key_payload_bytes = projected_len
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(CompactPreselectionError::AccountingOverflow)?;
    let mut projected_k = Vec::with_capacity(projected_len);
    for key in k.chunks_exact(shape.head_dim) {
        for &coordinate in coordinates {
            projected_k.push(key[coordinate]);
        }
    }
    let mut rows = Vec::with_capacity(shape.lse_len()?);
    let mut evaluated_pairs = 0usize;
    for (row, query) in q.chunks_exact(shape.head_dim).enumerate() {
        let query_position = row % shape.seq_len;
        let eligible = if config.causal {
            query_position + 1
        } else {
            shape.seq_len
        };
        let limit = budget.min(eligible);
        let mut best = BinaryHeap::<ScoredKey>::with_capacity(limit);
        if limit != 0 {
            evaluated_pairs = evaluated_pairs
                .checked_add(eligible)
                .ok_or(CompactPreselectionError::AccountingOverflow)?;
            let head_base = (row / shape.seq_len) * shape.seq_len * projection_dimension;
            for key_position in 0..eligible {
                let begin = head_base + key_position * projection_dimension;
                let key = &projected_k[begin..begin + projection_dimension];
                let score = coordinates
                    .iter()
                    .zip(key)
                    .map(|(&coordinate, &value)| f64::from(query[coordinate]) * f64::from(value))
                    .sum::<f64>()
                    * scale;
                if !score.is_finite() {
                    return Err(CompactPreselectionError::NonFiniteScore { row, key_position });
                }
                let candidate = ScoredKey {
                    score: if score == 0.0 { 0.0 } else { score },
                    key_position,
                };
                if best.len() < limit {
                    best.push(candidate);
                } else if best.peek().is_some_and(|worst| candidate < *worst) {
                    best.pop();
                    best.push(candidate);
                }
            }
        }
        rows.push(
            best.into_iter()
                .map(|candidate| candidate.key_position)
                .collect(),
        );
    }
    let candidates = StructuralCandidateSet::from_rows(shape, rows)?;
    let evaluated_score_components = evaluated_pairs
        .checked_mul(projection_dimension)
        .ok_or(CompactPreselectionError::AccountingOverflow)?;
    let selected_pairs = candidates.admitted_count();
    Ok(CompactPreselection {
        candidates,
        counters: CompactPreselectionCounters {
            projection_dimension,
            projected_key_payload_bytes,
            evaluated_pairs,
            evaluated_score_components,
            selected_pairs,
        },
    })
}

fn validate_tensor(
    tensor: &'static str,
    data: &[f32],
    expected: usize,
) -> Result<(), CompactPreselectionError> {
    if data.len() != expected {
        return Err(FlatAttentionError::LengthMismatch {
            tensor,
            actual: data.len(),
            expected,
        }
        .into());
    }
    if let Some(index) = data.iter().position(|value| !value.is_finite()) {
        return Err(FlatAttentionError::NonFiniteInput { tensor, index }.into());
    }
    Ok(())
}
