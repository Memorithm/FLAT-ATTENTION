//! Research-only structural candidate routing for masked FLAT attention.
//!
//! FA-V888-1 defines a deterministic CSR-like host representation for per-query
//! key candidates and two scalar numerical oracles:
//! - a dense masked reference that scans every key position;
//! - a sparse reference that visits only the canonical candidate list.
//!
//! The module contains no BANC data and makes no performance claim. It remains
//! outside the stable `api::v1` surface.

use core::fmt;

use crate::{AttentionShape, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput};

/// Version of the research structural-candidate representation.
pub const STRUCTURAL_CANDIDATE_SCHEMA_VERSION: u32 = 1;

/// Canonical CSR-like candidate keys for every attention query.
///
/// Rows are ordered by `[batch, head, query_position]`. Within one row key
/// positions are strictly increasing. Empty rows are representable so upstream
/// routing can be inspected, but numerical attention rejects an empty effective
/// survivor set explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralCandidateSet {
    seq_len: usize,
    query_rows: usize,
    offsets: Vec<usize>,
    key_positions: Vec<usize>,
}

impl StructuralCandidateSet {
    /// Canonicalize per-query key rows by sorting them and rejecting duplicates.
    pub fn from_rows(
        shape: AttentionShape,
        mut rows: Vec<Vec<usize>>,
    ) -> Result<Self, StructuralRoutingError> {
        validate_shape(shape)?;
        let query_rows = shape.lse_len()?;
        if rows.len() != query_rows {
            return Err(StructuralRoutingError::RowCountMismatch {
                actual: rows.len(),
                expected: query_rows,
            });
        }

        let mut offsets = Vec::with_capacity(query_rows + 1);
        let mut key_positions = Vec::new();
        offsets.push(0);

        for (row_index, row) in rows.iter_mut().enumerate() {
            row.sort_unstable();
            for &key_position in row.iter() {
                if key_position >= shape.seq_len {
                    return Err(StructuralRoutingError::KeyOutOfRange {
                        row: row_index,
                        key_position,
                        seq_len: shape.seq_len,
                    });
                }
            }
            for pair in row.windows(2) {
                if pair[0] == pair[1] {
                    return Err(StructuralRoutingError::DuplicateKey {
                        row: row_index,
                        key_position: pair[0],
                    });
                }
            }
            key_positions.extend_from_slice(row);
            offsets.push(key_positions.len());
        }

        Ok(Self {
            seq_len: shape.seq_len,
            query_rows,
            offsets,
            key_positions,
        })
    }

    /// Construct the all-accept structural control for one attention shape.
    pub fn all(shape: AttentionShape) -> Result<Self, StructuralRoutingError> {
        validate_shape(shape)?;
        let query_rows = shape.lse_len()?;
        let row: Vec<usize> = (0..shape.seq_len).collect();
        Self::from_rows(shape, vec![row; query_rows])
    }

    /// Number of logical query rows represented by this candidate set.
    #[must_use]
    pub const fn query_rows(&self) -> usize {
        self.query_rows
    }

    /// Sequence length used when the candidate set was canonicalized.
    #[must_use]
    pub const fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Canonical offsets; length is `query_rows + 1`.
    #[must_use]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Flat canonical key-position array.
    #[must_use]
    pub fn key_positions(&self) -> &[usize] {
        &self.key_positions
    }

    /// Exact number of structurally admitted query-key pairs before causal masking.
    #[must_use]
    pub fn admitted_count(&self) -> usize {
        self.key_positions.len()
    }

    /// Canonical key positions admitted for one query row.
    pub fn row(&self, row: usize) -> Result<&[usize], StructuralRoutingError> {
        if row >= self.query_rows {
            return Err(StructuralRoutingError::RowOutOfRange {
                row,
                query_rows: self.query_rows,
            });
        }
        Ok(&self.key_positions[self.offsets[row]..self.offsets[row + 1]])
    }

    /// Test one canonical row/key membership without scanning unrelated rows.
    pub fn is_admitted(
        &self,
        row: usize,
        key_position: usize,
    ) -> Result<bool, StructuralRoutingError> {
        if key_position >= self.seq_len {
            return Err(StructuralRoutingError::KeyOutOfRange {
                row,
                key_position,
                seq_len: self.seq_len,
            });
        }
        Ok(self.row(row)?.binary_search(&key_position).is_ok())
    }

    /// Intersect structural candidates with autoregressive causal eligibility.
    ///
    /// Query position is derived from the canonical row order. The result stays
    /// canonical and may contain empty rows; numerical execution handles those
    /// rows as an explicit error rather than fabricating an attention value.
    #[must_use]
    pub fn intersect_causal(&self) -> Self {
        let mut offsets = Vec::with_capacity(self.query_rows + 1);
        let mut key_positions = Vec::new();
        offsets.push(0);

        for row in 0..self.query_rows {
            let query_position = row % self.seq_len;
            let begin = self.offsets[row];
            let end = self.offsets[row + 1];
            key_positions.extend(
                self.key_positions[begin..end]
                    .iter()
                    .copied()
                    .take_while(|&key_position| key_position <= query_position),
            );
            offsets.push(key_positions.len());
        }

        Self {
            seq_len: self.seq_len,
            query_rows: self.query_rows,
            offsets,
            key_positions,
        }
    }

    fn validate_shape(&self, shape: AttentionShape) -> Result<(), StructuralRoutingError> {
        validate_shape(shape)?;
        let query_rows = shape.lse_len()?;
        if self.seq_len != shape.seq_len || self.query_rows != query_rows {
            return Err(StructuralRoutingError::GeometryMismatch {
                set_seq_len: self.seq_len,
                shape_seq_len: shape.seq_len,
                set_query_rows: self.query_rows,
                shape_query_rows: query_rows,
            });
        }
        Ok(())
    }
}

/// Structural-routing accounting emitted by host numerical oracles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuralRoutingCounters {
    /// Query-key pairs present in the structural candidate representation.
    pub admitted_pairs: usize,
    /// Candidate pairs actually evaluated after optional causal intersection.
    pub executed_pairs: usize,
}

/// Numerical output plus exact structural-routing counters.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralAttentionOutput {
    pub attention: FlatAttentionOutput,
    pub counters: StructuralRoutingCounters,
}

/// Dense masked scalar reference.
///
/// This path scans every key position and checks structural membership. It is
/// intentionally inefficient and serves as the independent masked reference for
/// the sparse traversal.
pub fn forward_reference_structural_dense(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
    candidates: &StructuralCandidateSet,
) -> Result<StructuralAttentionOutput, StructuralRoutingError> {
    let tensor_len = validate_inputs(q, k, v, shape, config, candidates)?;
    let scale = config.resolved_scale(shape.head_dim)?;
    let mut output = vec![0.0f32; tensor_len];
    let mut lse = vec![0.0f32; shape.lse_len()?];
    let mut executed_pairs = 0usize;
    let head_stride = shape.seq_len * shape.head_dim;

    for batch in 0..shape.batch {
        for head in 0..shape.heads {
            let bh = batch * shape.heads + head;
            let head_base = bh * head_stride;
            let lse_base = bh * shape.seq_len;

            for query_pos in 0..shape.seq_len {
                let row = bh * shape.seq_len + query_pos;
                let q_base = head_base + query_pos * shape.head_dim;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;
                let mut row_executed = 0usize;

                for key_pos in 0..shape.seq_len {
                    if config.causal && key_pos > query_pos {
                        break;
                    }
                    if !candidates.is_admitted(row, key_pos)? {
                        continue;
                    }
                    update_online_attention(
                        q,
                        k,
                        v,
                        &mut output,
                        shape.head_dim,
                        q_base,
                        head_base + key_pos * shape.head_dim,
                        scale,
                        &mut running_max,
                        &mut running_sum,
                    );
                    row_executed += 1;
                }

                finalize_row(
                    &mut output,
                    &mut lse,
                    shape.head_dim,
                    q_base,
                    lse_base + query_pos,
                    row,
                    row_executed,
                    running_max,
                    running_sum,
                )?;
                executed_pairs = executed_pairs
                    .checked_add(row_executed)
                    .ok_or(StructuralRoutingError::AccountingOverflow)?;
            }
        }
    }

    Ok(StructuralAttentionOutput {
        attention: FlatAttentionOutput { output, lse },
        counters: StructuralRoutingCounters {
            admitted_pairs: candidates.admitted_count(),
            executed_pairs,
        },
    })
}

/// Sparse scalar reference that visits only structurally admitted key positions.
///
/// Candidate ordering is canonical, so for the same effective survivor set this
/// performs floating-point operations in the same key order as the dense masked
/// reference.
pub fn forward_reference_structural_sparse(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
    candidates: &StructuralCandidateSet,
) -> Result<StructuralAttentionOutput, StructuralRoutingError> {
    let tensor_len = validate_inputs(q, k, v, shape, config, candidates)?;
    let scale = config.resolved_scale(shape.head_dim)?;
    let mut output = vec![0.0f32; tensor_len];
    let mut lse = vec![0.0f32; shape.lse_len()?];
    let mut executed_pairs = 0usize;
    let head_stride = shape.seq_len * shape.head_dim;

    for batch in 0..shape.batch {
        for head in 0..shape.heads {
            let bh = batch * shape.heads + head;
            let head_base = bh * head_stride;
            let lse_base = bh * shape.seq_len;

            for query_pos in 0..shape.seq_len {
                let row = bh * shape.seq_len + query_pos;
                let q_base = head_base + query_pos * shape.head_dim;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;
                let mut row_executed = 0usize;

                for &key_pos in candidates.row(row)? {
                    if config.causal && key_pos > query_pos {
                        break;
                    }
                    update_online_attention(
                        q,
                        k,
                        v,
                        &mut output,
                        shape.head_dim,
                        q_base,
                        head_base + key_pos * shape.head_dim,
                        scale,
                        &mut running_max,
                        &mut running_sum,
                    );
                    row_executed += 1;
                }

                finalize_row(
                    &mut output,
                    &mut lse,
                    shape.head_dim,
                    q_base,
                    lse_base + query_pos,
                    row,
                    row_executed,
                    running_max,
                    running_sum,
                )?;
                executed_pairs = executed_pairs
                    .checked_add(row_executed)
                    .ok_or(StructuralRoutingError::AccountingOverflow)?;
            }
        }
    }

    Ok(StructuralAttentionOutput {
        attention: FlatAttentionOutput { output, lse },
        counters: StructuralRoutingCounters {
            admitted_pairs: candidates.admitted_count(),
            executed_pairs,
        },
    })
}

fn validate_shape(shape: AttentionShape) -> Result<(), StructuralRoutingError> {
    if shape.batch == 0 || shape.heads == 0 || shape.seq_len == 0 || shape.head_dim == 0 {
        return Err(FlatAttentionError::ZeroDimension.into());
    }
    shape.tensor_len()?;
    shape.lse_len()?;
    Ok(())
}

fn validate_inputs(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
    candidates: &StructuralCandidateSet,
) -> Result<usize, StructuralRoutingError> {
    validate_shape(shape)?;
    candidates.validate_shape(shape)?;
    let tensor_len = shape.tensor_len()?;
    validate_tensor("Q", q, tensor_len)?;
    validate_tensor("K", k, tensor_len)?;
    validate_tensor("V", v, tensor_len)?;
    config.resolved_scale(shape.head_dim)?;
    Ok(tensor_len)
}

fn validate_tensor(
    name: &'static str,
    data: &[f32],
    expected: usize,
) -> Result<(), StructuralRoutingError> {
    if data.len() != expected {
        return Err(FlatAttentionError::LengthMismatch {
            tensor: name,
            actual: data.len(),
            expected,
        }
        .into());
    }
    if let Some(index) = data.iter().position(|value| !value.is_finite()) {
        return Err(FlatAttentionError::NonFiniteInput {
            tensor: name,
            index,
        }
        .into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_online_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    head_dim: usize,
    q_base: usize,
    kv_base: usize,
    scale: f32,
    running_max: &mut f32,
    running_sum: &mut f32,
) {
    let mut dot = 0.0f32;
    for dim in 0..head_dim {
        dot += q[q_base + dim] * k[kv_base + dim];
    }
    let score = dot * scale;
    let new_max = (*running_max).max(score);
    let alpha = if (*running_max).is_infinite() {
        0.0
    } else {
        (*running_max - new_max).exp()
    };
    let probability_numerator = (score - new_max).exp();

    for dim in 0..head_dim {
        output[q_base + dim] =
            output[q_base + dim] * alpha + probability_numerator * v[kv_base + dim];
    }
    *running_sum = *running_sum * alpha + probability_numerator;
    *running_max = new_max;
}

#[allow(clippy::too_many_arguments)]
fn finalize_row(
    output: &mut [f32],
    lse: &mut [f32],
    head_dim: usize,
    out_base: usize,
    lse_index: usize,
    row: usize,
    row_executed: usize,
    running_max: f32,
    running_sum: f32,
) -> Result<(), StructuralRoutingError> {
    if row_executed == 0 {
        return Err(StructuralRoutingError::EmptyEffectiveCandidates { row });
    }
    let inv_sum = running_sum.recip();
    for dim in 0..head_dim {
        output[out_base + dim] *= inv_sum;
    }
    lse[lse_index] = running_max + running_sum.ln();
    Ok(())
}

/// Fail-closed structural-routing errors.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StructuralRoutingError {
    Attention(FlatAttentionError),
    RowCountMismatch {
        actual: usize,
        expected: usize,
    },
    RowOutOfRange {
        row: usize,
        query_rows: usize,
    },
    KeyOutOfRange {
        row: usize,
        key_position: usize,
        seq_len: usize,
    },
    DuplicateKey {
        row: usize,
        key_position: usize,
    },
    GeometryMismatch {
        set_seq_len: usize,
        shape_seq_len: usize,
        set_query_rows: usize,
        shape_query_rows: usize,
    },
    EmptyEffectiveCandidates {
        row: usize,
    },
    AccountingOverflow,
}

impl From<FlatAttentionError> for StructuralRoutingError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Attention(value)
    }
}

impl fmt::Display for StructuralRoutingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attention(error) => write!(formatter, "{error}"),
            Self::RowCountMismatch { actual, expected } => write!(
                formatter,
                "structural candidate set has {actual} query rows, expected {expected}"
            ),
            Self::RowOutOfRange { row, query_rows } => write!(
                formatter,
                "structural candidate row {row} is outside 0..{query_rows}"
            ),
            Self::KeyOutOfRange {
                row,
                key_position,
                seq_len,
            } => write!(
                formatter,
                "structural candidate key {key_position} in row {row} is outside 0..{seq_len}"
            ),
            Self::DuplicateKey { row, key_position } => write!(
                formatter,
                "structural candidate row {row} contains duplicate key {key_position}"
            ),
            Self::GeometryMismatch {
                set_seq_len,
                shape_seq_len,
                set_query_rows,
                shape_query_rows,
            } => write!(
                formatter,
                "structural candidate geometry seq/query rows {set_seq_len}/{set_query_rows} does not match attention shape {shape_seq_len}/{shape_query_rows}"
            ),
            Self::EmptyEffectiveCandidates { row } => write!(
                formatter,
                "structural candidate row {row} has no effective key after masking"
            ),
            Self::AccountingOverflow => {
                formatter.write_str("structural routing accounting overflowed")
            }
        }
    }
}

impl std::error::Error for StructuralRoutingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_reference;

    fn shape() -> AttentionShape {
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 4,
            head_dim: 2,
        }
    }

    fn tensors() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![1.0, 0.5, -0.5, 1.0, 0.25, -1.0, 1.5, 0.75],
            vec![0.5, 1.0, -1.0, 0.25, 0.75, -0.5, 1.25, 0.5],
            vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0, -2.0, 1.5],
        )
    }

    #[test]
    fn canonicalizes_rows_and_rejects_duplicates() {
        let candidates = StructuralCandidateSet::from_rows(
            shape(),
            vec![
                vec![3, 1],
                vec![2],
                vec![3, 0, 2],
                vec![1, 0],
            ],
        )
        .unwrap();
        assert_eq!(candidates.row(0).unwrap(), &[1, 3]);
        assert_eq!(candidates.row(2).unwrap(), &[0, 2, 3]);
        assert_eq!(candidates.admitted_count(), 8);

        assert!(matches!(
            StructuralCandidateSet::from_rows(
                shape(),
                vec![vec![1, 1], vec![0], vec![0], vec![0]],
            ),
            Err(StructuralRoutingError::DuplicateKey {
                row: 0,
                key_position: 1,
            })
        ));
    }

    #[test]
    fn causal_intersection_is_explicit_and_canonical() {
        let candidates = StructuralCandidateSet::all(shape()).unwrap();
        let causal = candidates.intersect_causal();
        assert_eq!(causal.row(0).unwrap(), &[0]);
        assert_eq!(causal.row(1).unwrap(), &[0, 1]);
        assert_eq!(causal.row(2).unwrap(), &[0, 1, 2]);
        assert_eq!(causal.row(3).unwrap(), &[0, 1, 2, 3]);
        assert_eq!(causal.admitted_count(), 10);
    }

    #[test]
    fn dense_masked_and_sparse_oracles_are_bit_identical() {
        let candidates = StructuralCandidateSet::from_rows(
            shape(),
            vec![
                vec![0, 2],
                vec![1, 3],
                vec![0, 1, 3],
                vec![2, 3],
            ],
        )
        .unwrap();
        let (q, k, v) = tensors();
        let config = FlatAttentionConfig::default();

        let dense =
            forward_reference_structural_dense(&q, &k, &v, shape(), config, &candidates).unwrap();
        let sparse =
            forward_reference_structural_sparse(&q, &k, &v, shape(), config, &candidates).unwrap();

        assert_eq!(dense, sparse);
        assert_eq!(sparse.counters.admitted_pairs, 9);
        assert_eq!(sparse.counters.executed_pairs, 9);
    }

    #[test]
    fn all_accept_matches_existing_flat_oracle_bit_for_bit() {
        let candidates = StructuralCandidateSet::all(shape()).unwrap();
        let (q, k, v) = tensors();
        let config = FlatAttentionConfig::default();

        let existing = forward_reference(&q, &k, &v, shape(), config).unwrap();
        let sparse =
            forward_reference_structural_sparse(&q, &k, &v, shape(), config, &candidates).unwrap();

        assert_eq!(sparse.attention, existing);
        assert_eq!(sparse.counters.admitted_pairs, 16);
        assert_eq!(sparse.counters.executed_pairs, 16);
    }

    #[test]
    fn causal_all_accept_matches_existing_flat_oracle_bit_for_bit() {
        let candidates = StructuralCandidateSet::all(shape()).unwrap();
        let (q, k, v) = tensors();
        let config = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };

        let existing = forward_reference(&q, &k, &v, shape(), config).unwrap();
        let sparse =
            forward_reference_structural_sparse(&q, &k, &v, shape(), config, &candidates).unwrap();

        assert_eq!(sparse.attention, existing);
        assert_eq!(sparse.counters.admitted_pairs, 16);
        assert_eq!(sparse.counters.executed_pairs, 10);
    }

    #[test]
    fn causal_sparse_oracle_matches_dense_masked_oracle() {
        let candidates = StructuralCandidateSet::from_rows(
            shape(),
            vec![
                vec![0, 2],
                vec![0, 1, 3],
                vec![0, 2, 3],
                vec![1, 3],
            ],
        )
        .unwrap();
        let (q, k, v) = tensors();
        let config = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };

        let dense =
            forward_reference_structural_dense(&q, &k, &v, shape(), config, &candidates).unwrap();
        let sparse =
            forward_reference_structural_sparse(&q, &k, &v, shape(), config, &candidates).unwrap();

        assert_eq!(dense, sparse);
        assert_eq!(sparse.counters.admitted_pairs, 10);
        assert_eq!(sparse.counters.executed_pairs, 7);
    }

    #[test]
    fn empty_effective_survivor_is_explicit_error() {
        let candidates = StructuralCandidateSet::from_rows(
            shape(),
            vec![
                vec![1],
                vec![0],
                vec![0],
                vec![0],
            ],
        )
        .unwrap();
        let (q, k, v) = tensors();
        let config = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };

        assert!(matches!(
            forward_reference_structural_sparse(&q, &k, &v, shape(), config, &candidates),
            Err(StructuralRoutingError::EmptyEffectiveCandidates { row: 0 })
        ));
        assert!(matches!(
            forward_reference_structural_dense(&q, &k, &v, shape(), config, &candidates),
            Err(StructuralRoutingError::EmptyEffectiveCandidates { row: 0 })
        ));
    }

    #[test]
    fn malformed_geometry_and_keys_fail_closed() {
        assert!(matches!(
            StructuralCandidateSet::from_rows(
                shape(),
                vec![vec![0], vec![0], vec![0]],
            ),
            Err(StructuralRoutingError::RowCountMismatch {
                actual: 3,
                expected: 4,
            })
        ));
        assert!(matches!(
            StructuralCandidateSet::from_rows(
                shape(),
                vec![vec![4], vec![0], vec![0], vec![0]],
            ),
            Err(StructuralRoutingError::KeyOutOfRange {
                row: 0,
                key_position: 4,
                seq_len: 4,
            })
        ));
    }
}
