//! MAA-14d Gate-A exact device representation for structural candidates.
//!
//! This module converts the canonical host StructuralCandidateSet into two u32
//! arrays suitable for WGPU storage buffers. It does not execute a shader and
//! makes no performance claim. Exact candidate identity is the only authority.

use core::fmt;

use crate::{
    api::research_structural_routing::{StructuralCandidateSet, StructuralRoutingError},
    fingerprint::fnv1a64,
    AttentionShape,
};

/// Version of the structural device-buffer representation.
pub const STRUCTURAL_DEVICE_SCHEMA_VERSION: u32 = 1;

/// Exact device-facing structural candidate buffers.
///
/// Offsets retain the host CSR row partition and key positions retain canonical
/// ascending per-row order. No density-only or page-level approximation is
/// permitted by this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralDevicePlan {
    seq_len: u32,
    query_rows: u32,
    offsets: Vec<u32>,
    key_positions: Vec<u32>,
}

impl StructuralDevicePlan {
    /// Convert one canonical host candidate set into the portable WGSL u32
    /// index space.
    pub fn from_candidates(
        candidates: &StructuralCandidateSet,
    ) -> Result<Self, StructuralDevicePlanError> {
        let seq_len = checked_u32(candidates.seq_len(), "seq_len")?;
        let query_rows = checked_u32(candidates.query_rows(), "query_rows")?;
        let offsets = candidates
            .offsets()
            .iter()
            .copied()
            .map(|value| checked_u32(value, "offset"))
            .collect::<Result<Vec<_>, _>>()?;
        let key_positions = candidates
            .key_positions()
            .iter()
            .copied()
            .map(|value| checked_u32(value, "key_position"))
            .collect::<Result<Vec<_>, _>>()?;

        let plan = Self {
            seq_len,
            query_rows,
            offsets,
            key_positions,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Revalidate one raw device representation before it can be used for a
    /// WGPU upload or reconstructed into host semantics.
    pub fn from_u32_parts(
        seq_len: u32,
        query_rows: u32,
        offsets: Vec<u32>,
        key_positions: Vec<u32>,
    ) -> Result<Self, StructuralDevicePlanError> {
        let plan = Self {
            seq_len,
            query_rows,
            offsets,
            key_positions,
        };
        plan.validate()?;
        Ok(plan)
    }

    #[must_use]
    pub const fn seq_len(&self) -> u32 {
        self.seq_len
    }

    #[must_use]
    pub const fn query_rows(&self) -> u32 {
        self.query_rows
    }

    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    #[must_use]
    pub fn key_positions(&self) -> &[u32] {
        &self.key_positions
    }

    #[must_use]
    pub fn admitted_count(&self) -> usize {
        self.key_positions.len()
    }

    #[must_use]
    pub fn offsets_bytes(&self) -> u64 {
        (self.offsets.len() as u64) * 4
    }

    #[must_use]
    pub fn key_positions_bytes(&self) -> u64 {
        (self.key_positions.len() as u64) * 4
    }

    #[must_use]
    pub fn total_storage_bytes(&self) -> u64 {
        self.offsets_bytes() + self.key_positions_bytes()
    }

    /// Canonical little-endian bytes for deterministic evidence/fingerprinting.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            20 + self.offsets.len() * core::mem::size_of::<u32>()
                + self.key_positions.len() * core::mem::size_of::<u32>(),
        );
        bytes.extend_from_slice(&STRUCTURAL_DEVICE_SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.seq_len.to_le_bytes());
        bytes.extend_from_slice(&self.query_rows.to_le_bytes());
        bytes.extend_from_slice(&(self.offsets.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.key_positions.len() as u32).to_le_bytes());
        for value in &self.offsets {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.key_positions {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Deterministic structural fingerprint. This is not authentication.
    #[must_use]
    pub fn fingerprint_fnv1a64(&self) -> u64 {
        fnv1a64(&self.canonical_bytes())
    }

    /// Reconstruct the authoritative host candidate semantics for an explicitly
    /// supplied attention shape.
    pub fn to_candidates(
        &self,
        shape: AttentionShape,
    ) -> Result<StructuralCandidateSet, StructuralDevicePlanError> {
        self.validate()?;
        let expected_rows = shape.lse_len()?;
        if shape.seq_len != self.seq_len as usize || expected_rows != self.query_rows as usize {
            return Err(StructuralDevicePlanError::GeometryMismatch {
                plan_seq_len: self.seq_len,
                shape_seq_len: shape.seq_len,
                plan_query_rows: self.query_rows,
                shape_query_rows: expected_rows,
            });
        }

        let mut rows = Vec::with_capacity(expected_rows);
        for row in 0..expected_rows {
            let start = self.offsets[row] as usize;
            let end = self.offsets[row + 1] as usize;
            rows.push(
                self.key_positions[start..end]
                    .iter()
                    .map(|value| *value as usize)
                    .collect(),
            );
        }
        StructuralCandidateSet::from_rows(shape, rows).map_err(Into::into)
    }

    fn validate(&self) -> Result<(), StructuralDevicePlanError> {
        if self.seq_len == 0 || self.query_rows == 0 {
            return Err(StructuralDevicePlanError::ZeroGeometry);
        }
        let expected_offsets = self.query_rows as usize + 1;
        if self.offsets.len() != expected_offsets {
            return Err(StructuralDevicePlanError::OffsetCountMismatch {
                actual: self.offsets.len(),
                expected: expected_offsets,
            });
        }
        if self.offsets.first().copied() != Some(0) {
            return Err(StructuralDevicePlanError::FirstOffsetNonZero);
        }
        let terminal = u32::try_from(self.key_positions.len()).map_err(|_| {
            StructuralDevicePlanError::IndexSpaceExceeded {
                field: "key_count",
                value: self.key_positions.len(),
            }
        })?;
        if self.offsets.last().copied() != Some(terminal) {
            return Err(StructuralDevicePlanError::TerminalOffsetMismatch {
                actual: self.offsets.last().copied().unwrap_or_default(),
                expected: terminal,
            });
        }

        for row in 0..self.query_rows as usize {
            let start = self.offsets[row];
            let end = self.offsets[row + 1];
            if start > end {
                return Err(StructuralDevicePlanError::DecreasingOffset { row, start, end });
            }
            let start = start as usize;
            let end = end as usize;
            if end > self.key_positions.len() {
                return Err(StructuralDevicePlanError::OffsetOutOfRange {
                    row,
                    end,
                    key_count: self.key_positions.len(),
                });
            }
            let row_keys = &self.key_positions[start..end];
            for (position, &key) in row_keys.iter().enumerate() {
                if key >= self.seq_len {
                    return Err(StructuralDevicePlanError::KeyOutOfRange {
                        row,
                        key,
                        seq_len: self.seq_len,
                    });
                }
                if position > 0 && row_keys[position - 1] >= key {
                    return Err(StructuralDevicePlanError::NonIncreasingKey {
                        row,
                        previous: row_keys[position - 1],
                        current: key,
                    });
                }
            }
        }
        Ok(())
    }
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32, StructuralDevicePlanError> {
    u32::try_from(value).map_err(|_| StructuralDevicePlanError::IndexSpaceExceeded { field, value })
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StructuralDevicePlanError {
    Structural(StructuralRoutingError),
    Attention(crate::FlatAttentionError),
    ZeroGeometry,
    IndexSpaceExceeded {
        field: &'static str,
        value: usize,
    },
    OffsetCountMismatch {
        actual: usize,
        expected: usize,
    },
    FirstOffsetNonZero,
    TerminalOffsetMismatch {
        actual: u32,
        expected: u32,
    },
    DecreasingOffset {
        row: usize,
        start: u32,
        end: u32,
    },
    OffsetOutOfRange {
        row: usize,
        end: usize,
        key_count: usize,
    },
    KeyOutOfRange {
        row: usize,
        key: u32,
        seq_len: u32,
    },
    NonIncreasingKey {
        row: usize,
        previous: u32,
        current: u32,
    },
    GeometryMismatch {
        plan_seq_len: u32,
        shape_seq_len: usize,
        plan_query_rows: u32,
        shape_query_rows: usize,
    },
}

impl From<StructuralRoutingError> for StructuralDevicePlanError {
    fn from(error: StructuralRoutingError) -> Self {
        Self::Structural(error)
    }
}

impl From<crate::FlatAttentionError> for StructuralDevicePlanError {
    fn from(error: crate::FlatAttentionError) -> Self {
        Self::Attention(error)
    }
}

impl fmt::Display for StructuralDevicePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StructuralDevicePlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> AttentionShape {
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 4,
            head_dim: 2,
        }
    }

    #[test]
    fn host_device_round_trip_preserves_exact_candidate_identity() {
        let candidates = StructuralCandidateSet::from_rows(
            shape(),
            vec![vec![0, 2], vec![1, 3], vec![0, 1, 3], vec![2, 3]],
        )
        .unwrap();
        let plan = StructuralDevicePlan::from_candidates(&candidates).unwrap();

        assert_eq!(plan.offsets(), &[0, 2, 4, 7, 9]);
        assert_eq!(plan.key_positions(), &[0, 2, 1, 3, 0, 1, 3, 2, 3]);
        assert_eq!(plan.total_storage_bytes(), 56);
        assert_eq!(plan.to_candidates(shape()).unwrap(), candidates);
    }

    #[test]
    fn all_accept_round_trip_preserves_every_key() {
        let candidates = StructuralCandidateSet::all(shape()).unwrap();
        let plan = StructuralDevicePlan::from_candidates(&candidates).unwrap();
        assert_eq!(plan.admitted_count(), 16);
        assert_eq!(plan.to_candidates(shape()).unwrap(), candidates);
    }

    #[test]
    fn malformed_offsets_and_keys_fail_closed() {
        assert!(matches!(
            StructuralDevicePlan::from_u32_parts(4, 2, vec![0, 2], vec![0, 1]),
            Err(StructuralDevicePlanError::OffsetCountMismatch {
                actual: 2,
                expected: 3
            })
        ));
        assert!(matches!(
            StructuralDevicePlan::from_u32_parts(4, 1, vec![0, 2], vec![2, 2]),
            Err(StructuralDevicePlanError::NonIncreasingKey {
                row: 0,
                previous: 2,
                current: 2
            })
        ));
        assert!(matches!(
            StructuralDevicePlan::from_u32_parts(4, 1, vec![0, 1], vec![4]),
            Err(StructuralDevicePlanError::KeyOutOfRange {
                row: 0,
                key: 4,
                seq_len: 4
            })
        ));
    }

    #[test]
    fn fingerprint_is_deterministic_and_identity_sensitive() {
        let a =
            StructuralCandidateSet::from_rows(shape(), vec![vec![0], vec![1], vec![2], vec![3]])
                .unwrap();
        let b =
            StructuralCandidateSet::from_rows(shape(), vec![vec![0], vec![1], vec![2], vec![2, 3]])
                .unwrap();
        let a1 = StructuralDevicePlan::from_candidates(&a).unwrap();
        let a2 = StructuralDevicePlan::from_candidates(&a).unwrap();
        let b = StructuralDevicePlan::from_candidates(&b).unwrap();

        assert_eq!(a1.fingerprint_fnv1a64(), a2.fingerprint_fnv1a64());
        assert_ne!(a1.fingerprint_fnv1a64(), b.fingerprint_fnv1a64());
    }

    #[test]
    fn causal_intersection_round_trip_is_exact() {
        let original = StructuralCandidateSet::all(shape()).unwrap();
        let causal = original.intersect_causal();
        let plan = StructuralDevicePlan::from_candidates(&causal).unwrap();
        assert_eq!(plan.to_candidates(shape()).unwrap(), causal);
        assert_eq!(plan.key_positions(), &[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]);
    }
}
