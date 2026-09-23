//! Research-only monotonic repair transport for sparse FLAT attention.
//!
//! MAA-14a deliberately separates how a structural candidate set is widened
//! from why a row is declared insufficient. A caller supplies the rows that
//! require another repair round; this module only applies a frozen,
//! deterministic expansion schedule. No residual/confidence policy, timing
//! claim, or production routing decision is embedded here.

use core::fmt;

use crate::AttentionShape;

use super::research_structural_routing::{StructuralCandidateSet, StructuralRoutingError};

/// Version of the MAA-14a adaptive-repair transport contract.
pub const ADAPTIVE_REPAIR_SCHEMA_VERSION: u32 = 1;

/// Frozen monotonic expansion schedule for one structural candidate set.
///
/// The base set is round zero. Each row owns zero or more ordered expansion
/// tiers. Advancing a row exposes exactly its next tier. A key may occur only
/// once across the base row and all of that row's tiers, which makes every
/// materialized state a monotonic superset of its predecessor.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveRepairSchedule {
    shape: AttentionShape,
    base: StructuralCandidateSet,
    tiers: Vec<Vec<Vec<usize>>>,
}

/// Per-row repair depth for one AdaptiveRepairSchedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveRepairState {
    levels: Vec<usize>,
}

/// Result of one deterministic repair advance.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveRepairAdvance {
    state: AdaptiveRepairState,
    previous_admitted_pairs: usize,
    admitted_pairs: usize,
    rows_advanced: usize,
}

impl AdaptiveRepairSchedule {
    /// Validate and freeze one expansion schedule.
    ///
    /// The outer vector must contain one entry per canonical query row. Within a
    /// row, tiers are applied in order. Empty tiers, out-of-range keys, and keys
    /// repeated in the base or any earlier/later tier are rejected.
    pub fn new(
        shape: AttentionShape,
        base: StructuralCandidateSet,
        tiers: Vec<Vec<Vec<usize>>>,
    ) -> Result<Self, AdaptiveRepairError> {
        let shape_rows = shape
            .lse_len()
            .map_err(StructuralRoutingError::from)
            .map_err(AdaptiveRepairError::Structural)?;
        if base.seq_len() != shape.seq_len || base.query_rows() != shape_rows {
            return Err(AdaptiveRepairError::BaseGeometryMismatch {
                base_seq_len: base.seq_len(),
                shape_seq_len: shape.seq_len,
                base_query_rows: base.query_rows(),
                shape_query_rows: shape_rows,
            });
        }
        if tiers.len() != base.query_rows() {
            return Err(AdaptiveRepairError::ExpansionRowCountMismatch {
                actual: tiers.len(),
                expected: base.query_rows(),
            });
        }

        for (row, row_tiers) in tiers.iter().enumerate() {
            let mut seen = vec![false; base.seq_len()];
            for &key in base.row(row)? {
                seen[key] = true;
            }

            for (tier_index, tier) in row_tiers.iter().enumerate() {
                if tier.is_empty() {
                    return Err(AdaptiveRepairError::EmptyExpansionTier {
                        row,
                        tier: tier_index,
                    });
                }
                for &key in tier {
                    if key >= base.seq_len() {
                        return Err(AdaptiveRepairError::ScheduledKeyOutOfRange {
                            row,
                            tier: tier_index,
                            key,
                            seq_len: base.seq_len(),
                        });
                    }
                    if seen[key] {
                        return Err(AdaptiveRepairError::DuplicateScheduledKey {
                            row,
                            tier: tier_index,
                            key,
                        });
                    }
                    seen[key] = true;
                }
            }
        }

        Ok(Self { shape, base, tiers })
    }

    /// Return the immutable round-zero candidate set.
    #[must_use]
    pub const fn base(&self) -> &StructuralCandidateSet {
        &self.base
    }

    /// Return the frozen attention geometry.
    #[must_use]
    pub const fn shape(&self) -> AttentionShape {
        self.shape
    }

    /// Create the round-zero repair state.
    #[must_use]
    pub fn initial_state(&self) -> AdaptiveRepairState {
        AdaptiveRepairState {
            levels: vec![0; self.base.query_rows()],
        }
    }

    /// Number of expansion tiers available for one row.
    pub fn tier_count(&self, row: usize) -> Result<usize, AdaptiveRepairError> {
        self.tiers
            .get(row)
            .map(Vec::len)
            .ok_or(AdaptiveRepairError::RepairRowOutOfRange {
                row,
                query_rows: self.base.query_rows(),
            })
    }

    /// True when every row's complete schedule reaches the all-key set.
    ///
    /// This says nothing about whether dense completion is desirable; it only
    /// exposes whether the frozen schedule has a deterministic dense terminal
    /// state available as a fail-safe research control.
    #[must_use]
    pub fn has_dense_terminal_state(&self) -> bool {
        (0..self.base.query_rows()).all(|row| {
            let scheduled = self.tiers[row].iter().map(Vec::len).sum::<usize>();
            self.base
                .row(row)
                .map(|base| base.len() + scheduled == self.base.seq_len())
                .unwrap_or(false)
        })
    }

    /// Materialize the canonical structural candidate set for a repair state.
    pub fn materialize(
        &self,
        state: &AdaptiveRepairState,
    ) -> Result<StructuralCandidateSet, AdaptiveRepairError> {
        self.validate_state(state)?;

        let mut rows = Vec::with_capacity(self.base.query_rows());
        for row in 0..self.base.query_rows() {
            let mut keys = self.base.row(row)?.to_vec();
            for tier in self.tiers[row].iter().take(state.levels[row]) {
                keys.extend_from_slice(tier);
            }
            rows.push(keys);
        }

        StructuralCandidateSet::from_rows(self.shape, rows).map_err(AdaptiveRepairError::Structural)
    }

    /// Advance exactly the requested rows by one tier.
    ///
    /// The row set is an externally supplied decision. MAA-14a does not define
    /// or infer residuals. Duplicate row requests and attempts to advance past
    /// the frozen schedule fail closed.
    pub fn advance(
        &self,
        state: &AdaptiveRepairState,
        rows_to_repair: &[usize],
    ) -> Result<AdaptiveRepairAdvance, AdaptiveRepairError> {
        self.validate_state(state)?;
        let previous_admitted_pairs = self.materialize(state)?.admitted_count();
        let mut next = state.clone();
        let mut seen = vec![false; self.base.query_rows()];

        for &row in rows_to_repair {
            if row >= self.base.query_rows() {
                return Err(AdaptiveRepairError::RepairRowOutOfRange {
                    row,
                    query_rows: self.base.query_rows(),
                });
            }
            if seen[row] {
                return Err(AdaptiveRepairError::DuplicateRepairRow { row });
            }
            seen[row] = true;

            let level = next.levels[row];
            if level >= self.tiers[row].len() {
                return Err(AdaptiveRepairError::RepairExhausted {
                    row,
                    level,
                    tier_count: self.tiers[row].len(),
                });
            }
            next.levels[row] = level
                .checked_add(1)
                .ok_or(AdaptiveRepairError::LevelOverflow { row })?;
        }

        let admitted_pairs = self.materialize(&next)?.admitted_count();
        Ok(AdaptiveRepairAdvance {
            state: next,
            previous_admitted_pairs,
            admitted_pairs,
            rows_advanced: rows_to_repair.len(),
        })
    }

    fn validate_state(&self, state: &AdaptiveRepairState) -> Result<(), AdaptiveRepairError> {
        if state.levels.len() != self.base.query_rows() {
            return Err(AdaptiveRepairError::StateRowCountMismatch {
                actual: state.levels.len(),
                expected: self.base.query_rows(),
            });
        }
        for (row, &level) in state.levels.iter().enumerate() {
            if level > self.tiers[row].len() {
                return Err(AdaptiveRepairError::StateLevelOutOfRange {
                    row,
                    level,
                    tier_count: self.tiers[row].len(),
                });
            }
        }
        Ok(())
    }
}

impl AdaptiveRepairState {
    /// Per-row number of already-applied expansion tiers.
    #[must_use]
    pub fn levels(&self) -> &[usize] {
        &self.levels
    }
}

impl AdaptiveRepairAdvance {
    /// State after the repair advance.
    #[must_use]
    pub const fn state(&self) -> &AdaptiveRepairState {
        &self.state
    }

    /// Number of structurally admitted pairs before this advance.
    #[must_use]
    pub const fn previous_admitted_pairs(&self) -> usize {
        self.previous_admitted_pairs
    }

    /// Number of structurally admitted pairs after this advance.
    #[must_use]
    pub const fn admitted_pairs(&self) -> usize {
        self.admitted_pairs
    }

    /// Number of distinct rows advanced exactly one tier.
    #[must_use]
    pub const fn rows_advanced(&self) -> usize {
        self.rows_advanced
    }
}

/// Fail-closed errors for MAA-14a repair transport.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AdaptiveRepairError {
    /// Underlying structural-candidate validation failed.
    Structural(StructuralRoutingError),
    /// Base candidate geometry does not match the declared attention shape.
    BaseGeometryMismatch {
        base_seq_len: usize,
        shape_seq_len: usize,
        base_query_rows: usize,
        shape_query_rows: usize,
    },
    /// Expansion schedule row count does not match the base candidate geometry.
    ExpansionRowCountMismatch { actual: usize, expected: usize },
    /// One scheduled expansion tier is empty.
    EmptyExpansionTier { row: usize, tier: usize },
    /// A scheduled key lies outside the base sequence geometry.
    ScheduledKeyOutOfRange {
        row: usize,
        tier: usize,
        key: usize,
        seq_len: usize,
    },
    /// A key appears in both the base and expansion schedule, or in two tiers.
    DuplicateScheduledKey { row: usize, tier: usize, key: usize },
    /// State row count does not match the frozen schedule.
    StateRowCountMismatch { actual: usize, expected: usize },
    /// A state claims more applied tiers than the schedule contains.
    StateLevelOutOfRange {
        row: usize,
        level: usize,
        tier_count: usize,
    },
    /// A requested repair row is outside the frozen geometry.
    RepairRowOutOfRange { row: usize, query_rows: usize },
    /// The same row was requested twice in one repair step.
    DuplicateRepairRow { row: usize },
    /// No further frozen tier exists for a requested row.
    RepairExhausted {
        row: usize,
        level: usize,
        tier_count: usize,
    },
    /// A per-row repair level overflowed.
    LevelOverflow { row: usize },
}

impl From<StructuralRoutingError> for AdaptiveRepairError {
    fn from(value: StructuralRoutingError) -> Self {
        Self::Structural(value)
    }
}

impl fmt::Display for AdaptiveRepairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structural(error) => write!(f, "{error}"),
            Self::BaseGeometryMismatch {
                base_seq_len,
                shape_seq_len,
                base_query_rows,
                shape_query_rows,
            } => write!(
                f,
                "adaptive repair base geometry {base_seq_len}/{base_query_rows} does not match shape {shape_seq_len}/{shape_query_rows}"
            ),
            Self::ExpansionRowCountMismatch { actual, expected } => {
                write!(f, "adaptive repair has {actual} rows, expected {expected}")
            }
            Self::EmptyExpansionTier { row, tier } => {
                write!(f, "adaptive repair row {row} tier {tier} is empty")
            }
            Self::ScheduledKeyOutOfRange {
                row,
                tier,
                key,
                seq_len,
            } => write!(
                f,
                "adaptive repair key {key} in row {row} tier {tier} is outside 0..{seq_len}"
            ),
            Self::DuplicateScheduledKey { row, tier, key } => write!(
                f,
                "adaptive repair key {key} in row {row} tier {tier} was already admitted or scheduled"
            ),
            Self::StateRowCountMismatch { actual, expected } => {
                write!(f, "adaptive repair state has {actual} rows, expected {expected}")
            }
            Self::StateLevelOutOfRange {
                row,
                level,
                tier_count,
            } => write!(
                f,
                "adaptive repair row {row} is at level {level}, beyond {tier_count} tiers"
            ),
            Self::RepairRowOutOfRange { row, query_rows } => {
                write!(f, "adaptive repair row {row} is outside 0..{query_rows}")
            }
            Self::DuplicateRepairRow { row } => {
                write!(f, "adaptive repair row {row} was requested more than once")
            }
            Self::RepairExhausted {
                row,
                level,
                tier_count,
            } => write!(
                f,
                "adaptive repair row {row} cannot advance from level {level}; only {tier_count} tiers exist"
            ),
            Self::LevelOverflow { row } => {
                write!(f, "adaptive repair level overflowed for row {row}")
            }
        }
    }
}

impl std::error::Error for AdaptiveRepairError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Structural(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> AttentionShape {
        AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 4,
            head_dim: 1,
        }
    }

    fn schedule() -> AdaptiveRepairSchedule {
        let base =
            StructuralCandidateSet::from_rows(shape(), vec![vec![0], vec![1], vec![2], vec![3]])
                .unwrap();
        AdaptiveRepairSchedule::new(
            shape(),
            base,
            vec![
                vec![vec![1, 2], vec![3]],
                vec![vec![0], vec![2, 3]],
                vec![vec![0, 1], vec![3]],
                vec![vec![0], vec![1, 2]],
            ],
        )
        .unwrap()
    }

    #[test]
    fn initial_state_is_exactly_the_base_set() {
        let schedule = schedule();
        let state = schedule.initial_state();
        assert_eq!(state.levels(), &[0, 0, 0, 0]);
        assert_eq!(
            schedule.materialize(&state).unwrap(),
            schedule.base().clone()
        );
        assert!(schedule.has_dense_terminal_state());
    }

    #[test]
    fn repair_is_monotonic_and_row_local() {
        let schedule = schedule();
        let state = schedule.initial_state();
        let first = schedule.advance(&state, &[0, 2]).unwrap();
        assert_eq!(first.previous_admitted_pairs(), 4);
        assert_eq!(first.admitted_pairs(), 8);
        assert_eq!(first.rows_advanced(), 2);
        assert_eq!(first.state().levels(), &[1, 0, 1, 0]);

        let materialized = schedule.materialize(first.state()).unwrap();
        assert_eq!(materialized.row(0).unwrap(), &[0, 1, 2]);
        assert_eq!(materialized.row(1).unwrap(), &[1]);
        assert_eq!(materialized.row(2).unwrap(), &[0, 1, 2]);
        assert_eq!(materialized.row(3).unwrap(), &[3]);
    }

    #[test]
    fn frozen_schedule_can_reach_dense_control_without_removal() {
        let schedule = schedule();
        let state = schedule.initial_state();
        let first = schedule.advance(&state, &[0, 1, 2, 3]).unwrap();
        let second = schedule.advance(first.state(), &[0, 1, 2, 3]).unwrap();
        let dense = schedule.materialize(second.state()).unwrap();
        assert_eq!(dense.admitted_count(), 16);
        for row in 0..4 {
            assert_eq!(dense.row(row).unwrap(), &[0, 1, 2, 3]);
        }
    }

    #[test]
    fn duplicate_or_out_of_range_schedule_keys_fail_closed() {
        let base =
            StructuralCandidateSet::from_rows(shape(), vec![vec![0], vec![1], vec![2], vec![3]])
                .unwrap();

        assert!(matches!(
            AdaptiveRepairSchedule::new(
                shape(),
                base.clone(),
                vec![vec![vec![0]], vec![vec![0]], vec![vec![0]], vec![vec![0]]],
            ),
            Err(AdaptiveRepairError::DuplicateScheduledKey {
                row: 0,
                tier: 0,
                key: 0
            })
        ));

        assert!(matches!(
            AdaptiveRepairSchedule::new(
                shape(),
                base,
                vec![vec![vec![4]], vec![vec![0]], vec![vec![0]], vec![vec![0]]],
            ),
            Err(AdaptiveRepairError::ScheduledKeyOutOfRange {
                row: 0,
                tier: 0,
                key: 4,
                seq_len: 4
            })
        ));
    }

    #[test]
    fn duplicate_repair_request_and_exhaustion_fail_closed() {
        let schedule = schedule();
        let state = schedule.initial_state();
        assert!(matches!(
            schedule.advance(&state, &[1, 1]),
            Err(AdaptiveRepairError::DuplicateRepairRow { row: 1 })
        ));

        let first = schedule.advance(&state, &[0]).unwrap();
        let second = schedule.advance(first.state(), &[0]).unwrap();
        assert!(matches!(
            schedule.advance(second.state(), &[0]),
            Err(AdaptiveRepairError::RepairExhausted {
                row: 0,
                level: 2,
                tier_count: 2
            })
        ));
    }
}
