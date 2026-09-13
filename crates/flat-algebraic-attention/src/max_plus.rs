use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MaxPlusError {
    ZeroDimension,
    StorageLengthMismatch {
        rows: usize,
        cols: usize,
        expected: usize,
        actual: usize,
    },
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    NodeOutOfBounds {
        node: usize,
        node_count: usize,
    },
    NonForwardEdge {
        from: usize,
        to: usize,
    },
    NegativeDelay {
        delay: i64,
    },
    ArithmeticOverflow,
}

impl fmt::Display for MaxPlusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(formatter, "max-plus dimensions must be non-zero"),
            Self::StorageLengthMismatch {
                rows,
                cols,
                expected,
                actual,
            } => write!(
                formatter,
                "max-plus matrix {rows}x{cols} requires {expected} values, got {actual}"
            ),
            Self::DimensionMismatch { expected, actual } => write!(
                formatter,
                "max-plus dimension mismatch: expected {expected} values, got {actual}"
            ),
            Self::NodeOutOfBounds { node, node_count } => write!(
                formatter,
                "max-plus schedule node {node} is outside 0..{node_count}"
            ),
            Self::NonForwardEdge { from, to } => write!(
                formatter,
                "max-plus schedule edge {from}->{to} is not forward in topological order"
            ),
            Self::NegativeDelay { delay } => write!(
                formatter,
                "max-plus schedule delay must be non-negative, got {delay}"
            ),
            Self::ArithmeticOverflow => write!(formatter, "max-plus addition overflowed i64"),
        }
    }
}

impl std::error::Error for MaxPlusError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaxPlusValue {
    NegInfinity,
    Finite(i64),
}

impl MaxPlusValue {
    pub const ZERO: Self = Self::NegInfinity;
    pub const ONE: Self = Self::Finite(0);

    #[must_use]
    pub const fn finite(value: i64) -> Self {
        Self::Finite(value)
    }

    #[must_use]
    pub const fn from_positive_boolean(value: bool) -> Self {
        if value {
            Self::ONE
        } else {
            Self::ZERO
        }
    }

    #[must_use]
    pub const fn as_finite(self) -> Option<i64> {
        match self {
            Self::NegInfinity => None,
            Self::Finite(value) => Some(value),
        }
    }

    #[must_use]
    pub const fn is_reachable(self) -> bool {
        matches!(self, Self::Finite(_))
    }

    #[must_use]
    pub fn oplus(self, other: Self) -> Self {
        self.max(other)
    }

    pub fn otimes(self, other: Self) -> Result<Self, MaxPlusError> {
        match (self, other) {
            (Self::NegInfinity, _) | (_, Self::NegInfinity) => Ok(Self::NegInfinity),
            (Self::Finite(left), Self::Finite(right)) => left
                .checked_add(right)
                .map(Self::Finite)
                .ok_or(MaxPlusError::ArithmeticOverflow),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxPlusMatrix {
    rows: usize,
    cols: usize,
    values: Vec<MaxPlusValue>,
}

impl MaxPlusMatrix {
    pub fn new(
        rows: usize,
        cols: usize,
        values: Vec<MaxPlusValue>,
    ) -> Result<Self, MaxPlusError> {
        if rows == 0 || cols == 0 {
            return Err(MaxPlusError::ZeroDimension);
        }
        let expected = rows
            .checked_mul(cols)
            .ok_or(MaxPlusError::ArithmeticOverflow)?;
        if values.len() != expected {
            return Err(MaxPlusError::StorageLengthMismatch {
                rows,
                cols,
                expected,
                actual: values.len(),
            });
        }
        Ok(Self { rows, cols, values })
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[must_use]
    pub fn values(&self) -> &[MaxPlusValue] {
        &self.values
    }

    pub fn apply(&self, input: &[MaxPlusValue]) -> Result<Vec<MaxPlusValue>, MaxPlusError> {
        if input.len() != self.cols {
            return Err(MaxPlusError::DimensionMismatch {
                expected: self.cols,
                actual: input.len(),
            });
        }

        self.values
            .chunks_exact(self.cols)
            .map(|row| {
                row.iter().copied().zip(input.iter().copied()).try_fold(
                    MaxPlusValue::ZERO,
                    |accumulator, (weight, value)| {
                        Ok(accumulator.oplus(weight.otimes(value)?))
                    },
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaxPlusEdge {
    from: usize,
    to: usize,
    delay: i64,
}

impl MaxPlusEdge {
    pub fn new(from: usize, to: usize, delay: i64) -> Self {
        Self { from, to, delay }
    }

    #[must_use]
    pub fn from(&self) -> usize {
        self.from
    }

    #[must_use]
    pub fn to(&self) -> usize {
        self.to
    }

    #[must_use]
    pub fn delay(&self) -> i64 {
        self.delay
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxPlusSchedule {
    node_count: usize,
    edges: Vec<MaxPlusEdge>,
}

impl MaxPlusSchedule {
    pub fn new(node_count: usize, mut edges: Vec<MaxPlusEdge>) -> Result<Self, MaxPlusError> {
        if node_count == 0 {
            return Err(MaxPlusError::ZeroDimension);
        }
        for edge in &edges {
            if edge.from >= node_count {
                return Err(MaxPlusError::NodeOutOfBounds {
                    node: edge.from,
                    node_count,
                });
            }
            if edge.to >= node_count {
                return Err(MaxPlusError::NodeOutOfBounds {
                    node: edge.to,
                    node_count,
                });
            }
            if edge.from >= edge.to {
                return Err(MaxPlusError::NonForwardEdge {
                    from: edge.from,
                    to: edge.to,
                });
            }
            if edge.delay < 0 {
                return Err(MaxPlusError::NegativeDelay { delay: edge.delay });
            }
        }
        edges.sort_unstable();
        edges.dedup();
        Ok(Self { node_count, edges })
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    #[must_use]
    pub fn edges(&self) -> &[MaxPlusEdge] {
        &self.edges
    }

    pub fn earliest_feasible_times(
        &self,
        initial: &[MaxPlusValue],
    ) -> Result<Vec<MaxPlusValue>, MaxPlusError> {
        if initial.len() != self.node_count {
            return Err(MaxPlusError::DimensionMismatch {
                expected: self.node_count,
                actual: initial.len(),
            });
        }

        let mut times = initial.to_vec();
        for from in 0..self.node_count {
            for edge in self.edges.iter().filter(|edge| edge.from == from) {
                let candidate = times[from].otimes(MaxPlusValue::Finite(edge.delay))?;
                times[edge.to] = times[edge.to].oplus(candidate);
            }
        }
        Ok(times)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semiring_identities_hold() {
        let value = MaxPlusValue::Finite(7);

        assert_eq!(MaxPlusValue::ZERO.oplus(value), value);
        assert_eq!(MaxPlusValue::ONE.otimes(value).unwrap(), value);
        assert_eq!(MaxPlusValue::ZERO.otimes(value).unwrap(), MaxPlusValue::ZERO);
        assert_eq!(
            MaxPlusValue::Finite(5).oplus(MaxPlusValue::Finite(9)),
            MaxPlusValue::Finite(9)
        );
        assert_eq!(
            MaxPlusValue::Finite(5)
                .otimes(MaxPlusValue::Finite(9))
                .unwrap(),
            MaxPlusValue::Finite(14)
        );
    }

    #[test]
    fn positive_boolean_bridge_preserves_or_and_and() {
        for left in [false, true] {
            for right in [false, true] {
                let left_tropical = MaxPlusValue::from_positive_boolean(left);
                let right_tropical = MaxPlusValue::from_positive_boolean(right);

                assert_eq!(
                    left_tropical.oplus(right_tropical),
                    MaxPlusValue::from_positive_boolean(left || right)
                );
                assert_eq!(
                    left_tropical.otimes(right_tropical).unwrap(),
                    MaxPlusValue::from_positive_boolean(left && right)
                );
            }
        }
    }

    #[test]
    fn matrix_apply_uses_max_of_sums() {
        let matrix = MaxPlusMatrix::new(
            2,
            3,
            vec![
                MaxPlusValue::Finite(0),
                MaxPlusValue::Finite(2),
                MaxPlusValue::ZERO,
                MaxPlusValue::Finite(4),
                MaxPlusValue::Finite(1),
                MaxPlusValue::Finite(-3),
            ],
        )
        .unwrap();
        let input = [
            MaxPlusValue::Finite(5),
            MaxPlusValue::Finite(1),
            MaxPlusValue::Finite(10),
        ];

        assert_eq!(
            matrix.apply(&input).unwrap(),
            vec![MaxPlusValue::Finite(5), MaxPlusValue::Finite(9)]
        );
    }

    #[test]
    fn precedence_schedule_returns_critical_synchronization_times() {
        let schedule = MaxPlusSchedule::new(
            4,
            vec![
                MaxPlusEdge::new(0, 1, 3),
                MaxPlusEdge::new(0, 2, 2),
                MaxPlusEdge::new(1, 3, 4),
                MaxPlusEdge::new(2, 3, 10),
            ],
        )
        .unwrap();
        let initial = [
            MaxPlusValue::Finite(0),
            MaxPlusValue::ZERO,
            MaxPlusValue::ZERO,
            MaxPlusValue::ZERO,
        ];

        assert_eq!(
            schedule.earliest_feasible_times(&initial).unwrap(),
            vec![
                MaxPlusValue::Finite(0),
                MaxPlusValue::Finite(3),
                MaxPlusValue::Finite(2),
                MaxPlusValue::Finite(12),
            ]
        );
    }

    #[test]
    fn unreachable_nodes_remain_negative_infinity() {
        let schedule = MaxPlusSchedule::new(3, vec![MaxPlusEdge::new(0, 1, 2)]).unwrap();
        let initial = [
            MaxPlusValue::Finite(0),
            MaxPlusValue::ZERO,
            MaxPlusValue::ZERO,
        ];

        let times = schedule.earliest_feasible_times(&initial).unwrap();
        assert_eq!(times[2], MaxPlusValue::ZERO);
        assert!(!times[2].is_reachable());
    }

    #[test]
    fn invalid_schedule_edges_fail_closed() {
        assert_eq!(
            MaxPlusSchedule::new(3, vec![MaxPlusEdge::new(2, 1, 1)]),
            Err(MaxPlusError::NonForwardEdge { from: 2, to: 1 })
        );
        assert_eq!(
            MaxPlusSchedule::new(3, vec![MaxPlusEdge::new(0, 3, 1)]),
            Err(MaxPlusError::NodeOutOfBounds {
                node: 3,
                node_count: 3,
            })
        );
        assert_eq!(
            MaxPlusSchedule::new(3, vec![MaxPlusEdge::new(0, 1, -1)]),
            Err(MaxPlusError::NegativeDelay { delay: -1 })
        );
    }

    #[test]
    fn arithmetic_overflow_fails_closed() {
        assert_eq!(
            MaxPlusValue::Finite(i64::MAX).otimes(MaxPlusValue::Finite(1)),
            Err(MaxPlusError::ArithmeticOverflow)
        );
    }

    #[test]
    fn dimension_mismatch_fails_closed() {
        let matrix = MaxPlusMatrix::new(1, 2, vec![MaxPlusValue::ONE; 2]).unwrap();
        assert_eq!(
            matrix.apply(&[MaxPlusValue::ONE]),
            Err(MaxPlusError::DimensionMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }
}
