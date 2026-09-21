use core::fmt;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct RetainedSoftmaxMass {
    retained_mass: f64,
    dropped_mass: f64,
    dense_lse: f64,
    selected_lse: Option<f64>,
    lse_gap: Option<f64>,
}

impl RetainedSoftmaxMass {
    #[must_use]
    pub const fn retained_mass(&self) -> f64 {
        self.retained_mass
    }

    #[must_use]
    pub const fn dropped_mass(&self) -> f64 {
        self.dropped_mass
    }

    #[must_use]
    pub const fn dense_lse(&self) -> f64 {
        self.dense_lse
    }

    #[must_use]
    pub const fn selected_lse(&self) -> Option<f64> {
        self.selected_lse
    }

    #[must_use]
    pub const fn lse_gap(&self) -> Option<f64> {
        self.lse_gap
    }
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SoftmaxMassError {
    EmptyScores,
    NonFiniteScore {
        index: usize,
    },
    SelectedIndexOutOfBounds {
        index: usize,
        score_count: usize,
    },
    DuplicateSelectedIndex {
        index: usize,
    },
    NonFinitePartition,
    SelectedMassUnderflow,
    InvalidRetainedMass {
        value: f64,
    },
    InvalidValueBound {
        value: f64,
    },
}

impl fmt::Display for SoftmaxMassError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyScores => write!(formatter, "softmax mass requires at least one score"),
            Self::NonFiniteScore { index } => {
                write!(formatter, "softmax score at index {index} is non-finite")
            }
            Self::SelectedIndexOutOfBounds { index, score_count } => write!(
                formatter,
                "selected softmax index {index} is outside 0..{score_count}"
            ),
            Self::DuplicateSelectedIndex { index } => {
                write!(formatter, "selected softmax index {index} occurs more than once")
            }
            Self::NonFinitePartition => {
                write!(formatter, "softmax partition became non-finite or non-positive")
            }
            Self::SelectedMassUnderflow => write!(
                formatter,
                "non-empty selected softmax set underflowed to zero retained partition mass"
            ),
            Self::InvalidRetainedMass { value } => {
                write!(formatter, "retained softmax mass {value} is outside [0,1]")
            }
            Self::InvalidValueBound { value } => {
                write!(formatter, "softmax output value bound {value} is invalid")
            }
        }
    }
}

impl std::error::Error for SoftmaxMassError {}

pub fn retained_softmax_mass(
    scores: &[f64],
    selected: &[usize],
) -> Result<RetainedSoftmaxMass, SoftmaxMassError> {
    if scores.is_empty() {
        return Err(SoftmaxMassError::EmptyScores);
    }

    let mut max_score = f64::NEG_INFINITY;
    for (index, score) in scores.iter().copied().enumerate() {
        if !score.is_finite() {
            return Err(SoftmaxMassError::NonFiniteScore { index });
        }
        max_score = max_score.max(score);
    }

    let mut canonical_selected = BTreeSet::new();
    for &index in selected {
        if index >= scores.len() {
            return Err(SoftmaxMassError::SelectedIndexOutOfBounds {
                index,
                score_count: scores.len(),
            });
        }
        if !canonical_selected.insert(index) {
            return Err(SoftmaxMassError::DuplicateSelectedIndex { index });
        }
    }

    let weights = scores
        .iter()
        .map(|score| (*score - max_score).exp())
        .collect::<Vec<_>>();
    let dense_partition = weights.iter().sum::<f64>();
    if !dense_partition.is_finite() || dense_partition <= 0.0 {
        return Err(SoftmaxMassError::NonFinitePartition);
    }

    let selected_partition = canonical_selected
        .iter()
        .map(|index| weights[*index])
        .sum::<f64>();

    let dense_lse = max_score + dense_partition.ln();
    if canonical_selected.is_empty() {
        return Ok(RetainedSoftmaxMass {
            retained_mass: 0.0,
            dropped_mass: 1.0,
            dense_lse,
            selected_lse: None,
            lse_gap: None,
        });
    }

    if selected_partition <= 0.0 {
        return Err(SoftmaxMassError::SelectedMassUnderflow);
    }

    let mut retained_mass = selected_partition / dense_partition;
    let roundoff = 16.0 * f64::EPSILON;
    if retained_mass > 1.0 && retained_mass <= 1.0 + roundoff {
        retained_mass = 1.0;
    }
    if !(0.0..=1.0).contains(&retained_mass) {
        return Err(SoftmaxMassError::InvalidRetainedMass {
            value: retained_mass,
        });
    }

    let selected_lse = max_score + selected_partition.ln();
    let lse_gap = dense_lse - selected_lse;
    Ok(RetainedSoftmaxMass {
        retained_mass,
        dropped_mass: 1.0 - retained_mass,
        dense_lse,
        selected_lse: Some(selected_lse),
        lse_gap: Some(lse_gap),
    })
}

pub fn output_error_bound(
    retained_mass: f64,
    max_abs_value: f64,
) -> Result<f64, SoftmaxMassError> {
    if !retained_mass.is_finite() || !(0.0..=1.0).contains(&retained_mass) {
        return Err(SoftmaxMassError::InvalidRetainedMass {
            value: retained_mass,
        });
    }
    if !max_abs_value.is_finite() || max_abs_value < 0.0 {
        return Err(SoftmaxMassError::InvalidValueBound {
            value: max_abs_value,
        });
    }
    Ok(2.0 * (1.0 - retained_mass) * max_abs_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_selection_has_unit_mass_and_zero_gap() {
        let scores = [1.0, 2.0, -3.0];
        let result = retained_softmax_mass(&scores, &[0, 1, 2]).unwrap();

        assert!((result.retained_mass() - 1.0).abs() <= f64::EPSILON);
        assert!(result.dropped_mass().abs() <= f64::EPSILON);
        assert!(result.lse_gap().unwrap().abs() <= f64::EPSILON);
        assert!(
            (result.selected_lse().unwrap() - result.dense_lse()).abs() <= f64::EPSILON
        );
    }

    #[test]
    fn equal_scores_half_selection_has_half_mass_and_log_two_gap() {
        let result = retained_softmax_mass(&[0.0, 0.0], &[1]).unwrap();

        assert!((result.retained_mass() - 0.5).abs() <= f64::EPSILON);
        assert!((result.dropped_mass() - 0.5).abs() <= f64::EPSILON);
        assert!((result.lse_gap().unwrap() - core::f64::consts::LN_2).abs() < 1.0e-15);
    }

    #[test]
    fn empty_selection_is_explicit_zero_mass() {
        let result = retained_softmax_mass(&[2.0, 1.0], &[]).unwrap();

        assert!(result.retained_mass().abs() <= f64::EPSILON);
        assert!((result.dropped_mass() - 1.0).abs() <= f64::EPSILON);
        assert_eq!(result.selected_lse(), None);
        assert_eq!(result.lse_gap(), None);
    }

    #[test]
    fn selection_order_does_not_change_mass() {
        let scores = [0.25, 0.5, 0.75, 1.0];
        let left = retained_softmax_mass(&scores, &[3, 0, 2]).unwrap();
        let right = retained_softmax_mass(&scores, &[0, 2, 3]).unwrap();

        assert_eq!(left, right);
    }

    #[test]
    fn malformed_selection_and_scores_fail_closed() {
        assert_eq!(
            retained_softmax_mass(&[], &[]),
            Err(SoftmaxMassError::EmptyScores)
        );
        assert_eq!(
            retained_softmax_mass(&[0.0, f64::NAN], &[0]),
            Err(SoftmaxMassError::NonFiniteScore { index: 1 })
        );
        assert_eq!(
            retained_softmax_mass(&[0.0], &[1]),
            Err(SoftmaxMassError::SelectedIndexOutOfBounds {
                index: 1,
                score_count: 1,
            })
        );
        assert_eq!(
            retained_softmax_mass(&[0.0, 1.0], &[1, 1]),
            Err(SoftmaxMassError::DuplicateSelectedIndex { index: 1 })
        );
    }

    #[test]
    fn output_bound_matches_simple_limits() {
        assert!(output_error_bound(1.0, 2.0).unwrap().abs() <= f64::EPSILON);
        assert!(
            (output_error_bound(0.75, 2.0).unwrap() - 1.0).abs() <= f64::EPSILON
        );
    }
}
