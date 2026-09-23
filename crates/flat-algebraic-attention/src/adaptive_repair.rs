//! MAA-14b structural-only adaptive repair triggers.
//!
//! These policies consume candidate-count metadata only. They never inspect
//! dense Q.K scores, retained-softmax mass, V values, output error, or LSE.
//! The variants and ratios are frozen by the MAA-14b preregistration.

use core::fmt;

/// Version of the MAA-14b structural repair-trigger contract.
pub const STRUCTURAL_REPAIR_TRIGGER_SCHEMA_VERSION: u32 = 1;

/// Frozen MAA-14b trigger family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructuralRepairTrigger {
    /// Keep the tiered-recall base set.
    NeverRepair,
    /// Repair whenever the hamming-medium envelope contains any omitted key.
    RepairAnyGap,
    /// Repair when base/medium coverage is strictly below 7/8.
    CoverageBelowSevenEighths,
    /// Repair when base/medium coverage is strictly below 3/4.
    CoverageBelowThreeQuarters,
    /// Repair when base/medium coverage is strictly below 2/3.
    CoverageBelowTwoThirds,
    /// Always select the complete hamming-medium envelope.
    AlwaysRepair,
}

/// One validated trigger decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuralRepairDecision {
    /// Whether this trigger requests the full repair envelope.
    pub repair: bool,
    /// Candidate count in the frozen base route.
    pub base_count: usize,
    /// Candidate count in the frozen hamming-medium envelope.
    pub medium_count: usize,
    /// Number of candidates available to add.
    pub gap_count: usize,
}

impl StructuralRepairTrigger {
    /// Stable protocol label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::NeverRepair => "never_repair",
            Self::RepairAnyGap => "repair_any_gap",
            Self::CoverageBelowSevenEighths => "repair_if_coverage_below_7_8",
            Self::CoverageBelowThreeQuarters => "repair_if_coverage_below_3_4",
            Self::CoverageBelowTwoThirds => "repair_if_coverage_below_2_3",
            Self::AlwaysRepair => "always_repair",
        }
    }

    /// Frozen random-control identifier for deployable adaptive triggers.
    ///
    /// Endpoint controls deliberately have no random counterpart.
    #[must_use]
    pub const fn random_control_id(self) -> Option<u64> {
        match self {
            Self::RepairAnyGap => Some(1),
            Self::CoverageBelowSevenEighths => Some(2),
            Self::CoverageBelowThreeQuarters => Some(3),
            Self::CoverageBelowTwoThirds => Some(4),
            Self::NeverRepair | Self::AlwaysRepair => None,
        }
    }

    /// Decide from structural counts only.
    pub fn decide(
        self,
        base_count: usize,
        medium_count: usize,
    ) -> Result<StructuralRepairDecision, StructuralRepairTriggerError> {
        if base_count > medium_count {
            return Err(StructuralRepairTriggerError::BaseExceedsEnvelope {
                base_count,
                medium_count,
            });
        }
        let gap_count = medium_count - base_count;
        let repair = match self {
            Self::NeverRepair => false,
            Self::RepairAnyGap => gap_count > 0,
            Self::CoverageBelowSevenEighths => ratio_below(base_count, medium_count, 7, 8)?,
            Self::CoverageBelowThreeQuarters => ratio_below(base_count, medium_count, 3, 4)?,
            Self::CoverageBelowTwoThirds => ratio_below(base_count, medium_count, 2, 3)?,
            Self::AlwaysRepair => true,
        };
        Ok(StructuralRepairDecision {
            repair,
            base_count,
            medium_count,
            gap_count,
        })
    }
}

fn ratio_below(
    base_count: usize,
    medium_count: usize,
    numerator: usize,
    denominator: usize,
) -> Result<bool, StructuralRepairTriggerError> {
    let left = base_count
        .checked_mul(denominator)
        .ok_or(StructuralRepairTriggerError::ArithmeticOverflow)?;
    let right = medium_count
        .checked_mul(numerator)
        .ok_or(StructuralRepairTriggerError::ArithmeticOverflow)?;
    Ok(left < right)
}

/// Fail-closed MAA-14b trigger errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StructuralRepairTriggerError {
    /// The declared base route is not a subset in cardinality of its envelope.
    BaseExceedsEnvelope {
        base_count: usize,
        medium_count: usize,
    },
    /// Checked integer ratio arithmetic overflowed.
    ArithmeticOverflow,
}

impl fmt::Display for StructuralRepairTriggerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BaseExceedsEnvelope {
                base_count,
                medium_count,
            } => write!(
                formatter,
                "MAA-14b base count {base_count} exceeds repair envelope count {medium_count}"
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("MAA-14b structural repair ratio arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for StructuralRepairTriggerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_and_gap_controls_are_exact() {
        let never = StructuralRepairTrigger::NeverRepair.decide(7, 9).unwrap();
        let any_gap = StructuralRepairTrigger::RepairAnyGap.decide(7, 9).unwrap();
        let no_gap = StructuralRepairTrigger::RepairAnyGap.decide(9, 9).unwrap();
        let always = StructuralRepairTrigger::AlwaysRepair.decide(9, 9).unwrap();

        assert!(!never.repair);
        assert!(any_gap.repair);
        assert!(!no_gap.repair);
        assert!(always.repair);
    }

    #[test]
    fn coverage_boundaries_are_strict() {
        let seven_boundary = StructuralRepairTrigger::CoverageBelowSevenEighths
            .decide(7, 8)
            .unwrap();
        let seven_below = StructuralRepairTrigger::CoverageBelowSevenEighths
            .decide(6, 8)
            .unwrap();
        let three_boundary = StructuralRepairTrigger::CoverageBelowThreeQuarters
            .decide(3, 4)
            .unwrap();
        let three_below = StructuralRepairTrigger::CoverageBelowThreeQuarters
            .decide(2, 4)
            .unwrap();
        let two_boundary = StructuralRepairTrigger::CoverageBelowTwoThirds
            .decide(2, 3)
            .unwrap();
        let two_below = StructuralRepairTrigger::CoverageBelowTwoThirds
            .decide(1, 3)
            .unwrap();

        assert!(!seven_boundary.repair);
        assert!(seven_below.repair);
        assert!(!three_boundary.repair);
        assert!(three_below.repair);
        assert!(!two_boundary.repair);
        assert!(two_below.repair);
    }

    #[test]
    fn malformed_counts_fail_closed() {
        assert!(matches!(
            StructuralRepairTrigger::RepairAnyGap.decide(5, 4),
            Err(StructuralRepairTriggerError::BaseExceedsEnvelope {
                base_count: 5,
                medium_count: 4
            })
        ));
    }

    #[test]
    fn random_control_ids_match_preregistration() {
        assert_eq!(
            StructuralRepairTrigger::RepairAnyGap.random_control_id(),
            Some(1)
        );
        assert_eq!(
            StructuralRepairTrigger::CoverageBelowSevenEighths.random_control_id(),
            Some(2)
        );
        assert_eq!(
            StructuralRepairTrigger::CoverageBelowThreeQuarters.random_control_id(),
            Some(3)
        );
        assert_eq!(
            StructuralRepairTrigger::CoverageBelowTwoThirds.random_control_id(),
            Some(4)
        );
        assert_eq!(
            StructuralRepairTrigger::NeverRepair.random_control_id(),
            None
        );
        assert_eq!(
            StructuralRepairTrigger::AlwaysRepair.random_control_id(),
            None
        );
    }
}
