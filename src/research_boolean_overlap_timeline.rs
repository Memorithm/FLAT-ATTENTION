//! M13B.4 block/page overlap timeline in one declared timing domain.
//!
//! The preregistration defines overlap as measured temporal intersection between
//! Boolean-plane work for the next consumable block/page and numerical-plane
//! work for the current one. This module represents exactly that evidence. A
//! positive intersection proves only temporal overlap in the declared clock
//! domain; it does not prove an end-to-end latency improvement.

use core::fmt;

use super::trace::{M13B4SchedulingVariant, M13B4TimingSource};

/// Version of the research-only consumable-unit overlap timeline schema.
pub const M13B4_TIMELINE_SCHEMA_VERSION: u32 = 1;

/// Granularity of one consumable routing unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum M13B4ConsumableUnitKind {
    /// One routed attention block.
    Block,
    /// One routed K/V page.
    Page,
}

/// Boolean and numerical intervals for one unit in consumption order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct M13B4UnitIntervals {
    /// Monotonic sequence index inside this timeline.
    pub sequence_index: u64,
    /// Start of Q-signature/routing work for this unit.
    pub boolean_start_ns: u64,
    /// End of Boolean admission/routing work for this unit.
    pub boolean_end_ns: u64,
    /// Start of exact numerical work for this admitted unit.
    pub numerical_start_ns: u64,
    /// End of exact numerical work for this admitted unit.
    pub numerical_end_ns: u64,
}

/// Exact overlap between numerical unit `n` and Boolean unit `n + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct M13B4AdjacentOverlap {
    pub current_sequence_index: u64,
    pub next_sequence_index: u64,
    /// Nanoseconds in the common declared clock domain.
    pub overlap_ns: u64,
}

/// Correlated M13B.4 block/page intervals sharing one timing domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct M13B4OverlapTimeline {
    /// Host-wall-clock or backend/device timestamp provenance.
    pub timing_source: M13B4TimingSource,
    /// Scheduling variant represented by these intervals.
    pub scheduling_variant: M13B4SchedulingVariant,
    /// Block or page granularity; mixed granularities require separate records.
    pub unit_kind: M13B4ConsumableUnitKind,
    /// Experiment/backend-owned identifier asserting a common timestamp origin.
    pub clock_domain_id: String,
    /// Units in exact consumption order.
    pub units: Vec<M13B4UnitIntervals>,
}

impl M13B4OverlapTimeline {
    /// Validate interval shape, consumption order and scheduling invariants.
    ///
    /// # Errors
    ///
    /// Rejects an empty clock-domain identifier, fewer than two units,
    /// non-contiguous sequence indices, reversed intervals, numerical work that
    /// starts before its own Boolean decision on separate-dispatch variants, or
    /// a positive cross-unit overlap in the serial/matched baseline.
    pub fn validate(&self) -> Result<(), M13B4OverlapTimelineError> {
        if self.clock_domain_id.is_empty() {
            return Err(M13B4OverlapTimelineError::EmptyClockDomainId);
        }
        if self.units.len() < 2 {
            return Err(M13B4OverlapTimelineError::InsufficientUnits {
                actual: self.units.len(),
            });
        }

        for (position, unit) in self.units.iter().enumerate() {
            if unit.boolean_start_ns > unit.boolean_end_ns {
                return Err(M13B4OverlapTimelineError::ReversedBooleanInterval {
                    sequence_index: unit.sequence_index,
                });
            }
            if unit.numerical_start_ns > unit.numerical_end_ns {
                return Err(M13B4OverlapTimelineError::ReversedNumericalInterval {
                    sequence_index: unit.sequence_index,
                });
            }

            if self.scheduling_variant != M13B4SchedulingVariant::SameDispatchFusedCandidate
                && unit.boolean_end_ns > unit.numerical_start_ns
            {
                return Err(M13B4OverlapTimelineError::OwnDecisionNotReady {
                    sequence_index: unit.sequence_index,
                    boolean_end_ns: unit.boolean_end_ns,
                    numerical_start_ns: unit.numerical_start_ns,
                });
            }

            if let Some(previous) = position
                .checked_sub(1)
                .and_then(|index| self.units.get(index))
            {
                let expected = previous.sequence_index.checked_add(1).ok_or(
                    M13B4OverlapTimelineError::SequenceIndexOverflow {
                        sequence_index: previous.sequence_index,
                    },
                )?;
                if unit.sequence_index != expected {
                    return Err(M13B4OverlapTimelineError::NonContiguousSequence {
                        previous: previous.sequence_index,
                        expected,
                        actual: unit.sequence_index,
                    });
                }
            }
        }

        if self.scheduling_variant == M13B4SchedulingVariant::SerialMatched {
            for overlap in self.adjacent_overlaps_unchecked() {
                if overlap.overlap_ns != 0 {
                    return Err(M13B4OverlapTimelineError::SerialOverlapObserved {
                        current_sequence_index: overlap.current_sequence_index,
                        next_sequence_index: overlap.next_sequence_index,
                        overlap_ns: overlap.overlap_ns,
                    });
                }
            }
        }

        Ok(())
    }

    /// Compute exact adjacent-unit temporal intersections.
    ///
    /// # Errors
    ///
    /// Validates the full timeline before returning evidence.
    pub fn adjacent_overlaps(
        &self,
    ) -> Result<Vec<M13B4AdjacentOverlap>, M13B4OverlapTimelineError> {
        self.validate()?;
        Ok(self.adjacent_overlaps_unchecked())
    }

    /// Sum adjacent-unit overlap nanoseconds exactly.
    ///
    /// This is an overlap-duration diagnostic, not an end-to-end speedup metric.
    ///
    /// # Errors
    ///
    /// Validates the timeline and returns an explicit overflow error if the
    /// aggregate duration cannot be represented in `u64`.
    pub fn total_adjacent_overlap_ns(&self) -> Result<u64, M13B4OverlapTimelineError> {
        self.adjacent_overlaps()?.into_iter().try_fold(
            0u64,
            |total, overlap| {
                total
                    .checked_add(overlap.overlap_ns)
                    .ok_or(M13B4OverlapTimelineError::OverlapDurationOverflow)
            },
        )
    }

    fn adjacent_overlaps_unchecked(&self) -> Vec<M13B4AdjacentOverlap> {
        self.units
            .windows(2)
            .map(|pair| {
                let current = pair[0];
                let next = pair[1];
                let start = current.numerical_start_ns.max(next.boolean_start_ns);
                let end = current.numerical_end_ns.min(next.boolean_end_ns);
                M13B4AdjacentOverlap {
                    current_sequence_index: current.sequence_index,
                    next_sequence_index: next.sequence_index,
                    overlap_ns: end.saturating_sub(start),
                }
            })
            .collect()
    }
}

/// Fail-closed M13B.4 overlap-timeline validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum M13B4OverlapTimelineError {
    EmptyClockDomainId,
    InsufficientUnits {
        actual: usize,
    },
    ReversedBooleanInterval {
        sequence_index: u64,
    },
    ReversedNumericalInterval {
        sequence_index: u64,
    },
    OwnDecisionNotReady {
        sequence_index: u64,
        boolean_end_ns: u64,
        numerical_start_ns: u64,
    },
    SequenceIndexOverflow {
        sequence_index: u64,
    },
    NonContiguousSequence {
        previous: u64,
        expected: u64,
        actual: u64,
    },
    SerialOverlapObserved {
        current_sequence_index: u64,
        next_sequence_index: u64,
        overlap_ns: u64,
    },
    OverlapDurationOverflow,
}

impl fmt::Display for M13B4OverlapTimelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for M13B4OverlapTimelineError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(
        sequence_index: u64,
        boolean_start_ns: u64,
        boolean_end_ns: u64,
        numerical_start_ns: u64,
        numerical_end_ns: u64,
    ) -> M13B4UnitIntervals {
        M13B4UnitIntervals {
            sequence_index,
            boolean_start_ns,
            boolean_end_ns,
            numerical_start_ns,
            numerical_end_ns,
        }
    }

    #[test]
    fn multi_dispatch_records_exact_adjacent_overlap_without_claiming_speedup() {
        let timeline = M13B4OverlapTimeline {
            timing_source: M13B4TimingSource::DeviceTimestamp,
            scheduling_variant: M13B4SchedulingVariant::MultiDispatchOverlapCandidate,
            unit_kind: M13B4ConsumableUnitKind::Page,
            clock_domain_id: "gpu-query-set-17".to_owned(),
            units: vec![
                unit(0, 0, 10, 12, 30),
                unit(1, 20, 28, 31, 50),
                unit(2, 45, 55, 56, 70),
            ],
        };

        assert_eq!(
            timeline.adjacent_overlaps().unwrap(),
            vec![
                M13B4AdjacentOverlap {
                    current_sequence_index: 0,
                    next_sequence_index: 1,
                    overlap_ns: 8,
                },
                M13B4AdjacentOverlap {
                    current_sequence_index: 1,
                    next_sequence_index: 2,
                    overlap_ns: 5,
                },
            ]
        );
        assert_eq!(timeline.total_adjacent_overlap_ns().unwrap(), 13);
    }

    #[test]
    fn serial_matched_accepts_only_zero_cross_unit_overlap() {
        let timeline = M13B4OverlapTimeline {
            timing_source: M13B4TimingSource::HostWallClock,
            scheduling_variant: M13B4SchedulingVariant::SerialMatched,
            unit_kind: M13B4ConsumableUnitKind::Block,
            clock_domain_id: "host-run-1".to_owned(),
            units: vec![unit(4, 0, 10, 10, 20), unit(5, 20, 30, 30, 40)],
        };
        assert_eq!(timeline.validate(), Ok(()));
        assert_eq!(timeline.total_adjacent_overlap_ns().unwrap(), 0);
    }

    #[test]
    fn serial_matched_rejects_observed_cross_unit_overlap() {
        let timeline = M13B4OverlapTimeline {
            timing_source: M13B4TimingSource::HostWallClock,
            scheduling_variant: M13B4SchedulingVariant::SerialMatched,
            unit_kind: M13B4ConsumableUnitKind::Block,
            clock_domain_id: "host-run-2".to_owned(),
            units: vec![unit(0, 0, 10, 10, 20), unit(1, 15, 18, 21, 30)],
        };
        assert_eq!(
            timeline.validate(),
            Err(M13B4OverlapTimelineError::SerialOverlapObserved {
                current_sequence_index: 0,
                next_sequence_index: 1,
                overlap_ns: 3,
            })
        );
    }

    #[test]
    fn separate_dispatch_requires_own_boolean_decision_before_numerical_work() {
        let timeline = M13B4OverlapTimeline {
            timing_source: M13B4TimingSource::DeviceTimestamp,
            scheduling_variant: M13B4SchedulingVariant::MultiDispatchOverlapCandidate,
            unit_kind: M13B4ConsumableUnitKind::Page,
            clock_domain_id: "gpu-run".to_owned(),
            units: vec![unit(0, 0, 12, 10, 20), unit(1, 15, 18, 21, 30)],
        };
        assert_eq!(
            timeline.validate(),
            Err(M13B4OverlapTimelineError::OwnDecisionNotReady {
                sequence_index: 0,
                boolean_end_ns: 12,
                numerical_start_ns: 10,
            })
        );
    }

    #[test]
    fn rejects_missing_clock_domain_and_non_contiguous_units() {
        let missing_clock = M13B4OverlapTimeline {
            timing_source: M13B4TimingSource::HostWallClock,
            scheduling_variant: M13B4SchedulingVariant::SerialMatched,
            unit_kind: M13B4ConsumableUnitKind::Page,
            clock_domain_id: String::new(),
            units: vec![unit(0, 0, 1, 1, 2), unit(1, 2, 3, 3, 4)],
        };
        assert_eq!(
            missing_clock.validate(),
            Err(M13B4OverlapTimelineError::EmptyClockDomainId)
        );

        let gap = M13B4OverlapTimeline {
            timing_source: M13B4TimingSource::HostWallClock,
            scheduling_variant: M13B4SchedulingVariant::MultiDispatchOverlapCandidate,
            unit_kind: M13B4ConsumableUnitKind::Page,
            clock_domain_id: "host-run".to_owned(),
            units: vec![unit(7, 0, 1, 1, 2), unit(9, 2, 3, 3, 4)],
        };
        assert_eq!(
            gap.validate(),
            Err(M13B4OverlapTimelineError::NonContiguousSequence {
                previous: 7,
                expected: 8,
                actual: 9,
            })
        );
    }
}
