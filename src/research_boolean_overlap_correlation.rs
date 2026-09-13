//! M13B.4 correlated cross-unit overlap evidence.
//!
//! A single trace cannot establish temporal overlap between Boolean work for a
//! future decode unit and numerical work for the current unit. This module adds
//! the minimum explicit correlation metadata required to compare two validated
//! traces in one declared timing domain. It measures interval intersection only;
//! it does not infer latency improvement, throughput improvement, or useful GPU
//! concurrency from a non-zero intersection.

use core::fmt;

use super::trace::{
    M13B4SchedulingVariant, M13B4TimingSource, M13B4Trace, M13B4TraceError,
    M13B4TraceEventKind, M13B4TraceScope,
};

/// One validated trace plus the external correlation metadata required to place
/// it in a sequence of execution units sharing one timing domain.
#[derive(Debug, Clone, Copy)]
pub struct M13B4CorrelatedTraceRef<'a> {
    /// Trace for one decode execution unit.
    pub trace: &'a M13B4Trace,
    /// Stable experiment-owned identifier for the clock/timestamp domain.
    /// Equal timing-source enums alone are insufficient to prove comparability.
    pub timing_domain_id: &'a str,
    /// Monotonic execution-unit index assigned by the measurement harness.
    pub unit_index: u64,
}

/// Exact interval evidence for Boolean work of unit `n+1` intersecting numerical
/// attention work of unit `n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M13B4CrossUnitOverlapEvidence {
    pub timing_source: M13B4TimingSource,
    pub scheduling_variant: M13B4SchedulingVariant,
    pub current_unit_index: u64,
    pub next_unit_index: u64,
    pub current_numerical_start_ns: u64,
    pub current_numerical_end_ns: u64,
    pub next_boolean_start_ns: u64,
    pub next_boolean_end_ns: u64,
    /// Exact intersection duration of the two declared intervals.
    /// A positive value is evidence of temporal intersection only, not speedup.
    pub overlap_ns: u64,
}

/// Fail-closed errors for cross-unit M13B.4 correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum M13B4CorrelationError {
    CurrentTrace(M13B4TraceError),
    NextTrace(M13B4TraceError),
    EmptyTimingDomainId,
    TimingDomainMismatch,
    TimingSourceMismatch {
        current: M13B4TimingSource,
        next: M13B4TimingSource,
    },
    SchedulingVariantMismatch {
        current: M13B4SchedulingVariant,
        next: M13B4SchedulingVariant,
    },
    PrefillScopeNotCorrelatable,
    UnitIndexOverflow,
    NonAdjacentUnitIndices {
        current: u64,
        next: u64,
    },
}

/// Measure temporal intersection between numerical attention of the current
/// decode unit and Boolean front-end work of the immediately following unit.
///
/// The Boolean interval is `QSignatureStart..SurvivorMetadataReady`; this covers
/// Q-signature generation plus Boolean admission/routing under the preregistered
/// trace contract. The numerical interval is
/// `NumericalAttentionStart..NumericalAttentionEnd`.
///
/// # Errors
///
/// Fails closed unless both traces validate, both are decode scopes, timing
/// domain identifiers and timing sources match, scheduling variants match, and
/// the execution-unit indices are exactly adjacent.
pub fn measure_cross_unit_overlap(
    current: M13B4CorrelatedTraceRef<'_>,
    next: M13B4CorrelatedTraceRef<'_>,
) -> Result<M13B4CrossUnitOverlapEvidence, M13B4CorrelationError> {
    current
        .trace
        .validate()
        .map_err(M13B4CorrelationError::CurrentTrace)?;
    next.trace
        .validate()
        .map_err(M13B4CorrelationError::NextTrace)?;

    if current.trace.scope == M13B4TraceScope::Prefill
        || next.trace.scope == M13B4TraceScope::Prefill
    {
        return Err(M13B4CorrelationError::PrefillScopeNotCorrelatable);
    }
    if current.timing_domain_id.is_empty() || next.timing_domain_id.is_empty() {
        return Err(M13B4CorrelationError::EmptyTimingDomainId);
    }
    if current.timing_domain_id != next.timing_domain_id {
        return Err(M13B4CorrelationError::TimingDomainMismatch);
    }
    if current.trace.timing_source != next.trace.timing_source {
        return Err(M13B4CorrelationError::TimingSourceMismatch {
            current: current.trace.timing_source,
            next: next.trace.timing_source,
        });
    }
    if current.trace.scheduling_variant != next.trace.scheduling_variant {
        return Err(M13B4CorrelationError::SchedulingVariantMismatch {
            current: current.trace.scheduling_variant,
            next: next.trace.scheduling_variant,
        });
    }

    let expected_next = current
        .unit_index
        .checked_add(1)
        .ok_or(M13B4CorrelationError::UnitIndexOverflow)?;
    if next.unit_index != expected_next {
        return Err(M13B4CorrelationError::NonAdjacentUnitIndices {
            current: current.unit_index,
            next: next.unit_index,
        });
    }

    let current_numerical_start_ns = current
        .trace
        .timestamp_ns(M13B4TraceEventKind::NumericalAttentionStart)
        .map_err(M13B4CorrelationError::CurrentTrace)?;
    let current_numerical_end_ns = current
        .trace
        .timestamp_ns(M13B4TraceEventKind::NumericalAttentionEnd)
        .map_err(M13B4CorrelationError::CurrentTrace)?;
    let next_boolean_start_ns = next
        .trace
        .timestamp_ns(M13B4TraceEventKind::QSignatureStart)
        .map_err(M13B4CorrelationError::NextTrace)?;
    let next_boolean_end_ns = next
        .trace
        .timestamp_ns(M13B4TraceEventKind::SurvivorMetadataReady)
        .map_err(M13B4CorrelationError::NextTrace)?;

    let intersection_start = current_numerical_start_ns.max(next_boolean_start_ns);
    let intersection_end = current_numerical_end_ns.min(next_boolean_end_ns);
    let overlap_ns = intersection_end.saturating_sub(intersection_start);

    Ok(M13B4CrossUnitOverlapEvidence {
        timing_source: current.trace.timing_source,
        scheduling_variant: current.trace.scheduling_variant,
        current_unit_index: current.unit_index,
        next_unit_index: next.unit_index,
        current_numerical_start_ns,
        current_numerical_end_ns,
        next_boolean_start_ns,
        next_boolean_end_ns,
        overlap_ns,
    })
}

impl fmt::Display for M13B4CorrelationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for M13B4CorrelationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::trace::M13B4TraceEvent;

    fn decode_trace(offset_ns: u64, timing_source: M13B4TimingSource) -> M13B4Trace {
        let event = |kind, timestamp_ns| M13B4TraceEvent {
            kind,
            timestamp_ns: offset_ns + timestamp_ns,
        };
        M13B4Trace {
            timing_source,
            scheduling_variant: M13B4SchedulingVariant::SerialMatched,
            scope: M13B4TraceScope::SteadyStateDecode,
            events: vec![
                event(M13B4TraceEventKind::QueryRepresentationReady, 10),
                event(M13B4TraceEventKind::QSignatureStart, 11),
                event(M13B4TraceEventKind::QSignatureEnd, 15),
                event(M13B4TraceEventKind::BooleanRoutingStart, 16),
                event(M13B4TraceEventKind::BooleanRoutingEnd, 20),
                event(M13B4TraceEventKind::SurvivorMetadataReady, 21),
                event(M13B4TraceEventKind::NumericalKvStagingStart, 22),
                event(M13B4TraceEventKind::NumericalAttentionStart, 23),
                event(M13B4TraceEventKind::NumericalAttentionEnd, 30),
                event(M13B4TraceEventKind::OutputReady, 31),
            ],
        }
    }

    #[test]
    fn measures_exact_cross_unit_interval_intersection() {
        let current_trace = decode_trace(0, M13B4TimingSource::DeviceTimestamp);
        let next_trace = decode_trace(14, M13B4TimingSource::DeviceTimestamp);
        let evidence = measure_cross_unit_overlap(
            M13B4CorrelatedTraceRef {
                trace: &current_trace,
                timing_domain_id: "gpu-clock-0",
                unit_index: 41,
            },
            M13B4CorrelatedTraceRef {
                trace: &next_trace,
                timing_domain_id: "gpu-clock-0",
                unit_index: 42,
            },
        )
        .unwrap();

        assert_eq!(evidence.current_numerical_start_ns, 23);
        assert_eq!(evidence.current_numerical_end_ns, 30);
        assert_eq!(evidence.next_boolean_start_ns, 25);
        assert_eq!(evidence.next_boolean_end_ns, 35);
        assert_eq!(evidence.overlap_ns, 5);
    }

    #[test]
    fn disjoint_intervals_report_zero_without_claiming_failure() {
        let current_trace = decode_trace(0, M13B4TimingSource::HostWallClock);
        let next_trace = decode_trace(30, M13B4TimingSource::HostWallClock);
        let evidence = measure_cross_unit_overlap(
            M13B4CorrelatedTraceRef {
                trace: &current_trace,
                timing_domain_id: "host-monotonic-0",
                unit_index: 7,
            },
            M13B4CorrelatedTraceRef {
                trace: &next_trace,
                timing_domain_id: "host-monotonic-0",
                unit_index: 8,
            },
        )
        .unwrap();
        assert_eq!(evidence.overlap_ns, 0);
    }

    #[test]
    fn rejects_unproven_clock_correlation_and_non_adjacent_units() {
        let current_trace = decode_trace(0, M13B4TimingSource::DeviceTimestamp);
        let next_trace = decode_trace(14, M13B4TimingSource::DeviceTimestamp);

        assert_eq!(
            measure_cross_unit_overlap(
                M13B4CorrelatedTraceRef {
                    trace: &current_trace,
                    timing_domain_id: "queue-a-clock",
                    unit_index: 1,
                },
                M13B4CorrelatedTraceRef {
                    trace: &next_trace,
                    timing_domain_id: "queue-b-clock",
                    unit_index: 2,
                },
            ),
            Err(M13B4CorrelationError::TimingDomainMismatch)
        );

        assert_eq!(
            measure_cross_unit_overlap(
                M13B4CorrelatedTraceRef {
                    trace: &current_trace,
                    timing_domain_id: "device-0",
                    unit_index: 1,
                },
                M13B4CorrelatedTraceRef {
                    trace: &next_trace,
                    timing_domain_id: "device-0",
                    unit_index: 3,
                },
            ),
            Err(M13B4CorrelationError::NonAdjacentUnitIndices {
                current: 1,
                next: 3,
            })
        );
    }

    #[test]
    fn rejects_timing_source_mismatch_even_with_same_domain_label() {
        let current_trace = decode_trace(0, M13B4TimingSource::HostWallClock);
        let next_trace = decode_trace(14, M13B4TimingSource::DeviceTimestamp);
        assert!(matches!(
            measure_cross_unit_overlap(
                M13B4CorrelatedTraceRef {
                    trace: &current_trace,
                    timing_domain_id: "bad-shared-label",
                    unit_index: 1,
                },
                M13B4CorrelatedTraceRef {
                    trace: &next_trace,
                    timing_domain_id: "bad-shared-label",
                    unit_index: 2,
                },
            ),
            Err(M13B4CorrelationError::TimingSourceMismatch { .. })
        ));
    }
}