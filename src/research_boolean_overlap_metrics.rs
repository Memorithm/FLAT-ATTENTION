//! M13B.4 per-unit trace accounting derived from explicit evidence.
//!
//! The summary in this module is descriptive only. It turns one validated
//! first-token or steady-state decode trace into exact interval accounting for
//! the preregistered phases. It does not infer GPU concurrency, bandwidth
//! reduction, latency improvement, or useful overlap from the intervals.

use core::fmt;

use super::trace::{
    M13B4SchedulingVariant, M13B4TimingSource, M13B4Trace, M13B4TraceError, M13B4TraceEventKind,
    M13B4TraceScope,
};

/// Version of the research-only per-unit trace summary schema.
pub const M13B4_TRACE_SUMMARY_SCHEMA_VERSION: u32 = 1;

/// Exact interval accounting for one validated decode trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M13B4TraceSummary {
    /// Timing provenance copied from the validated trace.
    pub timing_source: M13B4TimingSource,
    /// Scheduling variant copied from the validated trace.
    pub scheduling_variant: M13B4SchedulingVariant,
    /// First-token or steady-state scope copied from the validated trace.
    pub scope: M13B4TraceScope,
    /// Dispatches observed by the measurement harness for this execution unit.
    pub dispatch_count: u32,
    /// Query-ready through output-ready interval.
    pub query_to_output_ns: u64,
    /// Q-signature generation interval.
    pub q_signature_ns: u64,
    /// Boolean routing interval.
    pub boolean_routing_ns: u64,
    /// Survivor-metadata-ready through numerical-K/V-staging-start gap.
    ///
    /// This is an observed dependency gap, not automatically a stall cause.
    pub survivor_to_kv_staging_gap_ns: u64,
    /// Numerical K/V staging start through numerical attention start.
    pub kv_staging_ns: u64,
    /// Numerical attention interval.
    pub numerical_attention_ns: u64,
    /// Numerical-attention-end through output-ready interval.
    pub output_finalize_ns: u64,
    /// Explicit synchronization wait when the trace contains the preregistered
    /// wait pair. `None` means no wait pair was recorded; it does not mean that
    /// implicit backend synchronization is impossible.
    pub synchronization_wait_ns: Option<u64>,
}

/// Fail-closed errors for M13B.4 trace summary construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum M13B4TraceSummaryError {
    /// The source trace itself is invalid.
    Trace(M13B4TraceError),
    /// Prefill uses a different event contract and cannot be summarized as a
    /// decode execution unit.
    PrefillScope,
    /// The measurement harness must report at least one dispatch.
    ZeroDispatchCount,
}

/// Build exact phase accounting from one validated decode trace.
///
/// `dispatch_count` is supplied by the execution harness because the trace
/// schema intentionally records semantic events rather than backend dispatch
/// objects. The value is evidence only; this function neither predicts nor
/// optimizes dispatch count.
///
/// # Errors
///
/// Returns an error when the trace fails its own contract, when the trace is a
/// prefill trace, or when the external harness reports zero dispatches.
pub fn summarize_decode_trace(
    trace: &M13B4Trace,
    dispatch_count: u32,
) -> Result<M13B4TraceSummary, M13B4TraceSummaryError> {
    trace.validate().map_err(M13B4TraceSummaryError::Trace)?;
    if trace.scope == M13B4TraceScope::Prefill {
        return Err(M13B4TraceSummaryError::PrefillScope);
    }
    if dispatch_count == 0 {
        return Err(M13B4TraceSummaryError::ZeroDispatchCount);
    }

    let interval = |start, end| {
        trace
            .interval_ns(start, end)
            .map_err(M13B4TraceSummaryError::Trace)
    };

    let synchronization_wait_ns = match (
        trace
            .events
            .iter()
            .any(|event| event.kind == M13B4TraceEventKind::SynchronizationWaitStart),
        trace
            .events
            .iter()
            .any(|event| event.kind == M13B4TraceEventKind::SynchronizationWaitEnd),
    ) {
        (true, true) => Some(interval(
            M13B4TraceEventKind::SynchronizationWaitStart,
            M13B4TraceEventKind::SynchronizationWaitEnd,
        )?),
        // `validate()` already rejects an incomplete pair.
        (false, false) => None,
        _ => unreachable!("validated trace cannot contain an incomplete synchronization pair"),
    };

    Ok(M13B4TraceSummary {
        timing_source: trace.timing_source,
        scheduling_variant: trace.scheduling_variant,
        scope: trace.scope,
        dispatch_count,
        query_to_output_ns: interval(
            M13B4TraceEventKind::QueryRepresentationReady,
            M13B4TraceEventKind::OutputReady,
        )?,
        q_signature_ns: interval(
            M13B4TraceEventKind::QSignatureStart,
            M13B4TraceEventKind::QSignatureEnd,
        )?,
        boolean_routing_ns: interval(
            M13B4TraceEventKind::BooleanRoutingStart,
            M13B4TraceEventKind::BooleanRoutingEnd,
        )?,
        survivor_to_kv_staging_gap_ns: interval(
            M13B4TraceEventKind::SurvivorMetadataReady,
            M13B4TraceEventKind::NumericalKvStagingStart,
        )?,
        kv_staging_ns: interval(
            M13B4TraceEventKind::NumericalKvStagingStart,
            M13B4TraceEventKind::NumericalAttentionStart,
        )?,
        numerical_attention_ns: interval(
            M13B4TraceEventKind::NumericalAttentionStart,
            M13B4TraceEventKind::NumericalAttentionEnd,
        )?,
        output_finalize_ns: interval(
            M13B4TraceEventKind::NumericalAttentionEnd,
            M13B4TraceEventKind::OutputReady,
        )?,
        synchronization_wait_ns,
    })
}

impl fmt::Display for M13B4TraceSummaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for M13B4TraceSummaryError {}

#[cfg(test)]
mod tests {
    use super::super::trace::M13B4TraceEvent;
    use super::*;

    fn event(kind: M13B4TraceEventKind, timestamp_ns: u64) -> M13B4TraceEvent {
        M13B4TraceEvent { kind, timestamp_ns }
    }

    fn decode_trace(scope: M13B4TraceScope, with_wait: bool) -> M13B4Trace {
        let mut events = vec![
            event(M13B4TraceEventKind::QueryRepresentationReady, 10),
            event(M13B4TraceEventKind::QSignatureStart, 12),
            event(M13B4TraceEventKind::QSignatureEnd, 17),
            event(M13B4TraceEventKind::BooleanRoutingStart, 18),
            event(M13B4TraceEventKind::BooleanRoutingEnd, 24),
            event(M13B4TraceEventKind::SurvivorMetadataReady, 25),
            event(M13B4TraceEventKind::NumericalKvStagingStart, 29),
            event(M13B4TraceEventKind::NumericalAttentionStart, 36),
        ];
        if with_wait {
            events.push(event(M13B4TraceEventKind::SynchronizationWaitStart, 37));
            events.push(event(M13B4TraceEventKind::SynchronizationWaitEnd, 40));
        }
        events.push(event(M13B4TraceEventKind::NumericalAttentionEnd, 48));
        events.push(event(M13B4TraceEventKind::OutputReady, 51));
        M13B4Trace {
            timing_source: M13B4TimingSource::DeviceTimestamp,
            scheduling_variant: if with_wait {
                M13B4SchedulingVariant::MultiDispatchOverlapCandidate
            } else {
                M13B4SchedulingVariant::SerialMatched
            },
            scope,
            events,
        }
    }

    #[test]
    fn summarizes_exact_preregistered_intervals() {
        let summary =
            summarize_decode_trace(&decode_trace(M13B4TraceScope::FirstDecode, true), 3).unwrap();
        assert_eq!(summary.scope, M13B4TraceScope::FirstDecode);
        assert_eq!(summary.dispatch_count, 3);
        assert_eq!(summary.query_to_output_ns, 41);
        assert_eq!(summary.q_signature_ns, 5);
        assert_eq!(summary.boolean_routing_ns, 6);
        assert_eq!(summary.survivor_to_kv_staging_gap_ns, 4);
        assert_eq!(summary.kv_staging_ns, 7);
        assert_eq!(summary.numerical_attention_ns, 12);
        assert_eq!(summary.output_finalize_ns, 3);
        assert_eq!(summary.synchronization_wait_ns, Some(3));
    }

    #[test]
    fn preserves_absent_wait_as_absent_instead_of_zero() {
        let summary =
            summarize_decode_trace(&decode_trace(M13B4TraceScope::SteadyStateDecode, false), 1)
                .unwrap();
        assert_eq!(summary.synchronization_wait_ns, None);
    }

    #[test]
    fn rejects_zero_dispatch_count() {
        assert_eq!(
            summarize_decode_trace(&decode_trace(M13B4TraceScope::FirstDecode, false), 0),
            Err(M13B4TraceSummaryError::ZeroDispatchCount)
        );
    }

    #[test]
    fn rejects_prefill_scope() {
        let prefill = M13B4Trace {
            timing_source: M13B4TimingSource::DeviceTimestamp,
            scheduling_variant: M13B4SchedulingVariant::SerialMatched,
            scope: M13B4TraceScope::Prefill,
            events: vec![
                event(M13B4TraceEventKind::NumericalKvCommit, 10),
                event(M13B4TraceEventKind::BooleanSignatureCommit, 12),
                event(M13B4TraceEventKind::DecodeVisible, 13),
            ],
        };
        assert_eq!(
            summarize_decode_trace(&prefill, 1),
            Err(M13B4TraceSummaryError::PrefillScope)
        );
    }

    #[test]
    fn invalid_trace_is_never_summarized() {
        let mut trace = decode_trace(M13B4TraceScope::SteadyStateDecode, false);
        trace.events.pop();
        assert!(matches!(
            summarize_decode_trace(&trace, 1),
            Err(M13B4TraceSummaryError::Trace(_))
        ));
    }
}
