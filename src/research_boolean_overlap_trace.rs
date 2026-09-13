//! M13B.4 backend-neutral trace/evidence contract.
//!
//! This module records the preregistered event order required to interpret
//! first-token and steady-state Boolean/numerical overlap experiments. It does
//! not schedule work, infer overlap from source structure, or claim a speedup.

use core::fmt;

/// Version of the research-only M13B.4 trace schema.
pub const M13B4_TRACE_SCHEMA_VERSION: u32 = 1;

/// Provenance of timestamps stored in an M13B.4 trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum M13B4TimingSource {
    /// Host wall-clock timestamps around explicitly declared intervals.
    HostWallClock,
    /// Backend/device timestamps emitted by the execution backend.
    DeviceTimestamp,
}

/// Scheduling variant named by the M13B.4 preregistration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum M13B4SchedulingVariant {
    /// Boolean routing completes before numerical survivor execution begins.
    SerialMatched,
    /// Separate Boolean and numerical dispatches with explicit synchronization.
    MultiDispatchOverlapCandidate,
    /// Backend-valid fused/same-dispatch candidate preserving oracle semantics.
    SameDispatchFusedCandidate,
}

/// Execution scope represented by one trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum M13B4TraceScope {
    /// Prefill boundary used to establish first-token readiness.
    Prefill,
    /// First decode token after the qualified prefill boundary.
    FirstDecode,
    /// Later steady-state decode token.
    SteadyStateDecode,
}

/// Preregistered trace events for M13B.4 qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum M13B4TraceEventKind {
    /// Query representation is available to the Boolean plane.
    QueryRepresentationReady,
    /// Q-signature generation starts.
    QSignatureStart,
    /// Q-signature generation ends.
    QSignatureEnd,
    /// Boolean admission/routing starts.
    BooleanRoutingStart,
    /// Boolean admission/routing ends.
    BooleanRoutingEnd,
    /// Survivor metadata is ready for numerical consumption.
    SurvivorMetadataReady,
    /// Numerical K/V staging starts.
    NumericalKvStagingStart,
    /// Numerical attention starts.
    NumericalAttentionStart,
    /// Numerical attention ends.
    NumericalAttentionEnd,
    /// Explicit synchronization wait starts.
    SynchronizationWaitStart,
    /// Explicit synchronization wait ends.
    SynchronizationWaitEnd,
    /// Final attention output is ready.
    OutputReady,
    /// Prefill numerical K/V commit is visible.
    NumericalKvCommit,
    /// Prefill Boolean K/KV signature commit is visible.
    BooleanSignatureCommit,
    /// Numerical and Boolean prefix state are jointly decode-visible.
    DecodeVisible,
}

/// One timestamped M13B.4 event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct M13B4TraceEvent {
    /// Event identity.
    pub kind: M13B4TraceEventKind,
    /// Timestamp in nanoseconds in the declared trace timing domain.
    pub timestamp_ns: u64,
}

/// Research-only ordered trace for one M13B.4 observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct M13B4Trace {
    /// Timestamp provenance shared by every event in this trace.
    pub timing_source: M13B4TimingSource,
    /// Scheduling variant represented by this trace.
    pub scheduling_variant: M13B4SchedulingVariant,
    /// Prefill/first-token/steady-state scope.
    pub scope: M13B4TraceScope,
    /// Events in observed chronological order.
    pub events: Vec<M13B4TraceEvent>,
}

impl M13B4Trace {
    /// Validate uniqueness, chronological order, mandatory events and the
    /// preregistered synchronization requirements.
    ///
    /// # Errors
    ///
    /// Returns a fail-closed error for empty traces, duplicated events,
    /// non-monotonic timestamps, missing mandatory events, invalid event order,
    /// or missing synchronization evidence for a multi-dispatch candidate.
    pub fn validate(&self) -> Result<(), M13B4TraceError> {
        if self.events.is_empty() {
            return Err(M13B4TraceError::EmptyTrace);
        }

        for (index, event) in self.events.iter().enumerate() {
            if let Some(previous) = index.checked_sub(1).and_then(|i| self.events.get(i)) {
                if event.timestamp_ns < previous.timestamp_ns {
                    return Err(M13B4TraceError::NonMonotonicTimestamp {
                        index,
                        previous_ns: previous.timestamp_ns,
                        current_ns: event.timestamp_ns,
                    });
                }
            }
            if self.events[..index].iter().any(|prior| prior.kind == event.kind) {
                return Err(M13B4TraceError::DuplicateEvent { kind: event.kind });
            }
        }

        match self.scope {
            M13B4TraceScope::Prefill => self.validate_prefill()?,
            M13B4TraceScope::FirstDecode | M13B4TraceScope::SteadyStateDecode => {
                self.validate_decode()?
            }
        }

        Ok(())
    }

    /// Return the timestamp for one unique event.
    ///
    /// # Errors
    ///
    /// Returns [`M13B4TraceError::MissingEvent`] if the event is absent.
    pub fn timestamp_ns(&self, kind: M13B4TraceEventKind) -> Result<u64, M13B4TraceError> {
        self.events
            .iter()
            .find(|event| event.kind == kind)
            .map(|event| event.timestamp_ns)
            .ok_or(M13B4TraceError::MissingEvent { kind })
    }

    /// Return the duration between two required ordered events.
    ///
    /// # Errors
    ///
    /// Returns an error if either event is missing or if the requested end event
    /// precedes the start event.
    pub fn interval_ns(
        &self,
        start: M13B4TraceEventKind,
        end: M13B4TraceEventKind,
    ) -> Result<u64, M13B4TraceError> {
        let start_ns = self.timestamp_ns(start)?;
        let end_ns = self.timestamp_ns(end)?;
        end_ns
            .checked_sub(start_ns)
            .ok_or(M13B4TraceError::InvalidEventOrder {
                before: start,
                after: end,
            })
    }

    fn validate_prefill(&self) -> Result<(), M13B4TraceError> {
        let numerical = self.timestamp_ns(M13B4TraceEventKind::NumericalKvCommit)?;
        let boolean = self.timestamp_ns(M13B4TraceEventKind::BooleanSignatureCommit)?;
        let visible = self.timestamp_ns(M13B4TraceEventKind::DecodeVisible)?;
        if numerical > visible {
            return Err(M13B4TraceError::InvalidEventOrder {
                before: M13B4TraceEventKind::NumericalKvCommit,
                after: M13B4TraceEventKind::DecodeVisible,
            });
        }
        if boolean > visible {
            return Err(M13B4TraceError::InvalidEventOrder {
                before: M13B4TraceEventKind::BooleanSignatureCommit,
                after: M13B4TraceEventKind::DecodeVisible,
            });
        }
        Ok(())
    }

    fn validate_decode(&self) -> Result<(), M13B4TraceError> {
        const REQUIRED: [M13B4TraceEventKind; 10] = [
            M13B4TraceEventKind::QueryRepresentationReady,
            M13B4TraceEventKind::QSignatureStart,
            M13B4TraceEventKind::QSignatureEnd,
            M13B4TraceEventKind::BooleanRoutingStart,
            M13B4TraceEventKind::BooleanRoutingEnd,
            M13B4TraceEventKind::SurvivorMetadataReady,
            M13B4TraceEventKind::NumericalKvStagingStart,
            M13B4TraceEventKind::NumericalAttentionStart,
            M13B4TraceEventKind::NumericalAttentionEnd,
            M13B4TraceEventKind::OutputReady,
        ];

        for pair in REQUIRED.windows(2) {
            let before = pair[0];
            let after = pair[1];
            if self.timestamp_ns(before)? > self.timestamp_ns(after)? {
                return Err(M13B4TraceError::InvalidEventOrder { before, after });
            }
        }

        let wait_start = self
            .events
            .iter()
            .find(|event| event.kind == M13B4TraceEventKind::SynchronizationWaitStart);
        let wait_end = self
            .events
            .iter()
            .find(|event| event.kind == M13B4TraceEventKind::SynchronizationWaitEnd);

        match (wait_start, wait_end) {
            (Some(start), Some(end)) => {
                if start.timestamp_ns > end.timestamp_ns {
                    return Err(M13B4TraceError::InvalidEventOrder {
                        before: M13B4TraceEventKind::SynchronizationWaitStart,
                        after: M13B4TraceEventKind::SynchronizationWaitEnd,
                    });
                }
            }
            (None, None) => {
                if self.scheduling_variant
                    == M13B4SchedulingVariant::MultiDispatchOverlapCandidate
                {
                    return Err(M13B4TraceError::MissingSynchronizationEvidence);
                }
            }
            _ => return Err(M13B4TraceError::IncompleteSynchronizationPair),
        }

        Ok(())
    }
}

/// Fail-closed M13B.4 trace validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum M13B4TraceError {
    /// No events were recorded.
    EmptyTrace,
    /// A timestamp moved backwards in the ordered event stream.
    NonMonotonicTimestamp {
        /// Index of the event that moved backwards.
        index: usize,
        /// Previous event timestamp.
        previous_ns: u64,
        /// Current event timestamp.
        current_ns: u64,
    },
    /// One event kind appeared more than once.
    DuplicateEvent {
        /// Duplicated event kind.
        kind: M13B4TraceEventKind,
    },
    /// A mandatory event was absent.
    MissingEvent {
        /// Missing event kind.
        kind: M13B4TraceEventKind,
    },
    /// Two required events were observed in an invalid order.
    InvalidEventOrder {
        /// Event that must not occur after `after`.
        before: M13B4TraceEventKind,
        /// Event that must not occur before `before`.
        after: M13B4TraceEventKind,
    },
    /// Only one synchronization endpoint was recorded.
    IncompleteSynchronizationPair,
    /// Multi-dispatch candidate omitted explicit synchronization evidence.
    MissingSynchronizationEvidence,
}

impl fmt::Display for M13B4TraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for M13B4TraceError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_events(include_wait: bool) -> Vec<M13B4TraceEvent> {
        let mut events = vec![
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::QueryRepresentationReady,
                timestamp_ns: 10,
            },
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::QSignatureStart,
                timestamp_ns: 11,
            },
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::QSignatureEnd,
                timestamp_ns: 15,
            },
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::BooleanRoutingStart,
                timestamp_ns: 16,
            },
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::BooleanRoutingEnd,
                timestamp_ns: 20,
            },
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::SurvivorMetadataReady,
                timestamp_ns: 21,
            },
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::NumericalKvStagingStart,
                timestamp_ns: 22,
            },
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::NumericalAttentionStart,
                timestamp_ns: 23,
            },
        ];
        if include_wait {
            events.push(M13B4TraceEvent {
                kind: M13B4TraceEventKind::SynchronizationWaitStart,
                timestamp_ns: 24,
            });
            events.push(M13B4TraceEvent {
                kind: M13B4TraceEventKind::SynchronizationWaitEnd,
                timestamp_ns: 25,
            });
        }
        events.extend([
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::NumericalAttentionEnd,
                timestamp_ns: 30,
            },
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::OutputReady,
                timestamp_ns: 31,
            },
        ]);
        events
    }

    #[test]
    fn first_decode_trace_validates_and_exposes_exact_intervals() {
        let trace = M13B4Trace {
            timing_source: M13B4TimingSource::DeviceTimestamp,
            scheduling_variant: M13B4SchedulingVariant::MultiDispatchOverlapCandidate,
            scope: M13B4TraceScope::FirstDecode,
            events: decode_events(true),
        };
        trace.validate().unwrap();
        assert_eq!(
            trace
                .interval_ns(
                    M13B4TraceEventKind::QSignatureStart,
                    M13B4TraceEventKind::QSignatureEnd,
                )
                .unwrap(),
            4
        );
        assert_eq!(
            trace
                .interval_ns(
                    M13B4TraceEventKind::BooleanRoutingStart,
                    M13B4TraceEventKind::BooleanRoutingEnd,
                )
                .unwrap(),
            4
        );
    }

    #[test]
    fn prefill_requires_both_commits_before_decode_visibility() {
        let trace = M13B4Trace {
            timing_source: M13B4TimingSource::HostWallClock,
            scheduling_variant: M13B4SchedulingVariant::SerialMatched,
            scope: M13B4TraceScope::Prefill,
            events: vec![
                M13B4TraceEvent {
                    kind: M13B4TraceEventKind::NumericalKvCommit,
                    timestamp_ns: 100,
                },
                M13B4TraceEvent {
                    kind: M13B4TraceEventKind::BooleanSignatureCommit,
                    timestamp_ns: 110,
                },
                M13B4TraceEvent {
                    kind: M13B4TraceEventKind::DecodeVisible,
                    timestamp_ns: 120,
                },
            ],
        };
        assert_eq!(trace.validate(), Ok(()));
    }

    #[test]
    fn multi_dispatch_requires_explicit_synchronization_pair() {
        let trace = M13B4Trace {
            timing_source: M13B4TimingSource::HostWallClock,
            scheduling_variant: M13B4SchedulingVariant::MultiDispatchOverlapCandidate,
            scope: M13B4TraceScope::SteadyStateDecode,
            events: decode_events(false),
        };
        assert_eq!(
            trace.validate(),
            Err(M13B4TraceError::MissingSynchronizationEvidence)
        );
    }

    #[test]
    fn serial_matched_decode_does_not_invent_sync_waits() {
        let trace = M13B4Trace {
            timing_source: M13B4TimingSource::HostWallClock,
            scheduling_variant: M13B4SchedulingVariant::SerialMatched,
            scope: M13B4TraceScope::SteadyStateDecode,
            events: decode_events(false),
        };
        assert_eq!(trace.validate(), Ok(()));
    }

    #[test]
    fn duplicate_or_backwards_events_fail_closed() {
        let mut duplicate = decode_events(false);
        duplicate.insert(
            1,
            M13B4TraceEvent {
                kind: M13B4TraceEventKind::QueryRepresentationReady,
                timestamp_ns: 10,
            },
        );
        let trace = M13B4Trace {
            timing_source: M13B4TimingSource::HostWallClock,
            scheduling_variant: M13B4SchedulingVariant::SerialMatched,
            scope: M13B4TraceScope::FirstDecode,
            events: duplicate,
        };
        assert_eq!(
            trace.validate(),
            Err(M13B4TraceError::DuplicateEvent {
                kind: M13B4TraceEventKind::QueryRepresentationReady,
            })
        );

        let mut backwards = decode_events(false);
        backwards[3].timestamp_ns = 9;
        let trace = M13B4Trace {
            timing_source: M13B4TimingSource::HostWallClock,
            scheduling_variant: M13B4SchedulingVariant::SerialMatched,
            scope: M13B4TraceScope::FirstDecode,
            events: backwards,
        };
        assert!(matches!(
            trace.validate(),
            Err(M13B4TraceError::NonMonotonicTimestamp { .. })
        ));
    }
}
