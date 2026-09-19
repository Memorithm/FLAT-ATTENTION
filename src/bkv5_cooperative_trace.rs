//! BKV-5 bounded cooperative timing-trace contract.
//!
//! This research-only module records caller-observed monotonic timing for one
//! qualified cooperative selection and, optionally, production of the next
//! Boolean decision while the current numerical consumer is active. It owns no
//! clock, thread, queue, WGPU submission, scheduler, or performance verdict.

use core::fmt;

use crate::api::bkv5_cooperative_selection_handoff::{
    CooperativeSelectionDisposition, CooperativeSelectionTicket,
};

/// Version of the bounded BKV-5 timing trace semantics.
pub const BKV5_COOPERATIVE_TRACE_SCHEMA_VERSION: u32 = 1;

/// Decode position represented by one trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CooperativeTraceMode {
    FirstToken,
    SteadyState,
}

/// Caller-observed monotonic timing in nanoseconds relative to one trace origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CooperativeTraceTiming {
    pub selection_ready_ns: u64,
    pub handoff_qualified_ns: u64,
    pub numerical_start_ns: u64,
    pub numerical_finish_ns: u64,
    pub next_boolean_start_ns: Option<u64>,
    pub next_boolean_finish_ns: Option<u64>,
}

/// Validated immutable trace bound to one qualified handoff ticket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CooperativeSelectionTrace {
    mode: CooperativeTraceMode,
    generation: u64,
    prefix_tokens: usize,
    mapped_pages: usize,
    selected_pages: usize,
    disposition: CooperativeSelectionDisposition,
    timing: CooperativeTraceTiming,
    observed_overlap_ns: Option<u64>,
}

impl CooperativeSelectionTrace {
    #[must_use]
    pub const fn mode(self) -> CooperativeTraceMode {
        self.mode
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }

    #[must_use]
    pub const fn prefix_tokens(self) -> usize {
        self.prefix_tokens
    }

    #[must_use]
    pub const fn mapped_pages(self) -> usize {
        self.mapped_pages
    }

    #[must_use]
    pub const fn selected_pages(self) -> usize {
        self.selected_pages
    }

    #[must_use]
    pub const fn disposition(self) -> CooperativeSelectionDisposition {
        self.disposition
    }

    #[must_use]
    pub const fn timing(self) -> CooperativeTraceTiming {
        self.timing
    }

    /// Intersection of next-Boolean production and current numerical execution.
    ///
    /// `None` means that the trace did not contain a complete next-Boolean
    /// interval. `Some(0)` is explicit evidence that both intervals existed but
    /// did not overlap. This is caller-observed timing, not a hardware claim.
    #[must_use]
    pub const fn observed_overlap_ns(self) -> Option<u64> {
        self.observed_overlap_ns
    }
}

/// Fail-closed trace validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CooperativeTraceError {
    HandoffBeforeSelectionReady,
    NumericalStartBeforeHandoff,
    NumericalFinishBeforeStart,
    IncompleteNextBooleanInterval,
    NextBooleanFinishBeforeStart,
}

impl fmt::Display for CooperativeTraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::HandoffBeforeSelectionReady => "handoff timestamp precedes selection readiness",
            Self::NumericalStartBeforeHandoff => "numerical start precedes qualified handoff",
            Self::NumericalFinishBeforeStart => "numerical finish precedes numerical start",
            Self::IncompleteNextBooleanInterval => {
                "next Boolean interval requires start and finish"
            }
            Self::NextBooleanFinishBeforeStart => "next Boolean finish precedes next Boolean start",
        };
        f.write_str(message)
    }
}

impl std::error::Error for CooperativeTraceError {}

fn interval_overlap_ns(a_start: u64, a_finish: u64, b_start: u64, b_finish: u64) -> u64 {
    let start = a_start.max(b_start);
    let finish = a_finish.min(b_finish);
    finish.saturating_sub(start)
}

/// Validate one caller-observed timing record and bind it to a qualified ticket.
///
/// This pure function owns no clock or execution resource. Its timing record is
/// evidence input for later target-backend qualification, not a speedup claim.
pub fn qualify_cooperative_trace(
    ticket: CooperativeSelectionTicket,
    mode: CooperativeTraceMode,
    timing: CooperativeTraceTiming,
) -> Result<CooperativeSelectionTrace, CooperativeTraceError> {
    if timing.handoff_qualified_ns < timing.selection_ready_ns {
        return Err(CooperativeTraceError::HandoffBeforeSelectionReady);
    }
    if timing.numerical_start_ns < timing.handoff_qualified_ns {
        return Err(CooperativeTraceError::NumericalStartBeforeHandoff);
    }
    if timing.numerical_finish_ns < timing.numerical_start_ns {
        return Err(CooperativeTraceError::NumericalFinishBeforeStart);
    }

    let observed_overlap_ns = match (timing.next_boolean_start_ns, timing.next_boolean_finish_ns) {
        (None, None) => None,
        (Some(_), None) | (None, Some(_)) => {
            return Err(CooperativeTraceError::IncompleteNextBooleanInterval)
        }
        (Some(start), Some(finish)) => {
            if finish < start {
                return Err(CooperativeTraceError::NextBooleanFinishBeforeStart);
            }
            Some(interval_overlap_ns(
                timing.numerical_start_ns,
                timing.numerical_finish_ns,
                start,
                finish,
            ))
        }
    };

    Ok(CooperativeSelectionTrace {
        mode,
        generation: ticket.generation(),
        prefix_tokens: ticket.prefix_tokens(),
        mapped_pages: ticket.mapped_pages(),
        selected_pages: ticket.selected_pages(),
        disposition: ticket.disposition(),
        timing,
        observed_overlap_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::bkv5_cooperative_selection_handoff::qualify_cooperative_selection;
    use crate::api::boolean_kv_paged_selection::{
        BooleanIndexedKvSelection, BooleanKvSelectedPage,
    };
    use crate::api::research_boolean_overlap::{
        qualify_first_decode_prefix, PrefillReadinessSnapshot,
    };

    fn ticket() -> CooperativeSelectionTicket {
        let ready = qualify_first_decode_prefix(PrefillReadinessSnapshot {
            numerical_generation: 5,
            boolean_generation: 5,
            prefix_tokens: 8,
            page_size: 4,
            numerical_mapped_pages: 2,
            boolean_signature_pages: 2,
        })
        .unwrap();
        let selection = BooleanIndexedKvSelection {
            generation: 5,
            signature_bits: 64,
            max_distance: 2,
            live_tokens: 8,
            mapped_pages: 2,
            boolean_pages_scanned: 2,
            boolean_key_bytes_read: 16,
            numerical_kv_bytes_per_token: 16,
            full_numerical_kv_bytes: 128,
            selected_numerical_kv_bytes: 64,
            avoided_numerical_kv_bytes: 64,
            selected_pages: vec![BooleanKvSelectedPage {
                logical_page: 0,
                physical_page: 3,
                live_tokens: 4,
                hamming_distance: 1,
                xnor_matches: 63,
            }],
        };
        qualify_cooperative_selection(ready, &selection, 0, 2).unwrap()
    }

    #[test]
    fn records_explicit_overlap_without_promoting_it() {
        let trace = qualify_cooperative_trace(
            ticket(),
            CooperativeTraceMode::FirstToken,
            CooperativeTraceTiming {
                selection_ready_ns: 10,
                handoff_qualified_ns: 12,
                numerical_start_ns: 20,
                numerical_finish_ns: 100,
                next_boolean_start_ns: Some(40),
                next_boolean_finish_ns: Some(70),
            },
        )
        .unwrap();
        assert_eq!(trace.generation(), 5);
        assert_eq!(trace.prefix_tokens(), 8);
        assert_eq!(trace.mapped_pages(), 2);
        assert_eq!(trace.selected_pages(), 1);
        assert_eq!(trace.mode(), CooperativeTraceMode::FirstToken);
        assert_eq!(trace.observed_overlap_ns(), Some(30));
    }

    #[test]
    fn distinguishes_missing_interval_from_zero_overlap() {
        let no_interval = qualify_cooperative_trace(
            ticket(),
            CooperativeTraceMode::SteadyState,
            CooperativeTraceTiming {
                selection_ready_ns: 1,
                handoff_qualified_ns: 2,
                numerical_start_ns: 3,
                numerical_finish_ns: 4,
                next_boolean_start_ns: None,
                next_boolean_finish_ns: None,
            },
        )
        .unwrap();
        assert_eq!(no_interval.observed_overlap_ns(), None);

        let disjoint = qualify_cooperative_trace(
            ticket(),
            CooperativeTraceMode::SteadyState,
            CooperativeTraceTiming {
                selection_ready_ns: 1,
                handoff_qualified_ns: 2,
                numerical_start_ns: 10,
                numerical_finish_ns: 20,
                next_boolean_start_ns: Some(21),
                next_boolean_finish_ns: Some(25),
            },
        )
        .unwrap();
        assert_eq!(disjoint.observed_overlap_ns(), Some(0));
    }

    #[test]
    fn rejects_incomplete_or_reversed_timings() {
        let base = CooperativeTraceTiming {
            selection_ready_ns: 10,
            handoff_qualified_ns: 12,
            numerical_start_ns: 20,
            numerical_finish_ns: 100,
            next_boolean_start_ns: None,
            next_boolean_finish_ns: None,
        };
        let mut timing = base;
        timing.handoff_qualified_ns = 9;
        assert_eq!(
            qualify_cooperative_trace(ticket(), CooperativeTraceMode::FirstToken, timing),
            Err(CooperativeTraceError::HandoffBeforeSelectionReady)
        );

        let mut timing = base;
        timing.numerical_start_ns = 11;
        assert_eq!(
            qualify_cooperative_trace(ticket(), CooperativeTraceMode::FirstToken, timing),
            Err(CooperativeTraceError::NumericalStartBeforeHandoff)
        );

        let mut timing = base;
        timing.numerical_finish_ns = 19;
        assert_eq!(
            qualify_cooperative_trace(ticket(), CooperativeTraceMode::FirstToken, timing),
            Err(CooperativeTraceError::NumericalFinishBeforeStart)
        );

        let mut timing = base;
        timing.next_boolean_start_ns = Some(30);
        assert_eq!(
            qualify_cooperative_trace(ticket(), CooperativeTraceMode::FirstToken, timing),
            Err(CooperativeTraceError::IncompleteNextBooleanInterval)
        );

        let mut timing = base;
        timing.next_boolean_start_ns = Some(50);
        timing.next_boolean_finish_ns = Some(40);
        assert_eq!(
            qualify_cooperative_trace(ticket(), CooperativeTraceMode::FirstToken, timing),
            Err(CooperativeTraceError::NextBooleanFinishBeforeStart)
        );
    }
}
