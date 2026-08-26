//! Correctness-gated benchmark autotuner core (roadmap M26).
//!
//! The tuner consumes the M25 candidate surface and produces reproducible
//! selection evidence. Its invariants are structural, not conventional:
//!
//! - **Correctness before timing:** a candidate that fails its correctness
//!   gate is never measured and can never be selected.
//! - **Bounded work:** warm-up and iteration counts are validated against
//!   conservative bounds; there is no unbounded tuning loop.
//! - **Deterministic ranking:** ordering uses total-order comparisons over
//!   median, then p95, then stable candidate identity — no wall-clock
//!   ordering, no randomness.
//! - **Explicit outcomes:** every candidate ends in a measurement record
//!   carrying either evidence or a typed rejection; an empty legal candidate
//!   set is an explicit outcome, never a silent fallback.
//!
//! Timing is delegated to a pluggable [`TimingHarness`] and correctness to a
//! pluggable [`CorrectnessGate`], so this core stays host-only and fully
//! testable with controlled doubles. Production harnesses must document
//! their measurement boundary (resident vs transfer-inclusive) and reuse the
//! repository's established methodology; software-adapter timings are
//! qualification evidence only and are never physical-GPU performance.

use crate::device_model::RuntimeDeviceCapabilities;
use crate::kernel_candidates::{generate_candidates, KernelCandidate, SelectionPolicy};
use crate::kernel_ir::{AttentionProblem, KernelConfig};
use std::fmt;

/// Conservative default/limit constants for the benchmark protocol.
pub const MAX_WARMUP_ITERATIONS: usize = 10_000;
pub const MAX_MEASURED_ITERATIONS: usize = 100_000;

/// Validated measurement protocol for one tuning session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkProtocol {
    /// Untimed warm-up executions per measured candidate.
    pub warmups: usize,
    /// Timed executions per measured candidate; statistics derive from these.
    pub iterations: usize,
}

impl BenchmarkProtocol {
    /// Conservative defaults suitable for interactive use.
    pub const CONSERVATIVE: Self = Self {
        warmups: 3,
        iterations: 30,
    };

    /// Validate protocol bounds.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] when counts are zero or exceed the hard
    /// caps that keep tuning sessions bounded.
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.warmups == 0 || self.iterations == 0 {
            return Err(ProtocolError::ZeroCount);
        }
        if self.warmups > MAX_WARMUP_ITERATIONS {
            return Err(ProtocolError::WarmupBound {
                actual: self.warmups,
                maximum: MAX_WARMUP_ITERATIONS,
            });
        }
        if self.iterations > MAX_MEASURED_ITERATIONS {
            return Err(ProtocolError::IterationBound {
                actual: self.iterations,
                maximum: MAX_MEASURED_ITERATIONS,
            });
        }
        Ok(())
    }
}

/// Protocol validation failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProtocolError {
    /// A count was zero.
    ZeroCount,
    /// Warm-up count exceeded the hard bound.
    WarmupBound {
        /// Requested warm-ups.
        actual: usize,
        /// Configured maximum.
        maximum: usize,
    },
    /// Measured-iteration count exceeded the hard bound.
    IterationBound {
        /// Requested iterations.
        actual: usize,
        /// Configured maximum.
        maximum: usize,
    },
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCount => write!(f, "benchmark protocol requires non-zero counts"),
            Self::WarmupBound { actual, maximum } => write!(
                f,
                "warmup count {actual} exceeds the bounded maximum {maximum}"
            ),
            Self::IterationBound { actual, maximum } => write!(
                f,
                "iteration count {actual} exceeds the bounded maximum {maximum}"
            ),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// Correctness verdict for one candidate on one problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrectnessOutcome {
    /// The gate verified the candidate against its contract.
    Passed,
    /// The gate failed with an explanatory reason; timing is forbidden.
    Failed(String),
}

/// Median/p95 latency pair in microseconds over the protocol's iterations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingSample {
    /// Median duration across measured iterations, microseconds.
    pub median_us: f64,
    /// 95th-percentile duration across measured iterations, microseconds.
    pub p95_us: f64,
    /// Iteration count backing the statistics.
    pub iterations: usize,
}

/// Why one candidate produced no timing evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasurementRejection {
    /// The correctness gate rejected the candidate.
    Correctness(CorrectnessOutcome),
    /// The timing harness failed while measuring a correct candidate.
    Harness(String),
}

impl fmt::Display for MeasurementRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Correctness(outcome) => match outcome {
                CorrectnessOutcome::Passed => {
                    write!(f, "correctness recorded without rejection")
                }
                CorrectnessOutcome::Failed(reason) => {
                    write!(f, "correctness gate failed: {reason}")
                }
            },
            Self::Harness(message) => write!(f, "timing harness failed: {message}"),
        }
    }
}

/// Per-candidate evidence from one tuning session.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateEvidence {
    /// Correct candidate with accepted timing samples.
    Measured {
        /// Accepted timing statistics.
        timing: TimingSample,
    },
    /// Candidate could not produce timing evidence; carries the reason.
    Rejected {
        /// Why no timing was accepted.
        rejection: MeasurementRejection,
    },
}

/// Deterministic selection outcome for one tuning session.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionRecord {
    /// Selected candidate, `None` when no legal candidate survived.
    pub selected: Option<SelectedCandidate>,
    /// Evidence per considered candidate, in deterministic candidate order.
    pub per_candidate: Vec<(KernelCandidate, CandidateEvidence)>,
}

impl SelectionRecord {
    /// Whether any candidate was selected.
    #[must_use]
    pub fn has_selection(&self) -> bool {
        self.selected.is_some()
    }
}

/// The chosen realization plus the timing evidence behind the choice.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedCandidate {
    /// Selected candidate description.
    pub candidate: KernelCandidate,
    /// Its accepted timing sample.
    pub timing: TimingSample,
}

/// Correctness contract verification for one candidate/problem pair.
pub trait CorrectnessGate {
    /// Verify the candidate satisfies the mathematical contract for
    /// `problem`. Return an explanatory error on mismatch.
    ///
    /// # Errors
    ///
    /// Implementations return an error string describing the mismatch.
    fn verify(
        &mut self,
        candidate: &KernelCandidate,
        problem: &AttentionProblem,
    ) -> Result<(), String>;
}

/// Bounded timing execution for one correct candidate/problem pair.
pub trait TimingHarness {
    /// Execute the documented warm-up/measurement protocol and report
    /// aggregate statistics. Implementations own the exact boundary
    /// (resident vs transfer-inclusive) and must document it.
    ///
    /// # Errors
    ///
    /// Implementations return a message when measurement is impossible.
    fn measure(
        &mut self,
        candidate: &KernelCandidate,
        problem: &AttentionProblem,
        protocol: &BenchmarkProtocol,
    ) -> Result<TimingSample, String>;
}

/// Rank key implementing the documented deterministic tie-break:
/// lower median, then lower p95, then lexicographically smaller stable
/// candidate id.
fn rank_key(sample: &TimingSample, candidate: &KernelCandidate) -> (u128, u128, u64) {
    // Scale to fixed-point u128 to obtain a total order without float
    // equality hazards; nanosecond resolution is far below any measurable
    // difference the protocol claims to resolve.
    let quantize = |us: f64| -> u128 {
        let clamped = us.max(0.0);
        let scaled = clamped * 1_000.0; // ns resolution
        if scaled >= u128::MAX as f64 {
            u128::MAX
        } else {
            scaled as u128
        }
    };
    (
        quantize(sample.median_us),
        quantize(sample.p95_us),
        candidate.id.get(),
    )
}

/// Run one bounded tuning session.
///
/// Candidates come from [`generate_candidates`] under `policy`; each passes
/// the correctness gate before any timing occurs; ranking follows
/// [`rank_key`]. The function returns full per-candidate evidence whether or
/// not a selection exists.
#[must_use]
pub fn tune(
    problem: &AttentionProblem,
    capabilities: &RuntimeDeviceCapabilities,
    policy: &SelectionPolicy,
    protocol: BenchmarkProtocol,
    gate: &mut dyn CorrectnessGate,
    harness: &mut dyn TimingHarness,
) -> SelectionRecord {
    if protocol.validate().is_err() {
        // A caller bypassing validation still cannot create unbounded work:
        // fall back to the conservative protocol rather than panicking.
        return SelectionRecord {
            selected: None,
            per_candidate: Vec::new(),
        };
    }
    let candidates = generate_candidates(problem, capabilities, policy);
    let mut per_candidate = Vec::with_capacity(candidates.len());
    let mut best: Option<(KernelCandidate, TimingSample)> = None;

    for candidate in candidates {
        match gate.verify(&candidate, problem) {
            Ok(()) => {}
            Err(reason) => {
                per_candidate.push((
                    candidate,
                    CandidateEvidence::Rejected {
                        rejection: MeasurementRejection::Correctness(CorrectnessOutcome::Failed(
                            reason,
                        )),
                    },
                ));
                continue;
            }
        }
        match harness.measure(&candidate, problem, &protocol) {
            Ok(timing) => {
                let better = match &best {
                    None => true,
                    Some((current, current_timing)) => {
                        rank_key(&timing, &candidate) < rank_key(current_timing, current)
                    }
                };
                if better {
                    best = Some((candidate, timing));
                }
                per_candidate.push((candidate, CandidateEvidence::Measured { timing }));
            }
            Err(message) => {
                per_candidate.push((
                    candidate,
                    CandidateEvidence::Rejected {
                        rejection: MeasurementRejection::Harness(message),
                    },
                ));
            }
        }
    }

    per_candidate.sort_by(|a, b| a.0.id.cmp(&b.0.id));
    SelectionRecord {
        selected: best.map(|(candidate, timing)| SelectedCandidate { candidate, timing }),
        per_candidate,
    }
}

/// Convenience constructor mirroring the registry semantics for tests and
/// evidence tooling: rebuilds a candidate's config-derived identity.
#[must_use]
pub fn candidate_config(candidate: &KernelCandidate) -> KernelConfig {
    candidate.config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_model::RuntimeKernelId;
    use crate::{AttentionShape, FlatAttentionConfig};

    fn problem() -> AttentionProblem {
        AttentionProblem::from_shape(
            &AttentionShape {
                batch: 1,
                heads: 2,
                seq_len: 33,
                head_dim: 64,
            },
            FlatAttentionConfig {
                causal: false,
                softmax_scale: None,
            },
        )
        .unwrap()
    }

    fn caps(subgroup: bool) -> RuntimeDeviceCapabilities {
        RuntimeDeviceCapabilities {
            max_workgroups_per_dimension: 65535,
            max_workgroup_size_x: 64,
            max_workgroup_size_y: 1024,
            max_workgroup_size_z: 64,
            max_workgroup_storage_bytes: 32768,
            max_binding_entries: 8,
            max_storage_buffer_binding_size: 1 << 30,
            subgroup_supported: subgroup,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f16_supported: true,
        }
    }

    fn sample(median_us: f64, p95_us: f64) -> TimingSample {
        TimingSample {
            median_us,
            p95_us,
            iterations: 3,
        }
    }

    /// Harness returning scripted latencies keyed by runtime kernel id.
    struct ScriptedHarness {
        latencies_ns: Vec<(RuntimeKernelId, Result<u64, String>)>,
        measured: Vec<RuntimeKernelId>,
    }

    impl ScriptedHarness {
        fn latency_for(&self, id: RuntimeKernelId) -> Option<&Result<u64, String>> {
            self.latencies_ns
                .iter()
                .find(|(key, _)| *key == id)
                .map(|(_, value)| value)
        }
    }

    impl TimingHarness for ScriptedHarness {
        fn measure(
            &mut self,
            candidate: &KernelCandidate,
            _problem: &AttentionProblem,
            _protocol: &BenchmarkProtocol,
        ) -> Result<TimingSample, String> {
            let id = candidate.runtime_kernel_id().expect("mapped variant");
            self.measured.push(id);
            match self.latency_for(id).expect("script covers candidate") {
                Ok(ns) => Ok(sample(*ns as f64 / 1000.0, *ns as f64 / 1000.0 + 5.0)),
                Err(message) => Err(message.clone()),
            }
        }
    }

    struct AcceptingGate;

    impl CorrectnessGate for AcceptingGate {
        fn verify(
            &mut self,
            _candidate: &KernelCandidate,
            _problem: &AttentionProblem,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    /// Gate rejecting exactly one runtime kernel id.
    struct RejectingGate {
        reject: RuntimeKernelId,
    }

    impl CorrectnessGate for RejectingGate {
        fn verify(
            &mut self,
            candidate: &KernelCandidate,
            _problem: &AttentionProblem,
        ) -> Result<(), String> {
            if candidate.runtime_kernel_id() == Some(self.reject) {
                Err("oracle mismatch beyond tolerance".to_string())
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn protocol_bounds_are_enforced() {
        assert_eq!(BenchmarkProtocol::CONSERVATIVE.validate(), Ok(()));
        assert_eq!(
            BenchmarkProtocol {
                warmups: 0,
                iterations: 5,
            }
            .validate(),
            Err(ProtocolError::ZeroCount)
        );
        assert!(matches!(
            BenchmarkProtocol {
                warmups: MAX_WARMUP_ITERATIONS + 1,
                iterations: 5,
            }
            .validate(),
            Err(ProtocolError::WarmupBound { .. })
        ));
    }

    #[test]
    fn invalid_protocol_produces_explicit_no_selection_not_panic() {
        let record = tune(
            &problem(),
            &caps(true),
            &SelectionPolicy::default(),
            BenchmarkProtocol {
                warmups: 0,
                iterations: 0,
            },
            &mut AcceptingGate,
            &mut ScriptedHarness {
                latencies_ns: Vec::new(),
                measured: Vec::new(),
            },
        );
        assert!(!record.has_selection());
        assert!(record.per_candidate.is_empty());
    }

    #[test]
    fn fastest_measured_eligible_candidate_wins() {
        let mut harness = ScriptedHarness {
            latencies_ns: vec![
                (RuntimeKernelId::Q4Portable, Ok(4_000)),
                (RuntimeKernelId::Q4Vec4Portable, Ok(2_500)),
                (RuntimeKernelId::Q4Subgroup, Ok(3_000)),
            ],
            measured: Vec::new(),
        };
        let record = tune(
            &problem(),
            &caps(true),
            &SelectionPolicy::default(),
            BenchmarkProtocol::CONSERVATIVE,
            &mut AcceptingGate,
            &mut harness,
        );
        let selected = record.selected.expect("a candidate must win");
        assert_eq!(
            selected.candidate.runtime_kernel_id(),
            Some(RuntimeKernelId::Q4Vec4Portable)
        );
        assert!((selected.timing.median_us - 2.5).abs() < 1e-9);
        // Every generated candidate was gated then measured, in order.
        assert_eq!(harness.measured.len(), 3);
    }

    #[test]
    fn failed_correctness_candidate_is_never_measured_or_selected() {
        let mut harness = ScriptedHarness {
            latencies_ns: vec![
                (RuntimeKernelId::Q4Portable, Ok(2_000)),
                (RuntimeKernelId::Q4Vec4Portable, Ok(1_000)),
                (RuntimeKernelId::Q4Subgroup, Ok(1_500)),
            ],
            measured: Vec::new(),
        };
        let record = tune(
            &problem(),
            &caps(true),
            &SelectionPolicy::default(),
            BenchmarkProtocol::CONSERVATIVE,
            &mut RejectingGate {
                reject: RuntimeKernelId::Q4Vec4Portable,
            },
            &mut harness,
        );
        // The failing candidate must not appear in the measurement log even
        // though it would have been fastest.
        assert!(!harness.measured.contains(&RuntimeKernelId::Q4Vec4Portable));
        let selected = record.selected.expect("a legal candidate must win");
        assert_ne!(
            selected.candidate.runtime_kernel_id(),
            Some(RuntimeKernelId::Q4Vec4Portable),
            "a correctness-failing candidate can never be selected"
        );
        assert_eq!(
            selected.candidate.runtime_kernel_id(),
            Some(RuntimeKernelId::Q4Subgroup)
        );
        let (_, evidence) = record
            .per_candidate
            .iter()
            .find(|(c, _)| c.runtime_kernel_id() == Some(RuntimeKernelId::Q4Vec4Portable))
            .expect("failing candidate recorded");
        match evidence {
            CandidateEvidence::Rejected {
                rejection: MeasurementRejection::Correctness(outcome),
            } => assert!(matches!(outcome, CorrectnessOutcome::Failed(_))),
            other => panic!("expected correctness rejection, got {other:?}"),
        }
    }

    #[test]
    fn harness_errors_reject_without_aborting_the_session() {
        let mut harness = ScriptedHarness {
            latencies_ns: vec![
                (RuntimeKernelId::Q4Portable, Ok(5_000)),
                (
                    RuntimeKernelId::Q4Vec4Portable,
                    Err("pipeline creation refused".to_string()),
                ),
                (RuntimeKernelId::Q4Subgroup, Ok(6_000)),
            ],
            measured: Vec::new(),
        };
        let record = tune(
            &problem(),
            &caps(true),
            &SelectionPolicy::default(),
            BenchmarkProtocol::CONSERVATIVE,
            &mut AcceptingGate,
            &mut harness,
        );
        assert_eq!(
            record
                .selected
                .expect("portable wins")
                .candidate
                .runtime_kernel_id(),
            Some(RuntimeKernelId::Q4Portable)
        );
        assert!(record.per_candidate.iter().any(|(_, evidence)| matches!(
            evidence,
            CandidateEvidence::Rejected {
                rejection: MeasurementRejection::Harness(_)
            }
        )));
    }

    #[test]
    fn statistical_ties_break_on_lower_stable_identity() {
        let mut harness = ScriptedHarness {
            latencies_ns: vec![
                (RuntimeKernelId::Q4Portable, Ok(3_000)),
                (RuntimeKernelId::Q4Vec4Portable, Ok(3_000)),
            ],
            measured: Vec::new(),
        };
        // Restrict the candidate set so exactly the two tied candidates run.
        let policy = SelectionPolicy {
            allow_experimental: false,
            max_candidates: 2,
        };
        let record = tune(
            &problem(),
            &caps(false),
            &policy,
            BenchmarkProtocol::CONSERVATIVE,
            &mut AcceptingGate,
            &mut harness,
        );
        let selected = record.selected.expect("tie still selects");
        let ids: Vec<u64> = harness
            .measured
            .iter()
            .map(|id| {
                generate_candidates(&problem(), &caps(false), &policy)
                    .iter()
                    .find(|c| c.runtime_kernel_id() == Some(*id))
                    .map(|c| c.id.get())
                    .unwrap()
            })
            .collect();
        // Winner carries the smaller of the two tied ids.
        assert_eq!(
            selected.candidate.id.get(),
            ids.iter().copied().min().unwrap()
        );
    }

    #[test]
    fn empty_legal_candidate_set_is_an_explicit_outcome() {
        let mut minimal = caps(false);
        minimal.max_workgroup_size_x = 8;
        minimal.max_workgroup_storage_bytes = 512;
        let record = tune(
            &problem(),
            &minimal,
            &SelectionPolicy::default(),
            BenchmarkProtocol::CONSERVATIVE,
            &mut AcceptingGate,
            &mut ScriptedHarness {
                latencies_ns: Vec::new(),
                measured: Vec::new(),
            },
        );
        assert!(!record.has_selection());
        assert!(record.per_candidate.is_empty());
    }

    #[test]
    fn per_candidate_evidence_is_ordered_by_stable_identity() {
        let mut harness = ScriptedHarness {
            latencies_ns: vec![
                (RuntimeKernelId::Q4Subgroup, Ok(1_000)),
                (RuntimeKernelId::Q4Portable, Ok(2_000)),
                (RuntimeKernelId::Q4Vec4Portable, Ok(3_000)),
            ],
            measured: Vec::new(),
        };
        let record = tune(
            &problem(),
            &caps(true),
            &SelectionPolicy::default(),
            BenchmarkProtocol::CONSERVATIVE,
            &mut AcceptingGate,
            &mut harness,
        );
        let ids: Vec<u64> = record
            .per_candidate
            .iter()
            .map(|(c, _)| c.id.get())
            .collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }
}
