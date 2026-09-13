//! Research-only machine-readable BIKV qualification evidence.
//!
//! BKV-K6.3 binds the generic M40 benchmark provenance schema to the exact
//! Boolean-KV accounting and gate semantics introduced by BKV-K6.1/K6.2.
//! The envelope is deterministic and dependency-free. It does not change
//! runtime routing and does not promote BIKV or BKV-7 by itself.

use core::fmt;
use std::fmt::Write as _;

use crate::benchmark_manifest::{BenchmarkManifest, BenchmarkManifestError};
use crate::research_bkv_qualification::{BikvPromotionDecision, BikvQualificationRecord};

pub const BIKV_EVIDENCE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BikvEvidenceScope {
    pub timing_scope: String,
    pub selection_policy: String,
    pub q_device_resident: bool,
    pub q_host_mirror_retained: bool,
    pub kv_device_resident: bool,
    pub uploads_readbacks_excluded: bool,
    pub resident_only_production_claim: bool,
    pub gpu_timestamp_claim: bool,
    pub physical_dram_traffic_claim: bool,
    pub model_quality_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BikvEvidenceGates {
    pub all_accept_k6_vs_m16: bool,
    pub sparse_k6_vs_restricted_oracle: bool,
    pub quality_gate_passed: bool,
}

impl BikvEvidenceGates {
    #[must_use]
    pub fn correctness_gate_passed(self) -> bool {
        self.all_accept_k6_vs_m16 && self.sparse_k6_vs_restricted_oracle
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BikvEvidenceManifest {
    pub candidate: BenchmarkManifest,
    pub dense_baseline: BenchmarkManifest,
    pub qualification: BikvQualificationRecord,
    pub signature_bits: usize,
    pub max_distance: usize,
    pub scope: BikvEvidenceScope,
    pub gates: BikvEvidenceGates,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BikvEvidenceError {
    CandidateManifest(BenchmarkManifestError),
    DenseManifest(BenchmarkManifestError),
    CommitMismatch,
    EnvironmentMismatch,
    ProblemMismatch,
    ProtocolMismatch,
    DuplicateBenchmarkId,
    ZeroSignatureBits,
    InvalidMaxDistance {
        max_distance: usize,
        signature_bits: usize,
    },
    MissingCandidateLatency,
    QualificationProblemMismatch(&'static str),
    DenseLatencyMismatch {
        qualification_ns: u64,
        manifest_ns: u64,
    },
    EmptyScopeField(&'static str),
    ContradictoryResidentScope,
}

impl fmt::Display for BikvEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CandidateManifest(error) => write!(f, "invalid BIKV candidate manifest: {error}"),
            Self::DenseManifest(error) => write!(f, "invalid M16 dense manifest: {error}"),
            Self::CommitMismatch => write!(f, "candidate and dense evidence must use the same commit"),
            Self::EnvironmentMismatch => {
                write!(f, "candidate and dense evidence must use the same environment")
            }
            Self::ProblemMismatch => {
                write!(f, "candidate and dense evidence must use the same attention problem")
            }
            Self::ProtocolMismatch => write!(
                f,
                "candidate and dense evidence must use the same warmup/measured iteration counts"
            ),
            Self::DuplicateBenchmarkId => {
                write!(f, "candidate and dense benchmark IDs must be distinct")
            }
            Self::ZeroSignatureBits => write!(f, "BIKV evidence signature width must be non-zero"),
            Self::InvalidMaxDistance {
                max_distance,
                signature_bits,
            } => write!(
                f,
                "BIKV evidence max Hamming distance {max_distance} exceeds signature width {signature_bits}"
            ),
            Self::MissingCandidateLatency => {
                write!(f, "BIKV candidate end-to-end median latency must be non-zero")
            }
            Self::QualificationProblemMismatch(field) => write!(
                f,
                "BIKV qualification accounting does not match benchmark problem field {field}"
            ),
            Self::DenseLatencyMismatch {
                qualification_ns,
                manifest_ns,
            } => write!(
                f,
                "BIKV qualification dense latency {qualification_ns} ns does not match dense manifest median {manifest_ns} ns"
            ),
            Self::EmptyScopeField(field) => {
                write!(f, "BIKV evidence scope field {field} must not be empty")
            }
            Self::ContradictoryResidentScope => write!(
                f,
                "BIKV evidence cannot claim resident-only production scope while retaining a host Q mirror"
            ),
        }
    }
}

impl std::error::Error for BikvEvidenceError {}

impl BikvEvidenceManifest {
    pub fn validate(&self) -> Result<(), BikvEvidenceError> {
        self.candidate
            .validate()
            .map_err(BikvEvidenceError::CandidateManifest)?;
        self.dense_baseline
            .validate()
            .map_err(BikvEvidenceError::DenseManifest)?;

        if self.candidate.commit_sha != self.dense_baseline.commit_sha {
            return Err(BikvEvidenceError::CommitMismatch);
        }
        if self.candidate.environment != self.dense_baseline.environment {
            return Err(BikvEvidenceError::EnvironmentMismatch);
        }
        if self.candidate.problem != self.dense_baseline.problem {
            return Err(BikvEvidenceError::ProblemMismatch);
        }
        if self.candidate.warmup_iterations != self.dense_baseline.warmup_iterations
            || self.candidate.measured_iterations != self.dense_baseline.measured_iterations
        {
            return Err(BikvEvidenceError::ProtocolMismatch);
        }
        if self.candidate.benchmark_id == self.dense_baseline.benchmark_id {
            return Err(BikvEvidenceError::DuplicateBenchmarkId);
        }
        if self.signature_bits == 0 {
            return Err(BikvEvidenceError::ZeroSignatureBits);
        }
        if self.max_distance > self.signature_bits {
            return Err(BikvEvidenceError::InvalidMaxDistance {
                max_distance: self.max_distance,
                signature_bits: self.signature_bits,
            });
        }
        if self.candidate.result.median_latency_ns == 0 {
            return Err(BikvEvidenceError::MissingCandidateLatency);
        }
        for (name, value) in [
            ("timing_scope", self.scope.timing_scope.as_str()),
            ("selection_policy", self.scope.selection_policy.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(BikvEvidenceError::EmptyScopeField(name));
            }
        }
        if self.scope.q_host_mirror_retained && self.scope.resident_only_production_claim {
            return Err(BikvEvidenceError::ContradictoryResidentScope);
        }

        let accounting = self.qualification.accounting();
        let problem = &self.candidate.problem;
        if accounting.live_tokens != problem.kv_len {
            return Err(BikvEvidenceError::QualificationProblemMismatch("kv_len"));
        }
        if accounting.kv_heads != problem.kv_heads {
            return Err(BikvEvidenceError::QualificationProblemMismatch("kv_heads"));
        }
        if accounting.head_dim != problem.head_dim {
            return Err(BikvEvidenceError::QualificationProblemMismatch("head_dim"));
        }
        let qualification_dense = self.qualification.latency().dense_attention_ns;
        let manifest_dense = self.dense_baseline.result.median_latency_ns;
        if qualification_dense != manifest_dense {
            return Err(BikvEvidenceError::DenseLatencyMismatch {
                qualification_ns: qualification_dense,
                manifest_ns: manifest_dense,
            });
        }
        Ok(())
    }

    /// Promotion disposition using the measured end-to-end candidate median.
    ///
    /// The phase-median sum stored in `BikvQualificationRecord` remains a
    /// diagnostic. It is deliberately not used for this decision because a sum
    /// of independently computed phase medians is not an end-to-end percentile.
    pub fn promotion_decision(&self) -> Result<BikvPromotionDecision, BikvEvidenceError> {
        self.validate()?;
        if !self.gates.correctness_gate_passed() {
            return Ok(BikvPromotionDecision::FallbackCorrectnessGate);
        }
        if !self.gates.quality_gate_passed {
            return Ok(BikvPromotionDecision::FallbackQualityGate);
        }
        if self.candidate.result.median_latency_ns >= self.dense_baseline.result.median_latency_ns {
            return Ok(BikvPromotionDecision::FallbackNoLatencyWin);
        }
        Ok(BikvPromotionDecision::Promote)
    }

    /// Deterministic evidence envelope suitable for retention or ingestion by a
    /// later evidence-normalization layer. Field order is part of schema v1.
    pub fn canonical_json(&self) -> Result<String, BikvEvidenceError> {
        self.validate()?;
        let candidate = self
            .candidate
            .canonical_json()
            .map_err(BikvEvidenceError::CandidateManifest)?;
        let dense = self
            .dense_baseline
            .canonical_json()
            .map_err(BikvEvidenceError::DenseManifest)?;
        let accounting = self.qualification.accounting();
        let latency = self.qualification.latency();
        let decision = decision_name(self.promotion_decision()?);

        let mut payload = String::with_capacity(candidate.len() + dense.len() + 1_024);
        write!(
            payload,
            "{{\"schema_version\":{},\"candidate\":{},\"dense_baseline\":{},",
            BIKV_EVIDENCE_SCHEMA_VERSION, candidate, dense
        )
        .expect("writing to String cannot fail");
        write!(
            payload,
            "\"selection\":{{\"signature_bits\":{},\"max_distance\":{},\"policy\":{}}},",
            self.signature_bits,
            self.max_distance,
            json_string(&self.scope.selection_policy),
        )
        .expect("writing to String cannot fail");
        write!(
            payload,
            "\"scope\":{{\"timing\":{},\"q_device_resident\":{},\"q_host_mirror_retained\":{},\"kv_device_resident\":{},\"uploads_readbacks_excluded\":{},\"resident_only_production_claim\":{},\"gpu_timestamp_claim\":{},\"physical_dram_traffic_claim\":{},\"model_quality_claim\":{}}},",
            json_string(&self.scope.timing_scope),
            self.scope.q_device_resident,
            self.scope.q_host_mirror_retained,
            self.scope.kv_device_resident,
            self.scope.uploads_readbacks_excluded,
            self.scope.resident_only_production_claim,
            self.scope.gpu_timestamp_claim,
            self.scope.physical_dram_traffic_claim,
            self.scope.model_quality_claim,
        )
        .expect("writing to String cannot fail");
        write!(
            payload,
            "\"accounting\":{{\"live_tokens\":{},\"selected_live_tokens\":{},\"mapped_pages\":{},\"selected_pages\":{},\"page_size\":{},\"kv_heads\":{},\"head_dim\":{},\"scalar_bytes\":{},\"boolean_index_bytes_read\":{},\"kv_bytes_per_token\":{},\"dense_numerical_kv_bytes\":{},\"selected_numerical_kv_bytes\":{},\"avoided_numerical_kv_bytes\":{}}},",
            accounting.live_tokens,
            accounting.selected_live_tokens,
            accounting.mapped_pages,
            accounting.selected_pages,
            accounting.page_size,
            accounting.kv_heads,
            accounting.head_dim,
            accounting.scalar_bytes,
            accounting.boolean_index_bytes_read,
            self.qualification.kv_bytes_per_token(),
            self.qualification.dense_numerical_kv_bytes(),
            self.qualification.selected_numerical_kv_bytes(),
            self.qualification.avoided_numerical_kv_bytes(),
        )
        .expect("writing to String cannot fail");
        write!(
            payload,
            "\"phase_medians_ns\":{{\"signature_generation\":{},\"boolean_search\":{},\"selected_attention\":{},\"synchronization\":{},\"dense_attention\":{},\"diagnostic_sum\":{}}},",
            latency.signature_generation_ns,
            latency.boolean_search_ns,
            latency.selected_attention_ns,
            latency.synchronization_ns,
            latency.dense_attention_ns,
            self.qualification.total_bikv_latency_ns(),
        )
        .expect("writing to String cannot fail");
        write!(
            payload,
            "\"gates\":{{\"all_accept_k6_vs_m16\":{},\"sparse_k6_vs_restricted_oracle\":{},\"correctness_gate_passed\":{},\"quality_gate_passed\":{}}},\"promotion_decision\":{}",
            self.gates.all_accept_k6_vs_m16,
            self.gates.sparse_k6_vs_restricted_oracle,
            self.gates.correctness_gate_passed(),
            self.gates.quality_gate_passed,
            json_string(decision),
        )
        .expect("writing to String cannot fail");

        let checksum = fnv1a64(payload.as_bytes());
        write!(
            payload,
            ",\"evidence_checksum\":{{\"algorithm\":\"fnv1a64\",\"value\":\"{checksum:016x}\"}}}}"
        )
        .expect("writing to String cannot fail");
        Ok(payload)
    }
}

fn decision_name(decision: BikvPromotionDecision) -> &'static str {
    match decision {
        BikvPromotionDecision::Promote => "promote",
        BikvPromotionDecision::FallbackQualityGate => "fallback_quality_gate",
        BikvPromotionDecision::FallbackCorrectnessGate => "fallback_correctness_gate",
        BikvPromotionDecision::FallbackNoLatencyWin => "fallback_no_latency_win",
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if character <= '\u{1f}' => {
                write!(out, "\\u{:04x}", character as u32).expect("writing to String cannot fail");
            }
            character => out.push(character),
        }
    }
    out.push('"');
    out
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
