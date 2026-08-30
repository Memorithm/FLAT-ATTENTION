//! Machine-readable evidence retention for the nonlocal attention research semantic.
//!
//! This module records what was evaluated, against which exact source revision
//! and configuration, and (when applicable) on which device/protocol.
//! Correctness and performance evidence are deliberately separate scopes: a
//! faster measurement is never interpreted as a correctness proof.

use core::fmt;
use std::fmt::Write as _;

use crate::{
    api::research_nonlocal::{
        HistoryBudgetPolicy, HistoryMode, HistorySchedule, HistoryWeighting,
        NonlocalAttentionConfig, NONLOCAL_ATTENTION_SEMANTIC_NAME,
        NONLOCAL_ATTENTION_SEMANTIC_REVISION,
    },
    kernel_autotune::BenchmarkProtocol,
    RuntimeDeviceFingerprint,
};

/// Version of the deterministic nonlocal research-evidence schema.
pub const NONLOCAL_EVIDENCE_SCHEMA_VERSION: u16 = 1;

/// What kind of claim one evidence record is allowed to address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResearchEvidenceScope {
    /// Numerical/semantic correctness evidence.
    Correctness,
    /// Latency, throughput, memory, or other performance evidence.
    Performance,
}

/// Outcome of evaluating the explicitly stated claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResearchEvidenceDisposition {
    /// The evidence supports the stated claim under the recorded conditions.
    Supports,
    /// The evidence rejects the stated claim under the recorded conditions.
    Rejects,
    /// The recorded evidence does not resolve the stated claim.
    Inconclusive,
}

/// Reproducible evidence record for `nonlocal-history-softmax@1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonlocalEvidenceManifest {
    /// Exact 40-hex source revision evaluated by the evidence-producing run.
    pub commit_sha: String,
    /// Stable identifier of the evidence-producing test, benchmark, or study.
    pub evidence_id: String,
    /// Non-empty claim/result statement whose outcome is recorded.
    pub statement: String,
    /// Exact structured-history configuration evaluated by the run.
    pub config: NonlocalAttentionConfig,
    /// Correctness and performance evidence are kept in separate scopes.
    pub scope: ResearchEvidenceScope,
    /// Whether the evidence supports, rejects, or leaves the statement unresolved.
    pub disposition: ResearchEvidenceDisposition,
    /// Executing device identity when the evidence depends on a device.
    ///
    /// Performance evidence requires this field. Device-independent scalar
    /// correctness evidence may omit it.
    pub device: Option<RuntimeDeviceFingerprint>,
    /// Exact bounded benchmark protocol, when timing is part of the evidence.
    ///
    /// Performance evidence requires this field even when a run ultimately
    /// rejects or fails to produce a usable timing sample.
    pub benchmark_protocol: Option<BenchmarkProtocol>,
}

/// Validation failures for a nonlocal research-evidence record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NonlocalEvidenceError {
    /// `commit_sha` was not exactly 40 hexadecimal digits.
    InvalidCommitSha,
    /// A required textual field was empty or whitespace-only.
    EmptyField(&'static str),
    /// The embedded nonlocal configuration failed its own validation.
    InvalidNonlocalConfig,
    /// Performance evidence did not identify the measured device.
    MissingPerformanceDevice,
    /// Performance evidence did not retain its exact benchmark protocol.
    MissingPerformanceProtocol,
    /// A supplied benchmark protocol failed its bounded-work validation.
    InvalidBenchmarkProtocol,
    /// A supplied device provenance field was empty or whitespace-only.
    EmptyDeviceField(&'static str),
}

impl fmt::Display for NonlocalEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCommitSha => {
                formatter.write_str("commit_sha must contain exactly 40 hex digits")
            }
            Self::EmptyField(field) => {
                write!(
                    formatter,
                    "nonlocal evidence field {field} must not be empty"
                )
            }
            Self::InvalidNonlocalConfig => {
                formatter.write_str("nonlocal evidence contains an invalid attention configuration")
            }
            Self::MissingPerformanceDevice => formatter
                .write_str("nonlocal performance evidence requires exact device provenance"),
            Self::MissingPerformanceProtocol => formatter
                .write_str("nonlocal performance evidence requires an exact benchmark protocol"),
            Self::InvalidBenchmarkProtocol => {
                formatter.write_str("nonlocal evidence contains an invalid benchmark protocol")
            }
            Self::EmptyDeviceField(field) => write!(
                formatter,
                "nonlocal evidence device field {field} must not be empty"
            ),
        }
    }
}

impl std::error::Error for NonlocalEvidenceError {}

impl NonlocalEvidenceManifest {
    /// Validate provenance and semantic invariants before evidence is retained.
    pub fn validate(&self) -> Result<(), NonlocalEvidenceError> {
        if self.commit_sha.len() != 40
            || !self.commit_sha.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(NonlocalEvidenceError::InvalidCommitSha);
        }
        for (field, value) in [
            ("evidence_id", self.evidence_id.as_str()),
            ("statement", self.statement.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(NonlocalEvidenceError::EmptyField(field));
            }
        }
        self.config
            .validate()
            .map_err(|_| NonlocalEvidenceError::InvalidNonlocalConfig)?;

        if let Some(protocol) = self.benchmark_protocol {
            protocol
                .validate()
                .map_err(|_| NonlocalEvidenceError::InvalidBenchmarkProtocol)?;
        }
        if matches!(self.scope, ResearchEvidenceScope::Performance) {
            if self.device.is_none() {
                return Err(NonlocalEvidenceError::MissingPerformanceDevice);
            }
            if self.benchmark_protocol.is_none() {
                return Err(NonlocalEvidenceError::MissingPerformanceProtocol);
            }
        }
        if let Some(device) = &self.device {
            for (field, value) in [
                ("name", device.name.as_str()),
                ("backend", device.backend.as_str()),
                ("driver", device.driver.as_str()),
            ] {
                if value.trim().is_empty() {
                    return Err(NonlocalEvidenceError::EmptyDeviceField(field));
                }
            }
        }
        Ok(())
    }

    /// Deterministic canonical representation of the embedded semantic config.
    ///
    /// This value is stored alongside its fingerprint so the fingerprint can
    /// always be recomputed from the human-auditable configuration.
    #[must_use]
    pub fn config_record(&self) -> String {
        nonlocal_config_record(self.config)
    }

    /// Stable FNV-1a-64 fingerprint of [`Self::config_record`].
    ///
    /// The fingerprint is for deterministic joining/deduplication only and is
    /// not a cryptographic authenticity primitive.
    #[must_use]
    pub fn config_fingerprint(&self) -> u64 {
        fnv1a64(self.config_record().as_bytes())
    }

    /// Canonical JSON payload. Field order is part of schema version 1.
    pub fn canonical_json(&self) -> Result<String, NonlocalEvidenceError> {
        self.validate()?;
        let core = self.core_json();
        let checksum = fnv1a64(core.as_bytes());
        let mut out = core;
        assert_eq!(out.pop(), Some('}'));
        write!(
            out,
            ",\"record_checksum\":{{\"algorithm\":\"fnv1a64\",\"value\":\"{checksum:016x}\"}}}}"
        )
        .expect("writing to String cannot fail");
        Ok(out)
    }

    /// Stable checksum over the canonical evidence payload excluding the
    /// checksum field itself.
    pub fn record_checksum(&self) -> Result<String, NonlocalEvidenceError> {
        self.validate()?;
        Ok(format!("{:016x}", fnv1a64(self.core_json().as_bytes())))
    }

    fn core_json(&self) -> String {
        format!(
            concat!(
                "{{\"schema_version\":{},\"commit_sha\":{},\"evidence_id\":{},",
                "\"semantic\":{{\"name\":{},\"revision\":{}}},",
                "\"config\":{},\"config_record\":{},\"config_fingerprint\":\"{:016x}\",",
                "\"scope\":{},\"disposition\":{},\"statement\":{},",
                "\"device\":{},\"benchmark_protocol\":{}}}"
            ),
            NONLOCAL_EVIDENCE_SCHEMA_VERSION,
            json_string(&self.commit_sha),
            json_string(&self.evidence_id),
            json_string(NONLOCAL_ATTENTION_SEMANTIC_NAME),
            NONLOCAL_ATTENTION_SEMANTIC_REVISION,
            config_json(self.config),
            json_string(&self.config_record()),
            self.config_fingerprint(),
            json_string(scope_name(self.scope)),
            json_string(disposition_name(self.disposition)),
            json_string(&self.statement),
            device_json(self.device.as_ref()),
            protocol_json(self.benchmark_protocol),
        )
    }
}

fn nonlocal_config_record(config: NonlocalAttentionConfig) -> String {
    format!(
        "history={};schedule={};weighting={};budget={}",
        history_mode_name(config.history_mode),
        schedule_name(config.history_schedule),
        weighting_name(config.history_weighting),
        budget_name(config.history_budget_policy),
    )
}

fn config_json(config: NonlocalAttentionConfig) -> String {
    format!(
        "{{\"history_mode\":{},\"history_schedule\":{},\"history_weighting\":{},\"history_budget_policy\":{}}}",
        json_string(&history_mode_name(config.history_mode)),
        json_string(schedule_name(config.history_schedule)),
        json_string(weighting_name(config.history_weighting)),
        json_string(&budget_name(config.history_budget_policy)),
    )
}

fn history_mode_name(mode: HistoryMode) -> String {
    match mode {
        HistoryMode::Complete => "complete".to_owned(),
        HistoryMode::Window { max_tokens } => format!("window:{max_tokens}"),
    }
}

const fn schedule_name(schedule: HistorySchedule) -> &'static str {
    match schedule {
        HistorySchedule::EveryToken => "every_token",
    }
}

const fn weighting_name(weighting: HistoryWeighting) -> &'static str {
    match weighting {
        HistoryWeighting::Identity => "identity",
    }
}

fn budget_name(policy: HistoryBudgetPolicy) -> String {
    match policy {
        HistoryBudgetPolicy::Unlimited => "unlimited".to_owned(),
        HistoryBudgetPolicy::RejectAbove {
            max_retained_tokens,
        } => format!("reject_above:{max_retained_tokens}"),
    }
}

const fn scope_name(scope: ResearchEvidenceScope) -> &'static str {
    match scope {
        ResearchEvidenceScope::Correctness => "correctness",
        ResearchEvidenceScope::Performance => "performance",
    }
}

const fn disposition_name(disposition: ResearchEvidenceDisposition) -> &'static str {
    match disposition {
        ResearchEvidenceDisposition::Supports => "supports",
        ResearchEvidenceDisposition::Rejects => "rejects",
        ResearchEvidenceDisposition::Inconclusive => "inconclusive",
    }
}

fn device_json(device: Option<&RuntimeDeviceFingerprint>) -> String {
    let Some(device) = device else {
        return "null".to_owned();
    };
    format!(
        "{{\"name\":{},\"backend\":{},\"driver\":{},\"driver_info\":{},\"vendor\":{},\"device\":{},\"canonical_record\":{},\"fingerprint\":\"{:016x}\"}}",
        json_string(&device.name),
        json_string(&device.backend),
        json_string(&device.driver),
        json_string(&device.driver_info),
        device.vendor,
        device.device,
        json_string(&device.canonical_record()),
        device.stable_fingerprint(),
    )
}

fn protocol_json(protocol: Option<BenchmarkProtocol>) -> String {
    match protocol {
        None => "null".to_owned(),
        Some(protocol) => format!(
            "{{\"warmups\":{},\"iterations\":{}}}",
            protocol.warmups, protocol.iterations
        ),
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
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

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> RuntimeDeviceFingerprint {
        RuntimeDeviceFingerprint {
            name: "Mesa llvmpipe".into(),
            backend: "Vulkan".into(),
            driver: "Mesa".into(),
            driver_info: "software adapter".into(),
            vendor: 0x10005,
            device: 0,
        }
    }

    fn correctness_record() -> NonlocalEvidenceManifest {
        NonlocalEvidenceManifest {
            commit_sha: "0123456789abcdef0123456789abcdef01234567".into(),
            evidence_id: "nonlocal-wgpu-oracle-parity".into(),
            statement: "WGPU candidate agrees with the scalar oracle within the declared tolerance"
                .into(),
            config: NonlocalAttentionConfig::default(),
            scope: ResearchEvidenceScope::Correctness,
            disposition: ResearchEvidenceDisposition::Supports,
            device: Some(device()),
            benchmark_protocol: None,
        }
    }

    #[test]
    fn canonical_record_is_bit_reproducible() {
        let first = correctness_record();
        let second = first.clone();
        assert_eq!(
            first.canonical_json().unwrap(),
            second.canonical_json().unwrap()
        );
        assert_eq!(
            first.record_checksum().unwrap(),
            second.record_checksum().unwrap()
        );
        let json = first.canonical_json().unwrap();
        assert!(json.contains("\"scope\":\"correctness\""));
        assert!(
            json.contains("\"semantic\":{\"name\":\"nonlocal-history-softmax\",\"revision\":1}")
        );
        assert!(json.contains("\"canonical_record\":\"name=Mesa llvmpipe;backend=Vulkan"));
    }

    #[test]
    fn config_fingerprint_changes_with_declared_history_rule() {
        let reference = correctness_record();
        let mut approximation = reference.clone();
        approximation.config.history_mode = HistoryMode::Window { max_tokens: 32 };
        assert_ne!(
            reference.config_fingerprint(),
            approximation.config_fingerprint()
        );
        assert!(approximation.config_record().contains("history=window:32"));
    }

    #[test]
    fn performance_evidence_requires_device_and_protocol() {
        let mut record = correctness_record();
        record.scope = ResearchEvidenceScope::Performance;
        record.device = None;
        assert_eq!(
            record.validate(),
            Err(NonlocalEvidenceError::MissingPerformanceDevice)
        );

        record.device = Some(device());
        assert_eq!(
            record.validate(),
            Err(NonlocalEvidenceError::MissingPerformanceProtocol)
        );

        record.benchmark_protocol = Some(BenchmarkProtocol {
            warmups: 3,
            iterations: 30,
        });
        assert_eq!(record.validate(), Ok(()));
        let json = record.canonical_json().unwrap();
        assert!(json.contains("\"benchmark_protocol\":{\"warmups\":3,\"iterations\":30}"));
    }

    #[test]
    fn invalid_protocol_fails_closed() {
        let mut record = correctness_record();
        record.benchmark_protocol = Some(BenchmarkProtocol {
            warmups: 0,
            iterations: 30,
        });
        assert_eq!(
            record.validate(),
            Err(NonlocalEvidenceError::InvalidBenchmarkProtocol)
        );
    }

    #[test]
    fn scalar_correctness_evidence_may_be_device_independent() {
        let mut record = correctness_record();
        record.device = None;
        assert_eq!(record.validate(), Ok(()));
        assert!(record.canonical_json().unwrap().contains("\"device\":null"));
    }

    #[test]
    fn negative_result_is_retained_explicitly() {
        let mut record = correctness_record();
        record.disposition = ResearchEvidenceDisposition::Rejects;
        record.statement = "window-32 preserves the complete-history result".into();
        let json = record.canonical_json().unwrap();
        assert!(json.contains("\"disposition\":\"rejects\""));
        assert!(json.contains("window-32 preserves the complete-history result"));
    }

    #[test]
    fn malformed_provenance_fails_closed() {
        let mut record = correctness_record();
        record.commit_sha = "main".into();
        assert_eq!(
            record.validate(),
            Err(NonlocalEvidenceError::InvalidCommitSha)
        );

        let mut record = correctness_record();
        record.statement = "   ".into();
        assert_eq!(
            record.validate(),
            Err(NonlocalEvidenceError::EmptyField("statement"))
        );

        let mut record = correctness_record();
        record.config.history_mode = HistoryMode::Window { max_tokens: 0 };
        assert_eq!(
            record.validate(),
            Err(NonlocalEvidenceError::InvalidNonlocalConfig)
        );
    }

    #[test]
    fn performance_scope_remains_distinct_from_correctness() {
        let mut record = correctness_record();
        record.scope = ResearchEvidenceScope::Performance;
        record.statement = "candidate median latency is lower under the recorded protocol".into();
        record.benchmark_protocol = Some(BenchmarkProtocol {
            warmups: 5,
            iterations: 15,
        });
        let json = record.canonical_json().unwrap();
        assert!(json.contains("\"scope\":\"performance\""));
        assert!(!json.contains("\"scope\":\"correctness\""));
    }
}
