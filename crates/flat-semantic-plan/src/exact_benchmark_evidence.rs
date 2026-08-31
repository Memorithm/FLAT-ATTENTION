//! Minimal versioned join between benchmark evidence and exact semantic provenance.
//!
//! `BenchmarkManifest` v1 and the E4q exact-forward provenance remain separate,
//! unchanged contracts. This envelope snapshots both canonical forms only after
//! validating facts represented on both sides. It never synthesizes run, source,
//! device, precision, timing-conversion, or other missing metadata.

use core::fmt;

use flat_attention::{
    benchmark_manifest::{
        BenchmarkManifest, BenchmarkManifestError, BENCHMARK_MANIFEST_SCHEMA_VERSION,
    },
    kernel_autotune::ProtocolError,
};

use crate::exact_selection::{ExactForwardTuningRecord, EXACT_FORWARD_TUNING_PROVENANCE_VERSION};

/// Version of this benchmark/semantic evidence envelope.
pub const EXACT_FORWARD_BENCHMARK_EVIDENCE_SCHEMA_VERSION: u16 = 1;

const BENCHMARK_SCHEMA: u16 = 1;
const SEMANTIC_PROVENANCE_SCHEMA: u16 = 1;

/// Canonical benchmark evidence paired with canonical exact semantic provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactForwardBenchmarkEvidenceEnvelope {
    schema_version: u16,
    benchmark_manifest_json: String,
    semantic_provenance: String,
}

/// Fail-closed join failures.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExactForwardBenchmarkEvidenceError {
    UnsupportedEnvelopeVersion { actual: u16, supported: u16 },
    UnsupportedBenchmarkVersion { actual: u16, supported: u16 },
    UnsupportedSemanticProvenanceVersion { actual: u16, supported: u16 },
    BenchmarkManifest(BenchmarkManifestError),
    TuningProtocol(ProtocolError),
    BenchmarkProtocolMismatch,
    BenchmarkProblemMismatch,
}

impl fmt::Display for ExactForwardBenchmarkEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedEnvelopeVersion { actual, supported } => write!(
                formatter,
                "evidence envelope version {actual} is unsupported; expected {supported}"
            ),
            Self::UnsupportedBenchmarkVersion { actual, supported } => write!(
                formatter,
                "benchmark manifest version {actual} is unsupported; expected {supported}"
            ),
            Self::UnsupportedSemanticProvenanceVersion { actual, supported } => write!(
                formatter,
                "semantic provenance version {actual} is unsupported; expected {supported}"
            ),
            Self::BenchmarkManifest(error) => {
                write!(formatter, "invalid benchmark manifest: {error}")
            }
            Self::TuningProtocol(error) => write!(formatter, "invalid tuning protocol: {error}"),
            Self::BenchmarkProtocolMismatch => {
                formatter.write_str("benchmark and semantic tuning protocols differ")
            }
            Self::BenchmarkProblemMismatch => {
                formatter.write_str("benchmark and semantic tuning problems differ")
            }
        }
    }
}

impl std::error::Error for ExactForwardBenchmarkEvidenceError {}

impl From<BenchmarkManifestError> for ExactForwardBenchmarkEvidenceError {
    fn from(value: BenchmarkManifestError) -> Self {
        Self::BenchmarkManifest(value)
    }
}

impl From<ProtocolError> for ExactForwardBenchmarkEvidenceError {
    fn from(value: ProtocolError) -> Self {
        Self::TuningProtocol(value)
    }
}

impl ExactForwardBenchmarkEvidenceEnvelope {
    /// Join using the current envelope version.
    ///
    /// # Errors
    ///
    /// Rejects unsupported component versions, invalid component records, and
    /// any shared protocol/problem fact that cannot be shown equal.
    pub fn join(
        benchmark: &BenchmarkManifest,
        semantic: &ExactForwardTuningRecord,
    ) -> Result<Self, ExactForwardBenchmarkEvidenceError> {
        Self::join_versioned(
            EXACT_FORWARD_BENCHMARK_EVIDENCE_SCHEMA_VERSION,
            benchmark,
            semantic,
        )
    }

    /// Join with an explicit envelope version discriminator.
    ///
    /// # Errors
    ///
    /// Unknown versions fail closed before either payload is emitted.
    pub fn join_versioned(
        schema_version: u16,
        benchmark: &BenchmarkManifest,
        semantic: &ExactForwardTuningRecord,
    ) -> Result<Self, ExactForwardBenchmarkEvidenceError> {
        if schema_version != EXACT_FORWARD_BENCHMARK_EVIDENCE_SCHEMA_VERSION {
            return Err(
                ExactForwardBenchmarkEvidenceError::UnsupportedEnvelopeVersion {
                    actual: schema_version,
                    supported: EXACT_FORWARD_BENCHMARK_EVIDENCE_SCHEMA_VERSION,
                },
            );
        }
        if BENCHMARK_MANIFEST_SCHEMA_VERSION != BENCHMARK_SCHEMA {
            return Err(
                ExactForwardBenchmarkEvidenceError::UnsupportedBenchmarkVersion {
                    actual: BENCHMARK_MANIFEST_SCHEMA_VERSION,
                    supported: BENCHMARK_SCHEMA,
                },
            );
        }
        if EXACT_FORWARD_TUNING_PROVENANCE_VERSION != SEMANTIC_PROVENANCE_SCHEMA {
            return Err(
                ExactForwardBenchmarkEvidenceError::UnsupportedSemanticProvenanceVersion {
                    actual: EXACT_FORWARD_TUNING_PROVENANCE_VERSION,
                    supported: SEMANTIC_PROVENANCE_SCHEMA,
                },
            );
        }

        // This validates and preserves BenchmarkManifest v1 as-is.
        let benchmark_manifest_json = benchmark.canonical_json()?;
        let tuning = semantic.tuning();
        let protocol = tuning.benchmark_protocol();
        protocol.validate()?;

        if usize::try_from(benchmark.warmup_iterations).ok() != Some(protocol.warmups)
            || usize::try_from(benchmark.measured_iterations).ok() != Some(protocol.iterations)
        {
            return Err(ExactForwardBenchmarkEvidenceError::BenchmarkProtocolMismatch);
        }

        let benchmark_problem = &benchmark.problem;
        let semantic_problem = tuning.problem();
        let batch_heads = benchmark_problem
            .batch
            .checked_mul(benchmark_problem.q_heads);

        // AttentionProblem retains only dense folded batch×heads and one shared
        // sequence length. GQA/asymmetric identities therefore cannot be proven
        // from the existing contracts and must fail closed rather than be guessed.
        if benchmark_problem.q_heads != benchmark_problem.kv_heads
            || benchmark_problem.query_len != benchmark_problem.kv_len
            || batch_heads != usize::try_from(semantic_problem.batch_heads).ok()
            || Some(benchmark_problem.query_len) != usize::try_from(semantic_problem.seq_len).ok()
            || Some(benchmark_problem.head_dim) != usize::try_from(semantic_problem.head_dim).ok()
            || benchmark_problem.causal != semantic_problem.causal
        {
            return Err(ExactForwardBenchmarkEvidenceError::BenchmarkProblemMismatch);
        }

        Ok(Self {
            schema_version,
            benchmark_manifest_json,
            semantic_provenance: semantic.canonical_provenance_record(),
        })
    }

    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    #[must_use]
    pub fn benchmark_manifest_json(&self) -> &str {
        &self.benchmark_manifest_json
    }

    #[must_use]
    pub fn semantic_provenance(&self) -> &str {
        &self.semantic_provenance
    }

    /// Canonical JSON containing the two existing canonical payloads.
    #[must_use]
    pub fn canonical_json(&self) -> String {
        format!(
            "{{\"schema_version\":{},\"benchmark_manifest\":{},\"semantic_provenance\":{}}}",
            self.schema_version,
            self.benchmark_manifest_json,
            json_string(&self.semantic_provenance),
        )
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
                use std::fmt::Write as _;
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
    use flat_attention::{
        benchmark_manifest::{BenchmarkEnvironment, BenchmarkProblem, BenchmarkResult},
        kernel_autotune::{BenchmarkProtocol, CorrectnessGate, TimingHarness, TimingSample},
        kernel_candidates::{KernelCandidate, SelectionPolicy},
        kernel_ir::AttentionProblem,
        AttentionShape, FlatAttentionConfig, RuntimeDeviceCapabilities,
    };
    use flat_semantic::v1::{SemanticFamily, SemanticId};
    use flat_semantic_execution::standard_softmax_runtime_catalog;
    use flat_semantic_registry::SemanticRegistry;
    use flat_semantic_selection::{ExactSemanticSelectionPolicy, SemanticSelectionRequest};

    use crate::exact_selection::{
        plan_exact_forward_execution, tune_exact_forward_execution_plan,
        ExactForwardPlanningOutcome,
    };

    fn record(protocol: BenchmarkProtocol) -> ExactForwardTuningRecord {
        let semantic =
            SemanticId::new(SemanticFamily::StandardSoftmax, "standard-softmax", 1).unwrap();
        let registry = SemanticRegistry::new([semantic.clone()]).unwrap();
        let selection = ExactSemanticSelectionPolicy
            .select(&registry, &SemanticSelectionRequest::new(semantic))
            .unwrap();
        let problem = AttentionProblem::from_shape(
            &AttentionShape {
                batch: 1,
                heads: 4,
                seq_len: 129,
                head_dim: 64,
            },
            FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
        )
        .unwrap();
        let capabilities = RuntimeDeviceCapabilities {
            max_workgroups_per_dimension: 65_535,
            max_workgroup_size_x: 64,
            max_workgroup_size_y: 1_024,
            max_workgroup_size_z: 64,
            max_workgroup_storage_bytes: 32_768,
            max_binding_entries: 8,
            max_storage_buffer_binding_size: 1 << 30,
            subgroup_supported: true,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f16_supported: true,
        };
        let outcome = plan_exact_forward_execution(
            &standard_softmax_runtime_catalog(),
            &selection,
            &problem,
            &capabilities,
            &SelectionPolicy::default(),
        );
        let ExactForwardPlanningOutcome::Ready(plan) = outcome else {
            panic!("exact StandardSoftmax plan expected");
        };

        struct Gate;
        impl CorrectnessGate for Gate {
            fn verify(&mut self, _: &KernelCandidate, _: &AttentionProblem) -> Result<(), String> {
                Ok(())
            }
        }
        struct Harness;
        impl TimingHarness for Harness {
            fn measure(
                &mut self,
                _: &KernelCandidate,
                _: &AttentionProblem,
                protocol: &BenchmarkProtocol,
            ) -> Result<TimingSample, String> {
                Ok(TimingSample {
                    median_us: 1.0,
                    p95_us: 1.5,
                    iterations: protocol.iterations,
                })
            }
        }

        tune_exact_forward_execution_plan(&plan, protocol, &mut Gate, &mut Harness).unwrap()
    }

    fn manifest(protocol: BenchmarkProtocol) -> BenchmarkManifest {
        BenchmarkManifest {
            commit_sha: "0123456789abcdef0123456789abcdef01234567".into(),
            benchmark_id: "a11-e4r-test".into(),
            command: "cargo test -p flat-semantic-plan".into(),
            environment: BenchmarkEnvironment {
                device: "test-device".into(),
                backend: "test-backend".into(),
                driver: "test-driver".into(),
                os: "test-os".into(),
                arch: "test-arch".into(),
            },
            problem: BenchmarkProblem {
                precision: "f32".into(),
                batch: 1,
                q_heads: 4,
                kv_heads: 4,
                query_len: 129,
                kv_len: 129,
                head_dim: 64,
                causal: true,
            },
            warmup_iterations: u32::try_from(protocol.warmups).unwrap(),
            measured_iterations: u32::try_from(protocol.iterations).unwrap(),
            result: BenchmarkResult {
                median_latency_ns: 1_000,
                p95_latency_ns: 1_500,
                tokens_per_second_milli: 129_000,
            },
        }
    }

    #[test]
    fn joins_without_changing_existing_v1_payloads() {
        let protocol = BenchmarkProtocol {
            warmups: 7,
            iterations: 11,
        };
        let benchmark = manifest(protocol);
        let semantic = record(protocol);
        let benchmark_json = benchmark.canonical_json().unwrap();
        let semantic_record = semantic.canonical_provenance_record();

        let first = ExactForwardBenchmarkEvidenceEnvelope::join(&benchmark, &semantic).unwrap();
        let second = ExactForwardBenchmarkEvidenceEnvelope::join(&benchmark, &semantic).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.benchmark_manifest_json(), benchmark_json);
        assert_eq!(first.semantic_provenance(), semantic_record);
        assert!(first
            .canonical_json()
            .starts_with("{\"schema_version\":1,\"benchmark_manifest\":{\"schema_version\":1,"));
    }

    #[test]
    fn unknown_envelope_version_fails_closed() {
        let protocol = BenchmarkProtocol {
            warmups: 1,
            iterations: 2,
        };
        assert_eq!(
            ExactForwardBenchmarkEvidenceEnvelope::join_versioned(
                2,
                &manifest(protocol),
                &record(protocol),
            ),
            Err(
                ExactForwardBenchmarkEvidenceError::UnsupportedEnvelopeVersion {
                    actual: 2,
                    supported: EXACT_FORWARD_BENCHMARK_EVIDENCE_SCHEMA_VERSION,
                }
            )
        );
    }

    #[test]
    fn invalid_or_mismatched_evidence_fails_closed() {
        let protocol = BenchmarkProtocol {
            warmups: 3,
            iterations: 5,
        };
        let semantic = record(protocol);

        let mut invalid = manifest(protocol);
        invalid.commit_sha = "main".into();
        assert!(matches!(
            ExactForwardBenchmarkEvidenceEnvelope::join(&invalid, &semantic),
            Err(ExactForwardBenchmarkEvidenceError::BenchmarkManifest(
                BenchmarkManifestError::InvalidCommitSha
            ))
        ));

        let invalid_protocol = BenchmarkProtocol {
            warmups: 0,
            iterations: 5,
        };
        assert_eq!(
            ExactForwardBenchmarkEvidenceEnvelope::join(
                &manifest(invalid_protocol),
                &record(invalid_protocol),
            ),
            Err(ExactForwardBenchmarkEvidenceError::TuningProtocol(
                ProtocolError::ZeroCount
            ))
        );

        let mut protocol_mismatch = manifest(protocol);
        protocol_mismatch.measured_iterations += 1;
        assert_eq!(
            ExactForwardBenchmarkEvidenceEnvelope::join(&protocol_mismatch, &semantic),
            Err(ExactForwardBenchmarkEvidenceError::BenchmarkProtocolMismatch)
        );

        let mut problem_mismatch = manifest(protocol);
        problem_mismatch.problem.q_heads = 2;
        problem_mismatch.problem.kv_heads = 2;
        problem_mismatch.validate().unwrap();
        assert_eq!(
            ExactForwardBenchmarkEvidenceEnvelope::join(&problem_mismatch, &semantic),
            Err(ExactForwardBenchmarkEvidenceError::BenchmarkProblemMismatch)
        );
    }
}
