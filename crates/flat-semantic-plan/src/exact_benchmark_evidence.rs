//! Versioned fail-closed join between benchmark evidence and exact semantic provenance.
//!
//! This module is deliberately additive. It does not change `BenchmarkManifest`
//! schema v1 or the exact-forward tuning provenance contract. Instead it binds
//! the canonical representations already produced by those contracts into one
//! small envelope after validating the facts they can actually share.
//!
//! The join checks only information represented on both sides: the bounded
//! warm-up/measurement protocol and the dense attention geometry representable
//! by [`flat_attention::kernel_ir::AttentionProblem`]. It does not synthesize a
//! run identifier, source revision, device identity, precision mapping, timing
//! conversion, or any other metadata that the exact semantic provenance does
//! not already carry.

use core::fmt;

use flat_attention::{
    benchmark_manifest::{
        BenchmarkManifest, BenchmarkManifestError, BENCHMARK_MANIFEST_SCHEMA_VERSION,
    },
    kernel_autotune::ProtocolError,
};

use crate::exact_selection::{
    ExactForwardTuningRecord, EXACT_FORWARD_TUNING_PROVENANCE_VERSION,
};

/// Version of the exact benchmark/semantic evidence envelope.
pub const EXACT_FORWARD_BENCHMARK_EVIDENCE_SCHEMA_VERSION: u16 = 1;

const SUPPORTED_BENCHMARK_MANIFEST_SCHEMA_VERSION: u16 = 1;
const SUPPORTED_SEMANTIC_PROVENANCE_VERSION: u16 = 1;

/// Canonical benchmark evidence paired with canonical exact semantic provenance.
///
/// The payloads are snapshots of the two existing contracts. Keeping the
/// canonical benchmark JSON intact makes `BenchmarkManifest` v1 an embedded
/// component rather than redefining or extending its schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactForwardBenchmarkEvidenceEnvelope {
    schema_version: u16,
    benchmark_manifest_json: String,
    semantic_provenance: String,
}

/// Failure while joining benchmark evidence to exact semantic provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExactForwardBenchmarkEvidenceError {
    /// The requested envelope schema is not understood by this implementation.
    UnsupportedSchemaVersion {
        /// Requested schema version.
        actual: u16,
        /// Only schema version accepted by this implementation.
        supported: u16,
    },
    /// The embedded benchmark contract changed beyond the version supported by
    /// this envelope.
    UnsupportedBenchmarkManifestSchemaVersion {
        /// Current benchmark-manifest schema version.
        actual: u16,
        /// Benchmark-manifest schema version bound by this envelope version.
        supported: u16,
    },
    /// The exact semantic provenance contract changed beyond the version
    /// supported by this envelope.
    UnsupportedSemanticProvenanceVersion {
        /// Current exact semantic provenance version.
        actual: u16,
        /// Semantic provenance version bound by this envelope version.
        supported: u16,
    },
    /// The existing benchmark manifest failed its own v1 validation.
    BenchmarkManifest(BenchmarkManifestError),
    /// The retained semantic tuning protocol is structurally invalid.
    TuningProtocol(ProtocolError),
    /// Warm-up or measured iteration counts disagree across the two records.
    BenchmarkProtocolMismatch,
    /// The benchmark workload cannot be proven to describe the semantic
    /// tuning problem using fields represented by both contracts.
    BenchmarkProblemMismatch,
}

impl fmt::Display for ExactForwardBenchmarkEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { actual, supported } => write!(
                formatter,
                "exact benchmark evidence schema version {actual} is unsupported; expected {supported}"
            ),
            Self::UnsupportedBenchmarkManifestSchemaVersion { actual, supported } => write!(
                formatter,
                "benchmark manifest schema version {actual} is unsupported by this envelope; expected {supported}"
            ),
            Self::UnsupportedSemanticProvenanceVersion { actual, supported } => write!(
                formatter,
                "exact semantic provenance version {actual} is unsupported by this envelope; expected {supported}"
            ),
            Self::BenchmarkManifest(error) => {
                write!(formatter, "benchmark manifest is invalid: {error}")
            }
            Self::TuningProtocol(error) => {
                write!(formatter, "semantic tuning protocol is invalid: {error}")
            }
            Self::BenchmarkProtocolMismatch => formatter.write_str(
                "benchmark manifest protocol does not match exact semantic tuning provenance",
            ),
            Self::BenchmarkProblemMismatch => formatter.write_str(
                "benchmark manifest problem does not match exact semantic tuning provenance",
            ),
        }
    }
}

impl std::error::Error for ExactForwardBenchmarkEvidenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BenchmarkManifest(error) => Some(error),
            Self::TuningProtocol(error) => Some(error),
            _ => None,
        }
    }
}

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
    /// Join one validated `BenchmarkManifest` v1 record to one exact semantic
    /// tuning provenance record using the current envelope schema.
    ///
    /// # Errors
    ///
    /// Fails closed when either component contract version is unsupported,
    /// either existing record is invalid, or the protocol/problem facts that
    /// are represented by both records disagree.
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

    /// Join using an explicitly supplied envelope schema discriminator.
    ///
    /// This exists so import/export callers can reject unknown versions rather
    /// than silently interpreting them as the current contract.
    ///
    /// # Errors
    ///
    /// Returns [`ExactForwardBenchmarkEvidenceError::UnsupportedSchemaVersion`]
    /// for any version other than the one implemented here, then applies all
    /// normal fail-closed join validation.
    pub fn join_versioned(
        schema_version: u16,
        benchmark: &BenchmarkManifest,
        semantic: &ExactForwardTuningRecord,
    ) -> Result<Self, ExactForwardBenchmarkEvidenceError> {
        if schema_version != EXACT_FORWARD_BENCHMARK_EVIDENCE_SCHEMA_VERSION {
            return Err(
                ExactForwardBenchmarkEvidenceError::UnsupportedSchemaVersion {
                    actual: schema_version,
                    supported: EXACT_FORWARD_BENCHMARK_EVIDENCE_SCHEMA_VERSION,
                },
            );
        }
        if BENCHMARK_MANIFEST_SCHEMA_VERSION != SUPPORTED_BENCHMARK_MANIFEST_SCHEMA_VERSION {
            return Err(
                ExactForwardBenchmarkEvidenceError::UnsupportedBenchmarkManifestSchemaVersion {
                    actual: BENCHMARK_MANIFEST_SCHEMA_VERSION,
                    supported: SUPPORTED_BENCHMARK_MANIFEST_SCHEMA_VERSION,
                },
            );
        }
        if EXACT_FORWARD_TUNING_PROVENANCE_VERSION != SUPPORTED_SEMANTIC_PROVENANCE_VERSION {
            return Err(
                ExactForwardBenchmarkEvidenceError::UnsupportedSemanticProvenanceVersion {
                    actual: EXACT_FORWARD_TUNING_PROVENANCE_VERSION,
                    supported: SUPPORTED_SEMANTIC_PROVENANCE_VERSION,
                },
            );
        }

        // Preserve BenchmarkManifest v1 byte-for-byte as its canonical JSON
        // component and delegate all of its established validation here.
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
        let benchmark_batch_heads = benchmark_problem.batch.checked_mul(benchmark_problem.q_heads);

        // AttentionProblem is the current dense Q=K=V semantic geometry: it
        // retains folded batch×heads and one shared sequence length. Therefore
        // GQA/asymmetric manifests cannot be proven equivalent and are rejected
        // rather than guessed into an identity.
        if benchmark_problem.q_heads != benchmark_problem.kv_heads
            || benchmark_problem.query_len != benchmark_problem.kv_len
            || benchmark_batch_heads != usize::try_from(semantic_problem.batch_heads).ok()
            || Some(benchmark_problem.query_len)
                != usize::try_from(semantic_problem.seq_len).ok()
            || Some(benchmark_problem.head_dim)
                != usize::try_from(semantic_problem.head_dim).ok()
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

    /// Envelope schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Existing canonical `BenchmarkManifest` v1 JSON, unchanged.
    #[must_use]
    pub fn benchmark_manifest_json(&self) -> &str {
        &self.benchmark_manifest_json
    }

    /// Existing canonical exact-forward semantic tuning provenance, unchanged.
    #[must_use]
    pub fn semantic_provenance(&self) -> &str {
        &self.semantic_provenance
    }

    /// Deterministic canonical JSON for the versioned envelope.
    ///
    /// No checksum, identifier, timestamp, device mapping, source mapping or
    /// timing conversion is added. The benchmark object remains its existing
    /// v1 canonical JSON and the semantic record remains its existing canonical
    /// string representation.
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
                write!(out, "\\u{:04x}", character as u32)
                    .expect("writing to String cannot fail");
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
        plan_exact_forward_execution, tune_exact_forward_execution_plan, ExactForwardExecutionPlan,
        ExactForwardPlanningOutcome,
    };

    fn semantic() -> SemanticId {
        SemanticId::new(SemanticFamily::StandardSoftmax, "standard-softmax", 1).unwrap()
    }

    fn problem() -> AttentionProblem {
        AttentionProblem::from_shape(
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
        .unwrap()
    }

    fn capabilities() -> RuntimeDeviceCapabilities {
        RuntimeDeviceCapabilities {
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
        }
    }

    fn ready_plan() -> ExactForwardExecutionPlan {
        let selected = semantic();
        let registry = SemanticRegistry::new([selected.clone()]).unwrap();
        let selection = ExactSemanticSelectionPolicy
            .select(&registry, &SemanticSelectionRequest::new(selected))
            .unwrap();
        let outcome = plan_exact_forward_execution(
            &standard_softmax_runtime_catalog(),
            &selection,
            &problem(),
            &capabilities(),
            &SelectionPolicy::default(),
        );
        let ExactForwardPlanningOutcome::Ready(plan) = outcome else {
            panic!("exact StandardSoftmax plan expected");
        };
        plan
    }

    struct PassGate;

    impl CorrectnessGate for PassGate {
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

    fn tune(protocol: BenchmarkProtocol) -> ExactForwardTuningRecord {
        let plan = ready_plan();
        let mut gate = PassGate;
        let mut harness = Harness;
        tune_exact_forward_execution_plan(&plan, protocol, &mut gate, &mut harness).unwrap()
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
    fn envelope_preserves_existing_v1_payloads() {
        let protocol = BenchmarkProtocol {
            warmups: 7,
            iterations: 11,
        };
        let benchmark = manifest(protocol);
        let semantic = tune(protocol);
        let benchmark_json = benchmark.canonical_json().unwrap();
        let semantic_record = semantic.canonical_provenance_record();

        let envelope = ExactForwardBenchmarkEvidenceEnvelope::join(&benchmark, &semantic).unwrap();

        assert_eq!(
            envelope.schema_version(),
            EXACT_FORWARD_BENCHMARK_EVIDENCE_SCHEMA_VERSION
        );
        assert_eq!(envelope.benchmark_manifest_json(), benchmark_json);
        assert_eq!(envelope.semantic_provenance(), semantic_record);
        assert_eq!(benchmark.canonical_json().unwrap(), benchmark_json);
        let json = envelope.canonical_json();
        assert!(json.starts_with(
            "{\"schema_version\":1,\"benchmark_manifest\":{\"schema_version\":1,"
        ));
        assert!(json.contains(
            "\"semantic_provenance\":\"flat-exact-forward-tuning-v1\\n"
        ));
    }

    #[test]
    fn unknown_envelope_version_fails_closed() {
        let protocol = BenchmarkProtocol {
            warmups: 1,
            iterations: 2,
        };
        let error = ExactForwardBenchmarkEvidenceEnvelope::join_versioned(
            2,
            &manifest(protocol),
            &tune(protocol),
        )
        .unwrap_err();

        assert_eq!(
            error,
            ExactForwardBenchmarkEvidenceError::UnsupportedSchemaVersion {
                actual: 2,
                supported: EXACT_FORWARD_BENCHMARK_EVIDENCE_SCHEMA_VERSION,
            }
        );
    }

    #[test]
    fn invalid_benchmark_manifest_fails_closed() {
        let protocol = BenchmarkProtocol {
            warmups: 1,
            iterations: 2,
        };
        let mut benchmark = manifest(protocol);
        benchmark.commit_sha = "main".into();

        assert_eq!(
            ExactForwardBenchmarkEvidenceEnvelope::join(&benchmark, &tune(protocol)),
            Err(ExactForwardBenchmarkEvidenceError::BenchmarkManifest(
                BenchmarkManifestError::InvalidCommitSha,
            ))
        );
    }

    #[test]
    fn invalid_retained_tuning_protocol_fails_closed() {
        let invalid = BenchmarkProtocol {
            warmups: 0,
            iterations: 2,
        };
        let semantic = tune(invalid);
        let mut benchmark = manifest(BenchmarkProtocol {
            warmups: 1,
            iterations: 2,
        });
        benchmark.warmup_iterations = 0;

        assert_eq!(
            ExactForwardBenchmarkEvidenceEnvelope::join(&benchmark, &semantic),
            Err(ExactForwardBenchmarkEvidenceError::TuningProtocol(
                ProtocolError::ZeroCount,
            ))
        );
    }

    #[test]
    fn protocol_mismatch_fails_closed() {
        let protocol = BenchmarkProtocol {
            warmups: 3,
            iterations: 5,
        };
        let semantic = tune(protocol);
        let mut benchmark = manifest(protocol);
        benchmark.measured_iterations += 1;

        assert_eq!(
            ExactForwardBenchmarkEvidenceEnvelope::join(&benchmark, &semantic),
            Err(ExactForwardBenchmarkEvidenceError::BenchmarkProtocolMismatch)
        );
    }

    #[test]
    fn unprovable_problem_identity_fails_closed() {
        let protocol = BenchmarkProtocol {
            warmups: 3,
            iterations: 5,
        };
        let semantic = tune(protocol);
        let mut benchmark = manifest(protocol);
        benchmark.problem.q_heads = 2;
        benchmark.problem.kv_heads = 2;
        benchmark.validate().unwrap();

        assert_eq!(
            ExactForwardBenchmarkEvidenceEnvelope::join(&benchmark, &semantic),
            Err(ExactForwardBenchmarkEvidenceError::BenchmarkProblemMismatch)
        );
    }
}
