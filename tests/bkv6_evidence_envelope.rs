#[path = "../src/benchmark_manifest.rs"]
pub mod benchmark_manifest;
#[path = "../src/research_bkv_evidence.rs"]
pub mod research_bkv_evidence;
#[path = "../src/research_bkv_qualification.rs"]
pub mod research_bkv_qualification;

use benchmark_manifest::{
    BenchmarkEnvironment, BenchmarkManifest, BenchmarkProblem, BenchmarkResult,
};
use research_bkv_evidence::{
    BikvEvidenceError, BikvEvidenceGates, BikvEvidenceManifest, BikvEvidenceScope,
    BIKV_EVIDENCE_SCHEMA_VERSION,
};
use research_bkv_qualification::{
    BikvAccountingInput, BikvLatencyInput, BikvPromotionDecision, BikvQualificationRecord,
};

fn environment() -> BenchmarkEnvironment {
    BenchmarkEnvironment {
        device: "Qualification Device".into(),
        backend: "Vulkan".into(),
        driver: "driver-1".into(),
        os: "Linux".into(),
        arch: "x86_64".into(),
    }
}

fn problem() -> BenchmarkProblem {
    BenchmarkProblem {
        precision: "f32".into(),
        batch: 1,
        q_heads: 4,
        kv_heads: 1,
        query_len: 1,
        kv_len: 128,
        head_dim: 64,
        causal: true,
    }
}

fn candidate() -> BenchmarkManifest {
    BenchmarkManifest {
        commit_sha: "0123456789abcdef0123456789abcdef01234567".into(),
        benchmark_id: "bkv-k6-candidate".into(),
        command: "cargo test --features wgpu --test bkv6_m16_qualification -- --nocapture".into(),
        environment: environment(),
        problem: problem(),
        warmup_iterations: 3,
        measured_iterations: 9,
        result: BenchmarkResult {
            median_latency_ns: 800_000,
            p95_latency_ns: 900_000,
            tokens_per_second_milli: 1_250_000,
        },
    }
}

fn dense() -> BenchmarkManifest {
    BenchmarkManifest {
        commit_sha: "0123456789abcdef0123456789abcdef01234567".into(),
        benchmark_id: "m16-dense-baseline".into(),
        command: "cargo test --features wgpu --test bkv6_m16_qualification -- --nocapture".into(),
        environment: environment(),
        problem: problem(),
        warmup_iterations: 3,
        measured_iterations: 9,
        result: BenchmarkResult {
            median_latency_ns: 1_000_000,
            p95_latency_ns: 1_100_000,
            tokens_per_second_milli: 1_000_000,
        },
    }
}

fn qualification() -> BikvQualificationRecord {
    BikvQualificationRecord::new(
        BikvAccountingInput {
            live_tokens: 128,
            selected_live_tokens: 64,
            mapped_pages: 16,
            selected_pages: 8,
            page_size: 8,
            kv_heads: 1,
            head_dim: 64,
            scalar_bytes: 4,
            boolean_index_bytes_read: 128,
        },
        BikvLatencyInput {
            signature_generation_ns: 100_000,
            boolean_search_ns: 200_000,
            synchronization_ns: 300_000,
            selected_attention_ns: 600_000,
            dense_attention_ns: 1_000_000,
        },
    )
    .unwrap()
}

fn scope() -> BikvEvidenceScope {
    BikvEvidenceScope {
        timing_scope: "host-observed host-mirrored-query".into(),
        selection_policy: "matched-density synthetic Boolean signatures".into(),
        q_device_resident: true,
        q_host_mirror_retained: true,
        kv_device_resident: true,
        uploads_readbacks_excluded: true,
        resident_only_production_claim: false,
        gpu_timestamp_claim: false,
        physical_dram_traffic_claim: false,
        model_quality_claim: false,
    }
}

fn gates() -> BikvEvidenceGates {
    BikvEvidenceGates {
        all_accept_k6_vs_m16: true,
        sparse_k6_vs_restricted_oracle: true,
        quality_gate_passed: true,
    }
}

fn manifest() -> BikvEvidenceManifest {
    BikvEvidenceManifest {
        candidate: candidate(),
        dense_baseline: dense(),
        qualification: qualification(),
        signature_bits: 8,
        max_distance: 1,
        scope: scope(),
        gates: gates(),
    }
}

#[test]
fn canonical_evidence_is_deterministic_and_self_describing() {
    let first = manifest();
    let second = first.clone();
    let first_json = first.canonical_json().unwrap();
    let second_json = second.canonical_json().unwrap();
    assert_eq!(first_json, second_json);
    assert!(first_json.contains(&format!(
        "\"schema_version\":{BIKV_EVIDENCE_SCHEMA_VERSION}"
    )));
    assert!(first_json.contains("\"candidate\":{"));
    assert!(first_json.contains("\"dense_baseline\":{"));
    assert!(first_json.contains("\"boolean_index_bytes_read\":128"));
    assert!(first_json.contains("\"promotion_decision\":\"promote\""));
    assert!(first_json.contains("\"evidence_checksum\":{\"algorithm\":\"fnv1a64\""));
}

#[test]
fn promotion_uses_end_to_end_candidate_median_not_phase_median_sum() {
    let evidence = manifest();
    assert_eq!(evidence.qualification.total_bikv_latency_ns(), 1_200_000);
    assert_eq!(
        evidence.qualification.promotion_decision(true, true),
        BikvPromotionDecision::FallbackNoLatencyWin
    );
    assert_eq!(
        evidence.promotion_decision().unwrap(),
        BikvPromotionDecision::Promote
    );
}

#[test]
fn independent_quality_and_correctness_gates_fail_closed() {
    let mut evidence = manifest();
    evidence.gates.quality_gate_passed = false;
    assert_eq!(
        evidence.promotion_decision().unwrap(),
        BikvPromotionDecision::FallbackQualityGate
    );

    evidence.gates.quality_gate_passed = true;
    evidence.gates.sparse_k6_vs_restricted_oracle = false;
    assert_eq!(
        evidence.promotion_decision().unwrap(),
        BikvPromotionDecision::FallbackCorrectnessGate
    );
}

#[test]
fn rejects_cross_commit_or_cross_problem_comparisons() {
    let mut evidence = manifest();
    evidence.dense_baseline.commit_sha = "fedcba9876543210fedcba9876543210fedcba98".into();
    assert_eq!(evidence.validate(), Err(BikvEvidenceError::CommitMismatch));

    let mut evidence = manifest();
    evidence.dense_baseline.problem.head_dim = 128;
    assert_eq!(evidence.validate(), Err(BikvEvidenceError::ProblemMismatch));
}

#[test]
fn rejects_contradictory_resident_only_scope() {
    let mut evidence = manifest();
    evidence.scope.resident_only_production_claim = true;
    assert_eq!(
        evidence.validate(),
        Err(BikvEvidenceError::ContradictoryResidentScope)
    );
}

#[test]
fn rejects_dense_median_drift_from_qualification_record() {
    let mut evidence = manifest();
    evidence.dense_baseline.result.median_latency_ns += 1;
    assert_eq!(
        evidence.validate(),
        Err(BikvEvidenceError::DenseLatencyMismatch {
            qualification_ns: 1_000_000,
            manifest_ns: 1_000_001,
        })
    );
}
