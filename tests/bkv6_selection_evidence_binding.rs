#![forbid(unsafe_code)]

pub mod api {
    pub use flat_attention::api::boolean_kv_paged_selection;
}

#[path = "../src/benchmark_manifest.rs"]
pub mod benchmark_manifest;
#[path = "../src/research_bkv_evidence.rs"]
pub mod research_bkv_evidence;
#[path = "../src/research_bkv_qualification.rs"]
pub mod research_bkv_qualification;
#[path = "../src/research_bkv_selection_binding.rs"]
pub mod research_bkv_selection_binding;

use benchmark_manifest::{
    BenchmarkEnvironment, BenchmarkManifest, BenchmarkProblem, BenchmarkResult,
};
use flat_attention::api::boolean_kv::{BooleanKvCache, PackedBooleanSignature};
use flat_attention::api::boolean_kv_paged_selection::{
    build_boolean_indexed_kv_selection, BooleanIndexedKvSelection, NumericalKvPageGeometry,
};
use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use research_bkv_evidence::{BikvEvidenceGates, BikvEvidenceManifest, BikvEvidenceScope};
use research_bkv_qualification::{BikvAccountingInput, BikvLatencyInput, BikvQualificationRecord};
use research_bkv_selection_binding::{
    BikvSelectionBindingError, BikvSelectionEvidenceBinding, BIKV_SELECTION_BINDING_SCHEMA,
};

fn signature(byte: u8) -> PackedBooleanSignature {
    let bits = (0..8).map(|bit| byte & (1 << bit) != 0).collect::<Vec<_>>();
    PackedBooleanSignature::from_bools(&bits).unwrap()
}

fn selection() -> BooleanIndexedKvSelection {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 4,
        physical_pages: 4,
    })
    .unwrap();
    table.append(10).unwrap();

    let mut cache = BooleanKvCache::new(8).unwrap();
    cache.append(signature(0), None).unwrap();
    cache.append(signature(0xff), None).unwrap();
    cache.append(signature(0x01), None).unwrap();

    build_boolean_indexed_kv_selection(
        &cache,
        &table,
        &signature(0),
        1,
        None,
        NumericalKvPageGeometry {
            kv_heads: 2,
            head_dim: 4,
            scalar_bytes: 4,
        },
    )
    .unwrap()
}

fn environment() -> BenchmarkEnvironment {
    BenchmarkEnvironment {
        device: "binding-fixture".into(),
        backend: "reference".into(),
        driver: "none".into(),
        os: "test".into(),
        arch: "test".into(),
    }
}

fn problem() -> BenchmarkProblem {
    BenchmarkProblem {
        precision: "f32".into(),
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        query_len: 1,
        kv_len: 10,
        head_dim: 4,
        causal: true,
    }
}

fn benchmark(id: &str, median_latency_ns: u64) -> BenchmarkManifest {
    BenchmarkManifest {
        commit_sha: "315716b1bf6c42cf5439b351aa9c1c3e7a1644ba".into(),
        benchmark_id: id.into(),
        command: "binding-fixture-only".into(),
        environment: environment(),
        problem: problem(),
        warmup_iterations: 1,
        measured_iterations: 3,
        result: BenchmarkResult {
            median_latency_ns,
            p95_latency_ns: median_latency_ns + 10,
            tokens_per_second_milli: 1_000_000,
        },
    }
}
fn evidence(
    boolean_bytes: u64,
    selected_live_tokens: usize,
    signature_bits: usize,
) -> BikvEvidenceManifest {
    let qualification = BikvQualificationRecord::new(
        BikvAccountingInput {
            live_tokens: 10,
            selected_live_tokens,
            mapped_pages: 3,
            selected_pages: 2,
            page_size: 4,
            kv_heads: 2,
            head_dim: 4,
            scalar_bytes: 4,
            boolean_index_bytes_read: boolean_bytes,
        },
        BikvLatencyInput {
            signature_generation_ns: 10,
            boolean_search_ns: 20,
            synchronization_ns: 30,
            selected_attention_ns: 40,
            dense_attention_ns: 200,
        },
    )
    .unwrap();
    BikvEvidenceManifest {
        candidate: benchmark("bkv-k6-selected", 100),
        dense_baseline: benchmark("m16-dense", 200),
        qualification,
        signature_bits,
        max_distance: 1,
        scope: BikvEvidenceScope {
            timing_scope: "synthetic binding fixture".into(),
            selection_policy: "exact selected-page fixture".into(),
            q_device_resident: false,
            q_host_mirror_retained: true,
            kv_device_resident: false,
            uploads_readbacks_excluded: false,
            resident_only_production_claim: false,
            gpu_timestamp_claim: false,
            physical_dram_traffic_claim: false,
            model_quality_claim: false,
        },
        gates: BikvEvidenceGates {
            all_accept_k6_vs_m16: true,
            sparse_k6_vs_restricted_oracle: true,
            quality_gate_passed: true,
        },
    }
}

#[test]
fn binds_exact_selection_to_qualification_evidence() {
    let selection = selection();
    assert_eq!(selection.selected_page_ids(), vec![0, 2]);
    assert_eq!(selection.boolean_key_bytes_read, 24);
    assert_eq!(selection.selected_numerical_kv_bytes, 384);

    let binding = BikvSelectionEvidenceBinding::new(&selection, &evidence(24, 6, 8)).unwrap();
    let json = binding.canonical_json();
    assert!(json.contains(BIKV_SELECTION_BINDING_SCHEMA));
    assert!(json.contains("\"schema\":\"flat.boolean-kv-selection.v2\""));
    assert!(json.contains("\"max_distance\":1"));
    assert!(json.contains("\"binding_checksum\":{\"algorithm\":\"fnv1a64\""));
    assert_eq!(
        binding.selection_json(),
        selection.canonical_evidence_json_v2().unwrap()
    );
}

#[test]
fn rejects_aggregate_boolean_byte_drift() {
    let selection = selection();
    assert!(matches!(
        BikvSelectionEvidenceBinding::new(&selection, &evidence(16, 6, 8)),
        Err(BikvSelectionBindingError::AccountingMismatch(
            "boolean_index_bytes_read"
        ))
    ));
}
#[test]
fn rejects_selected_live_token_drift() {
    let selection = selection();
    assert!(matches!(
        BikvSelectionEvidenceBinding::new(&selection, &evidence(24, 8, 8)),
        Err(BikvSelectionBindingError::AccountingMismatch(
            "selected_live_tokens"
        ))
    ));
}

#[test]
fn rejects_selection_outside_declared_hamming_threshold() {
    let selection = selection();
    let mut qualification = evidence(24, 6, 8);
    qualification.max_distance = 0;
    assert!(matches!(
        BikvSelectionEvidenceBinding::new(&selection, &qualification),
        Err(BikvSelectionBindingError::AccountingMismatch(
            "max_distance"
        ))
    ));
}

#[test]
fn rejects_widened_declared_hamming_threshold() {
    let selection = selection();
    assert_eq!(selection.max_distance, 1);
    let mut qualification = evidence(24, 6, 8);
    qualification.max_distance = 8;
    assert!(matches!(
        BikvSelectionEvidenceBinding::new(&selection, &qualification),
        Err(BikvSelectionBindingError::AccountingMismatch(
            "max_distance"
        ))
    ));
}

#[test]
fn rejects_signature_width_drift() {
    let selection = selection();
    assert!(matches!(
        BikvSelectionEvidenceBinding::new(&selection, &evidence(24, 6, 16)),
        Err(BikvSelectionBindingError::AccountingMismatch(
            "signature_bits"
        ))
    ));
}
