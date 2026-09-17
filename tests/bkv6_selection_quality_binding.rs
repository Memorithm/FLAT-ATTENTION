#![forbid(unsafe_code)]

pub mod api {
    pub use flat_attention::api::bkv6_selection_quality;
    pub use flat_attention::api::boolean_kv_paged_selection;
}

#[path = "../src/benchmark_manifest.rs"]
pub mod benchmark_manifest;
#[path = "../src/research_bkv_evidence.rs"]
pub mod research_bkv_evidence;
#[path = "../src/research_bkv_qualification.rs"]
pub mod research_bkv_qualification;
#[path = "../src/research_bkv_quality_binding.rs"]
pub mod research_bkv_quality_binding;
#[path = "../src/research_bkv_selection_binding.rs"]
pub mod research_bkv_selection_binding;

use benchmark_manifest::{
    BenchmarkEnvironment, BenchmarkManifest, BenchmarkProblem, BenchmarkResult,
};
use flat_attention::api::bkv6_selection_quality::Bkv6SelectionQualityEvidence;
use flat_attention::api::boolean_kv::{BooleanKvCache, PackedBooleanSignature};
use flat_attention::api::boolean_kv_paged_selection::{
    build_boolean_indexed_kv_selection, BooleanIndexedKvSelection, NumericalKvPageGeometry,
};
use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use research_bkv_evidence::{BikvEvidenceGates, BikvEvidenceManifest, BikvEvidenceScope};
use research_bkv_qualification::{BikvAccountingInput, BikvLatencyInput, BikvQualificationRecord};
use research_bkv_quality_binding::{
    BikvSelectionQualityBinding, BikvSelectionQualityBindingError,
    BIKV_SELECTION_QUALITY_BINDING_SCHEMA,
};

fn signature(byte: u8) -> PackedBooleanSignature {
    let bits = (0..8).map(|bit| byte & (1 << bit) != 0).collect::<Vec<_>>();
    PackedBooleanSignature::from_bools(&bits).unwrap()
}

fn selection(max_distance: usize) -> BooleanIndexedKvSelection {
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
        max_distance,
        None,
        NumericalKvPageGeometry {
            kv_heads: 2,
            head_dim: 4,
            scalar_bytes: 4,
        },
    )
    .unwrap()
}

fn benchmark(id: &str, median_latency_ns: u64) -> BenchmarkManifest {
    BenchmarkManifest {
        commit_sha: "dfd5fb7a90242b5e95c3946186864f56c83930bd".into(),
        benchmark_id: id.into(),
        command: "quality-binding-fixture".into(),
        environment: BenchmarkEnvironment {
            device: "binding-fixture".into(),
            backend: "reference".into(),
            driver: "none".into(),
            os: "test".into(),
            arch: "test".into(),
        },
        problem: BenchmarkProblem {
            precision: "f32".into(),
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            query_len: 1,
            kv_len: 10,
            head_dim: 4,
            causal: true,
        },
        warmup_iterations: 1,
        measured_iterations: 3,
        result: BenchmarkResult {
            median_latency_ns,
            p95_latency_ns: median_latency_ns + 10,
            tokens_per_second_milli: 1_000_000,
        },
    }
}

fn evidence(selection: &BooleanIndexedKvSelection) -> BikvEvidenceManifest {
    let selected_live_tokens = selection
        .selected_pages
        .iter()
        .map(|page| page.live_tokens)
        .sum();
    let qualification = BikvQualificationRecord::new(
        BikvAccountingInput {
            live_tokens: selection.live_tokens,
            selected_live_tokens,
            mapped_pages: selection.mapped_pages,
            selected_pages: selection.selected_pages.len(),
            page_size: 4,
            kv_heads: 2,
            head_dim: 4,
            scalar_bytes: 4,
            boolean_index_bytes_read: selection.boolean_key_bytes_read as u64,
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
        signature_bits: selection.signature_bits,
        max_distance: selection.max_distance,
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
fn binds_quality_to_the_exact_selection_and_qualification() {
    let selection = selection(1);
    let quality = Bkv6SelectionQualityEvidence::new(&selection, vec![0, 1]).unwrap();
    let binding =
        BikvSelectionQualityBinding::new(&selection, &quality, &evidence(&selection)).unwrap();

    let json = binding.canonical_json();
    assert!(json.contains(BIKV_SELECTION_QUALITY_BINDING_SCHEMA));
    assert!(json.contains("\"schema\":\"flat.bikv-selection-quality.v1\""));
    assert!(json.contains("\"schema\":\"flat.bikv-selection-binding.v1\""));
    assert!(json.contains("\"binding_checksum\":{\"algorithm\":\"fnv1a64\""));
}

#[test]
fn rejects_quality_from_a_different_selection() {
    let selected = selection(1);
    let different = selection(8);
    let quality = Bkv6SelectionQualityEvidence::new(&different, vec![0, 1]).unwrap();

    assert_eq!(
        BikvSelectionQualityBinding::new(&selected, &quality, &evidence(&selected)),
        Err(BikvSelectionQualityBindingError::QualitySelectionMismatch)
    );
}

#[test]
fn retains_target_choice_without_turning_it_into_a_default_threshold() {
    let selection = selection(1);
    let one_target = Bkv6SelectionQualityEvidence::new(&selection, vec![0]).unwrap();
    let two_targets = Bkv6SelectionQualityEvidence::new(&selection, vec![0, 1]).unwrap();
    let qualification = evidence(&selection);

    let one = BikvSelectionQualityBinding::new(&selection, &one_target, &qualification)
        .unwrap()
        .canonical_json();
    let two = BikvSelectionQualityBinding::new(&selection, &two_targets, &qualification)
        .unwrap()
        .canonical_json();

    assert_ne!(one, two);
    assert!(one.contains("\"quality_gate_passed\":true"));
    assert!(two.contains("\"quality_gate_passed\":true"));
}
