from pathlib import Path

p = Path("tests/bkv6_m16_qualification.rs")
s = p.read_text()


def replace_once(old: str, new: str) -> None:
    global s
    if old not in s:
        raise SystemExit(f"missing anchor: {old[:80]!r}")
    s = s.replace(old, new, 1)


replace_once(
    "use std::hint::black_box;\n",
    "use std::fs;\nuse std::hint::black_box;\n",
)
replace_once(
    '#[path = "../src/research_bkv_qualification.rs"]\npub mod research_bkv_qualification;\n',
    '#[path = "../src/benchmark_manifest.rs"]\npub mod benchmark_manifest;\n#[path = "../src/research_bkv_evidence.rs"]\npub mod research_bkv_evidence;\n#[path = "../src/research_bkv_qualification.rs"]\npub mod research_bkv_qualification;\n',
)
replace_once(
    "use boolean_kv::{BooleanKvCache, PackedBooleanSignature};\n",
    "use benchmark_manifest::{BenchmarkEnvironment, BenchmarkManifest, BenchmarkProblem, BenchmarkResult};\nuse boolean_kv::{BooleanKvCache, PackedBooleanSignature};\n",
)
replace_once(
    "use research_bkv_qualification::{\n    BikvAccountingInput, BikvLatencyInput, BikvPromotionDecision, BikvQualificationRecord,\n};\n",
    "use research_bkv_evidence::{BikvEvidenceGates, BikvEvidenceManifest, BikvEvidenceScope};\nuse research_bkv_qualification::{\n    BikvAccountingInput, BikvLatencyInput, BikvPromotionDecision, BikvQualificationRecord,\n};\n",
)
replace_once(
    "fn median(mut samples: Vec<u64>) -> u64 {\n    samples.sort_unstable();\n    samples[samples.len() / 2]\n}\n",
    "fn median(mut samples: Vec<u64>) -> u64 {\n    samples.sort_unstable();\n    samples[samples.len() / 2]\n}\n\nfn percentile_95(samples: &[u64]) -> u64 {\n    let mut sorted = samples.to_vec();\n    sorted.sort_unstable();\n    let rank = ((sorted.len() * 95).div_ceil(100)).saturating_sub(1);\n    sorted[rank.min(sorted.len() - 1)]\n}\n\nfn tokens_per_second_milli(latency_ns: u64) -> u64 {\n    1_000_000_000_000u64 / latency_ns.max(1)\n}\n",
)
replace_once(
    "    let signature_generation_ns = median(signature_samples);\n",
    "    let candidate_p95_ns = percentile_95(&candidate_total_samples);\n    let dense_p95_ns = percentile_95(&dense_samples);\n    let signature_generation_ns = median(signature_samples);\n",
)
anchor = """    let decision = if !correctness_gate_passed {
        BikvPromotionDecision::FallbackCorrectnessGate
    } else if !quality_gate_passed {
        BikvPromotionDecision::FallbackQualityGate
    } else if candidate_end_to_end_ns >= dense_attention_ns {
        BikvPromotionDecision::FallbackNoLatencyWin
    } else {
        BikvPromotionDecision::Promote
    };

"""
addition = anchor + """    let commit = git_head();
    let environment = BenchmarkEnvironment {
        device: info.name.clone(),
        backend: format!("{:?}", info.backend),
        driver: format!("{info:?}"),
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
    };
    let problem = BenchmarkProblem {
        precision: "f32".to_owned(),
        batch: 1,
        q_heads: geometry.q_heads,
        kv_heads: geometry.kv_heads,
        query_len: 1,
        kv_len: geometry.kv_len,
        head_dim: geometry.head_dim,
        causal: true,
    };
    let warmup_iterations = u32::try_from(warmup).expect("warmup count fits u32");
    let measured_iterations = u32::try_from(iterations).expect("iteration count fits u32");
    let command = "cargo test --features wgpu --test bkv6_m16_qualification -- --nocapture";
    let candidate_manifest = BenchmarkManifest {
        commit_sha: commit.clone(),
        benchmark_id: "bkv-k6-selected-candidate".to_owned(),
        command: command.to_owned(),
        environment: environment.clone(),
        problem: problem.clone(),
        warmup_iterations,
        measured_iterations,
        result: BenchmarkResult {
            median_latency_ns: candidate_end_to_end_ns,
            p95_latency_ns: candidate_p95_ns,
            tokens_per_second_milli: tokens_per_second_milli(candidate_end_to_end_ns),
        },
    };
    let dense_manifest = BenchmarkManifest {
        commit_sha: commit.clone(),
        benchmark_id: "m16-dense-baseline".to_owned(),
        command: command.to_owned(),
        environment,
        problem,
        warmup_iterations,
        measured_iterations,
        result: BenchmarkResult {
            median_latency_ns: dense_attention_ns,
            p95_latency_ns: dense_p95_ns,
            tokens_per_second_milli: tokens_per_second_milli(dense_attention_ns),
        },
    };
    let evidence = BikvEvidenceManifest {
        candidate: candidate_manifest,
        dense_baseline: dense_manifest,
        qualification: record,
        signature_bits: 8,
        max_distance,
        scope: BikvEvidenceScope {
            timing_scope: "host-observed host-mirrored-query".to_owned(),
            selection_policy: "matched-density synthetic Boolean signatures".to_owned(),
            q_device_resident: true,
            q_host_mirror_retained: true,
            kv_device_resident: true,
            uploads_readbacks_excluded: true,
            resident_only_production_claim: false,
            gpu_timestamp_claim: false,
            physical_dram_traffic_claim: false,
            model_quality_claim: false,
        },
        gates: BikvEvidenceGates {
            all_accept_k6_vs_m16: all_accept_parity,
            sparse_k6_vs_restricted_oracle: sparse_correctness,
            quality_gate_passed,
        },
    };
    assert_eq!(evidence.promotion_decision().unwrap(), decision);
    let evidence_json = evidence.canonical_json().unwrap();
    if let Ok(path) = std::env::var("FLAT_BKV_EVIDENCE_OUT") {
        fs::write(&path, format!("{evidence_json}\\n")).expect("write BKV evidence envelope");
        println!("evidence_output={path}");
    }

"""
replace_once(anchor, addition)
replace_once(
    '    println!("commit={}", git_head());\n',
    '    println!("commit={commit}");\n',
)
replace_once(
    '    println!("promotion_decision={decision:?}");\n',
    '    println!("promotion_decision={decision:?}");\n    println!("evidence_json={evidence_json}");\n',
)
p.write_text(s)
