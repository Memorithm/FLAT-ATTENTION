//! Device-level autotuner qualification: one real bounded tuning session on
//! the available adapter.
//!
//! This is a correctness/protocol qualification of the tuner machinery, not a
//! performance claim. Selected-candidate timings from software adapters are
//! never generalized as physical-GPU performance.

#![cfg(feature = "wgpu")]

use flat_attention::kernel_autotune::{tune, BenchmarkProtocol, SelectionRecord};
use flat_attention::kernel_candidates::SelectionPolicy;
use flat_attention::kernel_ir::AttentionProblem;
use flat_attention::{
    AttentionShape, FlatAttentionConfig, OracleParityGate, ResidentForwardHarness,
    WgpuFlatAttention,
};

fn problem() -> AttentionProblem {
    AttentionProblem::from_shape(
        &AttentionShape {
            batch: 1,
            heads: 2,
            seq_len: 64,
            head_dim: 64,
        },
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        },
    )
    .unwrap()
}

fn require_adapter() -> bool {
    if WgpuFlatAttention::new().is_ok() {
        return true;
    }
    if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
        panic!("autotuner device qualification requires a WGPU adapter");
    }
    eprintln!("WGPU adapter unavailable; optional autotuner device test skipped");
    false
}

fn run_session() -> SelectionRecord {
    let capabilities = flat_attention::probe_capabilities().expect("adapter probed above");
    tune(
        &problem(),
        &capabilities,
        &SelectionPolicy::default(),
        // Small protocol keeps CI cheap while exercising the full path.
        BenchmarkProtocol {
            warmups: 1,
            iterations: 5,
        },
        &mut OracleParityGate::new(
            &AttentionShape {
                batch: 1,
                heads: 2,
                seq_len: 64,
                head_dim: 64,
            },
            FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
        ),
        &mut ResidentForwardHarness,
    )
}

#[test]
fn tuning_session_produces_evidence_on_live_adapter() {
    if !require_adapter() {
        return;
    }
    let record = run_session();

    // Every considered candidate carries either measured timing or an
    // explicit rejection; nothing is silently dropped.
    assert!(
        !record.per_candidate.is_empty(),
        "a healthy adapter must admit at least one candidate"
    );
    for (_candidate, evidence) in &record.per_candidate {
        match evidence {
            flat_attention::kernel_autotune::CandidateEvidence::Measured { timing } => {
                assert_eq!(timing.iterations, 5);
                assert!(timing.median_us.is_finite() && timing.median_us >= 0.0);
                assert!(timing.p95_us >= timing.median_us - f64::EPSILON);
            }
            flat_attention::kernel_autotune::CandidateEvidence::Rejected { rejection } => {
                let message = rejection.to_string();
                assert!(!message.is_empty());
            }
        }
    }

    // A selection exists exactly when at least one candidate was measured,
    // and its identity matches one of the measured entries.
    let measured_ids: Vec<_> = record
        .per_candidate
        .iter()
        .filter(|(_, e)| {
            matches!(
                e,
                flat_attention::kernel_autotune::CandidateEvidence::Measured { .. }
            )
        })
        .map(|(c, _)| c.id)
        .collect();
    match &record.selected {
        Some(selected) => {
            assert!(measured_ids.contains(&selected.candidate.id));
        }
        None => assert!(measured_ids.is_empty()),
    }
}

#[test]
fn repeated_sessions_rank_identically_under_identical_conditions() {
    // Determinism contract at protocol level: same inputs produce the same
    // candidate consideration order and identity set. Timing values may vary
    // between sessions; identities and structure must not.
    if !require_adapter() {
        return;
    }
    let first = run_session();
    let second = run_session();
    let ids_a: Vec<_> = first.per_candidate.iter().map(|(c, _)| c.id).collect();
    let ids_b: Vec<_> = second.per_candidate.iter().map(|(c, _)| c.id).collect();
    assert_eq!(ids_a, ids_b);
}
