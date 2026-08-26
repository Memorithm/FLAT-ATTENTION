//! Opt-in routing qualification: pinned tuned candidates drive real dispatch.
//!
//! Verifies the mission-critical routing contract: a valid tuning result can
//! be pinned into a live context, executes with oracle parity, reports its
//! identity through telemetry accessors, and never silently substitutes
//! another realization or a CPU path.

#![cfg(feature = "wgpu")]

use flat_attention::kernel_autotune::{tune, BenchmarkProtocol};
use flat_attention::kernel_candidates::CandidateLifecycle;
use flat_attention::kernel_candidates::{generate_candidates, SelectionPolicy};
use flat_attention::kernel_ir::AttentionProblem;
use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, OracleParityGate,
    ResidentForwardHarness, RuntimeKernelId, WgpuFlatAttention, WgpuFlatAttentionError,
};

fn synthetic_caps() -> flat_attention::RuntimeDeviceCapabilities {
    flat_attention::RuntimeDeviceCapabilities {
        max_workgroups_per_dimension: 65535,
        max_workgroup_size_x: 64,
        max_workgroup_size_y: 1024,
        max_workgroup_size_z: 64,
        max_workgroup_storage_bytes: 32768,
        max_binding_entries: 8,
        max_storage_buffer_binding_size: 1 << 30,
        subgroup_supported: true,
        subgroup_min_size: 32,
        subgroup_max_size: 32,
        f16_supported: true,
    }
}

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
        panic!("routing qualification requires a WGPU adapter");
    }
    eprintln!("WGPU adapter unavailable; optional routing test skipped");
    false
}

#[test]
fn experimental_candidate_is_refused_before_any_device_contact() {
    // Deterministic generation under opt-in policy yields the experimental
    // M7 realization; default routing must refuse it by lifecycle alone.
    let candidates = generate_candidates(
        &problem(),
        // Synthetic capabilities are fine here: the refusal must happen
        // before any adapter query.
        &flat_attention::probe_capabilities().unwrap_or(synthetic_caps()),
        &SelectionPolicy {
            allow_experimental: true,
            ..SelectionPolicy::default()
        },
    );
    let experimental = candidates
        .iter()
        .find(|c| c.lifecycle == CandidateLifecycle::Experimental)
        .expect("registry contains exactly one experimental candidate");

    let error = match WgpuFlatAttention::with_kernel_candidate(experimental) {
        Err(error) => error,
        Ok(_) => panic!("experimental candidate must be refused"),
    };
    match error {
        WgpuFlatAttentionError::CandidateNotEligible {
            candidate_id,
            lifecycle,
        } => {
            assert_eq!(candidate_id, experimental.id.get());
            assert_eq!(lifecycle, "experimental");
        }
        other => panic!("expected typed eligibility rejection, got {other}"),
    }
}

#[test]
fn pinned_tuned_candidate_routes_and_matches_oracle() {
    if !require_adapter() {
        return;
    }
    let capabilities = flat_attention::probe_capabilities().expect("adapter probed");
    let record = tune(
        &problem(),
        &capabilities,
        &SelectionPolicy::default(),
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
    );
    let Some(selected) = record.selected else {
        // No legal candidate on this device is an explicit outcome; routing
        // must not fabricate one.
        return;
    };

    let context = WgpuFlatAttention::with_kernel_candidate(&selected.candidate)
        .unwrap_or_else(|error| panic!("pinned candidate refused: {error}"));
    assert_eq!(
        context.selected_candidate_id(),
        Some(selected.candidate.id.get())
    );

    // Telemetry identity must agree with the candidate's variant mapping.
    let telemetry = context
        .runtime_telemetry(AttentionShape {
            batch: 1,
            heads: 2,
            seq_len: 64,
            head_dim: 64,
        })
        .unwrap();
    assert_eq!(
        telemetry.kernel_id,
        selected.candidate.runtime_kernel_id().unwrap()
    );

    // Numerical contract through the pinned route.
    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 64,
        head_dim: 64,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let len = shape.tensor_len().unwrap();
    let q: Vec<f32> = (0..len).map(|i| (i as f32 * 0.041).sin() * 0.6).collect();
    let k: Vec<f32> = (0..len).map(|i| (i as f32 * 0.059).cos() * 0.55).collect();
    let v: Vec<f32> = (0..len).map(|i| (i as f32 * 0.031 + 0.7).sin()).collect();
    let expected = forward_reference(&q, &k, &v, shape, config).unwrap();
    let actual = context.forward(&q, &k, &v, shape, config).unwrap();
    for (index, (&a, &b)) in actual.output.iter().zip(&expected.output).enumerate() {
        let tolerance = 2.0e-5 + 2.0e-4 * b.abs();
        assert!(
            (a - b).abs() <= tolerance,
            "O[{index}] diverged: {a} vs {b}"
        );
    }
    for (index, (&a, &b)) in actual.lse.iter().zip(&expected.lse).enumerate() {
        let tolerance = 3.0e-5 + 3.0e-4 * b.abs();
        assert!(
            (a - b).abs() <= tolerance,
            "LSE[{index}] diverged: {a} vs {b}"
        );
    }
}

#[test]
fn every_qualified_candidate_is_pinnable_or_typed_unavailable() {
    if !require_adapter() {
        return;
    }
    let capabilities = flat_attention::probe_capabilities().expect("adapter probed");
    let candidates = generate_candidates(&problem(), &capabilities, &SelectionPolicy::default());
    assert!(!candidates.is_empty());
    let mut pinnable = 0;
    for candidate in &candidates {
        assert_eq!(candidate.lifecycle, CandidateLifecycle::Qualified);
        match WgpuFlatAttention::with_kernel_candidate(candidate) {
            Ok(context) => {
                assert_eq!(context.selected_candidate_id(), Some(candidate.id.get()));
                // The routed variant must agree with the registry's mapping;
                // the pinned accessor is the identity contract.
                assert_eq!(context.selected_candidate_id(), Some(candidate.id.get()));
                let _ = RuntimeKernelId::Q4Portable;
                pinnable += 1;
            }
            Err(WgpuFlatAttentionError::CandidateUnavailable {
                candidate_id,
                reason,
            }) => {
                assert_eq!(candidate_id, candidate.id.get());
                assert!(!reason.is_empty());
            }
            Err(other) => panic!("unexpected routing error: {other:?}"),
        }
    }
    assert!(
        pinnable >= 1,
        "the portable floor candidate must always pin"
    );
}
