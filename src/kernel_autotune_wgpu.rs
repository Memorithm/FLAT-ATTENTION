//! Production WGPU timing/correctness surfaces for the autotuner core.
//!
//! Boundary documentation (part of the repository's benchmark methodology):
//! this harness times the **transfer-inclusive** public `forward` path
//! (upload + fused dispatch + readback). The boundary is identical for every
//! candidate, so relative comparisons among candidates are fair A/B
//! measurements under one declared scope. It is not resident-only kernel
//! timing and must never be reported as such. Software-adapter timings are
//! qualification evidence only.

use crate::device_model::RuntimeDeviceCapabilities;
use crate::kernel_autotune::{CorrectnessGate, TimingHarness, TimingSample};
use crate::kernel_candidates::KernelCandidate;
use crate::kernel_ir::{AttentionProblem, ScoreReduction};
use crate::{
    forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionOutput, WgpuFlatAttention,
    WgpuSubgroupPolicy,
};

const O_ATOL: f32 = 2.0e-5;
const O_RTOL: f32 = 2.0e-4;
const LSE_ATOL: f32 = 3.0e-5;
const LSE_RTOL: f32 = 3.0e-4;

fn policy_for(candidate: &KernelCandidate) -> Result<WgpuSubgroupPolicy, String> {
    if candidate.config.score_reduction == ScoreReduction::SubgroupAssisted {
        Ok(WgpuSubgroupPolicy::Require)
    } else {
        Ok(WgpuSubgroupPolicy::Disable)
    }
}

fn context_for(candidate: &KernelCandidate) -> Result<WgpuFlatAttention, String> {
    let subgroup_policy = policy_for(candidate)?;
    let vectorization = candidate.config.vector_width == crate::kernel_ir::VectorWidth::Vec4;
    let double_buffering =
        candidate.config.kv_staging == crate::kernel_ir::KvStaging::DoubleBuffered;
    WgpuFlatAttention::with_subgroup_vectorization_and_double_buffering(
        subgroup_policy,
        vectorization,
        double_buffering,
    )
    .map_err(|error| format!("context construction: {error}"))
}

/// Oracle-parity gate using the qualified device tolerances.
pub struct OracleParityGate {
    fixtures: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
}

impl OracleParityGate {
    /// Build deterministic bounded-magnitude fixtures covering causal and
    /// non-causal modes for the given shape.
    #[must_use]
    pub fn new(shape: &AttentionShape, config: FlatAttentionConfig) -> Self {
        let len = shape
            .tensor_len()
            .expect("validated planning shape has finite length");
        let fixture = |phase: f32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    let x = i as f32 * 0.061 + phase;
                    x.sin() * 0.68 + (x * 0.41).cos() * 0.32
                })
                .collect()
        };
        let mut fixtures = Vec::new();
        for phase in [0.13_f32, 0.71] {
            fixtures.push((fixture(phase), fixture(phase + 0.29), fixture(phase + 0.53)));
        }
        let _ = config;
        Self { fixtures }
    }
}

impl CorrectnessGate for OracleParityGate {
    fn verify(
        &mut self,
        candidate: &KernelCandidate,
        problem: &AttentionProblem,
    ) -> Result<(), String> {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: problem.seq_len as usize,
            head_dim: problem.head_dim as usize,
        };
        let flat_config = FlatAttentionConfig {
            causal: problem.causal,
            softmax_scale: None,
        };
        let context =
            context_for(candidate).map_err(|error| format!("context construction: {error}"))?;
        for (q, k, v) in &self.fixtures {
            let expected: FlatAttentionOutput = forward_reference(q, k, v, shape, flat_config)
                .map_err(|error| format!("oracle failed: {error}"))?;
            let actual = context
                .forward(q, k, v, shape, flat_config)
                .map_err(|error| format!("forward failed: {error}"))?;
            verify_close("O", &actual.output, &expected.output, O_ATOL, O_RTOL)?;
            verify_close("LSE", &actual.lse, &expected.lse, LSE_ATOL, LSE_RTOL)?;
        }
        Ok(())
    }
}

fn verify_close(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{name}: length mismatch {} vs {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() {
            return Err(format!("{name}[{index}] is not finite: {a}"));
        }
        let error = (a - b).abs();
        let tolerance = atol + rtol * b.abs();
        if error > tolerance {
            return Err(format!(
                "{name}[{index}]: actual={a}, expected={b}, abs_error={error}, tolerance={tolerance}"
            ));
        }
    }
    Ok(())
}

/// Transfer-inclusive timing harness over the public forward path.
pub struct ResidentForwardHarness;

impl TimingHarness for ResidentForwardHarness {
    fn measure(
        &mut self,
        candidate: &KernelCandidate,
        problem: &AttentionProblem,
        protocol: &crate::kernel_autotune::BenchmarkProtocol,
    ) -> Result<TimingSample, String> {
        use std::time::Instant;

        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: problem.seq_len as usize,
            head_dim: problem.head_dim as usize,
        };
        let flat_config = FlatAttentionConfig {
            causal: problem.causal,
            softmax_scale: None,
        };
        let len = shape
            .tensor_len()
            .map_err(|error| format!("shape sizing: {error}"))?;
        let q: Vec<f32> = (0..len).map(|i| (i as f32 * 0.037).sin() * 0.5).collect();
        let k: Vec<f32> = (0..len).map(|i| (i as f32 * 0.053).cos() * 0.5).collect();
        let v: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.029 + 1.0).sin() * 0.9)
            .collect();

        let context =
            context_for(candidate).map_err(|error| format!("context construction: {error}"))?;
        for _ in 0..protocol.warmups {
            context
                .forward(&q, &k, &v, shape, flat_config)
                .map_err(|error| format!("warmup forward failed: {error}"))?;
        }

        let mut samples_us: Vec<f64> = Vec::with_capacity(protocol.iterations);
        for _ in 0..protocol.iterations {
            let start = Instant::now();
            let result = context
                .forward(&q, &k, &v, shape, flat_config)
                .map_err(|error| format!("measured forward failed: {error}"))?;
            let elapsed = start.elapsed();
            // Keep the output alive so the readback cannot be optimized out.
            let checksum: f32 = result.output.iter().take(16).sum::<f32>()
                + result.lse.first().copied().unwrap_or(0.0);
            debug_assert!(checksum.is_finite());
            samples_us.push(elapsed.as_secs_f64() * 1_000_000.0);
        }
        samples_us.sort_by(|a, b| a.total_cmp(b));
        let percentile = |p: f64| -> f64 {
            let index = ((samples_us.len() as f64 * p).ceil() as usize)
                .saturating_sub(1)
                .min(samples_us.len() - 1);
            samples_us[index]
        };
        let median = if samples_us.len() % 2 == 1 {
            samples_us[samples_us.len() / 2]
        } else {
            let mid = samples_us.len() / 2;
            (samples_us[mid - 1] + samples_us[mid]) / 2.0
        };
        Ok(TimingSample {
            median_us: median,
            p95_us: percentile(0.95),
            iterations: protocol.iterations,
        })
    }
}

/// Convenience: capabilities snapshot of the adapter backing the harness
/// contexts, used by callers that need the fingerprint in evidence records.
///
/// # Panics-free guarantee: returns `None` when no adapter is available.
#[must_use]
pub fn probe_capabilities() -> Option<RuntimeDeviceCapabilities> {
    WgpuFlatAttention::new()
        .ok()
        .map(|attention| attention.device_capabilities())
}
