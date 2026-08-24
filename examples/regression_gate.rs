//! Same-device performance regression gate.
//!
//! Compares opt-in kernel generations against the qualified Q4 portable
//! baseline **within one process on one adapter**, so thresholds are hardware
//! independent: any regression of a candidate relative to the reference is
//! visible on every adapter from Mesa lavapipe to physical GPUs.
//!
//! This is deliberately NOT a cross-run or cross-machine absolute benchmark.
//! Absolute timing claims remain governed by the physical-hardware
//! qualification protocol (`docs/RELEASE_BENCHMARK_SNAPSHOT.md`).
//!
//! Gates (median of iterations, end-to-end upload + dispatch + readback):
//!
//! - `q4_vec4_d64` must not exceed [`VEC4_D64_MAX_RATIO`] × `q4_portable_d64`;
//! - `q4_portable_d128` must not exceed [`D128_D64_MAX_RATIO`] ×
//!   `q4_portable_d64` (linear-in-d scaling sanity bound).
//!
//! Environment overrides:
//!
//! - `FLAT_REGRESSION_GATE_ITERATIONS` / `_WARMUP`: timing budget;
//! - `FLAT_REGRESSION_GATE_VEC4_RATIO` / `_D128_RATIO`: thresholds;
//! - `FLAT_REQUIRE_WGPU=1`: fail instead of skipping without an adapter.

use std::fmt::Write as _;

use flat_attention::{AttentionShape, FlatAttentionConfig, WgpuFlatAttention};

const SEQ_LEN: usize = 128;
const HEADS: usize = 2;
const BATCH: usize = 1;

/// A vec4 specialization must not be slower than this multiple of the scalar
/// staging path on the same adapter.
const VEC4_D64_MAX_RATIO: f64 = 1.20;
/// Head-dimension doubling is memory-bound here; allow generous headroom while
/// still catching pathological regressions.
const D128_D64_MAX_RATIO: f64 = 2.50;

struct Measurement {
    median_ms: f64,
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.017 + phase;
            x.sin() * 0.875 + (x * 0.41).cos() * 0.3125
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
    values[values.len() / 2]
}

fn run_workload(
    context: &WgpuFlatAttention,
    head_dim: usize,
    expect_vec4: bool,
    iterations: usize,
) -> Result<Measurement, String> {
    // The context under test must actually select the intended generation.
    let selected_variant = context.kernel_variant_for_head_dim(head_dim);
    let variant_name = format!("{selected_variant:?}");
    if expect_vec4 != variant_name.contains("Vec4") {
        return Err(format!(
            "kernel selection mismatch: expected_vec4={expect_vec4} head_dim={head_dim} selected={variant_name}"
        ));
    }
    let shape = AttentionShape {
        batch: BATCH,
        heads: HEADS,
        seq_len: SEQ_LEN,
        head_dim,
    };
    let tensor_len = shape.tensor_len().map_err(|error| error.to_string())?;
    let q = fixture(tensor_len, 0.11);
    let k = fixture(tensor_len, 0.47);
    let v = fixture(tensor_len, 0.83);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };

    // Warm-up outside timing.
    for _ in 0..3 {
        context
            .forward(&q, &k, &v, shape, config)
            .map_err(|error| error.to_string())?;
    }

    let mut timings_ms = Vec::with_capacity(iterations);
    let start_total = std::time::Instant::now();
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let output = context
            .forward(&q, &k, &v, shape, config)
            .map_err(|error| error.to_string())?;
        let elapsed = start.elapsed().as_secs_f64() * 1.0e3;
        // Touch the result so readback cannot be optimized away semantically.
        debug_assert!(!output.output.is_empty());
        timings_ms.push(elapsed);
    }
    let _ = start_total.elapsed();
    Ok(Measurement {
        median_ms: median(&mut timings_ms),
    })
}

fn main() {
    let Some(reference_context) = probe_context(false) else {
        return;
    };
    let Some(vec4_context) = probe_context(true) else {
        fail("vec4 context construction failed while the portable context succeeded");
    };
    let iterations = std::env::var("FLAT_REGRESSION_GATE_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(15);
    let vec4_ratio = std::env::var("FLAT_REGRESSION_GATE_VEC4_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(VEC4_D64_MAX_RATIO);
    let d128_ratio = std::env::var("FLAT_REGRESSION_GATE_D128_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(D128_D64_MAX_RATIO);

    println!("portable_adapter={}", reference_context.adapter_name());
    println!(
        "iterations={iterations} statistic=median measurement=end_to_end_including_upload_readback"
    );

    println!("vec4_adapter={}", vec4_context.adapter_name());

    let reference = match run_workload(&reference_context, 64, false, iterations) {
        Ok(measurement) => measurement,
        Err(error) => fail(&format!("reference q4_portable_d64 failed: {error}")),
    };
    report("q4_portable_d64", &reference, None, None);

    let vec4 = match run_workload(&vec4_context, 64, true, iterations) {
        Ok(measurement) => measurement,
        Err(error) => fail(&format!("candidate q4_vec4_d64 failed: {error}")),
    };
    let vec4_actual = vec4.median_ms / reference.median_ms;
    report(
        "q4_vec4_d64",
        &vec4,
        Some(vec4_actual),
        Some(("vs_q4_portable_d64", vec4_ratio)),
    );
    check("vec4_d64_ratio", vec4_actual, vec4_ratio);

    let d128 = match run_workload(&reference_context, 128, false, iterations) {
        Ok(measurement) => measurement,
        Err(error) => fail(&format!("candidate q4_portable_d128 failed: {error}")),
    };
    let d128_actual = d128.median_ms / reference.median_ms;
    report(
        "q4_portable_d128",
        &d128,
        Some(d128_actual),
        Some(("vs_q4_portable_d64", d128_ratio)),
    );
    check("d128_d64_ratio", d128_actual, d128_ratio);

    println!("regression_gate_verdict=pass");
}

fn probe_context(vectorization: bool) -> Option<WgpuFlatAttention> {
    match WgpuFlatAttention::with_subgroup_policy_and_vectorization(
        flat_attention::WgpuSubgroupPolicy::Disable,
        vectorization,
    ) {
        Ok(context) => Some(context),
        Err(error) => {
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
                panic!("regression gate requires a WGPU adapter: {error}");
            }
            eprintln!("WGPU adapter unavailable; regression gate skipped");
            None
        }
    }
}

fn report(name: &str, measurement: &Measurement, ratio: Option<f64>, bound: Option<(&str, f64)>) {
    let mut line = format!("workload={name} median_ms={:.3}", measurement.median_ms);
    if let Some(ratio) = ratio {
        let _ = write!(line, " ratio={ratio:.3}");
    }
    if let Some((bound_name, bound_value)) = bound {
        let _ = write!(line, " bound={bound_name} max_ratio={bound_value:.2}");
    }
    println!("{line}");
}

fn check(name: &str, actual: f64, maximum: f64) {
    if !actual.is_finite() || actual > maximum {
        fail(&format!(
            "{name}: observed ratio {actual:.3} exceeds maximum {maximum:.3}"
        ));
    }
}

fn fail(message: &str) -> ! {
    eprintln!("regression_gate_verdict=fail");
    eprintln!("regression_gate_reason={message}");
    std::process::exit(1);
}
