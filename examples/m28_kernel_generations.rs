#![cfg(feature = "wgpu")]

use std::{cmp::Ordering, hint::black_box, time::Instant};

use flat_attention::{
    forward_reference_grouped, AttentionShape, FlatAttentionConfig, GroupedAttentionShape,
    WgpuFlatAttention, WgpuKernelVariant, WgpuSubgroupPolicy,
};

const DEFAULT_WARMUP: usize = 3;
const DEFAULT_ITERATIONS: usize = 12;
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_HEADS: usize = 4;
const DEFAULT_HEAD_DIM: usize = 64;
const ATOL: f32 = 7.0e-4;
const RTOL: f32 = 3.0e-3;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.041 + phase;
            x.sin() * 0.68 + (x * 0.33).cos() * 0.32
        })
        .collect()
}

fn summarize(mut samples_us: Vec<f64>) -> (f64, f64) {
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = if samples_us.len() % 2 == 0 {
        let upper = samples_us.len() / 2;
        (samples_us[upper - 1] + samples_us[upper]) * 0.5
    } else {
        samples_us[samples_us.len() / 2]
    };
    let p95_index = ((samples_us.len() * 95).div_ceil(100)).saturating_sub(1);
    (median, samples_us[p95_index])
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let mut worst = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        let tolerance = ATOL + RTOL * actual.abs().max(expected.abs());
        assert!(
            actual.is_finite() && error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
        worst = worst.max(error);
    }
    worst
}

fn candidate(
    name: &'static str,
) -> (
    &'static str,
    Result<WgpuFlatAttention, flat_attention::WgpuFlatAttentionError>,
) {
    let attention = match name {
        "q4_portable" => WgpuFlatAttention::with_subgroup_policy_and_vectorization(
            WgpuSubgroupPolicy::Disable,
            false,
        ),
        "q4_vec4_portable" => WgpuFlatAttention::with_subgroup_policy_and_vectorization(
            WgpuSubgroupPolicy::Disable,
            true,
        ),
        "q4_vec4_double_buffered" => {
            WgpuFlatAttention::with_subgroup_vectorization_and_double_buffering(
                WgpuSubgroupPolicy::Disable,
                true,
                true,
            )
        }
        "auto" => WgpuFlatAttention::new(),
        _ => unreachable!(),
    };
    (name, attention)
}

fn main() {
    let warmup = env_usize("FLAT_M28_GENERATIONS_WARMUP", DEFAULT_WARMUP);
    let iterations = env_usize("FLAT_M28_GENERATIONS_ITERATIONS", DEFAULT_ITERATIONS);
    let seq_len = env_usize("FLAT_M28_GENERATIONS_SEQ_LEN", DEFAULT_SEQ_LEN);
    let heads = env_usize("FLAT_M28_GENERATIONS_HEADS", DEFAULT_HEADS);
    let head_dim = env_usize("FLAT_M28_GENERATIONS_HEAD_DIM", DEFAULT_HEAD_DIM);
    assert!(warmup > 0 && iterations > 0 && seq_len > 0 && heads > 0 && head_dim > 0);

    let shape = AttentionShape {
        batch: 1,
        heads,
        seq_len,
        head_dim,
    };
    let grouped_shape = GroupedAttentionShape {
        batch: 1,
        q_heads: heads,
        kv_heads: heads,
        seq_len,
        head_dim,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let len = shape.tensor_len().expect("M28 shape must fit");
    let q = fixture(len, 0.2);
    let k = fixture(len, 0.8);
    let v = fixture(len, 1.4);
    let expected = forward_reference_grouped(&q, &k, &v, grouped_shape, config)
        .expect("M28 scalar oracle failed");

    println!("benchmark=m28_kernel_generations");
    println!("commit_sha={}", option_env!("GITHUB_SHA").unwrap_or("unknown"));
    println!("timing_scope=public_forward_including_upload_dispatch_readback");
    println!("correctness_gate=O_and_LSE_match_scalar_oracle_before_timing");
    println!("performance_claim=none");
    println!("candidate,selected_variant,adapter,seq_len,heads,head_dim,warmup,iterations,median_us,p95_us,parity_o_max_abs,parity_lse_max_abs");

    for name in [
        "q4_portable",
        "q4_vec4_portable",
        "q4_vec4_double_buffered",
        "auto",
    ] {
        let (name, result) = candidate(name);
        let attention = result.unwrap_or_else(|error| panic!("{name}: {error}"));
        let selected = attention.kernel_variant_for_head_dim(head_dim);
        if name == "q4_portable" {
            assert_eq!(selected, WgpuKernelVariant::Q4Portable);
        }
        if name == "q4_vec4_portable" && matches!(head_dim, 64 | 128) {
            assert_eq!(selected, WgpuKernelVariant::Q4Vec4Portable);
        }
        if name == "q4_vec4_double_buffered" && matches!(head_dim, 64 | 128) {
            assert_eq!(selected, WgpuKernelVariant::Q4Vec4DoubleBuffered);
        }

        let actual = attention
            .forward(&q, &k, &v, shape, config)
            .unwrap_or_else(|error| panic!("{name}: correctness forward failed: {error}"));
        let parity_o = assert_close("O", &actual.output, &expected.output);
        let parity_lse = assert_close("LSE", &actual.lse, &expected.lse);

        for _ in 0..warmup {
            black_box(
                attention
                    .forward(&q, &k, &v, shape, config)
                    .unwrap_or_else(|error| panic!("{name}: warmup failed: {error}")),
            );
        }
        let samples = (0..iterations)
            .map(|_| {
                let start = Instant::now();
                black_box(
                    attention
                        .forward(&q, &k, &v, shape, config)
                        .unwrap_or_else(|error| panic!("{name}: timed forward failed: {error}")),
                );
                start.elapsed().as_secs_f64() * 1.0e6
            })
            .collect();
        let (median_us, p95_us) = summarize(samples);
        println!(
            "{name},{selected:?},{},{seq_len},{heads},{head_dim},{warmup},{iterations},{median_us:.3},{p95_us:.3},{parity_o:.8},{parity_lse:.8}",
            attention.adapter_name().replace(',', ";")
        );
    }
}
