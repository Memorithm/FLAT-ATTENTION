#[cfg(feature = "wgpu")]
use std::hint::black_box;
#[cfg(feature = "wgpu")]
use std::time::{Duration, Instant};

#[cfg(feature = "wgpu")]
use flat_attention::{AttentionShape, FlatAttentionConfig, WgpuFlatAttention, WgpuSubgroupPolicy};

#[cfg(feature = "wgpu")]
fn fixture(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| ((i as f32) * 0.013 - 0.7).sin() * 0.8)
        .collect();
    let k = (0..len)
        .map(|i| ((i as f32) * 0.019 + 0.2).cos() * 0.7)
        .collect();
    let v = (0..len)
        .map(|i| ((i as f32) * 0.011 - 0.3).sin() * 1.1)
        .collect();
    (q, k, v)
}

#[cfg(feature = "wgpu")]
fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[cfg(feature = "wgpu")]
fn measure(
    context: &WgpuFlatAttention,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
    iterations: usize,
) -> Duration {
    for _ in 0..3 {
        let output = context
            .forward(q, k, v, shape, config)
            .expect("warm-up forward");
        black_box(output.output[0]);
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let output = context
            .forward(q, k, v, shape, config)
            .expect("measured forward");
        black_box(output.output[0]);
        samples.push(start.elapsed());
    }
    median(samples)
}

#[cfg(feature = "wgpu")]
fn main() {
    let m6 = WgpuFlatAttention::with_subgroup_vectorization_and_double_buffering(
        WgpuSubgroupPolicy::Disable,
        true,
        false,
    )
    .expect("M6 WGPU context");
    let m7 = WgpuFlatAttention::with_subgroup_vectorization_and_double_buffering(
        WgpuSubgroupPolicy::Disable,
        true,
        true,
    )
    .expect("M7 WGPU context");

    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 128,
        head_dim: 64,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let (q, k, v) = fixture(shape);
    let iterations = 9;

    let m6_median = measure(&m6, &q, &k, &v, shape, config, iterations);
    let m7_median = measure(&m7, &q, &k, &v, shape, config, iterations);

    println!("adapter={}", m6.adapter_name());
    println!("shape=B1 H2 N128 D64 causal=true");
    println!("measurement=end-to-end upload + fused dispatch + readback");
    println!("iterations={iterations} warmup=3 statistic=median");
    println!("m6_variant={:?}", m6.kernel_variant_for_head_dim(64));
    println!("m7_variant={:?}", m7.kernel_variant_for_head_dim(64));
    println!("m6_median_ms={:.3}", m6_median.as_secs_f64() * 1_000.0);
    println!("m7_median_ms={:.3}", m7_median.as_secs_f64() * 1_000.0);
    println!(
        "m6_over_m7_ratio={:.3}x",
        m6_median.as_secs_f64() / m7_median.as_secs_f64()
    );
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("run with: cargo run --release --features wgpu --example double_buffer_bench");
}
