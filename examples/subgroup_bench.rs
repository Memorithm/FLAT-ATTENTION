#[cfg(feature = "wgpu")]
use std::hint::black_box;
#[cfg(feature = "wgpu")]
use std::time::{Duration, Instant};

#[cfg(feature = "wgpu")]
use flat_attention::{
    AttentionShape, FlatAttentionConfig, WgpuFlatAttention, WgpuFlatAttentionError,
    WgpuSubgroupPolicy,
};

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
fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
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
    let portable = WgpuFlatAttention::with_subgroup_policy(WgpuSubgroupPolicy::Disable)
        .expect("portable WGPU context");
    let subgroup = match WgpuFlatAttention::with_subgroup_policy(WgpuSubgroupPolicy::Require) {
        Ok(context) => context,
        Err(WgpuFlatAttentionError::RequiredSubgroupUnavailable) => {
            eprintln!(
                "adapter '{}' does not expose WGPU subgroup support; no subgroup timing is claimed",
                portable.adapter_name()
            );
            return;
        }
        Err(error) => panic!("subgroup WGPU context: {error}"),
    };

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

    let portable_median = measure(&portable, &q, &k, &v, shape, config, iterations);
    let subgroup_median = measure(&subgroup, &q, &k, &v, shape, config, iterations);

    println!("adapter={}", portable.adapter_name());
    println!("subgroup_range={:?}", subgroup.subgroup_size_range());
    println!("shape=B1 H2 N128 D64 causal=true");
    println!("measurement=end-to-end upload + fused dispatch + readback");
    println!("iterations={iterations} warmup=3 statistic=median");
    println!(
        "q4_portable_median_ms={:.3}",
        portable_median.as_secs_f64() * 1_000.0
    );
    println!(
        "q4_subgroup_median_ms={:.3}",
        subgroup_median.as_secs_f64() * 1_000.0
    );
    println!(
        "portable_over_subgroup_ratio={:.3}x",
        portable_median.as_secs_f64() / subgroup_median.as_secs_f64()
    );
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("run with: cargo run --release --features wgpu --example subgroup_bench");
}
