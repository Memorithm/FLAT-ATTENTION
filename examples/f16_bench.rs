#[cfg(feature = "wgpu")]
use std::hint::black_box;
#[cfg(feature = "wgpu")]
use std::time::{Duration, Instant};

#[cfg(feature = "wgpu")]
use flat_attention::{
    AttentionShape, FlatAttentionConfig, WgpuF16Attention, WgpuFlatAttention, WgpuSubgroupPolicy,
    F16,
};

#[cfg(feature = "wgpu")]
fn fixture(shape: AttentionShape) -> (Vec<F16>, Vec<F16>, Vec<F16>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| F16::from_f32(((i as f32) * 0.013 - 0.7).sin() * 0.8))
        .collect();
    let k = (0..len)
        .map(|i| F16::from_f32(((i as f32) * 0.019 + 0.2).cos() * 0.7))
        .collect();
    let v = (0..len)
        .map(|i| F16::from_f32(((i as f32) * 0.011 - 0.3).sin() * 1.1))
        .collect();
    (q, k, v)
}

#[cfg(feature = "wgpu")]
fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[cfg(feature = "wgpu")]
fn main() {
    let packed_f16 = WgpuF16Attention::new().expect("M8 packed-f16 WGPU context");
    let f32 = WgpuFlatAttention::with_subgroup_policy_and_vectorization(
        WgpuSubgroupPolicy::Disable,
        false,
    )
    .expect("f32 WGPU context");

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
    let (q16, k16, v16) = fixture(shape);
    let q32: Vec<f32> = q16.iter().copied().map(F16::to_f32).collect();
    let k32: Vec<f32> = k16.iter().copied().map(F16::to_f32).collect();
    let v32: Vec<f32> = v16.iter().copied().map(F16::to_f32).collect();

    let q16_gpu = packed_f16.upload_f16(&q16).expect("upload packed-f16 Q");
    let k16_gpu = packed_f16.upload_f16(&k16).expect("upload packed-f16 K");
    let v16_gpu = packed_f16.upload_f16(&v16).expect("upload packed-f16 V");
    let q32_gpu = f32.upload(&q32).expect("upload f32 Q");
    let k32_gpu = f32.upload(&k32).expect("upload f32 K");
    let v32_gpu = f32.upload(&v32).expect("upload f32 V");

    for _ in 0..3 {
        let resident = f32
            .forward_resident(&q32_gpu, &k32_gpu, &v32_gpu, shape, config)
            .expect("f32 warm-up dispatch");
        let output = f32
            .download_attention(&resident)
            .expect("f32 warm-up readback");
        black_box(output.output[0]);

        let resident = packed_f16
            .forward_resident(&q16_gpu, &k16_gpu, &v16_gpu, shape, config)
            .expect("packed-f16 warm-up dispatch");
        let output = packed_f16
            .download_attention(&resident)
            .expect("packed-f16 warm-up readback");
        black_box(output.output[0]);
    }

    let iterations = 9;
    let mut f32_samples = Vec::with_capacity(iterations);
    let mut packed_f16_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let resident = f32
            .forward_resident(&q32_gpu, &k32_gpu, &v32_gpu, shape, config)
            .expect("f32 measured dispatch");
        let output = f32
            .download_attention(&resident)
            .expect("f32 measured readback");
        black_box(output.output[0]);
        f32_samples.push(start.elapsed());

        let start = Instant::now();
        let resident = packed_f16
            .forward_resident(&q16_gpu, &k16_gpu, &v16_gpu, shape, config)
            .expect("packed-f16 measured dispatch");
        let output = packed_f16
            .download_attention(&resident)
            .expect("packed-f16 measured readback");
        black_box(output.output[0]);
        packed_f16_samples.push(start.elapsed());
    }

    let f32_median = median(f32_samples);
    let packed_f16_median = median(packed_f16_samples);
    let tensor_len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let lse_len = shape.batch * shape.heads * shape.seq_len;
    let f32_io_bytes = 4usize * (4 * tensor_len + lse_len);
    let packed_f16_io_bytes = 2usize * (4 * tensor_len) + 4usize * lse_len;

    println!("f32_adapter={}", f32.adapter_name());
    println!("packed_f16_adapter={}", packed_f16.adapter_name());
    println!("shape=B1 H2 N128 D64 causal=true");
    println!("measurement=resident inputs + dispatch + output readback");
    println!("iterations={iterations} warmup=3 statistic=median");
    println!("logical_qkvo_lse_bytes_f32={f32_io_bytes}");
    println!("logical_qkvo_lse_bytes_packed_f16={packed_f16_io_bytes}");
    println!("f32_median_ms={:.3}", f32_median.as_secs_f64() * 1_000.0);
    println!(
        "packed_f16_median_ms={:.3}",
        packed_f16_median.as_secs_f64() * 1_000.0
    );
    println!(
        "f32_over_packed_f16_ratio={:.3}x",
        f32_median.as_secs_f64() / packed_f16_median.as_secs_f64()
    );
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("run with: cargo run --release --features wgpu --example f16_bench");
}
