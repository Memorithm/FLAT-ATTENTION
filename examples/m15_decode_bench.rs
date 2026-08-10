#![cfg(feature = "wgpu")]

use std::time::Instant;

use flat_attention::{
    AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig,
    ExternalAsymmetricProjectionPass, ExternalAsymmetricProjectionRotaryGroupedPipeline,
    FlatAttentionConfig, ResidentDecodePass, WgpuResidentDecodePipeline, WgpuResidentKvCache,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.019 + phase;
            x.sin() * 1.25 + (x * 0.37).cos() * 0.3125
        })
        .collect()
}

fn rotate_k_projection(
    raw: &[f32],
    kv_len: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
) -> Vec<f32> {
    let mut rotated = raw.to_vec();
    let width = kv_heads * head_dim;
    for position in 0..kv_len {
        for head in 0..kv_heads {
            let head_base = position * width + head * head_dim;
            for pair in 0..head_dim / 2 {
                let dim = 2 * pair;
                let exponent = -2.0 * pair as f32 / head_dim as f32;
                let frequency = theta.powf(exponent);
                let angle = position as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                let even = raw[head_base + dim];
                let odd = raw[head_base + dim + 1];
                rotated[head_base + dim] = even * cos - odd * sin;
                rotated[head_base + dim + 1] = even * sin + odd * cos;
            }
        }
    }
    rotated
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn input_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m15-bench-input"),
        size: bytes.len().max(4) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

fn summarize(mut samples_us: Vec<f64>) -> (f64, f64) {
    samples_us.sort_by(f64::total_cmp);
    (percentile(&samples_us, 0.5), percentile(&samples_us, 0.95))
}

fn main() {
    let kv_len = env_usize("FLAT_BENCH_KV_LEN", 1024);
    let capacity = env_usize("FLAT_BENCH_CAPACITY", kv_len.max(2048));
    let warmup = env_usize("FLAT_BENCH_WARMUP", 10);
    let iterations = env_usize("FLAT_BENCH_ITERS", 50);
    assert!(
        capacity >= kv_len,
        "FLAT_BENCH_CAPACITY must be >= KV length"
    );

    let (q_heads, kv_heads, head_dim) = (8usize, 2usize, 64usize);
    let theta = 10_000.0;
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("M15 benchmark requires a WGPU adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m15-decode-bench"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .expect("M15 benchmark request_device failed");

    let q = fixture(q_heads * head_dim, 0.2);
    let raw_k = fixture(kv_len * kv_heads * head_dim, 0.8);
    let v = fixture(kv_len * kv_heads * head_dim, 1.4);
    let rotated_k = rotate_k_projection(&raw_k, kv_len, kv_heads, head_dim, theta);
    let q_gpu = input_buffer(&device, &queue, &q, wgpu::BufferUsages::STORAGE);
    let k_gpu = input_buffer(
        &device,
        &queue,
        &rotated_k,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let v_gpu = input_buffer(
        &device,
        &queue,
        &v,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let mut cache = WgpuResidentKvCache::new(&device, 1, kv_heads, capacity, head_dim).unwrap();
    let mut append_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m15-bench-cache-append"),
    });
    cache
        .record_append(&mut append_encoder, &k_gpu, &v_gpu, kv_len)
        .unwrap();
    queue.submit(Some(append_encoder.finish()));
    let _ = device.poll(wgpu::Maintain::Wait);

    let resident = WgpuResidentDecodePipeline::new(&device).unwrap();
    let resident_output = resident
        .create_output_buffer(&device, &cache, q_heads)
        .unwrap();

    let generic = ExternalAsymmetricProjectionRotaryGroupedPipeline::new(&device).unwrap();
    let generic_shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads,
        kv_heads,
        query_len: 1,
        kv_len,
        head_dim,
        query_position_offset: kv_len - 1,
    };
    let rotary = AsymmetricRotaryEmbeddingConfig {
        theta,
        query_position_offset: kv_len - 1,
        kv_position_offset: 0,
    };
    let generic_output = generic
        .create_output_buffer(&device, generic_shape)
        .unwrap();

    let run_resident = || {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m15-bench-resident"),
        });
        resident
            .encode(
                &device,
                &mut encoder,
                ResidentDecodePass {
                    q: &q_gpu,
                    out_and_lse: &resident_output,
                    cache: &cache,
                    q_heads,
                    config,
                    theta,
                    q_rope_position: kv_len - 1,
                },
            )
            .unwrap();
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::Maintain::Wait);
        start.elapsed().as_secs_f64() * 1.0e6
    };

    let run_generic = || {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m15-bench-generic"),
        });
        generic
            .encode_pre_rotated_k(
                &device,
                &mut encoder,
                ExternalAsymmetricProjectionPass {
                    q: &q_gpu,
                    k: &k_gpu,
                    v: &v_gpu,
                    out_and_lse: &generic_output,
                    shape: generic_shape,
                    config,
                    rotary,
                },
            )
            .unwrap();
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::Maintain::Wait);
        start.elapsed().as_secs_f64() * 1.0e6
    };

    for _ in 0..warmup {
        let _ = run_resident();
        let _ = run_generic();
    }

    let resident_samples = (0..iterations).map(|_| run_resident()).collect();
    let generic_samples = (0..iterations).map(|_| run_generic()).collect();
    let (resident_median, resident_p95) = summarize(resident_samples);
    let (generic_median, generic_p95) = summarize(generic_samples);
    let resident_tokens_s = 1.0e6 / resident_median;
    let generic_tokens_s = 1.0e6 / generic_median;

    println!("device={:?} backend={:?}", info.name, info.backend);
    println!(
        "shape=batch1 q_heads={q_heads} kv_heads={kv_heads} kv_len={kv_len} capacity={capacity} head_dim={head_dim}"
    );
    println!("warmup={warmup} iterations={iterations}");
    println!(
        "resident_decode median_us={resident_median:.3} p95_us={resident_p95:.3} tokens_s={resident_tokens_s:.3}"
    );
    println!(
        "generic_q1 median_us={generic_median:.3} p95_us={generic_p95:.3} tokens_s={generic_tokens_s:.3}"
    );
    println!(
        "resident_over_generic_median_ratio={:.6}",
        resident_median / generic_median
    );
}
