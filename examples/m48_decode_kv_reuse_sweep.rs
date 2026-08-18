#![cfg(feature = "wgpu")]

use std::{cmp::Ordering, sync::mpsc, time::Instant};

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, ExternalAsymmetricProjectionPass,
    ExternalAsymmetricProjectionRotaryGroupedPipeline, FlatAttentionConfig,
};

const Q_HEADS: usize = 16;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const THETA: f32 = 10_000.0;
const ATOL: f32 = 1.5e-4;
const RTOL: f32 = 1.0e-3;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.021 + phase;
            x.sin() * 1.75 + (x * 0.43).cos() * 0.34375
        })
        .collect()
}

fn rotate_k(raw: &[f32], kv_len: usize) -> Vec<f32> {
    let mut rotated = raw.to_vec();
    let width = KV_HEADS * HEAD_DIM;
    for position in 0..kv_len {
        for head in 0..KV_HEADS {
            let base = position * width + head * HEAD_DIM;
            for pair in 0..HEAD_DIM / 2 {
                let dim = 2 * pair;
                let frequency = THETA.powf(-2.0 * pair as f32 / HEAD_DIM as f32);
                let (sine, cosine) = (position as f32 * frequency).sin_cos();
                let even = raw[base + dim];
                let odd = raw[base + dim + 1];
                rotated[base + dim] = even * cosine - odd * sine;
                rotated[base + dim + 1] = even * sine + odd * cosine;
            }
        }
    }
    rotated
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn buffer(device: &wgpu::Device, queue: &wgpu::Queue, values: &[f32]) -> wgpu::Buffer {
    let contents = bytes(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m48-sweep-input"),
        size: contents.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &contents);
    buffer
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let byte_len = (len * size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m48-sweep-readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..byte_len);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let values = mapped
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    drop(mapped);
    staging.unmap();
    values
}

fn max_error(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| {
            let error = (actual - expected).abs();
            let tolerance = ATOL + RTOL * expected.abs();
            assert!(
                error <= tolerance,
                "parity failure actual={actual} expected={expected} error={error} tolerance={tolerance}"
            );
            error
        })
        .fold(0.0, f32::max)
}

fn summarize(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let median = if samples.len() % 2 == 0 {
        let upper = samples.len() / 2;
        (samples[upper - 1] + samples[upper]) * 0.5
    } else {
        samples[samples.len() / 2]
    };
    let p95 = samples[(samples.len() * 95).div_ceil(100).saturating_sub(1)];
    (median, p95)
}

fn pass<'a>(
    q: &'a wgpu::Buffer,
    k: &'a wgpu::Buffer,
    v: &'a wgpu::Buffer,
    output: &'a wgpu::Buffer,
    shape: AsymmetricGroupedAttentionShape,
    config: FlatAttentionConfig,
    rotary: AsymmetricRotaryEmbeddingConfig,
) -> ExternalAsymmetricProjectionPass<'a> {
    ExternalAsymmetricProjectionPass {
        q,
        k,
        v,
        out_and_lse: output,
        shape,
        config,
        rotary,
    }
}

fn run_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &ExternalAsymmetricProjectionRotaryGroupedPipeline,
    kv_len: usize,
    warmups: usize,
    repeats: usize,
) -> (f64, f64, f64, f64, f32, f32) {
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: Q_HEADS,
        kv_heads: KV_HEADS,
        query_len: 1,
        kv_len,
        head_dim: HEAD_DIM,
        query_position_offset: kv_len - 1,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let rotary = AsymmetricRotaryEmbeddingConfig {
        theta: THETA,
        query_position_offset: kv_len - 1,
        kv_position_offset: 0,
    };
    let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
    let raw_k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
    let k = rotate_k(&raw_k, kv_len);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
    let expected =
        forward_reference_projection_grouped_rope_asymmetric(&q, &raw_k, &v, shape, config, rotary)
            .unwrap();
    let q_gpu = buffer(device, queue, &q);
    let k_gpu = buffer(device, queue, &k);
    let v_gpu = buffer(device, queue, &v);
    let baseline = pipeline.create_output_buffer(device, shape).unwrap();
    let candidate = pipeline.create_output_buffer(device, shape).unwrap();
    let layout = ExternalAsymmetricProjectionRotaryGroupedPipeline::layout(shape).unwrap();

    let execute = |candidate_route: bool| {
        let started = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m48-sweep"),
        });
        if candidate_route {
            pipeline
                .encode_pre_rotated_k_decode_kv_reuse(
                    device,
                    &mut encoder,
                    pass(&q_gpu, &k_gpu, &v_gpu, &candidate, shape, config, rotary),
                )
                .unwrap();
        } else {
            pipeline
                .encode_pre_rotated_k(
                    device,
                    &mut encoder,
                    pass(&q_gpu, &k_gpu, &v_gpu, &baseline, shape, config, rotary),
                )
                .unwrap();
        }
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::Maintain::Wait);
        started.elapsed().as_secs_f64() * 1.0e6
    };

    let _ = execute(false);
    let _ = execute(true);
    let baseline_values = read(device, queue, &baseline, layout.combined_elements);
    let candidate_values = read(device, queue, &candidate, layout.combined_elements);
    let baseline_error = max_error(&baseline_values[..layout.output_elements], &expected.output)
        .max(max_error(
            &baseline_values[layout.output_elements..],
            &expected.lse,
        ));
    let candidate_error = max_error(
        &candidate_values[..layout.output_elements],
        &expected.output,
    )
    .max(max_error(
        &candidate_values[layout.output_elements..],
        &expected.lse,
    ));

    for iteration in 0..warmups {
        if iteration.is_multiple_of(2) {
            let _ = execute(false);
            let _ = execute(true);
        } else {
            let _ = execute(true);
            let _ = execute(false);
        }
    }

    let mut baseline_samples = Vec::with_capacity(repeats);
    let mut candidate_samples = Vec::with_capacity(repeats);
    for iteration in 0..repeats {
        if iteration.is_multiple_of(2) {
            baseline_samples.push(execute(false));
            candidate_samples.push(execute(true));
        } else {
            candidate_samples.push(execute(true));
            baseline_samples.push(execute(false));
        }
    }

    let (baseline_median, baseline_p95) = summarize(baseline_samples);
    let (candidate_median, candidate_p95) = summarize(candidate_samples);
    (
        baseline_median,
        baseline_p95,
        candidate_median,
        candidate_p95,
        baseline_error,
        candidate_error,
    )
}

fn main() {
    let warmups = env_usize("FLAT_M48_WARMUPS", 5).max(1);
    let repeats = env_usize("FLAT_M48_REPEATS", 20).max(3);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("M48 sweep requires a WGPU adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m48-sweep"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .expect("M48 request_device failed");
    let pipeline =
        ExternalAsymmetricProjectionRotaryGroupedPipeline::with_decode_kv_reuse(&device, true)
            .expect("M48 pipeline creation failed");

    println!("benchmark=m48_decode_kv_reuse_sweep");
    println!("adapter={}", info.name);
    println!("backend={:?}", info.backend);
    println!("q_heads={Q_HEADS} kv_heads={KV_HEADS} head_dim={HEAD_DIM} query_len=1");
    println!("pre_rotated_k=true same_wgpu_context=true resident_buffers=true");
    println!("uploads_in_timing=false readback_in_timing=false");
    println!("correctness_gate=scalar_oracle_before_timing");
    println!("measurement_order=alternating_m15_m48");
    println!("timing_scope=encode+queue_submit+device_poll");
    println!("kv_len,m15_median_us,m15_p95_us,m48_median_us,m48_p95_us,m15_over_m48,m15_max_abs,m48_max_abs");
    for kv_len in [128, 192, 256, 512, 1024, 2048, 4096, 8192] {
        let (m15, m15_p95, m48, m48_p95, m15_error, m48_error) =
            run_case(&device, &queue, &pipeline, kv_len, warmups, repeats);
        println!(
            "{kv_len},{m15:.3},{m15_p95:.3},{m48:.3},{m48_p95:.3},{:.6},{m15_error:.9},{m48_error:.9}",
            m15 / m48
        );
    }
    println!("performance_claim=none");
}
