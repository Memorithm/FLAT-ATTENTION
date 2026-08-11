#![cfg(feature = "wgpu")]

use std::{cmp::Ordering, sync::mpsc, time::Instant};

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, FlatAttentionConfig, ResidentDecodePass,
    WgpuResidentDecodePipeline, WgpuResidentKvCache,
};

const WARMUP: usize = 8;
const ITERATIONS: usize = 40;
const ATOL: f32 = 7.5e-4;
const RTOL: f32 = 3.0e-3;
const THETA: f32 = 10_000.0;

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.031 + phase;
            x.sin() * 0.8 + (x * 0.41).cos() * 0.2
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

fn bytes_f32(values: &[f32]) -> Vec<u8> {
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
    label: &'static str,
) -> wgpu::Buffer {
    let bytes = bytes_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn read_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    let bytes = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m27-decode-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m27-decode-readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..bytes);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let values = mapped
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    drop(mapped);
    staging.unmap();
    values
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
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

fn run_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &WgpuResidentDecodePipeline,
    q_heads: usize,
    kv_heads: usize,
    kv_len: usize,
    head_dim: usize,
) -> (f64, f64) {
    let q = fixture(q_heads * head_dim, 0.2);
    let raw_k = fixture(kv_len * kv_heads * head_dim, 0.8);
    let v = fixture(kv_len * kv_heads * head_dim, 1.4);
    let rotated_k = rotate_k_projection(&raw_k, kv_len, kv_heads, head_dim, THETA);
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads,
        kv_heads,
        query_len: 1,
        kv_len,
        head_dim,
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
    let expected = forward_reference_projection_grouped_rope_asymmetric(
        &q, &raw_k, &v, shape, config, rotary,
    )
    .unwrap();

    let q_gpu = input_buffer(
        device,
        queue,
        &q,
        wgpu::BufferUsages::STORAGE,
        "flat-m27-decode-q",
    );
    let k_gpu = input_buffer(
        device,
        queue,
        &rotated_k,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "flat-m27-decode-k",
    );
    let v_gpu = input_buffer(
        device,
        queue,
        &v,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "flat-m27-decode-v",
    );

    let mut cache = WgpuResidentKvCache::new(device, 1, kv_heads, kv_len, head_dim).unwrap();
    let mut append_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m27-decode-cache-append"),
    });
    cache
        .record_append(&mut append_encoder, &k_gpu, &v_gpu, kv_len)
        .unwrap();
    queue.submit(Some(append_encoder.finish()));
    let _ = device.poll(wgpu::Maintain::Wait);

    let output_gpu = pipeline
        .create_output_buffer(device, &cache, q_heads)
        .unwrap();
    let layout = WgpuResidentDecodePipeline::layout(&cache, q_heads).unwrap();

    let execute = || {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m27-resident-decode"),
        });
        pipeline
            .encode(
                device,
                &mut encoder,
                ResidentDecodePass {
                    q: &q_gpu,
                    out_and_lse: &output_gpu,
                    cache: &cache,
                    q_heads,
                    config,
                    theta: THETA,
                    q_rope_position: kv_len - 1,
                },
            )
            .unwrap();
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::Maintain::Wait);
        start.elapsed().as_secs_f64() * 1.0e6
    };

    let _ = execute();
    let actual = read_f32(device, queue, &output_gpu, layout.combined_elements);
    assert_close(
        "O",
        &actual[..layout.output_elements],
        &expected.output,
    );
    assert_close(
        "LSE",
        &actual[layout.output_elements..],
        &expected.lse,
    );

    for _ in 0..WARMUP {
        let _ = execute();
    }
    let samples = (0..ITERATIONS).map(|_| execute()).collect();
    summarize(samples)
}

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("M27 decode benchmark requires a WGPU adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m27-resident-decode-sweep"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .expect("M27 decode request_device failed");
    let pipeline =
        WgpuResidentDecodePipeline::new(&device).expect("M27 decode pipeline creation failed");

    println!("benchmark=m27_resident_decode_sweep");
    println!("device_name={}", info.name);
    println!("backend={:?}", info.backend);
    println!("driver={}", info.driver);
    println!("driver_info={}", info.driver_info);
    println!("precision=f32");
    println!("warmup={WARMUP}");
    println!("iterations={ITERATIONS}");
    println!("timing_scope=command_encoder+resident_decode_encode+queue_submit+device_poll");
    println!("uploads_in_timing=false");
    println!("cache_append_in_timing=false");
    println!("readback_in_timing=false");
    println!("correctness_gate=scalar_projection_rope_asymmetric_oracle_before_timing");
    println!("batch,q_heads,kv_heads,kv_len,head_dim,causal,median_us,p95_us,decode_tokens_per_s");

    for &(q_heads, kv_heads) in &[(4_usize, 4_usize), (4, 2), (4, 1)] {
        for &kv_len in &[32_usize, 128, 512, 2048] {
            for &head_dim in &[32_usize, 64, 80, 96, 128] {
                let (median_us, p95_us) =
                    run_case(&device, &queue, &pipeline, q_heads, kv_heads, kv_len, head_dim);
                let decode_tokens_per_s = 1.0e6 / median_us;
                println!(
                    "1,{q_heads},{kv_heads},{kv_len},{head_dim},true,{median_us:.3},{p95_us:.3},{decode_tokens_per_s:.3}"
                );
            }
        }
    }
    println!("performance_claim=none");
}
