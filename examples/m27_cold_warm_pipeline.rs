#![cfg(feature = "wgpu")]

use std::{cmp::Ordering, sync::mpsc, time::Instant};

use flat_attention::{
    forward_reference_grouped, FlatAttentionConfig, GroupedAttentionShape, GroupedForwardPass,
    WgpuGroupedForwardPipeline,
};

const WARMUP: usize = 8;
const ITERATIONS: usize = 40;
const PIPELINE_RECREATION_ITERATIONS: usize = 20;
const ATOL: f32 = 6.0e-4;
const RTOL: f32 = 2.5e-3;

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.043 + phase;
            x.sin() * 0.7 + (x * 0.37).cos() * 0.3
        })
        .collect()
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
    label: &'static str,
) -> wgpu::Buffer {
    let bytes = bytes_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        label: Some("flat-m27-cold-warm-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m27-cold-warm-readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..bytes);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range().expect("valid mapped range");
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

fn run_warm_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &WgpuGroupedForwardPipeline,
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
) -> (f64, f64) {
    let q = fixture(shape.q_tensor_len().unwrap(), 0.1);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.7);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.3);
    let expected = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();

    let q_gpu = input_buffer(device, queue, &q, "flat-m27-cold-warm-q");
    let k_gpu = input_buffer(device, queue, &k, "flat-m27-cold-warm-k");
    let v_gpu = input_buffer(device, queue, &v, "flat-m27-cold-warm-v");
    let output_gpu = pipeline.create_output_buffer(device, shape).unwrap();
    let layout = WgpuGroupedForwardPipeline::layout(shape).unwrap();

    let execute = || {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m27-cold-warm-resident-forward"),
        });
        pipeline
            .encode(
                device,
                &mut encoder,
                GroupedForwardPass {
                    q: &q_gpu,
                    k: &k_gpu,
                    v: &v_gpu,
                    output: &output_gpu,
                    shape,
                    config,
                },
            )
            .unwrap();
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        start.elapsed().as_secs_f64() * 1.0e6
    };

    let _ = execute();
    let actual = read_f32(device, queue, &output_gpu, layout.output_elements);
    assert_close(
        "O",
        &actual[layout.output_offset()..layout.lse_offset()],
        &expected.output,
    );
    assert_close("LSE", &actual[layout.lse_offset()..], &expected.lse);

    for _ in 0..WARMUP {
        let _ = execute();
    }
    let samples = (0..ITERATIONS).map(|_| execute()).collect();
    summarize(samples)
}

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("M27 cold/warm benchmark requires a WGPU adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-m27-cold-warm-pipeline"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .expect("M27 cold/warm request_device failed");

    let initial_creation_start = Instant::now();
    let pipeline = WgpuGroupedForwardPipeline::new(&device)
        .expect("M27 initial grouped-forward pipeline creation failed");
    let initial_pipeline_creation_us = initial_creation_start.elapsed().as_secs_f64() * 1.0e6;

    let recreated_samples = (0..PIPELINE_RECREATION_ITERATIONS)
        .map(|_| {
            let start = Instant::now();
            let recreated = WgpuGroupedForwardPipeline::new(&device)
                .expect("M27 grouped-forward pipeline recreation failed");
            let elapsed_us = start.elapsed().as_secs_f64() * 1.0e6;
            drop(recreated);
            elapsed_us
        })
        .collect();
    let (recreated_median_us, recreated_p95_us) = summarize(recreated_samples);

    println!("benchmark=m27_cold_warm_pipeline");
    println!("device_name={}", info.name);
    println!("backend={:?}", info.backend);
    println!("driver={}", info.driver);
    println!("driver_info={}", info.driver_info);
    println!("precision=f32");
    println!("warmup={WARMUP}");
    println!("iterations={ITERATIONS}");
    println!("pipeline_recreation_iterations={PIPELINE_RECREATION_ITERATIONS}");
    println!("device_creation_in_cold_timing=false");
    println!("cold_definition=first_WgpuGroupedForwardPipeline_new_after_device_creation");
    println!("recreated_definition=subsequent_WgpuGroupedForwardPipeline_new_same_process_device");
    println!("warm_definition=reuse_initial_pipeline_for_resident_encode_submit_poll");
    println!("uploads_in_warm_timing=false");
    println!("readback_in_warm_timing=false");
    println!("initial_pipeline_creation_us={initial_pipeline_creation_us:.3}");
    println!("recreated_pipeline_creation_median_us={recreated_median_us:.3}");
    println!("recreated_pipeline_creation_p95_us={recreated_p95_us:.3}");
    println!("correctness_gate=scalar_grouped_forward_oracle_before_warm_timing");
    println!("batch,q_heads,kv_heads,seq_len,head_dim,causal,warm_median_us,warm_p95_us,logical_query_tokens_per_s");

    for &(q_heads, kv_heads) in &[(4_usize, 4_usize), (4, 2), (4, 1)] {
        for &causal in &[false, true] {
            let shape = GroupedAttentionShape {
                batch: 1,
                q_heads,
                kv_heads,
                seq_len: 128,
                head_dim: 64,
            };
            let config = FlatAttentionConfig {
                causal,
                softmax_scale: None,
            };
            let (median_us, p95_us) = run_warm_case(&device, &queue, &pipeline, shape, config);
            let logical_query_tokens_per_s = shape.seq_len as f64 * 1.0e6 / median_us;
            println!(
                "1,{q_heads},{kv_heads},128,64,{causal},{median_us:.3},{p95_us:.3},{logical_query_tokens_per_s:.3}"
            );
        }
    }

    println!("performance_claim=none");
}
