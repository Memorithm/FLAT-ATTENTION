#![cfg(feature = "wgpu")]

use std::{cmp::Ordering, sync::mpsc, time::Instant};

use flat_attention::{
    forward_reference_grouped, FlatAttentionConfig, GroupedAttentionShape, GroupedForwardPass,
    WgpuGroupedForwardPipeline,
};

const WARMUP: usize = 6;
const ITERATIONS: usize = 30;
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

fn upload_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bytes: &[u8],
    label: &'static str,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

fn decode_mapped(mapped: &[u8]) -> Vec<f32> {
    mapped
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn readback_outside_timing(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    let bytes = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m27-io-parity-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m27-io-parity-readback"),
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
    let values = decode_mapped(&mapped);
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

struct CaseBuffers {
    q_bytes: Vec<u8>,
    k_bytes: Vec<u8>,
    v_bytes: Vec<u8>,
}

fn run_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &WgpuGroupedForwardPipeline,
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
) -> (f64, f64, f64, f64) {
    let q = fixture(shape.q_tensor_len().unwrap(), 0.1);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.7);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.3);
    let expected = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
    let host = CaseBuffers {
        q_bytes: bytes_f32(&q),
        k_bytes: bytes_f32(&k),
        v_bytes: bytes_f32(&v),
    };
    let layout = WgpuGroupedForwardPipeline::layout(shape).unwrap();

    let q_gpu = upload_bytes(device, queue, &host.q_bytes, "flat-m27-resident-q");
    let k_gpu = upload_bytes(device, queue, &host.k_bytes, "flat-m27-resident-k");
    let v_gpu = upload_bytes(device, queue, &host.v_bytes, "flat-m27-resident-v");
    let output_gpu = pipeline.create_output_buffer(device, shape).unwrap();

    let resident_execute = || {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m27-resident-only"),
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
        let _ = device.poll(wgpu::Maintain::Wait);
        start.elapsed().as_secs_f64() * 1.0e6
    };

    let _ = resident_execute();
    let resident_actual =
        readback_outside_timing(device, queue, &output_gpu, layout.output_elements);
    assert_close(
        "resident O",
        &resident_actual[layout.output_offset()..layout.lse_offset()],
        &expected.output,
    );
    assert_close(
        "resident LSE",
        &resident_actual[layout.lse_offset()..],
        &expected.lse,
    );

    let transfer_execute = || {
        let start = Instant::now();
        let q_gpu = upload_bytes(device, queue, &host.q_bytes, "flat-m27-transfer-q");
        let k_gpu = upload_bytes(device, queue, &host.k_bytes, "flat-m27-transfer-k");
        let v_gpu = upload_bytes(device, queue, &host.v_bytes, "flat-m27-transfer-v");
        let output_gpu = pipeline.create_output_buffer(device, shape).unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m27-transfer-forward"),
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

        let bytes = (layout.output_elements * std::mem::size_of::<f32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m27-transfer-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut readback = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m27-transfer-readback"),
        });
        readback.copy_buffer_to_buffer(&output_gpu, 0, &staging, 0, bytes);
        queue.submit(Some(readback.finish()));

        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();
        let elapsed_us = start.elapsed().as_secs_f64() * 1.0e6;
        let mapped = slice.get_mapped_range();
        let values = decode_mapped(&mapped);
        drop(mapped);
        staging.unmap();
        (elapsed_us, values)
    };

    let (_, transfer_actual) = transfer_execute();
    assert_close(
        "transfer-inclusive O",
        &transfer_actual[layout.output_offset()..layout.lse_offset()],
        &expected.output,
    );
    assert_close(
        "transfer-inclusive LSE",
        &transfer_actual[layout.lse_offset()..],
        &expected.lse,
    );

    for _ in 0..WARMUP {
        let _ = resident_execute();
        let _ = transfer_execute();
    }
    let resident_samples = (0..ITERATIONS).map(|_| resident_execute()).collect();
    let transfer_samples = (0..ITERATIONS).map(|_| transfer_execute().0).collect();
    let (resident_median, resident_p95) = summarize(resident_samples);
    let (transfer_median, transfer_p95) = summarize(transfer_samples);
    (resident_median, resident_p95, transfer_median, transfer_p95)
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
    .expect("M27 benchmark requires a WGPU adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m27-resident-vs-host-io"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .expect("M27 request_device failed");
    let pipeline = WgpuGroupedForwardPipeline::new(&device).expect("M27 pipeline creation failed");

    println!("benchmark=m27_resident_vs_host_io");
    println!("device_name={}", info.name);
    println!("backend={:?}", info.backend);
    println!("driver={}", info.driver);
    println!("driver_info={}", info.driver_info);
    println!("precision=f32");
    println!("warmup={WARMUP}");
    println!("iterations={ITERATIONS}");
    println!("resident_scope=encoder+public_encode+queue_submit+device_poll");
    println!("transfer_scope=input_buffer_create+queue_write+output_create+encode+submit+readback_copy+map+device_poll");
    println!("host_fixture_generation_in_timing=false");
    println!("host_f32_byte_packing_in_timing=false");
    println!("host_output_f32_decode_in_timing=false");
    println!("pipeline_creation_in_timing=false");
    println!("correctness_gate=both_scopes_match_scalar_grouped_forward_oracle_before_timing");
    println!("batch,q_heads,kv_heads,seq_len,head_dim,causal,resident_median_us,resident_p95_us,transfer_median_us,transfer_p95_us");

    for &(q_heads, kv_heads) in &[(4_usize, 4_usize), (4, 2), (4, 1)] {
        for &seq_len in &[32_usize, 128, 512] {
            for &causal in &[false, true] {
                let shape = GroupedAttentionShape {
                    batch: 1,
                    q_heads,
                    kv_heads,
                    seq_len,
                    head_dim: 64,
                };
                let config = FlatAttentionConfig {
                    causal,
                    softmax_scale: None,
                };
                let (resident_median, resident_p95, transfer_median, transfer_p95) =
                    run_case(&device, &queue, &pipeline, shape, config);
                println!(
                    "1,{q_heads},{kv_heads},{seq_len},64,{causal},{resident_median:.3},{resident_p95:.3},{transfer_median:.3},{transfer_p95:.3}"
                );
            }
        }
    }
    println!("performance_claim=none");
}
