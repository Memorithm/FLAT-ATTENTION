#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, ExternalAsymmetricProjectionPass,
    ExternalAsymmetricProjectionRotaryGroupedPipeline, FlatAttentionConfig,
};

const ATOL: f32 = 1.5e-4;
const RTOL: f32 = 1.0e-3;

struct DeviceHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn harness() -> Option<DeviceHarness> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
    }));
    let Some(adapter) = adapter else {
        if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
            panic!("M15 requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M15 device test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m15-prerotated-k-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M15 request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.021 + phase;
            x.sin() * 1.75 + (x * 0.43).cos() * 0.34375
        })
        .collect()
}

fn rotate_k_projection(
    raw: &[f32],
    kv_len: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Vec<f32> {
    let mut rotated = raw.to_vec();
    let width = kv_heads * head_dim;
    for position in 0..kv_len {
        let absolute_position = position_offset + position;
        for head in 0..kv_heads {
            let head_base = position * width + head * head_dim;
            for pair in 0..head_dim / 2 {
                let dim = 2 * pair;
                let exponent = -2.0 * pair as f32 / head_dim as f32;
                let frequency = theta.powf(exponent);
                let angle = absolute_position as f32 * frequency;
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

fn input_buffer(device: &wgpu::Device, queue: &wgpu::Queue, values: &[f32]) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m15-input"),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buffer, 0, &bytes);
    }
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
        label: Some("flat-m15-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m15-readback"),
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
    let mut values = Vec::with_capacity(len);
    for chunk in mapped.chunks_exact(4) {
        values.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
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

fn run_decode_case(q_heads: usize, kv_heads: usize, kv_len: usize, head_dim: usize, theta: f32) {
    let Some(harness) = harness() else {
        return;
    };
    let query_position = kv_len - 1;
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads,
        kv_heads,
        query_len: 1,
        kv_len,
        head_dim,
        query_position_offset: query_position,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let rotary = AsymmetricRotaryEmbeddingConfig {
        theta,
        query_position_offset: query_position,
        kv_position_offset: 0,
    };
    let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
    let raw_k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
    let rotated_k = rotate_k_projection(&raw_k, kv_len, kv_heads, head_dim, theta, 0);
    let expected =
        forward_reference_projection_grouped_rope_asymmetric(&q, &raw_k, &v, shape, config, rotary)
            .unwrap();

    let pipeline = ExternalAsymmetricProjectionRotaryGroupedPipeline::new(&harness.device).unwrap();
    let q_gpu = input_buffer(&harness.device, &harness.queue, &q);
    let k_gpu = input_buffer(&harness.device, &harness.queue, &rotated_k);
    let v_gpu = input_buffer(&harness.device, &harness.queue, &v);
    let output = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m15-prerotated-k-caller"),
        });
    let layout = pipeline
        .encode_pre_rotated_k(
            &harness.device,
            &mut encoder,
            ExternalAsymmetricProjectionPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                out_and_lse: &output,
                shape,
                config,
                rotary,
            },
        )
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));
    let values = read_f32(
        &harness.device,
        &harness.queue,
        &output,
        layout.combined_elements,
    );
    assert_close("M15 O", &values[..layout.output_elements], &expected.output);
    assert_close("M15 LSE", &values[layout.output_elements..], &expected.lse);
}

#[test]
fn prerotated_k_gqa_decode_matches_raw_k_m11_oracle() {
    run_decode_case(8, 2, 17, 64, 10_000.0);
}

#[test]
fn prerotated_k_mqa_decode_crosses_tile_boundary() {
    run_decode_case(8, 1, 65, 80, 500_000.0);
}
