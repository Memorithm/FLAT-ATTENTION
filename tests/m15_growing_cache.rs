#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, FlatAttentionConfig, ResidentDecodePass,
    WgpuResidentDecodePipeline, WgpuResidentKvCache,
};

const ATOL: f32 = 2.0e-4;
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
            panic!("M15 growing-cache test requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional M15 growing-cache test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m15-growing-cache-test"),
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
            let x = index as f32 * 0.023 + phase;
            x.sin() * 1.125 + (x * 0.29).cos() * 0.4375
        })
        .collect()
}

fn rotate_single_k_row(raw: &[f32], head_dim: usize, theta: f32, position: usize) -> Vec<f32> {
    let mut rotated = raw.to_vec();
    for pair in 0..head_dim / 2 {
        let dim = 2 * pair;
        let exponent = -2.0 * pair as f32 / head_dim as f32;
        let frequency = theta.powf(exponent);
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        let even = raw[dim];
        let odd = raw[dim + 1];
        rotated[dim] = even * cos - odd * sin;
        rotated[dim + 1] = even * sin + odd * cos;
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
        label: Some("flat-m15-growing-input"),
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
        label: Some("flat-m15-growing-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m15-growing-readback"),
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

#[test]
fn one_pipeline_tracks_token_by_token_cache_growth() {
    let Some(harness) = harness() else {
        return;
    };
    let (q_heads, kv_heads, capacity, head_dim) = (4usize, 1usize, 6usize, 32usize);
    let theta = 10_000.0;
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let pipeline = WgpuResidentDecodePipeline::new(&harness.device).unwrap();
    let mut cache =
        WgpuResidentKvCache::new(&harness.device, 1, kv_heads, capacity, head_dim).unwrap();
    let mut raw_k_prefix = Vec::new();
    let mut v_prefix = Vec::new();
    let mut output: Option<wgpu::Buffer> = None;

    for position in 0..capacity {
        let raw_k_row = fixture(head_dim, 0.4 + position as f32 * 0.13);
        let v_row = fixture(head_dim, 1.1 + position as f32 * 0.17);
        raw_k_prefix.extend_from_slice(&raw_k_row);
        v_prefix.extend_from_slice(&v_row);
        let rotated_k_row = rotate_single_k_row(&raw_k_row, head_dim, theta, position);
        let k_gpu = input_buffer(
            &harness.device,
            &harness.queue,
            &rotated_k_row,
            wgpu::BufferUsages::COPY_SRC,
        );
        let v_gpu = input_buffer(
            &harness.device,
            &harness.queue,
            &v_row,
            wgpu::BufferUsages::COPY_SRC,
        );
        let mut append_encoder =
            harness
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flat-m15-growing-append"),
                });
        cache
            .record_append(&mut append_encoder, &k_gpu, &v_gpu, 1)
            .unwrap();
        harness.queue.submit(Some(append_encoder.finish()));
        assert_eq!(cache.len(), position + 1);

        if output.is_none() {
            output = Some(
                pipeline
                    .create_output_buffer(&harness.device, &cache, q_heads)
                    .unwrap(),
            );
        }
        let output = output.as_ref().unwrap();
        let q = fixture(q_heads * head_dim, 2.0 + position as f32 * 0.19);
        let q_gpu = input_buffer(
            &harness.device,
            &harness.queue,
            &q,
            wgpu::BufferUsages::STORAGE,
        );
        let shape = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads,
            kv_heads,
            query_len: 1,
            kv_len: position + 1,
            head_dim,
            query_position_offset: position,
        };
        let rotary = AsymmetricRotaryEmbeddingConfig {
            theta,
            query_position_offset: position,
            kv_position_offset: 0,
        };
        let expected = forward_reference_projection_grouped_rope_asymmetric(
            &q,
            &raw_k_prefix,
            &v_prefix,
            shape,
            config,
            rotary,
        )
        .unwrap();

        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-m15-growing-decode"),
            });
        let layout = pipeline
            .encode(
                &harness.device,
                &mut encoder,
                ResidentDecodePass {
                    q: &q_gpu,
                    out_and_lse: output,
                    cache: &cache,
                    q_heads,
                    config,
                    theta,
                    q_rope_position: position,
                },
            )
            .unwrap();
        harness.queue.submit(Some(encoder.finish()));
        let actual = read_f32(
            &harness.device,
            &harness.queue,
            output,
            layout.combined_elements,
        );
        assert_close(
            "M15 growing O",
            &actual[..layout.output_elements],
            &expected.output,
        );
        assert_close(
            "M15 growing LSE",
            &actual[layout.output_elements..],
            &expected.lse,
        );
    }
}
