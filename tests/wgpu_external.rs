#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_projection_grouped_rope, ExternalProjectionPass,
    ExternalProjectionRotaryGroupedPipeline, ExternalWgpuError, FlatAttentionConfig,
    GroupedAttentionShape, RotaryEmbeddingConfig,
};

const ATOL: f32 = 1.5e-4;
const RTOL: f32 = 1.0e-3;

struct DeviceHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
}

fn harness() -> Option<DeviceHarness> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }));
    let Ok(adapter) = adapter else {
        if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
            panic!("FLAT-R2 requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional FLAT-R2 test skipped");
        return None;
    };
    let adapter_name = adapter.get_info().name;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-r2-external-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("FLAT-R2 request_device failed: {error}"));
    Some(DeviceHarness {
        device,
        queue,
        adapter_name,
    })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i as f32 * 0.019 + phase;
            x.sin() * 2.125 + (x * 0.47).cos() * 0.34375
        })
        .collect()
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
        label: Some("flat-r2-test-input"),
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
        label: Some("flat-r2-test-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-r2-test-readback"),
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
        assert!(actual.is_finite(), "{name}[{index}] is not finite");
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

#[test]
fn caller_can_encode_two_flat_passes_then_submit_once() {
    let Some(harness) = harness() else {
        return;
    };
    eprintln!("FLAT-R2 adapter: {}", harness.adapter_name);
    let pipeline = ExternalProjectionRotaryGroupedPipeline::new(&harness.device).unwrap();
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 8,
        kv_heads: 2,
        seq_len: 17,
        head_dim: 64,
    };
    let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
    let q_gpu = input_buffer(&harness.device, &harness.queue, &q);
    let k_gpu = input_buffer(&harness.device, &harness.queue, &k);
    let v_gpu = input_buffer(&harness.device, &harness.queue, &v);
    let out_causal = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let out_full = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let rotary = RotaryEmbeddingConfig {
        theta: 10_000.0,
        position_offset: 13,
    };
    let causal = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let full = FlatAttentionConfig {
        causal: false,
        softmax_scale: Some(0.125),
    };

    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-r2-caller-owned-encoder"),
        });
    let layout = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            ExternalProjectionPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                out_and_lse: &out_causal,
                shape,
                config: causal,
                rotary,
            },
        )
        .unwrap();
    pipeline
        .encode(
            &harness.device,
            &mut encoder,
            ExternalProjectionPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                out_and_lse: &out_full,
                shape,
                config: full,
                rotary,
            },
        )
        .unwrap();

    harness.queue.submit(Some(encoder.finish()));

    let causal_values = read_f32(
        &harness.device,
        &harness.queue,
        &out_causal,
        layout.combined_elements,
    );
    let full_values = read_f32(
        &harness.device,
        &harness.queue,
        &out_full,
        layout.combined_elements,
    );
    let causal_expected =
        forward_reference_projection_grouped_rope(&q, &k, &v, shape, causal, rotary).unwrap();
    let full_expected =
        forward_reference_projection_grouped_rope(&q, &k, &v, shape, full, rotary).unwrap();
    assert_close(
        "R2 causal O",
        &causal_values[..layout.output_elements],
        &causal_expected.output,
    );
    assert_close(
        "R2 causal LSE",
        &causal_values[layout.output_elements..],
        &causal_expected.lse,
    );
    assert_close(
        "R2 full O",
        &full_values[..layout.output_elements],
        &full_expected.output,
    );
    assert_close(
        "R2 full LSE",
        &full_values[layout.output_elements..],
        &full_expected.lse,
    );
}

#[test]
fn external_pipeline_handles_mqa_and_rejects_short_buffers() {
    let Some(harness) = harness() else {
        return;
    };
    let pipeline = ExternalProjectionRotaryGroupedPipeline::new(&harness.device).unwrap();
    let shape = GroupedAttentionShape {
        batch: 2,
        q_heads: 8,
        kv_heads: 1,
        seq_len: 9,
        head_dim: 80,
    };
    let q = fixture(shape.q_tensor_len().unwrap(), 0.4);
    let k = fixture(shape.kv_tensor_len().unwrap(), 1.0);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.6);
    let q_gpu = input_buffer(&harness.device, &harness.queue, &q);
    let k_gpu = input_buffer(&harness.device, &harness.queue, &k);
    let v_gpu = input_buffer(&harness.device, &harness.queue, &v);
    let output = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let rotary = RotaryEmbeddingConfig {
        theta: 500_000.0,
        position_offset: 21,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-r2-mqa"),
        });
    let layout = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            ExternalProjectionPass {
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
    let expected =
        forward_reference_projection_grouped_rope(&q, &k, &v, shape, config, rotary).unwrap();
    assert_close(
        "R2 MQA O",
        &values[..layout.output_elements],
        &expected.output,
    );
    assert_close(
        "R2 MQA LSE",
        &values[layout.output_elements..],
        &expected.lse,
    );

    let short = harness.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-r2-short-buffer"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-r2-short-buffer-check"),
        });
    let error = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            ExternalProjectionPass {
                q: &short,
                k: &k_gpu,
                v: &v_gpu,
                out_and_lse: &output,
                shape,
                config,
                rotary,
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ExternalWgpuError::BufferTooSmall { tensor: "Q", .. }
    ));
}
