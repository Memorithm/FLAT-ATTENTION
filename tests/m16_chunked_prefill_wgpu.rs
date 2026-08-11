#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    chunked_projection_prefill::{
        WgpuChunkedProjectionPrefillPass, WgpuChunkedProjectionPrefillPipeline,
    },
    forward_reference_projection_grouped_rope, FlatAttentionConfig, GroupedAttentionShape,
    RotaryEmbeddingConfig,
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
            panic!("M16 chunked prefill requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M16 chunked-prefill test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m16-chunked-prefill-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M16 request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.019 + phase;
            x.sin() * 1.1875 + (x * 0.37).cos() * 0.34375
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

fn input_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m16-chunked-prefill-input"),
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
        label: Some("flat-m16-chunked-prefill-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m16-chunked-prefill-readback"),
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

fn run_case(
    harness: &DeviceHarness,
    shape: GroupedAttentionShape,
    causal: bool,
    chunk_size: usize,
) {
    let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
    let config = FlatAttentionConfig {
        causal,
        softmax_scale: None,
    };
    let rotary = RotaryEmbeddingConfig {
        theta: 10_000.0,
        position_offset: 5,
    };
    let expected =
        forward_reference_projection_grouped_rope(&q, &k, &v, shape, config, rotary).unwrap();

    let q_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &q,
        wgpu::BufferUsages::COPY_SRC,
    );
    let k_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &k,
        wgpu::BufferUsages::STORAGE,
    );
    let v_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &v,
        wgpu::BufferUsages::STORAGE,
    );

    let pipeline = WgpuChunkedProjectionPrefillPipeline::new(&harness.device).unwrap();
    let output = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-chunked-prefill"),
        });
    let layout = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            WgpuChunkedProjectionPrefillPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                out_and_lse: &output,
                shape,
                config,
                rotary,
                query_chunk_size: chunk_size,
            },
        )
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));

    let actual = read_f32(
        &harness.device,
        &harness.queue,
        &output,
        layout.combined_elements,
    );
    assert_close(
        "M16 chunked O",
        &actual[..layout.output_elements],
        &expected.output,
    );
    assert_close(
        "M16 chunked LSE",
        &actual[layout.output_elements..],
        &expected.lse,
    );
}

#[test]
fn chunked_prefill_matches_contiguous_oracle_for_gqa_and_mqa() {
    let Some(harness) = harness() else {
        return;
    };

    run_case(
        &harness,
        GroupedAttentionShape {
            batch: 2,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 7,
            head_dim: 32,
        },
        true,
        3,
    );
    run_case(
        &harness,
        GroupedAttentionShape {
            batch: 2,
            q_heads: 4,
            kv_heads: 1,
            seq_len: 5,
            head_dim: 32,
        },
        false,
        2,
    );
}

#[test]
fn zero_chunk_is_rejected_before_recording() {
    let Some(harness) = harness() else {
        return;
    };
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 2,
        kv_heads: 1,
        seq_len: 2,
        head_dim: 8,
    };
    let q = input_buffer(
        &harness.device,
        &harness.queue,
        &fixture(shape.q_tensor_len().unwrap(), 0.1),
        wgpu::BufferUsages::COPY_SRC,
    );
    let k = input_buffer(
        &harness.device,
        &harness.queue,
        &fixture(shape.kv_tensor_len().unwrap(), 0.2),
        wgpu::BufferUsages::STORAGE,
    );
    let v = input_buffer(
        &harness.device,
        &harness.queue,
        &fixture(shape.kv_tensor_len().unwrap(), 0.3),
        wgpu::BufferUsages::STORAGE,
    );
    let pipeline = WgpuChunkedProjectionPrefillPipeline::new(&harness.device).unwrap();
    let output = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-zero-chunk"),
        });
    let error = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            WgpuChunkedProjectionPrefillPass {
                q: &q,
                k: &k,
                v: &v,
                out_and_lse: &output,
                shape,
                config: FlatAttentionConfig::default(),
                rotary: RotaryEmbeddingConfig {
                    theta: 10_000.0,
                    position_offset: 0,
                },
                query_chunk_size: 0,
            },
        )
        .expect_err("zero chunk must fail");
    assert!(matches!(
        error,
        flat_attention::chunked_projection_prefill::WgpuChunkedProjectionPrefillError::ZeroQueryChunkSize
    ));
}
