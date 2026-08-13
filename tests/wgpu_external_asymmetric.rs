#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, ExternalAsymmetricKernelVariant,
    ExternalAsymmetricProjectionPass, ExternalAsymmetricProjectionRotaryGroupedPipeline,
    ExternalWgpuError, FlatAttentionConfig,
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
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
    }));
    let Some(adapter) = adapter else {
        if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
            panic!("M11 requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M11 device test skipped");
        return None;
    };
    let adapter_name = adapter.get_info().name;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m11-asymmetric-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M11 request_device failed: {error}"));
    Some(DeviceHarness {
        device,
        queue,
        adapter_name,
    })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i as f32 * 0.023 + phase;
            x.sin() * 1.875 + (x * 0.41).cos() * 0.28125
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
        label: Some("flat-m11-input"),
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
        label: Some("flat-m11-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m11-readback"),
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
        assert!(actual.is_finite(), "{name}[{index}] is not finite");
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
    shape: AsymmetricGroupedAttentionShape,
    config: FlatAttentionConfig,
    rotary: AsymmetricRotaryEmbeddingConfig,
    phase: f32,
    vectorized: bool,
) {
    let pipeline = ExternalAsymmetricProjectionRotaryGroupedPipeline::with_vectorization(
        &harness.device,
        vectorized,
    )
    .unwrap();
    assert_eq!(
        pipeline.kernel_variant_for_shape(shape),
        if vectorized && matches!(shape.head_dim, 64 | 128) {
            ExternalAsymmetricKernelVariant::Vec4
        } else {
            ExternalAsymmetricKernelVariant::Portable
        }
    );
    let q = fixture(shape.q_tensor_len().unwrap(), phase);
    let k = fixture(shape.kv_tensor_len().unwrap(), phase + 0.6);
    let v = fixture(shape.kv_tensor_len().unwrap(), phase + 1.2);
    let q_gpu = input_buffer(&harness.device, &harness.queue, &q);
    let k_gpu = input_buffer(&harness.device, &harness.queue, &k);
    let v_gpu = input_buffer(&harness.device, &harness.queue, &v);
    let output = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m11-caller-encoder"),
        });
    let layout = pipeline
        .encode(
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
    let expected =
        forward_reference_projection_grouped_rope_asymmetric(&q, &k, &v, shape, config, rotary)
            .unwrap();
    assert_close("M11 O", &values[..layout.output_elements], &expected.output);
    assert_close("M11 LSE", &values[layout.output_elements..], &expected.lse);
}

#[test]
fn single_query_decode_reads_long_resident_gqa_cache() {
    let Some(harness) = harness() else {
        return;
    };
    eprintln!("M11 adapter: {}", harness.adapter_name);
    run_case(
        &harness,
        AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 8,
            kv_heads: 2,
            query_len: 1,
            kv_len: 17,
            head_dim: 64,
            query_position_offset: 16,
        },
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        },
        AsymmetricRotaryEmbeddingConfig {
            theta: 10_000.0,
            query_position_offset: 29,
            kv_position_offset: 13,
        },
        0.2,
        false,
    );
}

#[test]
fn rectangular_noncausal_mqa_matches_oracle() {
    let Some(harness) = harness() else {
        return;
    };
    run_case(
        &harness,
        AsymmetricGroupedAttentionShape {
            batch: 2,
            q_heads: 8,
            kv_heads: 1,
            query_len: 5,
            kv_len: 9,
            head_dim: 80,
            query_position_offset: 0,
        },
        FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(0.125),
        },
        AsymmetricRotaryEmbeddingConfig {
            theta: 500_000.0,
            query_position_offset: 3,
            kv_position_offset: 101,
        },
        0.4,
        false,
    );
}

#[test]
fn vec4_gqa_prefill_matches_oracle_for_product_shapes() {
    let Some(harness) = harness() else {
        return;
    };
    for (head_dim, causal) in [(64, false), (64, true), (128, false), (128, true)] {
        run_case(
            &harness,
            AsymmetricGroupedAttentionShape {
                batch: 1,
                q_heads: 8,
                kv_heads: 2,
                query_len: 17,
                kv_len: 17,
                head_dim,
                query_position_offset: 0,
            },
            FlatAttentionConfig {
                causal,
                softmax_scale: None,
            },
            AsymmetricRotaryEmbeddingConfig {
                theta: 10_000.0,
                query_position_offset: 0,
                kv_position_offset: 0,
            },
            0.7 + head_dim as f32 * 0.001,
            true,
        );
    }
}

#[test]
fn vectorized_pipeline_preserves_portable_fallback() {
    let Some(harness) = harness() else {
        return;
    };
    run_case(
        &harness,
        AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            query_len: 3,
            kv_len: 7,
            head_dim: 80,
            query_position_offset: 4,
        },
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        },
        AsymmetricRotaryEmbeddingConfig {
            theta: 10_000.0,
            query_position_offset: 4,
            kv_position_offset: 0,
        },
        1.1,
        true,
    );
}

#[test]
fn rectangular_pipeline_rejects_short_kv_buffer() {
    let Some(harness) = harness() else {
        return;
    };
    let pipeline = ExternalAsymmetricProjectionRotaryGroupedPipeline::new(&harness.device).unwrap();
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        query_len: 1,
        kv_len: 8,
        head_dim: 32,
        query_position_offset: 7,
    };
    let q = fixture(shape.q_tensor_len().unwrap(), 0.3);
    let kv = fixture(shape.kv_tensor_len().unwrap(), 0.9);
    let q_gpu = input_buffer(&harness.device, &harness.queue, &q);
    let v_gpu = input_buffer(&harness.device, &harness.queue, &kv);
    let short = harness.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m11-short-k"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m11-short-check"),
        });
    let error = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            ExternalAsymmetricProjectionPass {
                q: &q_gpu,
                k: &short,
                v: &v_gpu,
                out_and_lse: &output,
                shape,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                rotary: AsymmetricRotaryEmbeddingConfig {
                    theta: 10_000.0,
                    query_position_offset: 7,
                    kv_position_offset: 0,
                },
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ExternalWgpuError::BufferTooSmall { tensor: "K", .. }
    ));
}
