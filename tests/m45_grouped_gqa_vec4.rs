#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_grouped, FlatAttentionConfig, GroupedAttentionShape, GroupedForwardPass,
    WgpuGroupedForwardPipeline,
};
use naga::valid::{Capabilities, ValidationFlags, Validator};

const ATOL: f32 = 7.0e-4;
const RTOL: f32 = 3.0e-3;
const GROUPED_VEC4_WGSL: &str = include_str!("../shaders/flat_fwd_grouped_vec4.wgsl");
const GROUPED_KV_REUSE_WGSL: &str = include_str!("../shaders/flat_fwd_grouped_kv_reuse.wgsl");

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn harness() -> Option<Harness> {
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
            panic!("M45 grouped vec4 qualification requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional M45 grouped vec4 test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m45-grouped-gqa-vec4"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M45 request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.041 + phase;
            x.sin() * 0.68 + (x * 0.39).cos() * 0.32
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

fn storage(
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
        label: Some("flat-m45-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m45-readback"),
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

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        let tolerance = ATOL + RTOL * actual.abs().max(expected.abs());
        assert!(
            actual.is_finite() && error <= tolerance,
            "index={index} actual={actual} expected={expected} error={error} tolerance={tolerance}"
        );
    }
}

fn run_case(
    harness: &Harness,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
    kv_reuse: bool,
) {
    let shape = GroupedAttentionShape {
        batch: 2,
        q_heads,
        kv_heads,
        seq_len: 9,
        head_dim,
    };
    let config = FlatAttentionConfig {
        causal,
        softmax_scale: None,
    };
    let q_len = shape.q_tensor_len().unwrap();
    let kv_len = shape.kv_tensor_len().unwrap();
    assert_eq!(q_len, 2 * q_heads * 9 * head_dim);
    assert_eq!(kv_len, 2 * kv_heads * 9 * head_dim);
    assert!(
        kv_len < q_len,
        "native grouped case must keep K/V unexpanded"
    );

    let q = fixture(q_len, 0.2);
    let k = fixture(kv_len, 0.8);
    let v = fixture(kv_len, 1.4);
    let expected = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();

    let q_gpu = storage(&harness.device, &harness.queue, &q, "flat-m45-q");
    let k_gpu = storage(&harness.device, &harness.queue, &k, "flat-m45-k");
    let v_gpu = storage(&harness.device, &harness.queue, &v, "flat-m45-v");
    assert_eq!(q_gpu.size(), (q_len * 4) as u64);
    assert_eq!(k_gpu.size(), (kv_len * 4) as u64);
    assert_eq!(v_gpu.size(), (kv_len * 4) as u64);

    let pipeline = if kv_reuse {
        WgpuGroupedForwardPipeline::with_grouped_kv_reuse(&harness.device, true).unwrap()
    } else {
        WgpuGroupedForwardPipeline::with_grouped_vectorization(&harness.device, true).unwrap()
    };
    assert!(!pipeline.vectorization_enabled());
    assert_eq!(pipeline.grouped_vectorization_enabled(), !kv_reuse);
    assert_eq!(pipeline.grouped_kv_reuse_enabled(), kv_reuse);
    let expected_variant = if kv_reuse {
        "Q4Vec4GroupedKvReuse"
    } else {
        "Q4Vec4Grouped"
    };
    assert_eq!(
        format!("{:?}", pipeline.kernel_variant_for_shape(shape)),
        expected_variant
    );

    let layout = WgpuGroupedForwardPipeline::layout(shape).unwrap();
    assert_eq!(layout.q_elements, q_len);
    assert_eq!(layout.kv_elements, kv_len);
    let output = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let prepared = pipeline
        .prepare(
            &harness.device,
            GroupedForwardPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &output,
                shape,
                config,
            },
        )
        .unwrap();
    assert_eq!(format!("{:?}", prepared.kernel_variant()), expected_variant);

    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m45-grouped-gqa-vec4"),
        });
    let encoded = pipeline.encode_prepared(&mut encoder, &prepared);
    assert_eq!(encoded, layout);
    harness.queue.submit(Some(encoder.finish()));
    let _ = harness.device.poll(wgpu::Maintain::Wait);
    let actual = read_f32(
        &harness.device,
        &harness.queue,
        &output,
        layout.output_elements,
    );
    assert_close(&actual[..layout.lse_offset()], &expected.output);
    assert_close(&actual[layout.lse_offset()..], &expected.lse);
}

#[test]
fn grouped_vec4_shader_parses_and_validates_with_naga_020() {
    let module = naga::front::wgsl::parse_str(GROUPED_VEC4_WGSL)
        .unwrap_or_else(|error| panic!("M45 grouped vec4 WGSL parse failed: {error:?}"));
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|error| panic!("M45 grouped vec4 WGSL validation failed: {error:?}"));
}

#[test]
fn grouped_kv_reuse_shader_parses_and_validates_with_naga_020() {
    let module = naga::front::wgsl::parse_str(GROUPED_KV_REUSE_WGSL)
        .unwrap_or_else(|error| panic!("Phase O grouped K/V-reuse WGSL parse failed: {error:?}"));
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|error| panic!("Phase O grouped K/V-reuse validation failed: {error:?}"));
}

#[test]
fn native_gqa_and_mqa_vec4_match_oracle_without_kv_expansion() {
    let Some(harness) = harness() else {
        return;
    };
    for (q_heads, kv_heads) in [(4, 2), (4, 1)] {
        for head_dim in [64, 128] {
            for causal in [false, true] {
                run_case(&harness, q_heads, kv_heads, head_dim, causal, false);
            }
        }
    }
}

#[test]
fn native_gqa_and_mqa_kv_reuse_match_oracle_without_kv_expansion() {
    let Some(harness) = harness() else {
        return;
    };
    for (q_heads, kv_heads) in [(4, 2), (4, 1), (6, 2)] {
        for head_dim in [64, 128] {
            for causal in [false, true] {
                run_case(&harness, q_heads, kv_heads, head_dim, causal, true);
            }
        }
    }
}

#[test]
fn grouped_vec4_opt_in_is_independent_and_fails_back_by_shape() {
    let Some(harness) = harness() else {
        return;
    };
    let grouped =
        WgpuGroupedForwardPipeline::with_grouped_vectorization(&harness.device, true).unwrap();
    let mha = WgpuGroupedForwardPipeline::with_vectorization(&harness.device, true).unwrap();
    let reuse = WgpuGroupedForwardPipeline::with_grouped_kv_reuse(&harness.device, true).unwrap();

    let gqa_d64 = GroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        seq_len: 7,
        head_dim: 64,
    };
    let mha_d64 = GroupedAttentionShape {
        batch: 1,
        q_heads: 2,
        kv_heads: 2,
        seq_len: 7,
        head_dim: 64,
    };
    let gqa_d32 = GroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        seq_len: 7,
        head_dim: 32,
    };

    assert_eq!(
        format!("{:?}", grouped.kernel_variant_for_shape(gqa_d64)),
        "Q4Vec4Grouped"
    );
    assert_eq!(
        format!("{:?}", grouped.kernel_variant_for_shape(mha_d64)),
        "Q4PortableGrouped"
    );
    assert_eq!(
        format!("{:?}", grouped.kernel_variant_for_shape(gqa_d32)),
        "Q4PortableGrouped"
    );
    assert_eq!(
        format!("{:?}", mha.kernel_variant_for_shape(gqa_d64)),
        "Q4PortableGrouped"
    );
    assert_eq!(
        format!("{:?}", mha.kernel_variant_for_shape(mha_d64)),
        "Q4Vec4Mha"
    );
    assert_eq!(
        format!("{:?}", reuse.kernel_variant_for_shape(gqa_d64)),
        "Q4Vec4GroupedKvReuse"
    );
    assert_eq!(
        format!("{:?}", reuse.kernel_variant_for_shape(mha_d64)),
        "Q4PortableGrouped"
    );
    assert_eq!(
        format!("{:?}", reuse.kernel_variant_for_shape(gqa_d32)),
        "Q4PortableGrouped"
    );
}
