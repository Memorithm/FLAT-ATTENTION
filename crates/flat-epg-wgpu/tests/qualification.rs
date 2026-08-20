use std::sync::mpsc;

use epg_core::{EpgGeometryDescriptor, EpgPositionDomain, So4Geometry};
use flat_attention::{FlatAttentionConfig, GroupedAttentionShape};
use flat_epg_reference::{forward_reference_grouped_epg, EpgEmbeddingConfig};
use flat_epg_wgpu::{
    EpgQualificationPass, EpgVec4QualificationPipeline, EPG_GROUPED_VEC4_QUALIFY_WGSL,
};
use naga::valid::{Capabilities, ValidationFlags, Validator};

const ATOL: f32 = 1.2e-3;
const RTOL: f32 = 4.0e-3;

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
            panic!("EPG qualification requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional EPG device qualification skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-epg-qualification"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("EPG request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.037 + phase;
            x.sin() * 0.61 + (x * 0.43).cos() * 0.29
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
        size: bytes.len().max(16) as u64,
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
        label: Some("flat-epg-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-epg-readback"),
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
            actual.is_finite() && expected.is_finite() && error <= tolerance,
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
    position_offset: u64,
    geometry: EpgGeometryDescriptor,
) {
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads,
        kv_heads,
        seq_len: 5,
        head_dim,
    };
    let config = FlatAttentionConfig {
        causal,
        softmax_scale: None,
    };
    let q_len = shape.q_tensor_len().unwrap();
    let kv_len = shape.kv_tensor_len().unwrap();
    if q_heads != kv_heads {
        assert!(kv_len < q_len, "GQA/MQA must keep physical K/V cardinality");
    }

    let q = fixture(q_len, 0.17);
    let k = fixture(kv_len, 0.71);
    let v = fixture(kv_len, 1.31);
    let position = EpgPositionDomain::new(position_offset);
    let expected = forward_reference_grouped_epg(
        &q,
        &k,
        &v,
        shape,
        config,
        EpgEmbeddingConfig { geometry, position },
    )
    .unwrap();

    let q_gpu = storage(&harness.device, &harness.queue, &q, "flat-epg-q");
    let k_gpu = storage(&harness.device, &harness.queue, &k, "flat-epg-k");
    let v_gpu = storage(&harness.device, &harness.queue, &v, "flat-epg-v");
    let pipeline = EpgVec4QualificationPipeline::new(&harness.device).unwrap();
    let layout = EpgVec4QualificationPipeline::layout(shape).unwrap();
    assert_eq!(layout.q_elements, q_len);
    assert_eq!(layout.kv_elements, kv_len);
    let output = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let prepared = pipeline
        .prepare(
            &harness.device,
            EpgQualificationPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &output,
                shape,
                config,
                geometry,
                position,
            },
        )
        .unwrap();

    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-epg-qualification"),
        });
    assert_eq!(pipeline.encode_prepared(&mut encoder, &prepared), layout);
    harness.queue.submit(Some(encoder.finish()));
    let _ = harness.device.poll(wgpu::Maintain::Wait);
    let actual = read_f32(
        &harness.device,
        &harness.queue,
        &output,
        layout.combined_elements,
    );
    assert_close(&actual[..layout.lse_offset()], &expected.output);
    assert_close(&actual[layout.lse_offset()..], &expected.lse);
}

#[test]
fn qualification_shader_parses_and_validates_with_naga_020() {
    let module = naga::front::wgsl::parse_str(EPG_GROUPED_VEC4_QUALIFY_WGSL)
        .unwrap_or_else(|error| panic!("EPG qualification WGSL parse failed: {error:?}"));
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|error| panic!("EPG qualification WGSL validation failed: {error:?}"));
}

#[test]
fn vec4_gpu_matches_cpu_epg_controls_for_mha_gqa_and_mqa() {
    let Some(harness) = harness() else {
        return;
    };

    for (q_heads, kv_heads) in [(2, 2), (4, 2), (4, 1)] {
        for head_dim in [64, 128] {
            let so4_tail = (head_dim / 2) as u32;
            let geometries = [
                EpgGeometryDescriptor::so2(10_000.0).unwrap(),
                EpgGeometryDescriptor::hybrid_so4(10_000.0, so4_tail, So4Geometry::Biplanar)
                    .unwrap(),
                EpgGeometryDescriptor::hybrid_so4(10_000.0, so4_tail, So4Geometry::Isoclinic)
                    .unwrap(),
            ];
            for geometry in geometries {
                for causal in [false, true] {
                    for position_offset in [0, 23] {
                        run_case(
                            &harness,
                            q_heads,
                            kv_heads,
                            head_dim,
                            causal,
                            position_offset,
                            geometry,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_vec4_geometry_is_rejected_before_dispatch() {
    let Some(harness) = harness() else {
        return;
    };
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 2,
        kv_heads: 1,
        seq_len: 3,
        head_dim: 66,
    };
    assert!(EpgVec4QualificationPipeline::layout(shape).is_err());
}
