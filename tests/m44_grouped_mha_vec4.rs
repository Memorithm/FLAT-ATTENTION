#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_grouped, FlatAttentionConfig, GroupedAttentionShape, GroupedForwardPass,
    WgpuGroupedForwardPipeline,
};

const ATOL: f32 = 7.0e-4;
const RTOL: f32 = 3.0e-3;

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
            panic!("M44 grouped MHA vec4 qualification requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional M44 vec4 test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m44-grouped-mha-vec4"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M44 request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.039 + phase;
            x.sin() * 0.67 + (x * 0.43).cos() * 0.33
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
        label: Some("flat-m44-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m44-readback"),
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

fn run_case(harness: &Harness, head_dim: usize, causal: bool) {
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 2,
        kv_heads: 2,
        seq_len: 9,
        head_dim,
    };
    let config = FlatAttentionConfig {
        causal,
        softmax_scale: None,
    };
    let len = shape.q_tensor_len().unwrap();
    let q = fixture(len, 0.2);
    let k = fixture(len, 0.8);
    let v = fixture(len, 1.4);
    let expected = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();

    let q_gpu = storage(&harness.device, &harness.queue, &q, "flat-m44-q");
    let k_gpu = storage(&harness.device, &harness.queue, &k, "flat-m44-k");
    let v_gpu = storage(&harness.device, &harness.queue, &v, "flat-m44-v");

    for vectorized in [false, true] {
        let pipeline =
            WgpuGroupedForwardPipeline::with_vectorization(&harness.device, vectorized).unwrap();
        assert_eq!(pipeline.vectorization_enabled(), vectorized);
        let variant = format!("{:?}", pipeline.kernel_variant_for_shape(shape));
        assert_eq!(
            variant,
            if vectorized {
                "Q4Vec4Mha"
            } else {
                "Q4PortableGrouped"
            }
        );

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
        assert_eq!(format!("{:?}", prepared.kernel_variant()), variant);

        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-m44-grouped-mha-vec4"),
            });
        let layout = pipeline.encode_prepared(&mut encoder, &prepared);
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
}

#[test]
fn grouped_mha_vec4_matches_oracle_for_d64_d128() {
    let Some(harness) = harness() else {
        return;
    };
    for head_dim in [64, 128] {
        for causal in [false, true] {
            run_case(&harness, head_dim, causal);
        }
    }
}

#[test]
fn vectorization_never_claims_native_gqa_or_unqualified_dimensions() {
    let Some(harness) = harness() else {
        return;
    };
    let pipeline = WgpuGroupedForwardPipeline::with_vectorization(&harness.device, true).unwrap();
    for shape in [
        GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 7,
            head_dim: 64,
        },
        GroupedAttentionShape {
            batch: 1,
            q_heads: 2,
            kv_heads: 2,
            seq_len: 7,
            head_dim: 32,
        },
    ] {
        assert_eq!(
            format!("{:?}", pipeline.kernel_variant_for_shape(shape)),
            "Q4PortableGrouped"
        );
    }
}
