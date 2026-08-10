#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, ExternalAsymmetricProjectionPass, FlatAttentionConfig,
    ResidentDecodeError, WgpuResidentDecodePipeline,
};

const ATOL: f32 = 2.0e-4;
const RTOL: f32 = 1.0e-3;

struct DeviceHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[derive(Clone, Copy)]
struct PhysicalKvGeometry {
    batch: usize,
    kv_len: usize,
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
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
            panic!("M15 external-resident test requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional M15 external-resident test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m15-external-resident-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M15 external request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.027 + phase;
            x.sin() * 1.3125 + (x * 0.31).cos() * 0.40625
        })
        .collect()
}

fn rotate_k_row(raw: &[f32], kv_heads: usize, head_dim: usize, theta: f32, pos: usize) -> Vec<f32> {
    let mut rotated = raw.to_vec();
    for head in 0..kv_heads {
        let head_base = head * head_dim;
        for pair in 0..head_dim / 2 {
            let dim = 2 * pair;
            let exponent = -2.0 * pair as f32 / head_dim as f32;
            let frequency = theta.powf(exponent);
            let angle = pos as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let even = raw[head_base + dim];
            let odd = raw[head_base + dim + 1];
            rotated[head_base + dim] = even * cos - odd * sin;
            rotated[head_base + dim + 1] = even * sin + odd * cos;
        }
    }
    rotated
}

fn physical_kv(
    raw_k: &[f32],
    compact_v: &[f32],
    geometry: PhysicalKvGeometry,
) -> (Vec<f32>, Vec<f32>) {
    let width = geometry.kv_heads * geometry.head_dim;
    let mut k = vec![9_999.0f32; geometry.batch * geometry.capacity * width];
    let mut v = vec![-8_888.0f32; geometry.batch * geometry.capacity * width];
    for b in 0..geometry.batch {
        for pos in 0..geometry.kv_len {
            let compact = (b * geometry.kv_len + pos) * width;
            let physical = (b * geometry.capacity + pos) * width;
            let rotated = rotate_k_row(
                &raw_k[compact..compact + width],
                geometry.kv_heads,
                geometry.head_dim,
                geometry.theta,
                pos,
            );
            k[physical..physical + width].copy_from_slice(&rotated);
            v[physical..physical + width].copy_from_slice(&compact_v[compact..compact + width]);
        }
    }
    (k, v)
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
    // Deliberately over-allocate every external input. The M15 bridge must bind
    // only the validated logical tensor range, not this full backing allocation.
    let allocation_bytes = bytes.len().max(4) + 4096;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m15-external-input"),
        size: allocation_bytes as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn output_buffer(device: &wgpu::Device, elements: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m15-external-output"),
        size: (elements * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn read_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    let bytes = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m15-external-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m15-external-readback"),
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
fn specialized_decode_consumes_external_capacity_strided_kv_without_copy() {
    let Some(harness) = harness() else {
        return;
    };
    let (batch, q_heads, kv_heads, kv_len, capacity, head_dim) =
        (2usize, 4usize, 2usize, 3usize, 7usize, 32usize);
    let theta = 10_000.0;
    let q_width = q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let q = fixture(batch * q_width, 0.2);
    let raw_k = fixture(batch * kv_len * kv_width, 0.8);
    let compact_v = fixture(batch * kv_len * kv_width, 1.4);
    let (physical_k, physical_v) = physical_kv(
        &raw_k,
        &compact_v,
        PhysicalKvGeometry {
            batch,
            kv_len,
            capacity,
            kv_heads,
            head_dim,
            theta,
        },
    );

    let shape = AsymmetricGroupedAttentionShape {
        batch,
        q_heads,
        kv_heads,
        query_len: 1,
        kv_len,
        head_dim,
        query_position_offset: kv_len - 1,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let rotary = AsymmetricRotaryEmbeddingConfig {
        theta,
        query_position_offset: kv_len - 1,
        kv_position_offset: 0,
    };
    let expected = forward_reference_projection_grouped_rope_asymmetric(
        &q, &raw_k, &compact_v, shape, config, rotary,
    )
    .unwrap();

    let q_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &q,
        wgpu::BufferUsages::STORAGE,
    );
    let k_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &physical_k,
        wgpu::BufferUsages::STORAGE,
    );
    let v_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &physical_v,
        wgpu::BufferUsages::STORAGE,
    );
    let output_elements = batch * q_width;
    let lse_elements = batch * q_heads;
    let output = output_buffer(&harness.device, output_elements + lse_elements);

    let pipeline = WgpuResidentDecodePipeline::new(&harness.device).unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m15-external-decode"),
        });
    let layout = pipeline
        .encode_external_pre_rotated_k(
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
            capacity,
        )
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));

    let actual = read_f32(
        &harness.device,
        &harness.queue,
        &output,
        layout.combined_elements,
    );
    assert_eq!(layout.output_elements, output_elements);
    assert_eq!(layout.lse_elements, lse_elements);
    assert_close(
        "M15 external O",
        &actual[..layout.output_elements],
        &expected.output,
    );
    assert_close(
        "M15 external LSE",
        &actual[layout.output_elements..],
        &expected.lse,
    );
}

#[test]
fn external_decode_rejects_invalid_specialized_geometry_before_dispatch() {
    let Some(harness) = harness() else {
        return;
    };
    let pipeline = WgpuResidentDecodePipeline::new(&harness.device).unwrap();
    let q_heads = 4usize;
    let kv_heads = 2usize;
    let head_dim = 32usize;
    let kv_len = 3usize;
    let capacity = 5usize;
    let q = input_buffer(
        &harness.device,
        &harness.queue,
        &vec![0.0; q_heads * head_dim * 2],
        wgpu::BufferUsages::STORAGE,
    );
    let kv = input_buffer(
        &harness.device,
        &harness.queue,
        &vec![0.0; capacity * kv_heads * head_dim],
        wgpu::BufferUsages::STORAGE,
    );
    let output = output_buffer(&harness.device, 2 * q_heads * head_dim + 2 * q_heads);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let rotary = AsymmetricRotaryEmbeddingConfig {
        theta: 10_000.0,
        query_position_offset: kv_len - 1,
        kv_position_offset: 0,
    };

    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m15-external-invalid"),
        });
    let query_len_error = pipeline
        .encode_external_pre_rotated_k(
            &harness.device,
            &mut encoder,
            ExternalAsymmetricProjectionPass {
                q: &q,
                k: &kv,
                v: &kv,
                out_and_lse: &output,
                shape: AsymmetricGroupedAttentionShape {
                    batch: 1,
                    q_heads,
                    kv_heads,
                    query_len: 2,
                    kv_len,
                    head_dim,
                    query_position_offset: kv_len - 1,
                },
                config,
                rotary,
            },
            capacity,
        )
        .unwrap_err();
    assert_eq!(
        query_len_error,
        ResidentDecodeError::InvalidQueryLen { actual: 2 }
    );

    let visibility_error = pipeline
        .encode_external_pre_rotated_k(
            &harness.device,
            &mut encoder,
            ExternalAsymmetricProjectionPass {
                q: &q,
                k: &kv,
                v: &kv,
                out_and_lse: &output,
                shape: AsymmetricGroupedAttentionShape {
                    batch: 1,
                    q_heads,
                    kv_heads,
                    query_len: 1,
                    kv_len,
                    head_dim,
                    query_position_offset: 0,
                },
                config,
                rotary,
            },
            capacity,
        )
        .unwrap_err();
    assert_eq!(
        visibility_error,
        ResidentDecodeError::CausalVisibilityMismatch {
            query_position: 0,
            kv_len,
        }
    );

    let capacity_error = pipeline
        .encode_external_pre_rotated_k(
            &harness.device,
            &mut encoder,
            ExternalAsymmetricProjectionPass {
                q: &q,
                k: &kv,
                v: &kv,
                out_and_lse: &output,
                shape: AsymmetricGroupedAttentionShape {
                    batch: 1,
                    q_heads,
                    kv_heads,
                    query_len: 1,
                    kv_len,
                    head_dim,
                    query_position_offset: kv_len - 1,
                },
                config,
                rotary,
            },
            2,
        )
        .unwrap_err();
    assert_eq!(
        capacity_error,
        ResidentDecodeError::InvalidCacheLength {
            len: kv_len,
            capacity: 2,
        }
    );
}
