#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, FlatAttentionConfig, PagedDecodePass, WgpuPagedDecodePipeline,
    WgpuPagedKvTable, FLAT_DECODE_PAGED_WGSL,
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
            panic!("M16 paged decode requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M16 paged decode test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m16-paged-decode-test"),
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
            let x = index as f32 * 0.021 + phase;
            x.sin() * 1.1875 + (x * 0.43).cos() * 0.34375
        })
        .collect()
}

fn rotate_k_projection(
    raw: &[f32],
    kv_len: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
) -> Vec<f32> {
    let mut rotated = raw.to_vec();
    let width = kv_heads * head_dim;
    for position in 0..kv_len {
        for head in 0..kv_heads {
            let head_base = position * width + head * head_dim;
            for pair in 0..head_dim / 2 {
                let dim = 2 * pair;
                let exponent = -2.0 * pair as f32 / head_dim as f32;
                let angle = position as f32 * theta.powf(exponent);
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

fn input_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m16-paged-input"),
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
        label: Some("flat-m16-paged-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m16-paged-readback"),
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
fn paged_decode_shader_parses_and_validates() {
    let module = naga::front::wgsl::parse_str(FLAT_DECODE_PAGED_WGSL)
        .expect("M16 paged decode WGSL must parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("M16 paged decode WGSL must validate");
}

#[test]
fn paged_decode_matches_contiguous_oracle_across_page_boundaries() {
    let Some(harness) = harness() else {
        return;
    };
    let (q_heads, kv_heads, kv_len, page_size, physical_pages, head_dim) =
        (4usize, 2usize, 7usize, 3usize, 4usize, 64usize);
    let theta = 10_000.0;
    let q_position = kv_len - 1;
    let q = fixture(q_heads * head_dim, 0.2);
    let raw_k = fixture(kv_len * kv_heads * head_dim, 0.8);
    let v = fixture(kv_len * kv_heads * head_dim, 1.4);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads,
        kv_heads,
        query_len: 1,
        kv_len,
        head_dim,
        query_position_offset: q_position,
    };
    let rotary = AsymmetricRotaryEmbeddingConfig {
        theta,
        query_position_offset: q_position,
        kv_position_offset: 0,
    };
    let expected =
        forward_reference_projection_grouped_rope_asymmetric(&q, &raw_k, &v, shape, config, rotary)
            .unwrap();

    let rotated_k = rotate_k_projection(&raw_k, kv_len, kv_heads, head_dim, theta);
    let physical_rows = page_size * physical_pages;
    let width = kv_heads * head_dim;
    let mut physical_k = vec![17.0f32; physical_rows * width];
    let mut physical_v = vec![-19.0f32; physical_rows * width];
    physical_k[..rotated_k.len()].copy_from_slice(&rotated_k);
    physical_v[..v.len()].copy_from_slice(&v);

    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size,
        physical_pages,
    })
    .unwrap();
    table.append(kv_len).unwrap();
    assert_eq!(table.telemetry().unwrap().mapped_pages, 3);
    let device_table = WgpuPagedKvTable::from_table(&table).unwrap();

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
    let pipeline = WgpuPagedDecodePipeline::new(&harness.device).unwrap();
    let output = pipeline
        .create_output_buffer(&harness.device, q_heads, head_dim)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-decode"),
        });
    let layout = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            PagedDecodePass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                page_table: &device_table,
                out_and_lse: &output,
                q_heads,
                kv_heads,
                head_dim,
                config,
                theta,
                q_rope_position: q_position,
                q_causal_position: q_position,
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
        "M16 paged O",
        &actual[..layout.output_elements],
        &expected.output,
    );
    assert_close(
        "M16 paged LSE",
        &actual[layout.output_elements..],
        &expected.lse,
    );
}
