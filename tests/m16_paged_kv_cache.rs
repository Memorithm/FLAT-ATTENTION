#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric,
    paged_kv::{PagedKvConfig, WgpuPagedKvCache},
    AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig, FlatAttentionConfig,
    PagedDecodePass, WgpuPagedDecodePipeline,
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
            panic!("M16 paged KV cache requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M16 paged KV cache test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m16-paged-kv-cache-test"),
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
            let x = index as f32 * 0.031 + phase;
            x.sin() * 1.0625 + (x * 0.47).cos() * 0.28125
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
                let dim = pair * 2;
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
        label: Some("flat-m16-paged-kv-input"),
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
        label: Some("flat-m16-paged-kv-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m16-paged-kv-readback"),
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
fn append_crosses_pages_without_rewriting_prefix_and_reset_reuses_generation() {
    let Some(harness) = harness() else {
        return;
    };
    let config = PagedKvConfig {
        page_size: 3,
        physical_pages: 4,
    };
    let (kv_heads, head_dim) = (1usize, 8usize);
    let width = kv_heads * head_dim;
    let mut cache = WgpuPagedKvCache::new(&harness.device, config, kv_heads, head_dim).unwrap();

    let first_k = fixture(2 * width, 0.2);
    let first_v = fixture(2 * width, 0.7);
    let first_k_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &first_k,
        wgpu::BufferUsages::COPY_SRC,
    );
    let first_v_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &first_v,
        wgpu::BufferUsages::COPY_SRC,
    );
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-kv-first-append"),
        });
    cache
        .record_append(&mut encoder, &first_k_gpu, &first_v_gpu, 2)
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));

    let second_k = fixture(4 * width, 1.2);
    let second_v = fixture(4 * width, 1.7);
    let second_k_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &second_k,
        wgpu::BufferUsages::COPY_SRC,
    );
    let second_v_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &second_v,
        wgpu::BufferUsages::COPY_SRC,
    );
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-kv-second-append"),
        });
    cache
        .record_append(&mut encoder, &second_k_gpu, &second_v_gpu, 4)
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));
    let _ = harness.device.poll(wgpu::Maintain::Wait);

    assert_eq!(cache.len(), 6);
    assert_eq!(cache.table().telemetry().unwrap().mapped_pages, 2);
    let physical = read_f32(
        &harness.device,
        &harness.queue,
        cache.k_buffer(),
        config.capacity_tokens().unwrap() * width,
    );
    for logical in 0..6 {
        let expected_row = if logical < 2 {
            &first_k[logical * width..(logical + 1) * width]
        } else {
            let row = logical - 2;
            &second_k[row * width..(row + 1) * width]
        };
        let address = cache.table().address(logical).unwrap();
        let physical_row = address.physical_page * config.page_size + address.offset_in_page;
        assert_eq!(
            &physical[physical_row * width..(physical_row + 1) * width],
            expected_row
        );
    }

    let old_generation = cache.generation();
    cache.reset().unwrap();
    assert!(cache.is_empty());
    assert_ne!(cache.generation(), old_generation);

    let replacement = fixture(width, 3.1);
    let replacement_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &replacement,
        wgpu::BufferUsages::COPY_SRC,
    );
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-kv-reuse"),
        });
    cache
        .record_append(&mut encoder, &replacement_gpu, &replacement_gpu, 1)
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));
    let _ = harness.device.poll(wgpu::Maintain::Wait);
    let address = cache.table().address(0).unwrap();
    assert_eq!(address.physical_page, 0);
    assert_eq!(address.offset_in_page, 0);
}

#[test]
fn resident_paged_cache_feeds_qualified_paged_decode() {
    let Some(harness) = harness() else {
        return;
    };
    let (q_heads, kv_heads, kv_len, head_dim) = (4usize, 1usize, 5usize, 32usize);
    let theta = 10_000.0;
    let q_position = kv_len - 1;
    let q = fixture(q_heads * head_dim, 0.15);
    let raw_k = fixture(kv_len * kv_heads * head_dim, 0.75);
    let v = fixture(kv_len * kv_heads * head_dim, 1.35);
    let rotated_k = rotate_k_projection(&raw_k, kv_len, kv_heads, head_dim, theta);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let expected = forward_reference_projection_grouped_rope_asymmetric(
        &q,
        &raw_k,
        &v,
        AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads,
            kv_heads,
            query_len: 1,
            kv_len,
            head_dim,
            query_position_offset: q_position,
        },
        config,
        AsymmetricRotaryEmbeddingConfig {
            theta,
            query_position_offset: q_position,
            kv_position_offset: 0,
        },
    )
    .unwrap();

    let mut cache = WgpuPagedKvCache::new(
        &harness.device,
        PagedKvConfig {
            page_size: 2,
            physical_pages: 4,
        },
        kv_heads,
        head_dim,
    )
    .unwrap();
    let width = kv_heads * head_dim;
    for (start, len) in [(0usize, 2usize), (2, 3)] {
        let k_gpu = input_buffer(
            &harness.device,
            &harness.queue,
            &rotated_k[start * width..(start + len) * width],
            wgpu::BufferUsages::COPY_SRC,
        );
        let v_gpu = input_buffer(
            &harness.device,
            &harness.queue,
            &v[start * width..(start + len) * width],
            wgpu::BufferUsages::COPY_SRC,
        );
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-m16-paged-kv-decode-append"),
            });
        cache
            .record_append(&mut encoder, &k_gpu, &v_gpu, len)
            .unwrap();
        harness.queue.submit(Some(encoder.finish()));
    }

    let page_table = cache.decode_table().unwrap();
    let pipeline = WgpuPagedDecodePipeline::new(&harness.device).unwrap();
    let q_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &q,
        wgpu::BufferUsages::STORAGE,
    );
    let output = pipeline
        .create_output_buffer(&harness.device, q_heads, head_dim)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-kv-decode"),
        });
    let layout = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            PagedDecodePass {
                q: &q_gpu,
                k: cache.k_buffer(),
                v: cache.v_buffer(),
                page_table: &page_table,
                out_and_lse: &output,
                q_heads,
                kv_heads,
                head_dim,
                config,
                theta,
                q_rope_position: q_position,
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
        "M16 paged-cache O",
        &actual[..layout.output_elements],
        &expected.output,
    );
    assert_close(
        "M16 paged-cache LSE",
        &actual[layout.output_elements..],
        &expected.lse,
    );
}
