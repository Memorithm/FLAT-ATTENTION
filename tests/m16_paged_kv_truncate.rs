#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric,
    paged_kv::{PagedKvConfig, PagedKvError, WgpuPagedKvCache},
    AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig, FlatAttentionConfig,
    PagedDecodePass, WgpuPagedDecodePipeline, WgpuPagedKvTable,
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
            panic!("M16 paged KV truncate requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M16 truncate test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-m16-paged-kv-truncate-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("M16 truncate request_device failed: {error}"));
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
        label: Some("flat-m16-paged-kv-truncate-input"),
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
        label: Some("flat-m16-paged-kv-truncate-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m16-paged-kv-truncate-readback"),
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
fn paged_truncate_releases_tail_pages_and_reappend_overwrites_tail() {
    let Some(harness) = harness() else {
        return;
    };
    let config = PagedKvConfig {
        page_size: 2,
        physical_pages: 4,
    };
    let (kv_heads, head_dim) = (1usize, 4usize);
    let width = kv_heads * head_dim;
    let mut cache = WgpuPagedKvCache::new(&harness.device, config, kv_heads, head_dim).unwrap();

    let initial: Vec<f32> = (0..5 * width).map(|i| 10.0 + i as f32).collect();
    let initial_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &initial,
        wgpu::BufferUsages::COPY_SRC,
    );
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-kv-truncate-seed"),
        });
    cache
        .record_append(&mut encoder, &initial_gpu, &initial_gpu, 5)
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));
    assert_eq!(cache.len(), 5);
    assert_eq!(cache.table().telemetry().unwrap().mapped_pages, 3);

    let generation = cache.generation();
    cache.truncate(3).unwrap();
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.generation(), generation);
    let telemetry = cache.table().telemetry().unwrap();
    assert_eq!(telemetry.mapped_pages, 2);
    assert_eq!(telemetry.free_pages, 2);
    assert_eq!(telemetry.internal_fragmentation_tokens, 1);
    assert!(cache.table().address(3).is_none());
    assert_eq!(
        cache.truncate(4).unwrap_err().to_string(),
        PagedKvError::TruncateOutOfBounds {
            requested_len: 4,
            current_len: 3,
        }
        .to_string()
    );
    assert_eq!(cache.len(), 3);

    let replacement: Vec<f32> = (0..2 * width).map(|i| 100.0 + i as f32).collect();
    let replacement_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &replacement,
        wgpu::BufferUsages::COPY_SRC,
    );
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-kv-truncate-reappend"),
        });
    cache
        .record_append(&mut encoder, &replacement_gpu, &replacement_gpu, 2)
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));
    let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());

    assert_eq!(cache.len(), 5);
    assert_eq!(cache.table().address(4).unwrap().physical_page, 2);
    let physical = read_f32(
        &harness.device,
        &harness.queue,
        cache.k_buffer(),
        config.capacity_tokens().unwrap() * width,
    );

    for logical in 0..3 {
        let address = cache.table().address(logical).unwrap();
        let physical_row = address.physical_page * config.page_size + address.offset_in_page;
        assert_eq!(
            &physical[physical_row * width..(physical_row + 1) * width],
            &initial[logical * width..(logical + 1) * width]
        );
    }
    for logical in 3..5 {
        let replacement_row = logical - 3;
        let address = cache.table().address(logical).unwrap();
        let physical_row = address.physical_page * config.page_size + address.offset_in_page;
        assert_eq!(
            &physical[physical_row * width..(physical_row + 1) * width],
            &replacement[replacement_row * width..(replacement_row + 1) * width]
        );
    }
}

#[test]
fn paged_decode_after_truncate_uses_only_rewritten_live_tail() {
    let Some(harness) = harness() else {
        return;
    };
    let (q_heads, kv_heads, head_dim) = (4usize, 1usize, 32usize);
    let (initial_len, truncate_len, replacement_len) = (6usize, 3usize, 2usize);
    let final_len = truncate_len + replacement_len;
    let width = kv_heads * head_dim;
    let theta = 10_000.0;
    let q_position = final_len - 1;

    let prefix_k = fixture(truncate_len * width, 0.75);
    let prefix_v = fixture(truncate_len * width, 1.35);
    let stale_k = fixture((initial_len - truncate_len) * width, 7.75);
    let stale_v = fixture((initial_len - truncate_len) * width, 8.35);
    let replacement_k = fixture(replacement_len * width, 2.75);
    let replacement_v = fixture(replacement_len * width, 3.35);

    let mut initial_raw_k = prefix_k.clone();
    initial_raw_k.extend_from_slice(&stale_k);
    let mut initial_v = prefix_v.clone();
    initial_v.extend_from_slice(&stale_v);
    let initial_rotated_k =
        rotate_k_projection(&initial_raw_k, initial_len, kv_heads, head_dim, theta);

    let mut final_raw_k = prefix_k.clone();
    final_raw_k.extend_from_slice(&replacement_k);
    let mut final_v = prefix_v.clone();
    final_v.extend_from_slice(&replacement_v);
    let final_rotated_k = rotate_k_projection(&final_raw_k, final_len, kv_heads, head_dim, theta);

    let q = fixture(q_heads * head_dim, 0.15);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let expected = forward_reference_projection_grouped_rope_asymmetric(
        &q,
        &final_raw_k,
        &final_v,
        AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads,
            kv_heads,
            query_len: 1,
            kv_len: final_len,
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

    let mut stale_final_k = prefix_k.clone();
    stale_final_k.extend_from_slice(&stale_k[..replacement_len * width]);
    let mut stale_final_v = prefix_v.clone();
    stale_final_v.extend_from_slice(&stale_v[..replacement_len * width]);
    let stale_expected = forward_reference_projection_grouped_rope_asymmetric(
        &q,
        &stale_final_k,
        &stale_final_v,
        AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads,
            kv_heads,
            query_len: 1,
            kv_len: final_len,
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
    assert!(
        expected
            .output
            .iter()
            .zip(&stale_expected.output)
            .any(|(&fresh, &stale)| (fresh - stale).abs() > 1.0e-3),
        "truncate/reappend fixture must discriminate rewritten KV from stale tail"
    );

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
    let initial_k_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &initial_rotated_k,
        wgpu::BufferUsages::COPY_SRC,
    );
    let initial_v_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &initial_v,
        wgpu::BufferUsages::COPY_SRC,
    );
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-kv-truncate-decode-seed"),
        });
    cache
        .record_append(&mut encoder, &initial_k_gpu, &initial_v_gpu, initial_len)
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));

    cache.truncate(truncate_len).unwrap();
    let replacement_k_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &final_rotated_k[truncate_len * width..],
        wgpu::BufferUsages::COPY_SRC,
    );
    let replacement_v_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &final_v[truncate_len * width..],
        wgpu::BufferUsages::COPY_SRC,
    );
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-kv-truncate-decode-reappend"),
        });
    cache
        .record_append(
            &mut encoder,
            &replacement_k_gpu,
            &replacement_v_gpu,
            replacement_len,
        )
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));
    assert_eq!(cache.len(), final_len);

    let page_table = WgpuPagedKvTable::from_table(cache.table()).unwrap();
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
            label: Some("flat-m16-paged-kv-truncate-decode"),
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
        "M16 truncate paged decode O",
        &actual[..layout.output_elements],
        &expected.output,
    );
    assert_close(
        "M16 truncate paged decode LSE",
        &actual[layout.output_elements..],
        &expected.lse,
    );
}
