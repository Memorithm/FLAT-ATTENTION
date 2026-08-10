#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{WgpuResidentKvCache, WgpuResidentKvCacheError};

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
            panic!("M14 requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M14 device test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m14-resident-kv-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M14 request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn source_buffer(device: &wgpu::Device, queue: &wgpu::Queue, values: &[f32]) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m14-append-source"),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
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
        label: Some("flat-m14-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m14-readback"),
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

fn row(values: &[f32], row: usize, width: usize) -> &[f32] {
    &values[row * width..(row + 1) * width]
}

#[test]
fn resident_append_preserves_prefix_and_batch_capacity_stride() {
    let Some(harness) = harness() else {
        return;
    };
    let (batch, kv_heads, capacity, head_dim) = (2usize, 2usize, 5usize, 4usize);
    let width = kv_heads * head_dim;
    let mut cache =
        WgpuResidentKvCache::new(&harness.device, batch, kv_heads, capacity, head_dim).unwrap();
    assert!(cache.is_empty());
    assert_eq!(cache.remaining_capacity(), capacity);

    let first_k: Vec<f32> = (0..batch * 2 * width).map(|i| 100.0 + i as f32).collect();
    let first_v: Vec<f32> = (0..batch * 2 * width).map(|i| 500.0 + i as f32).collect();
    let first_k_gpu = source_buffer(&harness.device, &harness.queue, &first_k);
    let first_v_gpu = source_buffer(&harness.device, &harness.queue, &first_v);
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m14-first-append"),
        });
    assert_eq!(
        cache
            .record_append(&mut encoder, &first_k_gpu, &first_v_gpu, 2)
            .unwrap(),
        2
    );
    harness.queue.submit(Some(encoder.finish()));

    let next_k: Vec<f32> = (0..batch * width).map(|i| 900.0 + i as f32).collect();
    let next_v: Vec<f32> = (0..batch * width).map(|i| 1300.0 + i as f32).collect();
    let next_k_gpu = source_buffer(&harness.device, &harness.queue, &next_k);
    let next_v_gpu = source_buffer(&harness.device, &harness.queue, &next_v);
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m14-second-append"),
        });
    assert_eq!(
        cache
            .record_append(&mut encoder, &next_k_gpu, &next_v_gpu, 1)
            .unwrap(),
        3
    );
    harness.queue.submit(Some(encoder.finish()));
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.remaining_capacity(), 2);

    let tensor_len = batch * capacity * width;
    let actual_k = read_f32(
        &harness.device,
        &harness.queue,
        cache.k_buffer(),
        tensor_len,
    );
    let actual_v = read_f32(
        &harness.device,
        &harness.queue,
        cache.v_buffer(),
        tensor_len,
    );

    for b in 0..batch {
        for position in 0..2 {
            let source_row = b * 2 + position;
            let cache_row = b * capacity + position;
            assert_eq!(
                row(&actual_k, cache_row, width),
                row(&first_k, source_row, width)
            );
            assert_eq!(
                row(&actual_v, cache_row, width),
                row(&first_v, source_row, width)
            );
        }
        let cache_row = b * capacity + 2;
        assert_eq!(row(&actual_k, cache_row, width), row(&next_k, b, width));
        assert_eq!(row(&actual_v, cache_row, width), row(&next_v, b, width));
    }
}

#[test]
fn reset_reuses_prefix_without_copying_old_cache() {
    let Some(harness) = harness() else {
        return;
    };
    let (batch, kv_heads, capacity, head_dim) = (2usize, 1usize, 4usize, 8usize);
    let width = kv_heads * head_dim;
    let mut cache =
        WgpuResidentKvCache::new(&harness.device, batch, kv_heads, capacity, head_dim).unwrap();

    let initial = vec![3.0f32; batch * 2 * width];
    let initial_gpu = source_buffer(&harness.device, &harness.queue, &initial);
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m14-reset-seed"),
        });
    cache
        .record_append(&mut encoder, &initial_gpu, &initial_gpu, 2)
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));

    cache.reset();
    assert!(cache.is_empty());
    let replacement_k: Vec<f32> = (0..batch * width).map(|i| 10.0 + i as f32).collect();
    let replacement_v: Vec<f32> = (0..batch * width).map(|i| 40.0 + i as f32).collect();
    let replacement_k_gpu = source_buffer(&harness.device, &harness.queue, &replacement_k);
    let replacement_v_gpu = source_buffer(&harness.device, &harness.queue, &replacement_v);
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m14-reset-reuse"),
        });
    cache
        .record_append(&mut encoder, &replacement_k_gpu, &replacement_v_gpu, 1)
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));
    assert_eq!(cache.len(), 1);

    let actual_k = read_f32(
        &harness.device,
        &harness.queue,
        cache.k_buffer(),
        batch * capacity * width,
    );
    let actual_v = read_f32(
        &harness.device,
        &harness.queue,
        cache.v_buffer(),
        batch * capacity * width,
    );
    for b in 0..batch {
        let cache_row = b * capacity;
        assert_eq!(
            row(&actual_k, cache_row, width),
            row(&replacement_k, b, width)
        );
        assert_eq!(
            row(&actual_v, cache_row, width),
            row(&replacement_v, b, width)
        );
    }
}

#[test]
fn append_rejects_capacity_overflow_and_short_sources() {
    let Some(harness) = harness() else {
        return;
    };
    let mut cache = WgpuResidentKvCache::new(&harness.device, 1, 2, 2, 4).unwrap();
    let row = vec![1.0f32; 8];
    let source = source_buffer(&harness.device, &harness.queue, &row);
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m14-validation"),
        });
    cache
        .record_append(&mut encoder, &source, &source, 1)
        .unwrap();
    let error = cache
        .record_append(&mut encoder, &source, &source, 2)
        .unwrap_err();
    assert_eq!(
        error,
        WgpuResidentKvCacheError::CapacityExceeded {
            current_len: 1,
            append_len: 2,
            capacity: 2,
        }
    );

    let mut fresh = WgpuResidentKvCache::new(&harness.device, 2, 2, 4, 4).unwrap();
    let error = fresh
        .record_append(&mut encoder, &source, &source, 1)
        .unwrap_err();
    assert!(matches!(
        error,
        WgpuResidentKvCacheError::BufferTooSmall { tensor: "K", .. }
    ));
}
