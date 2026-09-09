#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::paged_kv::{PagedKvConfig, PagedKvError, WgpuPagedKvCache};

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
        label: Some("flat-m16-paged-kv-truncate-input"),
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
    let initial_gpu = input_buffer(&harness.device, &harness.queue, &initial);
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
    let replacement_gpu = input_buffer(&harness.device, &harness.queue, &replacement);
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
