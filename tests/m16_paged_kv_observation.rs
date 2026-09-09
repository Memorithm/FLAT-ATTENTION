#![cfg(feature = "wgpu")]

use flat_attention::paged_kv::{PagedKvConfig, WgpuPagedKvCache};

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
            panic!("M16 paged KV observation contract requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional M16 observation test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-m16-paged-kv-observation-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("M16 observation request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn source_buffer(device: &wgpu::Device, queue: &wgpu::Queue, rows: usize, width: usize) -> wgpu::Buffer {
    let values = vec![0.0_f32; rows * width];
    let mut bytes = Vec::with_capacity(values.len() * core::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m16-paged-kv-observation-source"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

#[test]
fn checkpoint_and_page_telemetry_bind_the_same_logical_state() {
    let Some(harness) = harness() else {
        return;
    };
    let mut cache = WgpuPagedKvCache::new(
        &harness.device,
        PagedKvConfig {
            page_size: 4,
            physical_pages: 4,
        },
        2,
        8,
    )
    .unwrap();

    let width = cache.kv_heads() * cache.head_dim();
    let source = source_buffer(&harness.device, &harness.queue, 5, width);
    let (new_len, _) = cache
        .append_and_submit(
            &harness.device,
            &harness.queue,
            &source,
            &source,
            5,
        )
        .unwrap();
    assert_eq!(new_len, 5);

    let checkpoint = cache.checkpoint();
    let telemetry = cache.table().telemetry().unwrap();

    assert_eq!(checkpoint.len(), telemetry.live_tokens);
    assert_eq!(checkpoint.generation(), telemetry.generation);
    assert_eq!(checkpoint.branch_epoch(), cache.branch_epoch());
    assert_eq!(telemetry.mapped_pages, 2);
    assert_eq!(telemetry.internal_fragmentation_tokens, 3);
}
