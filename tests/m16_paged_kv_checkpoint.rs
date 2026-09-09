#![cfg(feature = "wgpu")]

use flat_attention::paged_kv::{PagedKvConfig, WgpuPagedKvCache, WgpuPagedKvCacheError};

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
            panic!("M16 paged KV checkpoint requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M16 checkpoint test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-m16-paged-kv-checkpoint-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("M16 checkpoint request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn source_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rows: usize,
    width: usize,
    phase: f32,
) -> wgpu::Buffer {
    let values: Vec<f32> = (0..rows * width)
        .map(|index| phase + index as f32 * 0.03125)
        .collect();
    let mut bytes = Vec::with_capacity(values.len() * core::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m16-paged-kv-checkpoint-source"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn append(
    harness: &DeviceHarness,
    cache: &mut WgpuPagedKvCache,
    rows: usize,
    phase: f32,
) {
    let width = cache.kv_heads() * cache.head_dim();
    let source = source_buffer(&harness.device, &harness.queue, rows, width, phase);
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-kv-checkpoint-append"),
        });
    cache
        .record_append(&mut encoder, &source, &source, rows)
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));
}

#[test]
fn checkpoint_restores_append_only_suffix_and_is_consumed_by_branch_change() {
    let Some(harness) = harness() else {
        return;
    };
    let mut cache = WgpuPagedKvCache::new(
        &harness.device,
        PagedKvConfig {
            page_size: 2,
            physical_pages: 4,
        },
        1,
        4,
    )
    .unwrap();

    append(&harness, &mut cache, 3, 1.0);
    let checkpoint = cache.checkpoint();
    assert_eq!(checkpoint.len(), 3);
    assert_eq!(checkpoint.generation(), cache.generation());
    assert_eq!(checkpoint.branch_epoch(), cache.branch_epoch());

    append(&harness, &mut cache, 2, 2.0);
    assert_eq!(cache.len(), 5);
    assert_eq!(cache.branch_epoch(), checkpoint.branch_epoch());

    cache.restore(&checkpoint).unwrap();
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.branch_epoch(), checkpoint.branch_epoch() + 1);

    append(&harness, &mut cache, 1, 3.0);
    assert_eq!(
        cache.restore(&checkpoint),
        Err(WgpuPagedKvCacheError::CheckpointBranchMismatch {
            checkpoint_branch_epoch: checkpoint.branch_epoch(),
            current_branch_epoch: cache.branch_epoch(),
        })
    );
}

#[test]
fn checkpoint_fails_closed_after_truncate_reappend_reset_and_foreign_cache() {
    let Some(harness) = harness() else {
        return;
    };
    let config = PagedKvConfig {
        page_size: 2,
        physical_pages: 4,
    };
    let mut cache = WgpuPagedKvCache::new(&harness.device, config, 1, 4).unwrap();
    append(&harness, &mut cache, 3, 1.0);
    let checkpoint = cache.checkpoint();

    append(&harness, &mut cache, 2, 2.0);
    cache.truncate(2).unwrap();
    append(&harness, &mut cache, 3, 4.0);
    assert_eq!(cache.len(), 5);
    assert_eq!(
        cache.restore(&checkpoint),
        Err(WgpuPagedKvCacheError::CheckpointBranchMismatch {
            checkpoint_branch_epoch: checkpoint.branch_epoch(),
            current_branch_epoch: cache.branch_epoch(),
        })
    );

    let post_branch = cache.checkpoint();
    let generation = cache.generation();
    cache.reset().unwrap();
    assert_eq!(
        cache.restore(&post_branch),
        Err(WgpuPagedKvCacheError::CheckpointGenerationMismatch {
            checkpoint_generation: generation,
            current_generation: cache.generation(),
        })
    );

    let foreign = cache.checkpoint();
    let mut other = WgpuPagedKvCache::new(&harness.device, config, 1, 4).unwrap();
    assert_eq!(
        other.restore(&foreign),
        Err(WgpuPagedKvCacheError::ForeignCheckpoint)
    );
}
