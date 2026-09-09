#![cfg(feature = "wgpu")]

use flat_attention::paged_kv::{PagedKvConfig, WgpuPagedKvCache};

fn device() -> Option<wgpu::Device> {
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
    let (device, _) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-m16-paged-kv-observation-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("M16 observation request_device failed: {error}"));
    Some(device)
}

#[test]
fn checkpoint_and_page_telemetry_bind_the_same_logical_state() {
    let Some(device) = device() else {
        return;
    };
    let mut cache = WgpuPagedKvCache::new(
        &device,
        PagedKvConfig {
            page_size: 4,
            physical_pages: 4,
        },
        2,
        8,
    )
    .unwrap();

    // Metadata-only table growth is sufficient for this observation contract;
    // no K/V bytes are read, copied back to the host, or interpreted here.
    cache.table_mut_for_test_only().append(5).unwrap();
    let checkpoint = cache.checkpoint();
    let telemetry = cache.table().telemetry().unwrap();

    assert_eq!(checkpoint.len(), telemetry.live_tokens);
    assert_eq!(checkpoint.generation(), telemetry.generation);
    assert_eq!(checkpoint.branch_epoch(), cache.branch_epoch());
    assert_eq!(telemetry.mapped_pages, 2);
    assert_eq!(telemetry.internal_fragmentation_tokens, 3);
}
