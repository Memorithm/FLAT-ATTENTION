#![cfg(feature = "wgpu")]

use std::sync::{Mutex, MutexGuard};

use flat_attention::{
    FlatAttentionConfig, FlatAttentionError, GroupedAttentionShape, GroupedBackwardRecomputeError,
    GroupedBackwardRecomputePass, WgpuGroupedBackwardRecomputePipeline,
};

struct Harness {
    device: wgpu::Device,
    _live_wgpu_guard: MutexGuard<'static, ()>,
}

fn live_wgpu_guard() -> MutexGuard<'static, ()> {
    // Each M22 test creates an independent live WGPU instance, adapter, and
    // device. Some physical drivers do not tolerate those initializations
    // concurrently inside one Rust test binary, so keep this device-level
    // qualification deterministic without changing the runtime under test.
    static LIVE_WGPU_LOCK: Mutex<()> = Mutex::new(());
    LIVE_WGPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn harness() -> Option<Harness> {
    let live_wgpu_guard = live_wgpu_guard();
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
            panic!("M22 prepared validation requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M22 prepared validation skipped");
        return None;
    };
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-m22-prepared-validation"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("M22 request_device failed: {error}"));
    Some(Harness {
        device,
        _live_wgpu_guard: live_wgpu_guard,
    })
}

fn storage_buffer(device: &wgpu::Device, size: u64, label: &'static str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn shape() -> GroupedAttentionShape {
    GroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 1,
        seq_len: 5,
        head_dim: 8,
    }
}

#[test]
fn prepare_rejects_undersized_forward_buffer_before_encoding() {
    let Some(harness) = harness() else {
        return;
    };
    let shape = shape();
    let pipeline = WgpuGroupedBackwardRecomputePipeline::new(&harness.device).unwrap();
    let layout = WgpuGroupedBackwardRecomputePipeline::layout(shape).unwrap();
    let forward = storage_buffer(&harness.device, 4, "flat-m22-small-forward");
    let grads = storage_buffer(
        &harness.device,
        layout.gradient_bytes,
        "flat-m22-valid-grads",
    );

    let error = pipeline
        .prepare(
            &harness.device,
            GroupedBackwardRecomputePass {
                packed_forward: &forward,
                packed_grads: &grads,
                shape,
                config: FlatAttentionConfig::default(),
            },
        )
        .unwrap_err();

    match error {
        GroupedBackwardRecomputeError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        } => {
            assert_eq!(tensor, "Q|K|V|dO|O|LSE");
            assert_eq!(actual_bytes, 4);
            assert_eq!(required_bytes, layout.packed_forward_bytes);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn prepare_rejects_undersized_gradient_buffer_before_encoding() {
    let Some(harness) = harness() else {
        return;
    };
    let shape = shape();
    let pipeline = WgpuGroupedBackwardRecomputePipeline::new(&harness.device).unwrap();
    let layout = WgpuGroupedBackwardRecomputePipeline::layout(shape).unwrap();
    let forward = storage_buffer(
        &harness.device,
        layout.packed_forward_bytes,
        "flat-m22-valid-forward",
    );
    let grads = storage_buffer(&harness.device, 4, "flat-m22-small-grads");

    let error = pipeline
        .prepare(
            &harness.device,
            GroupedBackwardRecomputePass {
                packed_forward: &forward,
                packed_grads: &grads,
                shape,
                config: FlatAttentionConfig::default(),
            },
        )
        .unwrap_err();

    match error {
        GroupedBackwardRecomputeError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        } => {
            assert_eq!(tensor, "dQ|dK|dV");
            assert_eq!(actual_bytes, 4);
            assert_eq!(required_bytes, layout.gradient_bytes);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn prepare_rejects_invalid_scale_and_accepts_overprovisioned_buffers() {
    let Some(harness) = harness() else {
        return;
    };
    let shape = shape();
    let pipeline = WgpuGroupedBackwardRecomputePipeline::new(&harness.device).unwrap();
    let layout = WgpuGroupedBackwardRecomputePipeline::layout(shape).unwrap();
    let forward = storage_buffer(
        &harness.device,
        layout.packed_forward_bytes + 64,
        "flat-m22-overprovisioned-forward",
    );
    let grads = storage_buffer(
        &harness.device,
        layout.gradient_bytes + 64,
        "flat-m22-overprovisioned-grads",
    );

    let error = pipeline
        .prepare(
            &harness.device,
            GroupedBackwardRecomputePass {
                packed_forward: &forward,
                packed_grads: &grads,
                shape,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: Some(0.0),
                },
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        GroupedBackwardRecomputeError::Core(FlatAttentionError::InvalidScale(scale)) if scale == 0.0
    ));

    let prepared = pipeline
        .prepare(
            &harness.device,
            GroupedBackwardRecomputePass {
                packed_forward: &forward,
                packed_grads: &grads,
                shape,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: Some(0.25),
                },
            },
        )
        .unwrap();
    assert_eq!(prepared.layout(), layout);
}
