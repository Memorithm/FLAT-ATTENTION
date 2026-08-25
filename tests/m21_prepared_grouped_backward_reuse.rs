#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    backward_reference_grouped, forward_reference_grouped, pack_grouped_backward_recompute_inputs,
    FlatAttentionConfig, GroupedAttentionShape, GroupedBackwardRecomputePass,
    WgpuGroupedBackwardRecomputePipeline,
};

const ATOL: f32 = 4.0e-4;
const RTOL: f32 = 2.0e-3;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn harness() -> Option<Harness> {
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
            panic!("M21 prepared grouped backward reuse requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional M21 prepared reuse test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-m21-prepared-reuse-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("M21 request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.053 + phase;
            x.sin() * 0.7 + (x * 0.37).cos() * 0.3
        })
        .collect()
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn initialized_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    usage: wgpu::BufferUsages,
    label: &'static str,
) -> wgpu::Buffer {
    let bytes = bytes_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
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
        label: Some("flat-m21-prepared-reuse-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m21-prepared-reuse-readback"),
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

fn run_case(harness: &Harness, shape: GroupedAttentionShape, config: FlatAttentionConfig) {
    let q = fixture(shape.q_tensor_len().unwrap(), 0.1);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.7);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.3);
    let d_out = fixture(shape.q_tensor_len().unwrap(), 1.9);
    let forward = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
    let expected = backward_reference_grouped(&q, &k, &v, &d_out, shape, config, &forward).unwrap();
    let packed =
        pack_grouped_backward_recompute_inputs(&q, &k, &v, &d_out, &forward, shape).unwrap();

    let pipeline = WgpuGroupedBackwardRecomputePipeline::new(&harness.device).unwrap();
    let packed_gpu = initialized_buffer(
        &harness.device,
        &harness.queue,
        &packed,
        wgpu::BufferUsages::STORAGE,
        "flat-m21-prepared-reuse-input",
    );
    let grads_gpu = pipeline
        .create_gradient_buffer(&harness.device, shape)
        .unwrap();
    let prepared = pipeline
        .prepare(
            &harness.device,
            GroupedBackwardRecomputePass {
                packed_forward: &packed_gpu,
                packed_grads: &grads_gpu,
                shape,
                config,
            },
        )
        .unwrap();
    let layout = prepared.layout();

    for repeat in 0..4 {
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-m21-prepared-reuse-dispatch"),
            });
        assert_eq!(pipeline.encode_prepared(&mut encoder, &prepared), layout);
        harness.queue.submit(Some(encoder.finish()));
        let actual = read_f32(
            &harness.device,
            &harness.queue,
            &grads_gpu,
            layout.gradient_elements,
        );
        let dq_end = layout.q_elements;
        let dk_end = dq_end + layout.kv_elements;
        assert_close(
            &format!("repeat-{repeat}-dQ"),
            &actual[..dq_end],
            &expected.dq,
        );
        assert_close(
            &format!("repeat-{repeat}-dK"),
            &actual[dq_end..dk_end],
            &expected.dk,
        );
        assert_close(
            &format!("repeat-{repeat}-dV"),
            &actual[dk_end..],
            &expected.dv,
        );
    }
}

#[test]
fn prepared_grouped_backward_reuse_matches_oracle_across_head_groupings() {
    let Some(harness) = harness() else {
        return;
    };
    let cases = [
        (
            GroupedAttentionShape {
                batch: 1,
                q_heads: 4,
                kv_heads: 4,
                seq_len: 7,
                head_dim: 8,
            },
            FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
        ),
        (
            GroupedAttentionShape {
                batch: 1,
                q_heads: 8,
                kv_heads: 2,
                seq_len: 9,
                head_dim: 16,
            },
            FlatAttentionConfig {
                causal: false,
                softmax_scale: Some(0.27),
            },
        ),
        (
            GroupedAttentionShape {
                batch: 2,
                q_heads: 4,
                kv_heads: 1,
                seq_len: 5,
                head_dim: 12,
            },
            FlatAttentionConfig {
                causal: true,
                softmax_scale: Some(0.31),
            },
        ),
    ];

    for (shape, config) in cases {
        run_case(&harness, shape, config);
    }
}
