#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    backward_reference_grouped, forward_reference_grouped, pack_grouped_backward_recompute_inputs,
    FlatAttentionConfig, GroupedAttentionShape, GroupedBackwardRecomputePass,
    WgpuGroupedBackwardRecomputePipeline,
};

const ATOL: f32 = 5.0e-4;
const RTOL: f32 = 2.5e-3;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn harness() -> Option<Harness> {
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
            panic!("M20 grouped backward stress gate requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional M20 grouped backward stress test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m20-grouped-backward-stress"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M20 stress request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.031 + phase;
            x.sin() * 0.63 + (x * 0.47).cos() * 0.37
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
        label: Some("flat-m20-grouped-backward-stress-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m20-grouped-backward-stress-readback"),
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

fn assert_close(case: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{case}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{case}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

fn run_case(
    harness: &Harness,
    pipeline: &WgpuGroupedBackwardRecomputePipeline,
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    phase: f32,
) {
    let q = fixture(shape.q_tensor_len().unwrap(), phase + 0.1);
    let k = fixture(shape.kv_tensor_len().unwrap(), phase + 0.7);
    let v = fixture(shape.kv_tensor_len().unwrap(), phase + 1.3);
    let d_out = fixture(shape.q_tensor_len().unwrap(), phase + 1.9);
    let forward = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
    let expected = backward_reference_grouped(&q, &k, &v, &d_out, shape, config, &forward).unwrap();
    let packed =
        pack_grouped_backward_recompute_inputs(&q, &k, &v, &d_out, &forward, shape).unwrap();
    let packed_gpu = initialized_buffer(
        &harness.device,
        &harness.queue,
        &packed,
        wgpu::BufferUsages::STORAGE,
        "flat-m20-grouped-backward-stress-forward",
    );
    let grads_gpu = pipeline
        .create_gradient_buffer(&harness.device, shape)
        .unwrap();

    for repetition in 0..3 {
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-m20-grouped-backward-stress"),
            });
        let layout = pipeline
            .encode(
                &harness.device,
                &mut encoder,
                GroupedBackwardRecomputePass {
                    packed_forward: &packed_gpu,
                    packed_grads: &grads_gpu,
                    shape,
                    config,
                },
            )
            .unwrap();
        harness.queue.submit(Some(encoder.finish()));
        let actual = read_f32(
            &harness.device,
            &harness.queue,
            &grads_gpu,
            layout.gradient_elements,
        );
        let dq_end = layout.q_elements;
        let dk_end = dq_end + layout.kv_elements;
        let tag = format!(
            "b{}-qh{}-kvh{}-s{}-d{}-causal{}-rep{}",
            shape.batch,
            shape.q_heads,
            shape.kv_heads,
            shape.seq_len,
            shape.head_dim,
            config.causal,
            repetition
        );
        assert_close(&format!("{tag}-dQ"), &actual[..dq_end], &expected.dq);
        assert_close(&format!("{tag}-dK"), &actual[dq_end..dk_end], &expected.dk);
        assert_close(&format!("{tag}-dV"), &actual[dk_end..], &expected.dv);
    }
}

#[test]
fn grouped_backward_stress_matrix_matches_oracle_repeatedly() {
    let Some(harness) = harness() else {
        return;
    };
    let pipeline = WgpuGroupedBackwardRecomputePipeline::new(&harness.device).unwrap();
    let cases = [
        (
            GroupedAttentionShape {
                batch: 1,
                q_heads: 1,
                kv_heads: 1,
                seq_len: 1,
                head_dim: 2,
            },
            false,
            None,
        ),
        (
            GroupedAttentionShape {
                batch: 1,
                q_heads: 4,
                kv_heads: 4,
                seq_len: 3,
                head_dim: 8,
            },
            true,
            None,
        ),
        (
            GroupedAttentionShape {
                batch: 2,
                q_heads: 8,
                kv_heads: 2,
                seq_len: 5,
                head_dim: 16,
            },
            false,
            Some(0.23),
        ),
        (
            GroupedAttentionShape {
                batch: 1,
                q_heads: 8,
                kv_heads: 1,
                seq_len: 9,
                head_dim: 32,
            },
            true,
            Some(0.17),
        ),
        (
            GroupedAttentionShape {
                batch: 2,
                q_heads: 12,
                kv_heads: 3,
                seq_len: 7,
                head_dim: 6,
            },
            true,
            None,
        ),
    ];

    for (index, (shape, causal, softmax_scale)) in cases.into_iter().enumerate() {
        run_case(
            &harness,
            &pipeline,
            shape,
            FlatAttentionConfig {
                causal,
                softmax_scale,
            },
            index as f32 * 0.37,
        );
    }
}
