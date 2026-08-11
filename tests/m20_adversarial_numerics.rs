#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    backward_reference_grouped, forward_reference_grouped, pack_grouped_backward_recompute_inputs,
    FlatAttentionConfig, GroupedAttentionShape, GroupedBackwardRecomputePass,
    WgpuGroupedBackwardRecomputePipeline,
};

const ATOL: f32 = 1.0e-3;
const RTOL: f32 = 5.0e-3;

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
            panic!("M20 adversarial numerical gate requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional M20 adversarial numerical test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m20-adversarial-numerics"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M20 adversarial request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn adversarial_fixture(len: usize, phase: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let bucket = ((index * 17 + phase * 11) % 23) as i32 - 11;
            let sign = if (index + phase).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            sign * bucket as f32 * 0.19 + ((index + phase) as f32 * 0.013).sin() * 0.07
        })
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
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
    label: &'static str,
) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        label: Some("flat-m20-adversarial-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m20-adversarial-readback"),
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

fn assert_finite_and_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "{name}[{index}] is not finite: {actual}"
        );
        assert!(
            expected.is_finite(),
            "{name}[{index}] oracle is not finite: {expected}"
        );
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

fn run_case(
    harness: &Harness,
    pipeline: &WgpuGroupedBackwardRecomputePipeline,
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    phase: usize,
) {
    let q = adversarial_fixture(shape.q_tensor_len().unwrap(), phase + 1);
    let k = adversarial_fixture(shape.kv_tensor_len().unwrap(), phase + 3);
    let v = adversarial_fixture(shape.kv_tensor_len().unwrap(), phase + 5);
    let d_out = adversarial_fixture(shape.q_tensor_len().unwrap(), phase + 7);
    let forward = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
    assert!(forward.output.iter().all(|value| value.is_finite()));
    assert!(forward.lse.iter().all(|value| value.is_finite()));
    let expected = backward_reference_grouped(&q, &k, &v, &d_out, shape, config, &forward).unwrap();
    let packed =
        pack_grouped_backward_recompute_inputs(&q, &k, &v, &d_out, &forward, shape).unwrap();
    let packed_gpu = initialized_buffer(
        &harness.device,
        &harness.queue,
        &packed,
        "flat-m20-adversarial-forward",
    );
    let grads_gpu = pipeline
        .create_gradient_buffer(&harness.device, shape)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m20-adversarial-numerics"),
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
    assert_finite_and_close("dQ", &actual[..dq_end], &expected.dq);
    assert_finite_and_close("dK", &actual[dq_end..dk_end], &expected.dk);
    assert_finite_and_close("dV", &actual[dk_end..], &expected.dv);
}

#[test]
fn grouped_backward_handles_adversarial_but_finite_inputs() {
    let Some(harness) = harness() else {
        return;
    };
    let pipeline = WgpuGroupedBackwardRecomputePipeline::new(&harness.device).unwrap();
    let cases = [
        (
            GroupedAttentionShape {
                batch: 1,
                q_heads: 8,
                kv_heads: 2,
                seq_len: 17,
                head_dim: 64,
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
                kv_heads: 1,
                seq_len: 33,
                head_dim: 32,
            },
            FlatAttentionConfig {
                causal: false,
                softmax_scale: Some(0.31),
            },
        ),
    ];

    for (index, (shape, config)) in cases.into_iter().enumerate() {
        run_case(&harness, &pipeline, shape, config, index * 13);
    }
}
