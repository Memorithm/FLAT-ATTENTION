#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    backward_reference_grouped, forward_reference_grouped, FlatAttentionConfig,
    GroupedAttentionShape, GroupedBackwardRecomputePass, GroupedForwardPass,
    WgpuGroupedBackwardRecomputePipeline, WgpuGroupedForwardPipeline,
};

const ATOL: f32 = 6.0e-4;
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
            panic!("M26 public resident training chain requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M26 public resident training-chain test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m26-public-resident-training-chain"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M26 request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.043 + phase;
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

fn empty_buffer(
    device: &wgpu::Device,
    elements: usize,
    usage: wgpu::BufferUsages,
    label: &'static str,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (elements * std::mem::size_of::<f32>()).max(4) as u64,
        usage,
        mapped_at_creation: false,
    })
}

fn read_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    let bytes = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m26-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m26-readback"),
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

fn run_case(shape: GroupedAttentionShape, config: FlatAttentionConfig) {
    let Some(harness) = harness() else {
        return;
    };
    let q_len = shape.q_tensor_len().unwrap();
    let kv_len = shape.kv_tensor_len().unwrap();
    let lse_len = shape.lse_len().unwrap();

    let q = fixture(q_len, 0.1);
    let k = fixture(kv_len, 0.7);
    let v = fixture(kv_len, 1.3);
    let d_out = fixture(q_len, 1.9);

    let expected_forward = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
    let expected_backward =
        backward_reference_grouped(&q, &k, &v, &d_out, shape, config, &expected_forward).unwrap();

    let resident_input_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let q_gpu = initialized_buffer(
        &harness.device,
        &harness.queue,
        &q,
        resident_input_usage,
        "flat-m26-q",
    );
    let k_gpu = initialized_buffer(
        &harness.device,
        &harness.queue,
        &k,
        resident_input_usage,
        "flat-m26-k",
    );
    let v_gpu = initialized_buffer(
        &harness.device,
        &harness.queue,
        &v,
        resident_input_usage,
        "flat-m26-v",
    );
    let d_out_gpu = initialized_buffer(
        &harness.device,
        &harness.queue,
        &d_out,
        wgpu::BufferUsages::COPY_SRC,
        "flat-m26-do",
    );

    let forward_pipeline = WgpuGroupedForwardPipeline::new(&harness.device).unwrap();
    let forward_gpu = forward_pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();

    let backward_pipeline = WgpuGroupedBackwardRecomputePipeline::new(&harness.device).unwrap();
    let backward_layout = WgpuGroupedBackwardRecomputePipeline::layout(shape).unwrap();
    let packed_gpu = empty_buffer(
        &harness.device,
        backward_layout.packed_forward_elements,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "flat-m26-packed-forward",
    );
    let grads_gpu = backward_pipeline
        .create_gradient_buffer(&harness.device, shape)
        .unwrap();

    let f32_bytes = std::mem::size_of::<f32>() as u64;
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m26-public-forward-backward-chain"),
        });

    let forward_layout = forward_pipeline
        .encode(
            &harness.device,
            &mut encoder,
            GroupedForwardPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &forward_gpu,
                shape,
                config,
            },
        )
        .unwrap();

    encoder.copy_buffer_to_buffer(&q_gpu, 0, &packed_gpu, 0, q_len as u64 * f32_bytes);
    encoder.copy_buffer_to_buffer(
        &k_gpu,
        0,
        &packed_gpu,
        backward_layout.k_offset() as u64 * f32_bytes,
        kv_len as u64 * f32_bytes,
    );
    encoder.copy_buffer_to_buffer(
        &v_gpu,
        0,
        &packed_gpu,
        backward_layout.v_offset() as u64 * f32_bytes,
        kv_len as u64 * f32_bytes,
    );
    encoder.copy_buffer_to_buffer(
        &d_out_gpu,
        0,
        &packed_gpu,
        backward_layout.d_out_offset() as u64 * f32_bytes,
        q_len as u64 * f32_bytes,
    );
    encoder.copy_buffer_to_buffer(
        &forward_gpu,
        forward_layout.output_offset() as u64 * f32_bytes,
        &packed_gpu,
        backward_layout.output_offset() as u64 * f32_bytes,
        q_len as u64 * f32_bytes,
    );
    encoder.copy_buffer_to_buffer(
        &forward_gpu,
        forward_layout.lse_offset() as u64 * f32_bytes,
        &packed_gpu,
        backward_layout.lse_offset() as u64 * f32_bytes,
        lse_len as u64 * f32_bytes,
    );

    backward_pipeline
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
    let _ = harness.device.poll(wgpu::Maintain::Wait);

    let actual_forward = read_f32(
        &harness.device,
        &harness.queue,
        &forward_gpu,
        forward_layout.output_elements,
    );
    assert_close(
        "O",
        &actual_forward[forward_layout.output_offset()..forward_layout.lse_offset()],
        &expected_forward.output,
    );
    assert_close(
        "LSE",
        &actual_forward[forward_layout.lse_offset()..],
        &expected_forward.lse,
    );

    let actual_grads = read_f32(
        &harness.device,
        &harness.queue,
        &grads_gpu,
        backward_layout.gradient_elements,
    );
    let dq_end = backward_layout.q_elements;
    let dk_end = dq_end + backward_layout.kv_elements;
    assert_close("dQ", &actual_grads[..dq_end], &expected_backward.dq);
    assert_close("dK", &actual_grads[dq_end..dk_end], &expected_backward.dk);
    assert_close("dV", &actual_grads[dk_end..], &expected_backward.dv);
}

#[test]
fn public_resident_gqa_training_chain_matches_oracle() {
    run_case(
        GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 7,
            head_dim: 16,
        },
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        },
    );
}

#[test]
fn public_resident_mqa_training_chain_matches_oracle() {
    run_case(
        GroupedAttentionShape {
            batch: 2,
            q_heads: 4,
            kv_heads: 1,
            seq_len: 5,
            head_dim: 8,
        },
        FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(0.31),
        },
    );
}
