#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    backward_reference, forward_reference, pack_backward_recompute_inputs, AttentionShape,
    BackwardRecomputePass, FlatAttentionConfig, WgpuBackwardRecomputePipeline,
};

fn harness() -> Option<(wgpu::Device, wgpu::Queue)> {
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
            panic!("M18 host pipeline requires a WGPU adapter in the mandatory device gate");
        }
        return None;
    };
    Some(
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flat-m18-host-pipeline-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .unwrap(),
    )
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
        label: Some("flat-m18-host-packed-input"),
        size: bytes.len() as u64,
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
        label: Some("flat-m18-host-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m18-host-readback"),
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
fn public_pipeline_matches_backward_oracle() {
    let Some((device, queue)) = harness() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 3,
        head_dim: 4,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: Some(0.57),
    };
    let tensor_elements = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let fixture = |phase: f32| {
        (0..tensor_elements)
            .map(|index| {
                let x = index as f32 * 0.083 + phase;
                x.sin() * 0.625 + (x * 0.31).cos() * 0.25
            })
            .collect::<Vec<_>>()
    };
    let q = fixture(0.1);
    let k = fixture(0.7);
    let v = fixture(1.2);
    let d_out = fixture(1.9);
    let forward = forward_reference(&q, &k, &v, shape, config).unwrap();
    let expected = backward_reference(&q, &k, &v, &d_out, shape, config, &forward).unwrap();
    let packed = pack_backward_recompute_inputs(&q, &k, &v, &d_out, &forward, shape).unwrap();

    let pipeline = WgpuBackwardRecomputePipeline::new(&device).unwrap();
    let layout = WgpuBackwardRecomputePipeline::layout(shape).unwrap();
    assert_eq!(packed.len(), layout.packed_forward_elements);
    assert_eq!(layout.k_offset(), tensor_elements);
    assert_eq!(layout.lse_offset(), 5 * tensor_elements);
    assert_eq!(layout.dv_offset(), 2 * tensor_elements);

    let packed_gpu = input_buffer(&device, &queue, &packed);
    let gradients = pipeline.create_gradient_buffer(&device, shape).unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m18-public-backward"),
    });
    pipeline
        .encode(
            &device,
            &mut encoder,
            BackwardRecomputePass {
                packed_forward: &packed_gpu,
                packed_grads: &gradients,
                shape,
                config,
            },
        )
        .unwrap();
    queue.submit(Some(encoder.finish()));

    let actual = read_f32(&device, &queue, &gradients, layout.gradient_elements);
    let expected_all = expected
        .dq
        .iter()
        .chain(&expected.dk)
        .chain(&expected.dv)
        .copied()
        .collect::<Vec<_>>();
    for (index, (&actual, &expected)) in actual.iter().zip(&expected_all).enumerate() {
        let tolerance = 4.0e-4 + 2.0e-3 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "gradient[{index}] actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
