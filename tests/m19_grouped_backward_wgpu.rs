#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    backward_reference_grouped, forward_reference_grouped, FlatAttentionConfig,
    GroupedAttentionShape,
};

const SHADER: &str = include_str!("../shaders/flat_backward_grouped_recompute.wgsl");
const WORKGROUP_SIZE: usize = 64;
const ATOL: f32 = 4.0e-4;
const RTOL: f32 = 2.0e-3;

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
            panic!("M19 grouped backward requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M19 grouped backward test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m19-grouped-backward-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M19 request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.071 + phase;
            x.sin() * 0.75 + (x * 0.37).cos() * 0.25
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

fn bytes_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn initialized_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
    label: &'static str,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
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
        label: Some("flat-m19-grouped-backward-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m19-grouped-backward-readback"),
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

#[test]
fn grouped_backward_shader_parses_and_validates() {
    let module =
        naga::front::wgsl::parse_str(SHADER).expect("M19 grouped backward WGSL must parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("M19 grouped backward WGSL must validate");
}

fn run_case(kv_heads: usize, causal: bool) {
    let Some(harness) = harness() else {
        return;
    };
    let shape = GroupedAttentionShape {
        batch: 2,
        q_heads: 4,
        kv_heads,
        seq_len: 5,
        head_dim: 8,
    };
    let config = FlatAttentionConfig {
        causal,
        softmax_scale: Some(0.41),
    };
    let q_len = shape.q_tensor_len().unwrap();
    let kv_len = shape.kv_tensor_len().unwrap();
    let q = fixture(q_len, 0.1);
    let k = fixture(kv_len, 0.7);
    let v = fixture(kv_len, 1.3);
    let d_out = fixture(q_len, 1.9);
    let forward = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
    let expected = backward_reference_grouped(&q, &k, &v, &d_out, shape, config, &forward).unwrap();

    let mut packed_forward = Vec::with_capacity(3 * q_len + 2 * kv_len + forward.lse.len());
    packed_forward.extend_from_slice(&q);
    packed_forward.extend_from_slice(&k);
    packed_forward.extend_from_slice(&v);
    packed_forward.extend_from_slice(&d_out);
    packed_forward.extend_from_slice(&forward.output);
    packed_forward.extend_from_slice(&forward.lse);
    let gradient_len = q_len + 2 * kv_len;

    let shader = harness
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m19-grouped-backward"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });
    let pipeline = harness
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m19-grouped-backward"),
            layout: None,
            module: &shader,
            entry_point: "flat_attention_backward_grouped",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

    let packed_gpu = initialized_buffer(
        &harness.device,
        &harness.queue,
        &bytes_f32(&packed_forward),
        wgpu::BufferUsages::STORAGE,
        "flat-m19-grouped-packed-forward",
    );
    let grads_gpu = harness.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m19-grouped-gradients"),
        size: (gradient_len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [
        u32::try_from(shape.batch).unwrap(),
        u32::try_from(shape.q_heads).unwrap(),
        u32::try_from(shape.kv_heads).unwrap(),
        u32::try_from(shape.seq_len).unwrap(),
        u32::try_from(shape.head_dim).unwrap(),
        u32::from(causal),
        config.resolved_scale(shape.head_dim).unwrap().to_bits(),
        u32::try_from(q_len).unwrap(),
        u32::try_from(kv_len).unwrap(),
        u32::try_from(forward.lse.len()).unwrap(),
    ];
    let params_gpu = initialized_buffer(
        &harness.device,
        &harness.queue,
        &bytes_u32(&params),
        wgpu::BufferUsages::UNIFORM,
        "flat-m19-grouped-params",
    );
    let bind_group = harness
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-m19-grouped-backward-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: packed_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grads_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_gpu.as_entire_binding(),
                },
            ],
        });
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m19-grouped-backward-dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-m19-grouped-backward"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            u32::try_from(gradient_len.div_ceil(WORKGROUP_SIZE)).unwrap(),
            1,
            1,
        );
    }
    harness.queue.submit(Some(encoder.finish()));

    let actual = read_f32(&harness.device, &harness.queue, &grads_gpu, gradient_len);
    assert_close("dQ", &actual[..q_len], &expected.dq);
    assert_close("dK", &actual[q_len..q_len + kv_len], &expected.dk);
    assert_close("dV", &actual[q_len + kv_len..], &expected.dv);
}

#[test]
fn grouped_backward_gqa_matches_m19_oracle() {
    run_case(2, true);
}

#[test]
fn grouped_backward_mqa_matches_m19_oracle() {
    run_case(1, false);
}
