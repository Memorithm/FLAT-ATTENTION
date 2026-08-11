#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    backward_reference, forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionOutput,
};

const SHADER: &str = include_str!("../shaders/flat_backward_recompute.wgsl");
const WORKGROUP_SIZE: usize = 64;
const ATOL: f32 = 4.0e-4;
const RTOL: f32 = 2.0e-3;

struct DeviceHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn harness() -> Option<DeviceHarness> {
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
            panic!("M18 requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M18 device test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m18-backward-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M18 request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.071 + phase;
            x.sin() * 0.75 + (x * 0.43).cos() * 0.1875
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

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn input_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m18-packed-forward"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn uniform_buffer(device: &wgpu::Device, values: &[u32]) -> wgpu::Buffer {
    let bytes = encode_u32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m18-params"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped.copy_from_slice(&bytes);
    }
    buffer.unmap();
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
        label: Some("flat-m18-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m18-readback"),
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

fn pack_forward_contract(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    forward: &FlatAttentionOutput,
) -> Vec<f32> {
    let mut packed = Vec::with_capacity(
        q.len() + k.len() + v.len() + d_out.len() + forward.output.len() + forward.lse.len(),
    );
    packed.extend_from_slice(q);
    packed.extend_from_slice(k);
    packed.extend_from_slice(v);
    packed.extend_from_slice(d_out);
    packed.extend_from_slice(&forward.output);
    packed.extend_from_slice(&forward.lse);
    packed
}

fn qualify_case(harness: &DeviceHarness, causal: bool) {
    let shape = AttentionShape {
        batch: 2,
        heads: 2,
        seq_len: 3,
        head_dim: 4,
    };
    let config = FlatAttentionConfig {
        causal,
        softmax_scale: Some(0.61),
    };
    let tensor_elements = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let lse_elements = shape.batch * shape.heads * shape.seq_len;
    let q = fixture(tensor_elements, 0.1);
    let k = fixture(tensor_elements, 0.7);
    let v = fixture(tensor_elements, 1.3);
    let d_out = fixture(tensor_elements, 2.1);
    let forward = forward_reference(&q, &k, &v, shape, config).unwrap();
    let expected = backward_reference(&q, &k, &v, &d_out, shape, config, &forward).unwrap();
    let packed = pack_forward_contract(&q, &k, &v, &d_out, &forward);

    let packed_gpu = input_buffer(&harness.device, &harness.queue, &packed);
    let gradient_elements = 3 * tensor_elements;
    let gradients_gpu = harness.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m18-gradients"),
        size: (gradient_elements * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = uniform_buffer(
        &harness.device,
        &[
            shape.batch as u32,
            shape.heads as u32,
            shape.seq_len as u32,
            shape.head_dim as u32,
            u32::from(causal),
            config.resolved_scale(shape.head_dim).unwrap().to_bits(),
            tensor_elements as u32,
            lse_elements as u32,
        ],
    );

    harness.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = harness.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flat-m18-backward-recompute"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
    });
    let pipeline = harness
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m18-backward-recompute"),
            layout: None,
            module: &module,
            entry_point: "flat_attention_backward",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
    if let Some(error) = pollster::block_on(harness.device.pop_error_scope()) {
        panic!("M18 pipeline validation failed: {error}");
    }

    let bind_group = harness.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("flat-m18-backward-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: packed_gpu.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gradients_gpu.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m18-backward-dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-m18-backward-recompute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(gradient_elements.div_ceil(WORKGROUP_SIZE) as u32, 1, 1);
    }
    harness.queue.submit(Some(encoder.finish()));

    let actual = read_f32(
        &harness.device,
        &harness.queue,
        &gradients_gpu,
        gradient_elements,
    );
    assert_close("M18 dQ", &actual[..tensor_elements], &expected.dq);
    assert_close(
        "M18 dK",
        &actual[tensor_elements..2 * tensor_elements],
        &expected.dk,
    );
    assert_close(
        "M18 dV",
        &actual[2 * tensor_elements..],
        &expected.dv,
    );
}

#[test]
fn backward_shader_parses_and_validates() {
    let module = naga::front::wgsl::parse_str(SHADER).expect("M18 WGSL must parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).expect("M18 WGSL must validate");
}

#[test]
fn recomputation_backward_matches_m17_oracle() {
    let Some(harness) = harness() else {
        return;
    };
    qualify_case(&harness, false);
    qualify_case(&harness, true);
}
