use std::borrow::Cow;
use std::sync::mpsc;

use flat_ada_a1_candidate::ADA_A1_FWD_WGSL;
use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, FLAT_FWD_WGSL, WGSL_QUERY_ROWS,
};
use naga::valid::{Capabilities, ValidationFlags, Validator};

const ORACLE_ATOL: f32 = 1.0e-3;
const ORACLE_RTOL: f32 = 4.0e-3;
const AB_ATOL: f32 = 5.0e-5;
const AB_RTOL: f32 = 5.0e-4;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[derive(Clone, Copy)]
struct Case {
    shape: AttentionShape,
    config: FlatAttentionConfig,
    phase: f32,
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
            panic!("ADA-A1 GPU parity requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional ADA-A1 device parity skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-ada-a1-parity"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("ADA-A1 request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let bounded = u16::try_from(index).expect("qualification fixtures stay below u16::MAX");
            let x = f32::from(bounded) * 0.031 + phase;
            x.sin() * 0.63 + (x * 0.47).cos() * 0.27
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

fn input_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    label: &'static str,
) -> wgpu::Buffer {
    let bytes = bytes_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::try_from(bytes.len()).expect("fixture byte length fits u64"),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn create_pipeline(
    device: &wgpu::Device,
    source: &'static str,
    entry_point: Option<&'static str>,
    label: &'static str,
) -> wgpu::ComputePipeline {
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    if let Some(error) = pollster::block_on(error_scope.pop()) {
        panic!("{label} pipeline validation failed: {error}");
    }
    pipeline
}

fn read_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
    label: &'static str,
) -> Vec<f32> {
    let bytes = u64::try_from(len)
        .expect("readback length fits u64")
        .checked_mul(4)
        .expect("readback byte length fits u64");
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
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

fn run_shader(
    harness: &Harness,
    pipeline: &wgpu::ComputePipeline,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    shape: AttentionShape,
    config: FlatAttentionConfig,
) -> Vec<f32> {
    let label = "flat-ada-a1-dispatch";
    let tensor_len = shape.tensor_len().unwrap();
    let lse_len = shape.lse_len().unwrap();
    let combined_len = tensor_len.checked_add(lse_len).unwrap();
    let combined_bytes = u64::try_from(combined_len).unwrap().checked_mul(4).unwrap();
    let output = harness.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: combined_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let batch_heads = shape.batch.checked_mul(shape.heads).unwrap();
    let scale = config.resolved_scale(shape.head_dim).unwrap();
    let params = [
        u32::try_from(shape.seq_len).unwrap(),
        u32::try_from(shape.head_dim).unwrap(),
        u32::try_from(batch_heads).unwrap(),
        u32::from(config.causal),
        scale.to_bits(),
        0,
        0,
        0,
    ];
    let mut param_bytes = Vec::with_capacity(std::mem::size_of_val(&params));
    for value in params {
        param_bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let params_buffer = harness.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-ada-a1-params"),
        size: u64::try_from(param_bytes.len()).unwrap(),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    harness.queue.write_buffer(&params_buffer, 0, &param_bytes);

    let bind_group = harness
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let dispatch_x = shape.seq_len.div_ceil(WGSL_QUERY_ROWS);
        pass.dispatch_workgroups(
            u32::try_from(dispatch_x).unwrap(),
            u32::try_from(batch_heads).unwrap(),
            1,
        );
    }
    harness.queue.submit(Some(encoder.finish()));
    let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
    read_f32(
        &harness.device,
        &harness.queue,
        &output,
        combined_len,
        label,
    )
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        let tolerance = atol + rtol * actual.abs().max(expected.abs());
        assert!(
            actual.is_finite() && expected.is_finite() && error <= tolerance,
            "{name}: index={index} actual={actual} expected={expected} error={error} tolerance={tolerance}"
        );
    }
}

#[test]
fn ada_a1_shader_parses_and_validates_with_naga_020() {
    let module = naga::front::wgsl::parse_str(ADA_A1_FWD_WGSL)
        .unwrap_or_else(|error| panic!("ADA-A1 WGSL parse failed: {error:?}"));
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|error| panic!("ADA-A1 WGSL validation failed: {error:?}"));
}

#[test]
fn ada_a1_source_is_isolated_from_qualified_q4() {
    assert_ne!(ADA_A1_FWD_WGSL, FLAT_FWD_WGSL);
    assert!(ADA_A1_FWD_WGSL.contains("flat_attention_forward_ada_a1"));
    assert!(ADA_A1_FWD_WGSL.contains("score <= previous_max"));
    assert!(!ADA_A1_FWD_WGSL
        .contains("select(\n                            exp(previous_max - new_max)"));
}

#[test]
fn ada_a1_matches_cpu_oracle_and_qualified_q4_gpu() {
    let Some(harness) = harness() else {
        return;
    };
    let baseline = create_pipeline(
        &harness.device,
        FLAT_FWD_WGSL,
        Some("flat_attention_forward"),
        "flat-q4-qualified-baseline",
    );
    let candidate = create_pipeline(
        &harness.device,
        ADA_A1_FWD_WGSL,
        Some("flat_attention_forward_ada_a1"),
        "flat-ada-a1-candidate",
    );

    let cases = [
        Case {
            shape: AttentionShape {
                batch: 1,
                heads: 2,
                seq_len: 7,
                head_dim: 8,
            },
            config: FlatAttentionConfig {
                causal: false,
                softmax_scale: None,
            },
            phase: 0.13,
        },
        Case {
            shape: AttentionShape {
                batch: 1,
                heads: 2,
                seq_len: 9,
                head_dim: 64,
            },
            config: FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
            phase: 0.73,
        },
        Case {
            shape: AttentionShape {
                batch: 2,
                heads: 1,
                seq_len: 8,
                head_dim: 128,
            },
            config: FlatAttentionConfig {
                causal: false,
                softmax_scale: Some(0.03125),
            },
            phase: 1.37,
        },
    ];

    for case in cases {
        let len = case.shape.tensor_len().unwrap();
        let q = fixture(len, case.phase);
        let k = fixture(len, case.phase + 0.41);
        let v = fixture(len, case.phase + 0.97);
        let expected = forward_reference(&q, &k, &v, case.shape, case.config).unwrap();
        let mut expected_combined = expected.output;
        expected_combined.extend_from_slice(&expected.lse);

        let q_gpu = input_buffer(&harness.device, &harness.queue, &q, "flat-ada-a1-q");
        let k_gpu = input_buffer(&harness.device, &harness.queue, &k, "flat-ada-a1-k");
        let v_gpu = input_buffer(&harness.device, &harness.queue, &v, "flat-ada-a1-v");

        let baseline_actual = run_shader(
            &harness,
            &baseline,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            case.shape,
            case.config,
        );
        let candidate_actual = run_shader(
            &harness,
            &candidate,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            case.shape,
            case.config,
        );

        assert_close(
            "qualified Q4 vs CPU oracle",
            &baseline_actual,
            &expected_combined,
            ORACLE_ATOL,
            ORACLE_RTOL,
        );
        assert_close(
            "ADA-A1 vs CPU oracle",
            &candidate_actual,
            &expected_combined,
            ORACLE_ATOL,
            ORACLE_RTOL,
        );
        assert_close(
            "ADA-A1 vs qualified Q4",
            &candidate_actual,
            &baseline_actual,
            AB_ATOL,
            AB_RTOL,
        );
    }
}
