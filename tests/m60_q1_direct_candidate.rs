#![cfg(feature = "wgpu")]

use std::borrow::Cow;
use std::error::Error;
use std::sync::mpsc;

use flat_attention::{
    forward_reference_grouped, FlatAttentionConfig, GroupedAttentionShape, GroupedForwardLayout,
    GroupedForwardPass, WgpuGroupedForwardPipeline,
};

const SHADER: &str = include_str!("../shaders/flat_fwd_q1_direct_vec4.wgsl");
const ATOL: f32 = 7.0e-4;
const RTOL: f32 = 3.0e-3;

struct Candidate {
    pipeline: wgpu::ComputePipeline,
}

struct Prepared {
    layout: GroupedForwardLayout,
    bind_group: wgpu::BindGroup,
    dispatch_x: u32,
    dispatch_y: u32,
    _params: wgpu::Buffer,
}

impl Candidate {
    fn new(device: &wgpu::Device) -> Result<Self, Box<dyn Error>> {
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m60-q1-direct"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m60-q1-direct"),
            layout: None,
            module: &shader,
            entry_point: Some("flat_attention_forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(format!("M60 pipeline validation failed: {error}").into());
        }
        Ok(Self { pipeline })
    }

    fn validate_shape(shape: GroupedAttentionShape) -> Result<GroupedForwardLayout, Box<dyn Error>> {
        if shape.q_heads != shape.kv_heads {
            return Err(format!(
                "M60 requires MHA q_heads == kv_heads, got {} and {}",
                shape.q_heads, shape.kv_heads
            )
            .into());
        }
        if !matches!(shape.head_dim, 64 | 128) {
            return Err(format!("M60 supports only D64/D128, got {}", shape.head_dim).into());
        }
        Ok(WgpuGroupedForwardPipeline::layout(shape)?)
    }

    fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        shape: GroupedAttentionShape,
    ) -> Result<wgpu::Buffer, Box<dyn Error>> {
        let layout = Self::validate_shape(shape)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m60-q1-direct-o-lse"),
            size: layout.output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    fn prepare(
        &self,
        device: &wgpu::Device,
        pass: GroupedForwardPass<'_>,
    ) -> Result<Prepared, Box<dyn Error>> {
        let layout = Self::validate_shape(pass.shape)?;
        for (name, buffer, required) in [
            ("Q", pass.q, layout.q_bytes),
            ("K", pass.k, layout.kv_bytes),
            ("V", pass.v, layout.kv_bytes),
            ("O|LSE", pass.output, layout.output_bytes),
        ] {
            if buffer.size() < required {
                return Err(format!(
                    "M60 {name} buffer has {} bytes, requires {required}",
                    buffer.size()
                )
                .into());
            }
        }

        let dispatch_x = u32::try_from(pass.shape.seq_len)?;
        let batch_heads = pass
            .shape
            .batch
            .checked_mul(pass.shape.q_heads)
            .ok_or("batch-head overflow")?;
        let dispatch_y = u32::try_from(batch_heads)?;
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        if dispatch_x > maximum || dispatch_y > maximum {
            return Err("M60 dispatch exceeds device workgroup limits".into());
        }

        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;
        let values = [
            u32::try_from(pass.shape.seq_len)?,
            u32::try_from(pass.shape.head_dim)?,
            dispatch_y,
            u32::from(pass.config.causal),
            scale.to_bits(),
            0,
            0,
            0,
        ];
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(&values));
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m60-q1-direct-params"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        {
            let mut mapped = params.slice(..).get_mapped_range_mut()?;
            mapped.copy_from_slice(&bytes);
        }
        params.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-m60-q1-direct-bind-group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pass.q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pass.k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pass.v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pass.output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        Ok(Prepared {
            layout,
            bind_group,
            dispatch_x,
            dispatch_y,
            _params: params,
        })
    }

    fn encode_prepared(&self, encoder: &mut wgpu::CommandEncoder, prepared: &Prepared) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-m60-q1-direct"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &prepared.bind_group, &[]);
        pass.dispatch_workgroups(prepared.dispatch_x, prepared.dispatch_y, 1);
    }
}

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
            panic!("M60 direct-load qualification requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional M60 device test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-m60-q1-direct-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("M60 request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.039 + phase;
            x.sin() * 0.67 + (x * 0.43).cos() * 0.33
        })
        .collect()
}

fn storage(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    label: &'static str,
) -> wgpu::Buffer {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
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
        label: Some("flat-m60-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m60-readback"),
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

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        let tolerance = ATOL + RTOL * actual.abs().max(expected.abs());
        assert!(
            actual.is_finite() && error <= tolerance,
            "index={index} actual={actual} expected={expected} error={error} tolerance={tolerance}"
        );
    }
}

fn run_case(harness: &Harness, head_dim: usize, causal: bool) {
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 2,
        kv_heads: 2,
        seq_len: 9,
        head_dim,
    };
    let config = FlatAttentionConfig {
        causal,
        softmax_scale: None,
    };
    let len = shape.q_tensor_len().unwrap();
    let q = fixture(len, 0.2);
    let k = fixture(len, 0.8);
    let v = fixture(len, 1.4);
    let expected = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
    let q_gpu = storage(&harness.device, &harness.queue, &q, "flat-m60-q");
    let k_gpu = storage(&harness.device, &harness.queue, &k, "flat-m60-k");
    let v_gpu = storage(&harness.device, &harness.queue, &v, "flat-m60-v");
    let candidate = Candidate::new(&harness.device).unwrap();
    let output = candidate
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let prepared = candidate
        .prepare(
            &harness.device,
            GroupedForwardPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &output,
                shape,
                config,
            },
        )
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m60-q1-direct"),
        });
    candidate.encode_prepared(&mut encoder, &prepared);
    harness.queue.submit(Some(encoder.finish()));
    let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
    let actual = read_f32(
        &harness.device,
        &harness.queue,
        &output,
        prepared.layout.output_elements,
    );
    assert_close(&actual[..prepared.layout.lse_offset()], &expected.output);
    assert_close(&actual[prepared.layout.lse_offset()..], &expected.lse);
}

#[test]
fn shader_removes_non_reused_kv_workgroup_staging() {
    assert!(!SHADER.contains("k_shared"));
    assert!(!SHADER.contains("v_shared"));
    assert!(SHADER.contains("let k_slice = k[global4]"));
    assert!(SHADER.contains("v_slice = v[global4]"));
}

#[test]
fn candidate_rejects_gqa_and_non_vec4_dimensions() {
    let gqa = GroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        seq_len: 7,
        head_dim: 64,
    };
    assert!(Candidate::validate_shape(gqa).is_err());
    let d32 = GroupedAttentionShape {
        batch: 1,
        q_heads: 2,
        kv_heads: 2,
        seq_len: 7,
        head_dim: 32,
    };
    assert!(Candidate::validate_shape(d32).is_err());
}

#[test]
fn q1_direct_matches_grouped_oracle_for_d64_d128() {
    let Some(harness) = harness() else {
        return;
    };
    for head_dim in [64, 128] {
        for causal in [false, true] {
            run_case(&harness, head_dim, causal);
        }
    }
}
