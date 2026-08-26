//! Device-level qualification of Kernel-IR-generated WGSL (roadmap M21).
//!
//! For every generated dense Q4 configuration this suite proves:
//!
//! 1. pipeline creation succeeds under validation error scopes;
//! 2. O and LSE match the deterministic scalar oracle;
//! 3. generated output matches the corresponding qualified handwritten
//!    shader's output on identical inputs (tight cross-tolerance).
//!
//! This is correctness qualification only; no performance claim is made and
//! software-adapter timings are not promoted anywhere.

#![cfg(feature = "wgpu")]

use std::borrow::Cow;
use std::error::Error;

use flat_attention::kernel_ir::{AttentionProblem, KernelConfig, KernelFamily, KernelModule};
use flat_attention::kernel_wgsl::{emit, GENERATED_ENTRY_POINT};
use flat_attention::{forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionOutput};

const FLAT_FWD_WGSL: &str = include_str!("../shaders/flat_fwd.wgsl");
const FLAT_FWD_VEC4_WGSL: &str = include_str!("../shaders/flat_fwd_vec4.wgsl");
const FLAT_FWD_DOUBLE_BUFFER_WGSL: &str = include_str!("../shaders/flat_fwd_double_buffer.wgsl");

// Oracle tolerances follow the qualified device-parity suite.
const O_ATOL: f32 = 2.0e-5;
const O_RTOL: f32 = 2.0e-4;
const LSE_ATOL: f32 = 3.0e-5;
const LSE_RTOL: f32 = 3.0e-4;
// Generated-vs-handwritten cross tolerance: same algorithm and operation
// order, but the sources are distinct text going through the same backend
// compiler. Tighter than oracle tolerances by design.
const CROSS_ATOL: f32 = 1.0e-6;
const CROSS_RTOL: f32 = 1.0e-5;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    subgroup_supported: bool,
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
            panic!("generated-kernel qualification requires a WGPU adapter");
        }
        eprintln!("WGPU adapter unavailable; optional generated-kernel test skipped");
        return None;
    };
    let subgroup_supported = adapter.features().contains(wgpu::Features::SUBGROUP);
    let required_features = if subgroup_supported {
        wgpu::Features::SUBGROUP
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-generated-q4-test"),
        required_features,
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("generated-kernel request_device failed: {error}"));
    Some(Harness {
        device,
        queue,
        subgroup_supported,
    })
}

struct Runner {
    pipeline: wgpu::ComputePipeline,
}

impl Runner {
    fn new(device: &wgpu::Device, source: &str, label: &str) -> Result<Self, Box<dyn Error>> {
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source.to_string())),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &shader,
            entry_point: Some(GENERATED_ENTRY_POINT),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(format!("{label} pipeline validation failed: {error}").into());
        }
        Ok(Self { pipeline })
    }

    fn run(
        &self,
        harness: &Harness,
        shape: AttentionShape,
        config: FlatAttentionConfig,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<FlatAttentionOutput, Box<dyn Error>> {
        let device = &harness.device;
        let o_lse_len = shape.tensor_len()? + shape.lse_len()?;
        let storage =
            |values: &[f32], label: &'static str| -> wgpu::Buffer { staged(device, values, label) };
        let q_buf = storage(q, "gen-q");
        let k_buf = storage(k, "gen-k");
        let v_buf = storage(v, "gen-v");
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gen-o-lse"),
            size: (o_lse_len * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scale = config.resolved_scale(shape.head_dim)?;
        let batch_heads = shape
            .batch
            .checked_mul(shape.heads)
            .ok_or("batch-head overflow")?;
        let params_values: [u32; 8] = [
            u32::try_from(shape.seq_len)?,
            u32::try_from(shape.head_dim)?,
            u32::try_from(batch_heads)?,
            u32::from(config.causal),
            scale.to_bits(),
            0,
            0,
            0,
        ];
        let mut params_bytes = Vec::with_capacity(32);
        for value in params_values {
            params_bytes.extend_from_slice(&value.to_ne_bytes());
        }
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gen-params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        {
            let mut mapped = params
                .slice(..)
                .get_mapped_range_mut()
                .expect("params map failed");
            mapped.copy_from_slice(&params_bytes);
        }
        params.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gen-bind-group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let dispatch_x = shape.seq_len.div_ceil(4) as u32;
        let dispatch_y = u32::try_from(batch_heads)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gen-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gen-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gen-readback"),
            size: out.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, out.size());
        harness.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result)
                .expect("readback callback after receiver dropped")
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("map callback lost")
            .expect("generated readback map failed");
        let data = slice.get_mapped_range().expect("readback range map failed");
        let mut scalars = Vec::with_capacity(o_lse_len);
        for chunk in data.chunks_exact(4) {
            scalars.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        drop(data);
        readback.unmap();

        let o_len = shape.tensor_len()?;
        Ok(FlatAttentionOutput {
            output: scalars[..o_len].to_vec(),
            lse: scalars[o_len..].to_vec(),
        })
    }
}

/// Create a STORAGE|COPY_SRC buffer pre-filled with native-endian f32 values.
fn staged(device: &wgpu::Device, values: &[f32], label: &'static str) -> wgpu::Buffer {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (bytes.len() as u64).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    if !bytes.is_empty() {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("staged buffer map failed");
        mapped.copy_from_slice(&bytes);
    }
    buffer.unmap();
    buffer
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite(), "{name}[{index}] is not finite: {a}");
        let error = (a - b).abs();
        let tolerance = atol + rtol * b.abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={a}, expected={b}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.053 + phase;
            x.sin() * 0.71 + (x * 0.37).cos() * 0.29
        })
        .collect()
}

struct Case {
    name: &'static str,
    config: KernelConfig,
    handwritten: &'static str,
    handwritten_label: &'static str,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    causal: bool,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "scalar D64 causal",
            config: KernelConfig::PORTABLE_SCALAR,
            handwritten: FLAT_FWD_WGSL,
            handwritten_label: "handwritten-flat_fwd",
            batch: 2,
            heads: 4,
            seq_len: 129,
            head_dim: 64,
            causal: true,
        },
        Case {
            name: "scalar D80 non-causal tail",
            config: KernelConfig::PORTABLE_SCALAR,
            handwritten: FLAT_FWD_WGSL,
            handwritten_label: "handwritten-flat_fwd",
            batch: 1,
            heads: 2,
            seq_len: 17,
            head_dim: 80,
            causal: false,
        },
        Case {
            name: "vec4 D64 causal",
            config: KernelConfig::PORTABLE_VEC4,
            handwritten: FLAT_FWD_VEC4_WGSL,
            handwritten_label: "handwritten-flat_fwd_vec4",
            batch: 2,
            heads: 4,
            seq_len: 129,
            head_dim: 64,
            causal: true,
        },
        Case {
            name: "double-buffered D128 causal",
            config: KernelConfig::DOUBLE_BUFFERED_VEC4,
            handwritten: FLAT_FWD_DOUBLE_BUFFER_WGSL,
            handwritten_label: "handwritten-double-buffer",
            batch: 1,
            heads: 2,
            seq_len: 70,
            head_dim: 128,
            causal: true,
        },
    ]
}

#[test]
fn generated_kernels_match_oracle_and_handwritten_counterparts() {
    let Some(harness) = harness() else {
        return;
    };
    for case in cases() {
        let shape = AttentionShape {
            batch: case.batch,
            heads: case.heads,
            seq_len: case.seq_len,
            head_dim: case.head_dim,
        };
        let config = FlatAttentionConfig {
            causal: case.causal,
            softmax_scale: None,
        };
        let problem = AttentionProblem::from_shape(&shape, config)
            .unwrap_or_else(|err| panic!("{}: problem build failed: {err}", case.name));
        let module = KernelModule::build(KernelFamily::DenseQ4Forward, problem, case.config)
            .unwrap_or_else(|err| panic!("{}: IR build failed: {err}", case.name));
        let generated =
            emit(&module).unwrap_or_else(|err| panic!("{}: emission failed: {err}", case.name));

        let runner = Runner::new(&harness.device, &generated.source, case.name)
            .unwrap_or_else(|err| panic!("{}: {err}", case.name));
        let handwritten = Runner::new(&harness.device, case.handwritten, case.handwritten_label)
            .unwrap_or_else(|err| panic!("{}: handwritten setup failed: {err}", case.name));

        let len = shape.tensor_len().unwrap();
        let q = fixture(len, 0.11);
        let k = fixture(len, 0.47);
        let v = fixture(len, 0.83);

        let expected = forward_reference(&q, &k, &v, shape, config)
            .unwrap_or_else(|err| panic!("{}: oracle failed: {err}", case.name));
        let actual = runner
            .run(&harness, shape, config, &q, &k, &v)
            .unwrap_or_else(|err| panic!("{}: generated run failed: {err}", case.name));

        assert_close("O", &actual.output, &expected.output, O_ATOL, O_RTOL);
        assert_close("LSE", &actual.lse, &expected.lse, LSE_ATOL, LSE_RTOL);

        let reference = handwritten
            .run(&harness, shape, config, &q, &k, &v)
            .unwrap_or_else(|err| panic!("{}: handwritten run failed: {err}", case.name));
        assert_close(
            "O-vs-handwritten",
            &actual.output,
            &reference.output,
            CROSS_ATOL,
            CROSS_RTOL,
        );
        assert_close(
            "LSE-vs-handwritten",
            &actual.lse,
            &reference.lse,
            CROSS_ATOL,
            CROSS_RTOL,
        );
    }
}

#[test]
fn generated_subgroup_kernel_matches_oracle_when_supported() {
    let Some(harness) = harness() else {
        return;
    };
    if !harness.subgroup_supported {
        if std::env::var_os("FLAT_REQUIRE_SUBGROUP").is_some() {
            panic!("subgroup qualification requested but adapter lacks Features::SUBGROUP");
        }
        eprintln!("adapter lacks subgroup support; generated subgroup test skipped");
        return;
    }
    let shape = AttentionShape {
        batch: 2,
        heads: 4,
        seq_len: 129,
        head_dim: 64,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let problem = AttentionProblem::from_shape(&shape, config).unwrap();
    let module = KernelModule::build(
        KernelFamily::DenseQ4Forward,
        problem,
        KernelConfig::SUBGROUP_ASSISTED,
    )
    .unwrap();
    let generated = emit(&module).unwrap();
    let runner = Runner::new(&harness.device, &generated.source, "gen-subgroup").unwrap();

    let len = shape.tensor_len().unwrap();
    let q = fixture(len, 0.19);
    let k = fixture(len, 0.53);
    let v = fixture(len, 0.91);
    let expected = forward_reference(&q, &k, &v, shape, config).unwrap();
    let actual = runner.run(&harness, shape, config, &q, &k, &v).unwrap();
    assert_close("O", &actual.output, &expected.output, O_ATOL, O_RTOL);
    assert_close("LSE", &actual.lse, &expected.lse, LSE_ATOL, LSE_RTOL);
}
