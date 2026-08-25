#[cfg(not(feature = "wgpu"))]
fn main() {}

#[cfg(feature = "wgpu")]
mod enabled {
    use std::borrow::Cow;
    use std::error::Error;
    use std::hint::black_box;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use flat_attention::api::wgpu::PreparedGroupedForward;
    use flat_attention::{
        forward_reference_grouped, FlatAttentionConfig, GroupedAttentionShape,
        GroupedForwardLayout, GroupedForwardPass, WgpuGroupedForwardPipeline,
    };

    const SHADER: &str = include_str!("../shaders/flat_fwd_q1_direct_vec4.wgsl");
    const DEFAULT_WARMUPS: usize = 5;
    const DEFAULT_REPEATS: usize = 20;
    const ATOL: f32 = 8.0e-4;
    const RTOL: f32 = 3.0e-3;

    struct CaseResult {
        m58_median: u128,
        m58_p95: u128,
        m60_median: u128,
        m60_p95: u128,
        ratio: f64,
        m58_parity: f32,
        m60_parity: f32,
    }

    struct DirectPipeline {
        pipeline: wgpu::ComputePipeline,
    }

    struct DirectPrepared {
        layout: GroupedForwardLayout,
        bind_group: wgpu::BindGroup,
        dispatch_x: u32,
        dispatch_y: u32,
        _params: wgpu::Buffer,
    }

    impl DirectPipeline {
        fn new(device: &wgpu::Device) -> Result<Self, Box<dyn Error>> {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("flat-m60-ab-direct"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flat-m60-ab-direct"),
                layout: None,
                module: &shader,
                entry_point: Some("flat_attention_forward"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            if let Some(error) = pollster::block_on(scope.pop()) {
                return Err(format!("M60 pipeline validation failed: {error}").into());
            }
            Ok(Self { pipeline })
        }

        fn create_output(
            &self,
            device: &wgpu::Device,
            shape: GroupedAttentionShape,
        ) -> Result<wgpu::Buffer, Box<dyn Error>> {
            validate_shape(shape)?;
            let layout = WgpuGroupedForwardPipeline::layout(shape)?;
            Ok(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flat-m60-ab-o-lse"),
                size: layout.output_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        }

        fn prepare(
            &self,
            device: &wgpu::Device,
            pass: GroupedForwardPass<'_>,
        ) -> Result<DirectPrepared, Box<dyn Error>> {
            validate_shape(pass.shape)?;
            let layout = WgpuGroupedForwardPipeline::layout(pass.shape)?;
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
                label: Some("flat-m60-ab-params"),
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
                label: Some("flat-m60-ab-bind-group"),
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

            Ok(DirectPrepared {
                layout,
                bind_group,
                dispatch_x,
                dispatch_y,
                _params: params,
            })
        }

        fn encode_prepared(&self, encoder: &mut wgpu::CommandEncoder, prepared: &DirectPrepared) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-m60-ab-direct"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &prepared.bind_group, &[]);
            pass.dispatch_workgroups(prepared.dispatch_x, prepared.dispatch_y, 1);
        }
    }

    fn validate_shape(shape: GroupedAttentionShape) -> Result<(), Box<dyn Error>> {
        if shape.q_heads != shape.kv_heads || !matches!(shape.head_dim, 64 | 128) {
            return Err("M60 benchmark requires MHA with D64/D128".into());
        }
        Ok(())
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.037 + phase;
                x.sin() * 0.65 + (x * 0.41).cos() * 0.35
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

    fn storage(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        values: &[f32],
        label: &'static str,
    ) -> wgpu::Buffer {
        let bytes = bytes_f32(values);
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
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let bytes = (len * std::mem::size_of::<f32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m60-ab-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m60-ab-readback"),
        });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv()??;
        let mapped = slice.get_mapped_range()?;
        let values = mapped
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        drop(mapped);
        staging.unmap();
        Ok(values)
    }

    fn max_abs_error(actual: &[f32], expected: &[f32]) -> Result<f32, Box<dyn Error>> {
        if actual.len() != expected.len() {
            return Err(format!("length {} != {}", actual.len(), expected.len()).into());
        }
        let mut worst = 0.0f32;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            let tolerance = ATOL + RTOL * actual.abs().max(expected.abs());
            if !actual.is_finite() || error > tolerance {
                return Err(format!(
                    "index={index} actual={actual} expected={expected} error={error} tolerance={tolerance}"
                )
                .into());
            }
            worst = worst.max(error);
        }
        Ok(worst)
    }

    fn percentile_ns(samples: &[Duration], percentile: usize) -> u128 {
        let mut values: Vec<u128> = samples.iter().map(Duration::as_nanos).collect();
        values.sort_unstable();
        let rank = percentile.saturating_mul(values.len()).div_ceil(100).max(1);
        values[rank - 1]
    }

    fn median_ns(samples: &[Duration]) -> u128 {
        percentile_ns(samples, 50)
    }

    fn time_m58(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &WgpuGroupedForwardPipeline,
        prepared: &PreparedGroupedForward,
    ) -> Duration {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m60-ab-m58"),
        });
        black_box(pipeline.encode_prepared(&mut encoder, prepared));
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        start.elapsed()
    }

    fn time_m60(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &DirectPipeline,
        prepared: &DirectPrepared,
    ) -> Duration {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m60-ab-direct"),
        });
        pipeline.encode_prepared(&mut encoder, prepared);
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        start.elapsed()
    }

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        seq_len: usize,
        head_dim: usize,
        causal: bool,
        warmups: usize,
        repeats: usize,
    ) -> Result<CaseResult, Box<dyn Error>> {
        let shape = GroupedAttentionShape {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            seq_len,
            head_dim,
        };
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let len = shape.q_tensor_len()?;
        let q = fixture(len, 0.2);
        let k = fixture(len, 0.8);
        let v = fixture(len, 1.4);
        let expected = forward_reference_grouped(&q, &k, &v, shape, config)?;
        let mut expected_combined = expected.output.clone();
        expected_combined.extend_from_slice(&expected.lse);

        let q_gpu = storage(device, queue, &q, "flat-m60-ab-q");
        let k_gpu = storage(device, queue, &k, "flat-m60-ab-k");
        let v_gpu = storage(device, queue, &v, "flat-m60-ab-v");

        let m58 = WgpuGroupedForwardPipeline::with_q1_vec4_mha(device, true)?;
        let m60 = DirectPipeline::new(device)?;
        let m58_output = m58.create_output_buffer(device, shape)?;
        let m60_output = m60.create_output(device, shape)?;
        let m58_prepared = m58.prepare(
            device,
            GroupedForwardPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &m58_output,
                shape,
                config,
            },
        )?;
        let m60_prepared = m60.prepare(
            device,
            GroupedForwardPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &m60_output,
                shape,
                config,
            },
        )?;

        let _ = time_m58(device, queue, &m58, &m58_prepared);
        let _ = time_m60(device, queue, &m60, &m60_prepared);
        let m58_host = read_f32(
            device,
            queue,
            &m58_output,
            m58_prepared.layout().output_elements,
        )?;
        let m60_host = read_f32(
            device,
            queue,
            &m60_output,
            m60_prepared.layout.output_elements,
        )?;
        let m58_parity = max_abs_error(&m58_host, &expected_combined)?;
        let m60_parity = max_abs_error(&m60_host, &expected_combined)?;

        for iteration in 0..warmups {
            if iteration % 2 == 0 {
                let _ = time_m58(device, queue, &m58, &m58_prepared);
                let _ = time_m60(device, queue, &m60, &m60_prepared);
            } else {
                let _ = time_m60(device, queue, &m60, &m60_prepared);
                let _ = time_m58(device, queue, &m58, &m58_prepared);
            }
        }

        let mut m58_samples = Vec::with_capacity(repeats);
        let mut m60_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats {
            if iteration % 2 == 0 {
                m58_samples.push(time_m58(device, queue, &m58, &m58_prepared));
                m60_samples.push(time_m60(device, queue, &m60, &m60_prepared));
            } else {
                m60_samples.push(time_m60(device, queue, &m60, &m60_prepared));
                m58_samples.push(time_m58(device, queue, &m58, &m58_prepared));
            }
        }

        let m58_median = median_ns(&m58_samples);
        let m60_median = median_ns(&m60_samples);
        Ok(CaseResult {
            m58_median,
            m58_p95: percentile_ns(&m58_samples, 95),
            m60_median,
            m60_p95: percentile_ns(&m60_samples, 95),
            ratio: m58_median as f64 / m60_median.max(1) as f64,
            m58_parity,
            m60_parity,
        })
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let warmups = env_usize("FLAT_M60_WARMUPS", DEFAULT_WARMUPS);
        let repeats = env_usize("FLAT_M60_REPEATS", DEFAULT_REPEATS);
        if warmups == 0 || repeats == 0 {
            return Err("warmups and repeats must be non-zero".into());
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))?;
        let info = adapter.get_info();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("flat-m60-direct-ab"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            }))?;

        eprintln!("benchmark=m60_q1_direct_vs_m58");
        eprintln!("mechanism=remove_non_reused_kv_workgroup_staging");
        eprintln!("adapter={}", info.name);
        eprintln!("backend={:?}", info.backend);
        eprintln!("driver={}", info.driver);
        eprintln!("warmups={warmups}");
        eprintln!("repeats={repeats}");
        eprintln!("measurement_order=alternating_m58_m60");
        println!(
            "adapter,backend,seq_len,head_dim,causal,warmups,repeats,m58_median_us,m58_p95_us,m60_median_us,m60_p95_us,m58_over_m60,m58_parity_max_abs,m60_parity_max_abs,performance_claim"
        );

        for seq_len in [128_usize, 512] {
            for head_dim in [64_usize, 128] {
                for causal in [false, true] {
                    let result =
                        run_case(&device, &queue, seq_len, head_dim, causal, warmups, repeats)?;
                    println!(
                        "{},{:?},{seq_len},{head_dim},{causal},{warmups},{repeats},{:.3},{:.3},{:.3},{:.3},{:.6},{:.8},{:.8},measurement_only_no_production_routing_change",
                        info.name.replace(',', ";"),
                        info.backend,
                        result.m58_median as f64 / 1_000.0,
                        result.m58_p95 as f64 / 1_000.0,
                        result.m60_median as f64 / 1_000.0,
                        result.m60_p95 as f64 / 1_000.0,
                        result.ratio,
                        result.m58_parity,
                        result.m60_parity,
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "wgpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    enabled::run()
}
