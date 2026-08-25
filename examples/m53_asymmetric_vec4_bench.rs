#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("m53_asymmetric_vec4_bench requires --features wgpu");
}

#[cfg(feature = "wgpu")]
fn main() {
    bench::run();
}

#[cfg(feature = "wgpu")]
mod bench {
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use flat_attention::{
        AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig,
        ExternalAsymmetricKernelVariant, ExternalAsymmetricProjectionPass,
        ExternalAsymmetricProjectionRotaryGroupedPipeline, FlatAttentionConfig,
    };

    const ATOL: f32 = 8.0e-4;
    const RTOL: f32 = 3.0e-3;

    struct DeviceHarness {
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_name: String,
        backend: wgpu::Backend,
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    pub fn run() {
        let harness = harness().expect("M53 benchmark requires a WGPU adapter");
        let warmups = env_usize("FLAT_M53_WARMUPS", 3);
        let repeats = env_usize("FLAT_M53_REPEATS", 12);
        assert!(warmups > 0 && repeats >= 4);
        println!(
            "adapter,backend,causal,q_heads,kv_heads,seq_len,head_dim,warmups,repeats,portable_median_us,portable_p95_us,vec4_median_us,vec4_p95_us,portable_over_vec4,parity_max_abs,performance_claim"
        );
        for seq_len in [128usize, 512] {
            for head_dim in [64usize, 128] {
                for causal in [false, true] {
                    run_case(&harness, seq_len, head_dim, causal, warmups, repeats);
                }
            }
        }
    }

    fn harness() -> Option<DeviceHarness> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flat-m53-asymmetric-vec4-bench"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .ok()?;
        Some(DeviceHarness {
            device,
            queue,
            adapter_name: info.name,
            backend: info.backend,
        })
    }

    fn run_case(
        harness: &DeviceHarness,
        seq_len: usize,
        head_dim: usize,
        causal: bool,
        warmups: usize,
        repeats: usize,
    ) {
        let shape = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 8,
            kv_heads: 2,
            query_len: seq_len,
            kv_len: seq_len,
            head_dim,
            query_position_offset: 0,
        };
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let rotary = AsymmetricRotaryEmbeddingConfig {
            theta: 10_000.0,
            query_position_offset: 0,
            kv_position_offset: 0,
        };
        let portable = ExternalAsymmetricProjectionRotaryGroupedPipeline::new(&harness.device)
            .expect("portable pipeline");
        let vec4 = ExternalAsymmetricProjectionRotaryGroupedPipeline::with_vectorization(
            &harness.device,
            true,
        )
        .expect("M53 vec4 pipeline");
        assert_eq!(
            vec4.kernel_variant_for_shape(shape),
            ExternalAsymmetricKernelVariant::Vec4
        );

        let q = fixture(shape.q_tensor_len().expect("Q shape"), 0.2);
        let k = fixture(shape.kv_tensor_len().expect("K shape"), 0.8);
        let v = fixture(shape.kv_tensor_len().expect("V shape"), 1.4);
        let q_gpu = input_buffer(&harness.device, &harness.queue, &q, "M53 Q");
        let k_gpu = input_buffer(&harness.device, &harness.queue, &k, "M53 K");
        let v_gpu = input_buffer(&harness.device, &harness.queue, &v, "M53 V");
        let portable_output = portable
            .create_output_buffer(&harness.device, shape)
            .expect("portable output");
        let vec4_output = vec4
            .create_output_buffer(&harness.device, shape)
            .expect("vec4 output");
        let layout =
            ExternalAsymmetricProjectionRotaryGroupedPipeline::layout(shape).expect("M53 layout");

        dispatch(
            harness,
            &portable,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &portable_output,
            shape,
            config,
            rotary,
        );
        dispatch(
            harness,
            &vec4,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &vec4_output,
            shape,
            config,
            rotary,
        );
        let portable_values = read_f32(
            &harness.device,
            &harness.queue,
            &portable_output,
            layout.combined_elements,
        );
        let vec4_values = read_f32(
            &harness.device,
            &harness.queue,
            &vec4_output,
            layout.combined_elements,
        );
        let parity = max_error(&vec4_values, &portable_values);

        for iteration in 0..warmups {
            if iteration.is_multiple_of(2) {
                dispatch(
                    harness,
                    &portable,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &portable_output,
                    shape,
                    config,
                    rotary,
                );
                dispatch(
                    harness,
                    &vec4,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &vec4_output,
                    shape,
                    config,
                    rotary,
                );
            } else {
                dispatch(
                    harness,
                    &vec4,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &vec4_output,
                    shape,
                    config,
                    rotary,
                );
                dispatch(
                    harness,
                    &portable,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &portable_output,
                    shape,
                    config,
                    rotary,
                );
            }
        }
        let mut portable_samples = Vec::with_capacity(repeats);
        let mut vec4_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats {
            if iteration.is_multiple_of(2) {
                portable_samples.push(timed_dispatch(
                    harness,
                    &portable,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &portable_output,
                    shape,
                    config,
                    rotary,
                ));
                vec4_samples.push(timed_dispatch(
                    harness,
                    &vec4,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &vec4_output,
                    shape,
                    config,
                    rotary,
                ));
            } else {
                vec4_samples.push(timed_dispatch(
                    harness,
                    &vec4,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &vec4_output,
                    shape,
                    config,
                    rotary,
                ));
                portable_samples.push(timed_dispatch(
                    harness,
                    &portable,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &portable_output,
                    shape,
                    config,
                    rotary,
                ));
            }
        }
        let portable_median = percentile_ns(&portable_samples, 50);
        let vec4_median = percentile_ns(&vec4_samples, 50);
        println!(
            "{},{:?},{causal},8,2,{seq_len},{head_dim},{warmups},{repeats},{:.3},{:.3},{:.3},{:.3},{:.6},{parity:.8},none",
            harness.adapter_name.replace(',', ";"),
            harness.backend,
            portable_median as f64 / 1_000.0,
            percentile_ns(&portable_samples, 95) as f64 / 1_000.0,
            vec4_median as f64 / 1_000.0,
            percentile_ns(&vec4_samples, 95) as f64 / 1_000.0,
            portable_median as f64 / vec4_median.max(1) as f64,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn timed_dispatch(
        harness: &DeviceHarness,
        pipeline: &ExternalAsymmetricProjectionRotaryGroupedPipeline,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        output: &wgpu::Buffer,
        shape: AsymmetricGroupedAttentionShape,
        config: FlatAttentionConfig,
        rotary: AsymmetricRotaryEmbeddingConfig,
    ) -> Duration {
        let start = Instant::now();
        dispatch(harness, pipeline, q, k, v, output, shape, config, rotary);
        start.elapsed()
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        harness: &DeviceHarness,
        pipeline: &ExternalAsymmetricProjectionRotaryGroupedPipeline,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        output: &wgpu::Buffer,
        shape: AsymmetricGroupedAttentionShape,
        config: FlatAttentionConfig,
        rotary: AsymmetricRotaryEmbeddingConfig,
    ) {
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("M53 dispatch"),
            });
        pipeline
            .encode(
                &harness.device,
                &mut encoder,
                ExternalAsymmetricProjectionPass {
                    q,
                    k,
                    v,
                    out_and_lse: output,
                    shape,
                    config,
                    rotary,
                },
            )
            .expect("M53 encode");
        harness.queue.submit(Some(encoder.finish()));
        let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
    }

    fn max_error(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(actual.len(), expected.len());
        let mut worst = 0.0f32;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            let tolerance = ATOL + RTOL * actual.abs().max(expected.abs());
            assert!(
                actual.is_finite() && error <= tolerance,
                "M53 parity[{index}] actual={actual} expected={expected} error={error} tolerance={tolerance}"
            );
            worst = worst.max(error);
        }
        worst
    }

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.037 + phase;
                x.sin() * 0.65 + (x * 0.41).cos() * 0.35
            })
            .collect()
    }

    fn input_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        values: &[f32],
        label: &'static str,
    ) -> wgpu::Buffer {
        let bytes = encode_f32(values);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
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
            label: Some("M53 readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("M53 readback encoder"),
        });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().expect("M53 map callback").expect("M53 map");
        let mapped = slice.get_mapped_range().expect("valid mapped range");
        let values = mapped
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        drop(mapped);
        staging.unmap();
        values
    }

    fn encode_f32(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for &value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn percentile_ns(samples: &[Duration], percentile: usize) -> u128 {
        let mut values: Vec<u128> = samples.iter().map(Duration::as_nanos).collect();
        values.sort_unstable();
        let rank = percentile.saturating_mul(values.len()).div_ceil(100).max(1);
        values[rank - 1]
    }
}
