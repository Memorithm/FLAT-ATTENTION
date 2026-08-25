#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("m11_decode_bench requires --features wgpu");
}

#[cfg(feature = "wgpu")]
fn main() {
    bench::run();
}

#[cfg(feature = "wgpu")]
mod bench {
    use std::time::{Duration, Instant};

    use flat_attention::{
        AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig,
        ExternalAsymmetricProjectionPass, ExternalAsymmetricProjectionRotaryGroupedPipeline,
        FlatAttentionConfig,
    };

    const WARMUP_ITERS: usize = 20;
    const MEASURE_ITERS: usize = 200;

    struct DeviceHarness {
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_name: String,
    }

    pub fn run() {
        let Some(harness) = harness() else {
            eprintln!("WGPU adapter unavailable; benchmark not run");
            return;
        };

        println!("adapter={}", harness.adapter_name);
        println!("measurement=end-to-end-submit-and-device-poll");
        println!("warmup_iters={WARMUP_ITERS}");
        println!("measure_iters={MEASURE_ITERS}");
        println!("query_len=1");
        println!("q_heads=8 kv_heads=2 head_dim=64 batch=1 causal=true");

        for kv_len in [16usize, 64, 256, 1024, 4096] {
            run_case(&harness, kv_len);
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
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flat-m12-decode-bench"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .ok()?;
        Some(DeviceHarness {
            device,
            queue,
            adapter_name,
        })
    }

    fn run_case(harness: &DeviceHarness, kv_len: usize) {
        let shape = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 8,
            kv_heads: 2,
            query_len: 1,
            kv_len,
            head_dim: 64,
            query_position_offset: kv_len - 1,
        };
        let config = FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        };
        let rotary = AsymmetricRotaryEmbeddingConfig {
            theta: 10_000.0,
            query_position_offset: kv_len - 1,
            kv_position_offset: 0,
        };

        let pipeline = ExternalAsymmetricProjectionRotaryGroupedPipeline::new(&harness.device)
            .expect("create M11 pipeline");
        let q = fixture(shape.q_tensor_len().expect("Q shape"), 0.2);
        let k = fixture(shape.kv_tensor_len().expect("K shape"), 0.8);
        let v = fixture(shape.kv_tensor_len().expect("V shape"), 1.4);
        let q_gpu = input_buffer(&harness.device, &harness.queue, &q);
        let k_gpu = input_buffer(&harness.device, &harness.queue, &k);
        let v_gpu = input_buffer(&harness.device, &harness.queue, &v);
        let output = pipeline
            .create_output_buffer(&harness.device, shape)
            .expect("create M11 output");

        for _ in 0..WARMUP_ITERS {
            dispatch(
                harness, &pipeline, &q_gpu, &k_gpu, &v_gpu, &output, shape, config, rotary,
            );
        }

        let mut samples = Vec::with_capacity(MEASURE_ITERS);
        for _ in 0..MEASURE_ITERS {
            let start = Instant::now();
            dispatch(
                harness, &pipeline, &q_gpu, &k_gpu, &v_gpu, &output, shape, config, rotary,
            );
            samples.push(start.elapsed());
        }
        samples.sort_unstable();

        let min = samples[0];
        let median = samples[samples.len() / 2];
        let p95_index = (samples.len() * 95).div_ceil(100).saturating_sub(1);
        let p95 = samples[p95_index];
        let mean = samples.iter().copied().sum::<Duration>() / samples.len() as u32;

        println!(
            "kv_len={kv_len} min_us={:.3} median_us={:.3} mean_us={:.3} p95_us={:.3}",
            micros(min),
            micros(median),
            micros(mean),
            micros(p95),
        );
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
                label: Some("flat-m12-decode-bench-dispatch"),
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
            .expect("encode M11 dispatch");
        harness.queue.submit(Some(encoder.finish()));
        let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
    }

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.023 + phase;
                x.sin() * 1.875 + (x * 0.41).cos() * 0.28125
            })
            .collect()
    }

    fn input_buffer(device: &wgpu::Device, queue: &wgpu::Queue, values: &[f32]) -> wgpu::Buffer {
        let bytes = encode_f32(values);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m12-decode-bench-input"),
            size: bytes.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            queue.write_buffer(&buffer, 0, &bytes);
        }
        buffer
    }

    fn encode_f32(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for &value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn micros(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1_000_000.0
    }
}
