#![cfg(feature = "wgpu")]

use std::time::Instant;

use flat_attention::{
    forward_reference_grouped, pack_grouped_backward_recompute_inputs, FlatAttentionConfig,
    GroupedAttentionShape, GroupedBackwardRecomputePass, WgpuGroupedBackwardRecomputePipeline,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.017 + phase;
            x.sin() * 0.75 + (x * 0.29).cos() * 0.25
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

fn input_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    label: &'static str,
) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

fn summarize(mut samples_us: Vec<f64>) -> (f64, f64) {
    samples_us.sort_by(f64::total_cmp);
    (percentile(&samples_us, 0.5), percentile(&samples_us, 0.95))
}

fn main() {
    let q_heads = env_usize("FLAT_BENCH_Q_HEADS", 8);
    let kv_heads = env_usize("FLAT_BENCH_KV_HEADS", 2);
    let seq_len = env_usize("FLAT_BENCH_SEQ_LEN", 128);
    let head_dim = env_usize("FLAT_BENCH_HEAD_DIM", 64);
    let warmup = env_usize("FLAT_BENCH_WARMUP", 20);
    let iterations = env_usize("FLAT_BENCH_ITERS", 200);
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads,
        kv_heads,
        seq_len,
        head_dim,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("M20 host-overhead benchmark requires a WGPU adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m20-grouped-backward-host-overhead"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .expect("M20 host-overhead request_device failed");

    let pipeline = WgpuGroupedBackwardRecomputePipeline::new(&device).unwrap();
    let q = fixture(shape.q_tensor_len().unwrap(), 0.1);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.7);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.3);
    let d_out = fixture(shape.q_tensor_len().unwrap(), 1.9);
    let forward = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
    let packed =
        pack_grouped_backward_recompute_inputs(&q, &k, &v, &d_out, &forward, shape).unwrap();
    let packed_gpu = input_buffer(&device, &queue, &packed, "flat-m20-host-overhead-input");
    let grads_gpu = pipeline.create_gradient_buffer(&device, shape).unwrap();

    let measure_once = || {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m20-host-overhead-encoder"),
        });
        pipeline
            .encode(
                &device,
                &mut encoder,
                GroupedBackwardRecomputePass {
                    packed_forward: &packed_gpu,
                    packed_grads: &grads_gpu,
                    shape,
                    config,
                },
            )
            .unwrap();
        let _commands = encoder.finish();
        start.elapsed().as_secs_f64() * 1.0e6
    };

    for _ in 0..warmup {
        let _ = measure_once();
    }
    let samples = (0..iterations).map(|_| measure_once()).collect();
    let (median_us, p95_us) = summarize(samples);

    println!("device={:?} backend={:?}", info.name, info.backend);
    println!(
        "benchmark=m20_grouped_backward_host_overhead batch=1 q_heads={q_heads} kv_heads={kv_heads} seq_len={seq_len} head_dim={head_dim} warmup={warmup} iterations={iterations}"
    );
    println!("timing_scope=command_encoder+public_encode+finish_no_submit");
    println!("median_us={median_us:.3} p95_us={p95_us:.3}");
    println!("performance_claim=none");
}
