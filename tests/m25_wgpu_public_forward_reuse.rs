#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_grouped, FlatAttentionConfig, GroupedAttentionShape, GroupedForwardPass,
    WgpuGroupedForwardPipeline,
};

const ATOL: f32 = 6.0e-4;
const RTOL: f32 = 2.5e-3;

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
            panic!("M25 public grouped-forward reuse requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M25 public grouped-forward reuse test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m25-public-grouped-forward-reuse"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M25 request_device failed: {error}"));
    Some(Harness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.043 + phase;
            x.sin() * 0.7 + (x * 0.37).cos() * 0.3
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

fn initialized_storage_buffer(
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
) -> Vec<f32> {
    let bytes = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m25-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m25-readback"),
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

fn run_case(
    harness: &Harness,
    pipeline: &WgpuGroupedForwardPipeline,
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    phase: f32,
) {
    let q_len = shape.q_tensor_len().unwrap();
    let kv_len = shape.kv_tensor_len().unwrap();
    let q = fixture(q_len, phase);
    let k = fixture(kv_len, phase + 0.6);
    let v = fixture(kv_len, phase + 1.2);
    let expected = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();

    let q_gpu = initialized_storage_buffer(
        &harness.device,
        &harness.queue,
        &q,
        "flat-m25-q",
    );
    let k_gpu = initialized_storage_buffer(
        &harness.device,
        &harness.queue,
        &k,
        "flat-m25-k",
    );
    let v_gpu = initialized_storage_buffer(
        &harness.device,
        &harness.queue,
        &v,
        "flat-m25-v",
    );
    let output_gpu = pipeline.create_output_buffer(&harness.device, shape).unwrap();
    let layout = WgpuGroupedForwardPipeline::layout(shape).unwrap();

    for reuse in 0..3 {
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-m25-public-grouped-forward-reuse"),
            });
        let encoded_layout = pipeline
            .encode(
                &harness.device,
                &mut encoder,
                GroupedForwardPass {
                    q: &q_gpu,
                    k: &k_gpu,
                    v: &v_gpu,
                    output: &output_gpu,
                    shape,
                    config,
                },
            )
            .unwrap();
        assert_eq!(encoded_layout, layout, "reuse {reuse}: layout drift");
        harness.queue.submit(Some(encoder.finish()));
        let _ = harness.device.poll(wgpu::Maintain::Wait);

        let actual = read_f32(
            &harness.device,
            &harness.queue,
            &output_gpu,
            layout.output_elements,
        );
        assert_close(
            &format!("reuse {reuse} O"),
            &actual[layout.output_offset()..layout.lse_offset()],
            &expected.output,
        );
        assert_close(
            &format!("reuse {reuse} LSE"),
            &actual[layout.lse_offset()..],
            &expected.lse,
        );
    }
}

#[test]
fn public_grouped_forward_pipeline_reuses_across_gqa_and_mqa() {
    let Some(harness) = harness() else {
        return;
    };
    let pipeline = WgpuGroupedForwardPipeline::new(&harness.device).unwrap();

    run_case(
        &harness,
        &pipeline,
        GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 7,
            head_dim: 16,
        },
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        },
        0.1,
    );

    run_case(
        &harness,
        &pipeline,
        GroupedAttentionShape {
            batch: 2,
            q_heads: 4,
            kv_heads: 1,
            seq_len: 5,
            head_dim: 8,
        },
        FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(0.31),
        },
        1.7,
    );
}
