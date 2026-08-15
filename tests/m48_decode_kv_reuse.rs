#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, ExternalAsymmetricProjectionPass,
    ExternalAsymmetricProjectionRotaryGroupedPipeline, ExternalWgpuError, FlatAttentionConfig,
};

const ATOL: f32 = 1.5e-4;
const RTOL: f32 = 1.0e-3;

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.021 + phase;
            x.sin() * 1.75 + (x * 0.43).cos() * 0.34375
        })
        .collect()
}

fn rotate_k(raw: &[f32], kv_len: usize, kv_heads: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    let mut rotated = raw.to_vec();
    let width = kv_heads * head_dim;
    for position in 0..kv_len {
        for head in 0..kv_heads {
            let base = position * width + head * head_dim;
            for pair in 0..head_dim / 2 {
                let dim = 2 * pair;
                let frequency = theta.powf(-2.0 * pair as f32 / head_dim as f32);
                let (sine, cosine) = (position as f32 * frequency).sin_cos();
                let even = raw[base + dim];
                let odd = raw[base + dim + 1];
                rotated[base + dim] = even * cosine - odd * sine;
                rotated[base + dim + 1] = even * sine + odd * cosine;
            }
        }
    }
    rotated
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn buffer(device: &wgpu::Device, queue: &wgpu::Queue, values: &[f32]) -> wgpu::Buffer {
    let contents = bytes(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m48-input"),
        size: contents.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &contents);
    buffer
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let byte_len = (len * size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m48-readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m48-readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..byte_len);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let output = mapped
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    drop(mapped);
    staging.unmap();
    output
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = ATOL + RTOL * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name}[{index}] actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

#[test]
fn m48_sciagent_decode_matches_oracle() {
    let instance = wgpu::Instance::default();
    let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
            panic!("M48 requires a WGPU adapter");
        }
        return;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m48-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap();
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 16,
        kv_heads: 4,
        query_len: 1,
        kv_len: 192,
        head_dim: 64,
        query_position_offset: 191,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let rotary = AsymmetricRotaryEmbeddingConfig {
        theta: 10_000.0,
        query_position_offset: 191,
        kv_position_offset: 0,
    };
    let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
    let raw_k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
    let k = rotate_k(&raw_k, 192, 4, 64, 10_000.0);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
    let expected =
        forward_reference_projection_grouped_rope_asymmetric(&q, &raw_k, &v, shape, config, rotary)
            .unwrap();
    let q_gpu = buffer(&device, &queue, &q);
    let k_gpu = buffer(&device, &queue, &k);
    let v_gpu = buffer(&device, &queue, &v);
    let pipeline =
        ExternalAsymmetricProjectionRotaryGroupedPipeline::with_decode_kv_reuse(&device, true)
            .unwrap();
    let output = pipeline.create_output_buffer(&device, shape).unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m48-test"),
    });
    let layout = pipeline
        .encode_pre_rotated_k_decode_kv_reuse(
            &device,
            &mut encoder,
            ExternalAsymmetricProjectionPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                out_and_lse: &output,
                shape,
                config,
                rotary,
            },
        )
        .unwrap();
    queue.submit(Some(encoder.finish()));
    let actual = read(&device, &queue, &output, layout.combined_elements);
    assert_close("M48 O", &actual[..layout.output_elements], &expected.output);
    assert_close("M48 LSE", &actual[layout.output_elements..], &expected.lse);
}

#[test]
fn m48_is_opt_in() {
    let instance = wgpu::Instance::default();
    let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        return;
    };
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .unwrap();
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 16,
        kv_heads: 4,
        query_len: 1,
        kv_len: 1,
        head_dim: 64,
        query_position_offset: 0,
    };
    let input = buffer(&device, &queue, &vec![0.0; shape.q_tensor_len().unwrap()]);
    let kv = buffer(&device, &queue, &vec![0.0; shape.kv_tensor_len().unwrap()]);
    let pipeline = ExternalAsymmetricProjectionRotaryGroupedPipeline::new(&device).unwrap();
    let output = pipeline.create_output_buffer(&device, shape).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let error = pipeline
        .encode_pre_rotated_k_decode_kv_reuse(
            &device,
            &mut encoder,
            ExternalAsymmetricProjectionPass {
                q: &input,
                k: &kv,
                v: &kv,
                out_and_lse: &output,
                shape,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                rotary: AsymmetricRotaryEmbeddingConfig {
                    theta: 10_000.0,
                    query_position_offset: 0,
                    kv_position_offset: 0,
                },
            },
        )
        .unwrap_err();
    assert_eq!(
        error,
        ExternalWgpuError::CandidateNotEnabled { candidate: "M48" }
    );
}
