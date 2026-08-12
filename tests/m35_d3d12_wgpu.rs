#![cfg(all(feature = "wgpu", target_os = "windows"))]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric_biased, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, AttentionBias, ExternalAsymmetricProjectionPass,
    ExternalAsymmetricProjectionRotaryGroupedPipeline, FlatAttentionConfig,
};

const ATOL: f32 = 2.0e-4;
const RTOL: f32 = 1.0e-3;

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.019 + phase;
            x.sin() * 1.625 + (x * 0.37).cos() * 0.3125
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

fn input_buffer(device: &wgpu::Device, queue: &wgpu::Queue, values: &[f32]) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m35-input"),
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
        label: Some("flat-m35-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m35-readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..bytes);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .expect("M35 map callback")
        .expect("M35 map read");
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

#[test]
fn d3d12_warp_asymmetric_gqa_alibi_matches_scalar_oracle() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: true,
    }))
    .expect("M35 requires a Direct3D 12 fallback adapter (WARP)");
    let info = adapter.get_info();
    assert_eq!(
        info.backend,
        wgpu::Backend::Dx12,
        "M35 must execute on D3D12"
    );
    eprintln!(
        "M35 D3D12 adapter: name={} vendor={:#x} device={:#x} driver={} info={}",
        info.name, info.vendor, info.device, info.driver, info.driver_info
    );
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m35-d3d12-warp"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .expect("request M35 D3D12 WARP device");

    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 8,
        kv_heads: 2,
        query_len: 3,
        kv_len: 17,
        head_dim: 64,
        query_position_offset: 14,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let rotary = AsymmetricRotaryEmbeddingConfig {
        theta: 10_000.0,
        query_position_offset: 41,
        kv_position_offset: 7,
    };
    let slopes = [
        0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25,
    ];
    let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
    let expected = forward_reference_projection_grouped_rope_asymmetric_biased(
        &q,
        &k,
        &v,
        shape,
        config,
        rotary,
        AttentionBias::Alibi {
            slopes: &slopes,
            query_position_offset: 103,
            kv_position_offset: 97,
        },
    )
    .expect("M35 scalar oracle");

    let pipeline = ExternalAsymmetricProjectionRotaryGroupedPipeline::new(&device)
        .expect("M35 D3D12 pipeline creation");
    let q_gpu = input_buffer(&device, &queue, &q);
    let k_gpu = input_buffer(&device, &queue, &k);
    let v_gpu = input_buffer(&device, &queue, &v);
    let output = pipeline
        .create_output_buffer(&device, shape)
        .expect("M35 output buffer");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m35-d3d12"),
    });
    let pass = ExternalAsymmetricProjectionPass {
        q: &q_gpu,
        k: &k_gpu,
        v: &v_gpu,
        out_and_lse: &output,
        shape,
        config,
        rotary,
    };
    let layout = pipeline
        .encode_alibi(&device, &mut encoder, pass, &slopes, 103, 97)
        .expect("M35 encode ALiBi");
    queue.submit(Some(encoder.finish()));
    let values = read_f32(&device, &queue, &output, layout.combined_elements);
    assert_close(
        "M35 D3D12 O",
        &values[..layout.output_elements],
        &expected.output,
    );
    assert_close(
        "M35 D3D12 LSE",
        &values[layout.output_elements..],
        &expected.lse,
    );
}
