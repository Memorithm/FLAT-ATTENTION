#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig, ExternalVariableProjectionPass,
    ExternalVariableProjectionRotaryGroupedPipeline, ExternalWgpuError, FlatAttentionConfig,
    FlatAttentionError, VariableLengthRotaryEmbeddingConfig, VariableLengthSequenceMetadata,
};

const ATOL: f32 = 1.5e-4;
const RTOL: f32 = 1.0e-3;

struct DeviceHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
}

fn harness() -> Option<DeviceHarness> {
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
            panic!("M12 requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional M12 device test skipped");
        return None;
    };
    let adapter_name = adapter.get_info().name;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m12-variable-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .unwrap_or_else(|error| panic!("M12 request_device failed: {error}"));
    Some(DeviceHarness {
        device,
        queue,
        adapter_name,
    })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i as f32 * 0.019 + phase;
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
        label: Some("flat-m12-input"),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buffer, 0, &bytes);
    }
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
        label: Some("flat-m12-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m12-readback"),
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
    let mut values = Vec::with_capacity(len);
    for chunk in mapped.chunks_exact(4) {
        values.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    drop(mapped);
    staging.unmap();
    values
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "{name}[{index}] is not finite");
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

fn compact_rows(
    values: &[f32],
    batch: usize,
    padded_rows: usize,
    active_rows: usize,
    width: usize,
) -> Vec<f32> {
    let start = batch * padded_rows * width;
    values[start..start + active_rows * width].to_vec()
}

fn poison_padding(
    values: &mut [f32],
    batch: usize,
    padded_rows: usize,
    active_rows: usize,
    width: usize,
    value: f32,
) {
    let batch_start = batch * padded_rows * width;
    let start = batch_start + active_rows * width;
    let end = batch_start + padded_rows * width;
    values[start..end].fill(value);
}

fn run_case(
    harness: &DeviceHarness,
    shape: AsymmetricGroupedAttentionShape,
    metadata: &[VariableLengthSequenceMetadata],
    config: FlatAttentionConfig,
    rotary: VariableLengthRotaryEmbeddingConfig,
    phase: f32,
) {
    let pipeline = ExternalVariableProjectionRotaryGroupedPipeline::new(&harness.device).unwrap();
    let q_width = shape.q_heads * shape.head_dim;
    let kv_width = shape.kv_heads * shape.head_dim;
    let mut q = fixture(shape.q_tensor_len().unwrap(), phase);
    let mut k = fixture(shape.kv_tensor_len().unwrap(), phase + 0.6);
    let mut v = fixture(shape.kv_tensor_len().unwrap(), phase + 1.2);

    for (batch, entry) in metadata.iter().enumerate() {
        poison_padding(
            &mut q,
            batch,
            shape.query_len,
            entry.active_query_len,
            q_width,
            1.0e20,
        );
        poison_padding(
            &mut k,
            batch,
            shape.kv_len,
            entry.active_kv_len,
            kv_width,
            -1.0e20,
        );
        poison_padding(
            &mut v,
            batch,
            shape.kv_len,
            entry.active_kv_len,
            kv_width,
            1.0e20,
        );
    }

    let q_gpu = input_buffer(&harness.device, &harness.queue, &q);
    let k_gpu = input_buffer(&harness.device, &harness.queue, &k);
    let v_gpu = input_buffer(&harness.device, &harness.queue, &v);
    let output = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m12-caller-encoder"),
        });
    let layout = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            ExternalVariableProjectionPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                out_and_lse: &output,
                shape,
                metadata,
                config,
                rotary,
            },
        )
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));
    let values = read_f32(
        &harness.device,
        &harness.queue,
        &output,
        layout.combined_elements,
    );
    let actual_output = &values[..layout.output_elements];
    let actual_lse = &values[layout.output_elements..];

    for (batch, entry) in metadata.iter().enumerate() {
        let q_one = compact_rows(
            &q,
            batch,
            shape.query_len,
            entry.active_query_len,
            q_width,
        );
        let k_one = compact_rows(
            &k,
            batch,
            shape.kv_len,
            entry.active_kv_len,
            kv_width,
        );
        let v_one = compact_rows(
            &v,
            batch,
            shape.kv_len,
            entry.active_kv_len,
            kv_width,
        );
        let compact_shape = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: shape.q_heads,
            kv_heads: shape.kv_heads,
            query_len: entry.active_query_len,
            kv_len: entry.active_kv_len,
            head_dim: shape.head_dim,
            query_position_offset: entry.query_position_offset,
        };
        let expected = forward_reference_projection_grouped_rope_asymmetric(
            &q_one,
            &k_one,
            &v_one,
            compact_shape,
            config,
            AsymmetricRotaryEmbeddingConfig {
                theta: rotary.theta,
                query_position_offset: entry.query_rope_position_offset,
                kv_position_offset: rotary.kv_position_offset,
            },
        )
        .unwrap();

        let output_start = batch * shape.query_len * q_width;
        assert_close(
            "M12 active O",
            &actual_output[output_start..output_start + entry.active_query_len * q_width],
            &expected.output,
        );
        for row in entry.active_query_len..shape.query_len {
            let row_start = output_start + row * q_width;
            assert!(actual_output[row_start..row_start + q_width]
                .iter()
                .all(|&x| x == 0.0));
        }

        for head in 0..shape.q_heads {
            let actual_head_base = (batch * shape.q_heads + head) * shape.query_len;
            let expected_head_base = head * entry.active_query_len;
            assert_close(
                "M12 active LSE",
                &actual_lse[actual_head_base..actual_head_base + entry.active_query_len],
                &expected.lse[expected_head_base..expected_head_base + entry.active_query_len],
            );
            for row in entry.active_query_len..shape.query_len {
                assert_eq!(actual_lse[actual_head_base + row], f32::NEG_INFINITY);
            }
        }
    }
}

#[test]
fn mixed_length_causal_gqa_matches_independent_m11_oracles() {
    let Some(harness) = harness() else {
        return;
    };
    eprintln!("M12 adapter: {}", harness.adapter_name);
    let shape = AsymmetricGroupedAttentionShape {
        batch: 2,
        q_heads: 8,
        kv_heads: 2,
        query_len: 5,
        kv_len: 9,
        head_dim: 64,
        query_position_offset: 0,
    };
    let metadata = [
        VariableLengthSequenceMetadata {
            active_query_len: 3,
            active_kv_len: 6,
            query_position_offset: 3,
            query_rope_position_offset: 17,
        },
        VariableLengthSequenceMetadata {
            active_query_len: 5,
            active_kv_len: 9,
            query_position_offset: 4,
            query_rope_position_offset: 29,
        },
    ];
    run_case(
        &harness,
        shape,
        &metadata,
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        },
        VariableLengthRotaryEmbeddingConfig {
            theta: 10_000.0,
            kv_position_offset: 11,
        },
        0.2,
    );
}

#[test]
fn mixed_length_noncausal_mqa_matches_independent_m11_oracles() {
    let Some(harness) = harness() else {
        return;
    };
    let shape = AsymmetricGroupedAttentionShape {
        batch: 3,
        q_heads: 8,
        kv_heads: 1,
        query_len: 6,
        kv_len: 10,
        head_dim: 80,
        query_position_offset: 0,
    };
    let metadata = [
        VariableLengthSequenceMetadata {
            active_query_len: 1,
            active_kv_len: 4,
            query_position_offset: 0,
            query_rope_position_offset: 7,
        },
        VariableLengthSequenceMetadata {
            active_query_len: 4,
            active_kv_len: 7,
            query_position_offset: 0,
            query_rope_position_offset: 19,
        },
        VariableLengthSequenceMetadata {
            active_query_len: 6,
            active_kv_len: 10,
            query_position_offset: 0,
            query_rope_position_offset: 31,
        },
    ];
    run_case(
        &harness,
        shape,
        &metadata,
        FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(0.125),
        },
        VariableLengthRotaryEmbeddingConfig {
            theta: 500_000.0,
            kv_position_offset: 5,
        },
        0.4,
    );
}

#[test]
fn variable_pipeline_rejects_metadata_length_and_causal_overflow() {
    let Some(harness) = harness() else {
        return;
    };
    let pipeline = ExternalVariableProjectionRotaryGroupedPipeline::new(&harness.device).unwrap();
    let shape = AsymmetricGroupedAttentionShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        query_len: 2,
        kv_len: 4,
        head_dim: 32,
        query_position_offset: 0,
    };
    let q_data = vec![0.0; shape.q_tensor_len().unwrap()];
    let kv_data = vec![0.0; shape.kv_tensor_len().unwrap()];
    let q = input_buffer(&harness.device, &harness.queue, &q_data);
    let kv = input_buffer(&harness.device, &harness.queue, &kv_data);
    let output = pipeline
        .create_output_buffer(&harness.device, shape)
        .unwrap();
    let rotary = VariableLengthRotaryEmbeddingConfig {
        theta: 10_000.0,
        kv_position_offset: 0,
    };

    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m12-validation"),
        });
    let error = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            ExternalVariableProjectionPass {
                q: &q,
                k: &kv,
                v: &kv,
                out_and_lse: &output,
                shape,
                metadata: &[],
                config: FlatAttentionConfig::default(),
                rotary,
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ExternalWgpuError::Core(FlatAttentionError::LengthMismatch {
            tensor: "active sequence metadata",
            ..
        })
    ));

    let overflowing = [VariableLengthSequenceMetadata {
        active_query_len: 1,
        active_kv_len: 1,
        query_position_offset: u32::MAX as usize,
        query_rope_position_offset: 0,
    }];
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m12-overflow-validation"),
        });
    let error = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            ExternalVariableProjectionPass {
                q: &q,
                k: &kv,
                v: &kv,
                out_and_lse: &output,
                shape,
                metadata: &overflowing,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                rotary,
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ExternalWgpuError::Core(FlatAttentionError::PositionOverflow)
    ));
}
