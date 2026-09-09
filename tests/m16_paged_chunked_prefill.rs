#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::paged_kv::{
    PagedChunkedPrefillError, PagedChunkedPrefillPass, PagedKvConfig,
    WgpuPagedChunkedPrefillPipeline, WgpuPagedKvCache,
};
use flat_attention::{
    forward_reference_projection_grouped_rope, FlatAttentionConfig, GroupedAttentionShape,
    PagedDecodeError, RotaryEmbeddingConfig,
};

const ATOL: f32 = 2.0e-4;
const RTOL: f32 = 1.0e-3;

struct DeviceHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[derive(Clone, Copy)]
struct Case {
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    page_size: usize,
    physical_pages: usize,
    causal: bool,
    chunk_size: usize,
    position_offset: usize,
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
    }));
    let Ok(adapter) = adapter else {
        if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
            panic!(
                "M16 paged chunked prefill requires a WGPU adapter in the mandatory device gate"
            );
        }
        eprintln!("WGPU adapter unavailable; optional M16 paged chunked-prefill test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-m16-paged-chunked-prefill-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("M16 request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.017 + phase;
            x.sin() * 1.15625 + (x * 0.41).cos() * 0.328125
        })
        .collect()
}

fn rotate_k_projection(
    raw: &[f32],
    kv_len: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Vec<f32> {
    let mut rotated = raw.to_vec();
    let width = kv_heads * head_dim;
    for position in 0..kv_len {
        let absolute_position = position_offset + position;
        for head in 0..kv_heads {
            let head_base = position * width + head * head_dim;
            for pair in 0..head_dim / 2 {
                let dim = 2 * pair;
                let exponent = -2.0 * pair as f32 / head_dim as f32;
                let angle = absolute_position as f32 * theta.powf(exponent);
                let (sin, cos) = angle.sin_cos();
                let even = raw[head_base + dim];
                let odd = raw[head_base + dim + 1];
                rotated[head_base + dim] = even * cos - odd * sin;
                rotated[head_base + dim + 1] = even * sin + odd * cos;
            }
        }
    }
    rotated
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
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m16-paged-chunked-prefill-input"),
        size: bytes.len().max(4) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
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
        label: Some("flat-m16-paged-chunked-prefill-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m16-paged-chunked-prefill-readback"),
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

fn run_case(harness: &DeviceHarness, case: Case) {
    let theta = 10_000.0;
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: case.q_heads,
        kv_heads: case.kv_heads,
        seq_len: case.seq_len,
        head_dim: case.head_dim,
    };
    let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
    let raw_k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
    let config = FlatAttentionConfig {
        causal: case.causal,
        softmax_scale: None,
    };
    let rotary = RotaryEmbeddingConfig {
        theta,
        position_offset: case.position_offset,
    };
    let expected =
        forward_reference_projection_grouped_rope(&q, &raw_k, &v, shape, config, rotary).unwrap();
    let rotated_k = rotate_k_projection(
        &raw_k,
        case.seq_len,
        case.kv_heads,
        case.head_dim,
        theta,
        case.position_offset,
    );

    let q_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &q,
        wgpu::BufferUsages::COPY_SRC,
    );
    let k_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &rotated_k,
        wgpu::BufferUsages::COPY_SRC,
    );
    let v_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &v,
        wgpu::BufferUsages::COPY_SRC,
    );

    let mut cache = WgpuPagedKvCache::new(
        &harness.device,
        PagedKvConfig {
            page_size: case.page_size,
            physical_pages: case.physical_pages,
        },
        case.kv_heads,
        case.head_dim,
    )
    .unwrap();
    let (new_len, _) = cache
        .append_and_submit(
            &harness.device,
            &harness.queue,
            &k_gpu,
            &v_gpu,
            case.seq_len,
        )
        .unwrap();
    assert_eq!(new_len, case.seq_len);
    assert!(cache.table().telemetry().unwrap().mapped_pages > 1);

    let pipeline = WgpuPagedChunkedPrefillPipeline::new(&harness.device).unwrap();
    let output = pipeline
        .create_output_buffer(&harness.device, &cache, case.q_heads)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-chunked-prefill"),
        });
    let layout = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            PagedChunkedPrefillPass {
                q: &q_gpu,
                cache: &cache,
                out_and_lse: &output,
                q_heads: case.q_heads,
                config,
                theta,
                query_position_offset: case.position_offset,
                query_chunk_size: case.chunk_size,
            },
        )
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));

    let actual = read_f32(
        &harness.device,
        &harness.queue,
        &output,
        layout.combined_elements,
    );
    assert_close(
        "M16 paged chunked O",
        &actual[..layout.output_elements],
        &expected.output,
    );
    assert_close(
        "M16 paged chunked LSE",
        &actual[layout.output_elements..],
        &expected.lse,
    );
}

#[test]
fn paged_chunked_prefill_matches_contiguous_oracle_across_page_boundaries() {
    let Some(harness) = harness() else {
        return;
    };

    for case in [
        Case {
            q_heads: 4,
            kv_heads: 2,
            seq_len: 7,
            head_dim: 32,
            page_size: 3,
            physical_pages: 4,
            causal: true,
            chunk_size: 2,
            position_offset: 5,
        },
        Case {
            q_heads: 4,
            kv_heads: 1,
            seq_len: 5,
            head_dim: 32,
            page_size: 2,
            physical_pages: 4,
            causal: true,
            chunk_size: 3,
            position_offset: 7,
        },
        Case {
            q_heads: 4,
            kv_heads: 2,
            seq_len: 7,
            head_dim: 32,
            page_size: 3,
            physical_pages: 4,
            causal: false,
            chunk_size: 4,
            position_offset: 3,
        },
    ] {
        run_case(&harness, case);
    }
}

#[test]
fn paged_chunked_prefill_rejects_cache_with_unsubmitted_recorded_writes() {
    let Some(harness) = harness() else {
        return;
    };
    let q_heads = 2usize;
    let kv_heads = 1usize;
    let seq_len = 3usize;
    let head_dim = 32usize;
    let theta = 10_000.0;
    let q = fixture(seq_len * q_heads * head_dim, 0.2);
    let raw_k = fixture(seq_len * kv_heads * head_dim, 0.8);
    let rotated_k = rotate_k_projection(&raw_k, seq_len, kv_heads, head_dim, theta, 0);
    let v = fixture(seq_len * kv_heads * head_dim, 1.4);

    let q_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &q,
        wgpu::BufferUsages::COPY_SRC,
    );
    let k_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &rotated_k,
        wgpu::BufferUsages::COPY_SRC,
    );
    let v_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &v,
        wgpu::BufferUsages::COPY_SRC,
    );
    let mut cache = WgpuPagedKvCache::new(
        &harness.device,
        PagedKvConfig {
            page_size: 2,
            physical_pages: 2,
        },
        kv_heads,
        head_dim,
    )
    .unwrap();
    let mut append_encoder =
        harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-m16-paged-chunked-prefill-unsubmitted-append"),
            });
    cache
        .record_append(&mut append_encoder, &k_gpu, &v_gpu, seq_len)
        .unwrap();
    assert!(cache.has_unsubmitted_recorded_writes());

    let pipeline = WgpuPagedChunkedPrefillPipeline::new(&harness.device).unwrap();
    let output = pipeline
        .create_output_buffer(&harness.device, &cache, q_heads)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-chunked-prefill-taint-rejection"),
        });
    let error = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            PagedChunkedPrefillPass {
                q: &q_gpu,
                cache: &cache,
                out_and_lse: &output,
                q_heads,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                theta,
                query_position_offset: 0,
                query_chunk_size: 2,
            },
        )
        .expect_err("unsubmitted cache writes must fail closed");
    assert!(matches!(
        error,
        PagedChunkedPrefillError::CacheHasUnsubmittedRecordedWrites
    ));

    drop(append_encoder);
}

#[test]
fn paged_chunked_prefill_layout_rejects_odd_rotary_head_dim() {
    let Some(harness) = harness() else {
        return;
    };
    let head_dim = 31usize;
    let k_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &fixture(head_dim, 0.8),
        wgpu::BufferUsages::COPY_SRC,
    );
    let v_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &fixture(head_dim, 1.4),
        wgpu::BufferUsages::COPY_SRC,
    );
    let mut cache = WgpuPagedKvCache::new(
        &harness.device,
        PagedKvConfig {
            page_size: 1,
            physical_pages: 1,
        },
        1,
        head_dim,
    )
    .unwrap();
    cache
        .append_and_submit(&harness.device, &harness.queue, &k_gpu, &v_gpu, 1)
        .unwrap();

    let error = WgpuPagedChunkedPrefillPipeline::layout(&cache, 2)
        .expect_err("odd rotary head dimensions must be rejected by layout");
    assert!(matches!(
        error,
        PagedChunkedPrefillError::Decode(PagedDecodeError::Core(
            flat_attention::FlatAttentionError::InvalidRotaryHeadDim { head_dim: 31 }
        ))
    ));
}

#[test]
fn paged_chunked_prefill_preflights_rope_u32_range_before_recording() {
    if usize::BITS <= 32 {
        return;
    }
    let Some(harness) = harness() else {
        return;
    };
    let q_heads = 2usize;
    let kv_heads = 1usize;
    let seq_len = 2usize;
    let head_dim = 32usize;
    let q = fixture(seq_len * q_heads * head_dim, 0.2);
    let k = fixture(seq_len * kv_heads * head_dim, 0.8);
    let v = fixture(seq_len * kv_heads * head_dim, 1.4);
    let q_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &q,
        wgpu::BufferUsages::COPY_SRC,
    );
    let k_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &k,
        wgpu::BufferUsages::COPY_SRC,
    );
    let v_gpu = input_buffer(
        &harness.device,
        &harness.queue,
        &v,
        wgpu::BufferUsages::COPY_SRC,
    );
    let mut cache = WgpuPagedKvCache::new(
        &harness.device,
        PagedKvConfig {
            page_size: 1,
            physical_pages: 2,
        },
        kv_heads,
        head_dim,
    )
    .unwrap();
    cache
        .append_and_submit(&harness.device, &harness.queue, &k_gpu, &v_gpu, seq_len)
        .unwrap();

    let pipeline = WgpuPagedChunkedPrefillPipeline::new(&harness.device).unwrap();
    let output = pipeline
        .create_output_buffer(&harness.device, &cache, q_heads)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-chunked-prefill-u32-preflight"),
        });
    let error = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            PagedChunkedPrefillPass {
                q: &q_gpu,
                cache: &cache,
                out_and_lse: &output,
                q_heads,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                theta: 10_000.0,
                query_position_offset: u32::MAX as usize,
                query_chunk_size: 2,
            },
        )
        .expect_err("RoPE positions outside the shader u32 domain must fail in preflight");
    assert!(matches!(
        error,
        PagedChunkedPrefillError::Decode(PagedDecodeError::IndexSpaceExceeded { elements })
            if elements == u32::MAX as usize + 1
    ));

    harness.queue.submit(Some(encoder.finish()));
}
