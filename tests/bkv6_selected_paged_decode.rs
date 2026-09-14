#![cfg(feature = "wgpu")]

use std::sync::mpsc;

pub use flat_attention::{FlatAttentionConfig, FlatAttentionError, WGSL_MAX_HEAD_DIM};

pub mod paged_kv {
    pub use flat_attention::paged_kv::*;
}

#[path = "../src/boolean_kv.rs"]
pub mod boolean_kv;

pub mod api {
    pub use crate::boolean_kv;
    pub use crate::boolean_kv_paged_selection;
}

#[path = "../src/boolean_kv_paged_selection.rs"]
pub mod boolean_kv_paged_selection;

#[path = "../src/wgpu_boolean_selected_paged_decode.rs"]
pub mod wgpu_boolean_selected_paged_decode;

use boolean_kv::{BooleanKvCache, PackedBooleanSignature};
use boolean_kv_paged_selection::{
    build_boolean_indexed_kv_selection, BooleanIndexedKvSelection, NumericalKvPageGeometry,
};
use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricGroupedAttentionShape,
    AsymmetricRotaryEmbeddingConfig,
};
use wgpu_boolean_selected_paged_decode::{
    BooleanSelectedDecodePass, BooleanSelectedPagedDecodeError, BooleanSelectedPagedKvTable,
    WgpuBooleanSelectedPagedDecodePipeline, BOOLEAN_SELECTED_PAGED_DECODE_WGSL,
};

const ATOL: f32 = 2.0e-4;
const RTOL: f32 = 1.0e-3;

struct DeviceHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
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
            panic!("BKV-K6 selected paged decode requires a WGPU adapter in the mandatory gate");
        }
        eprintln!("WGPU adapter unavailable; optional BKV-K6 device test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-bkv-k6-selected-paged-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("BKV-K6 request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.019 + phase;
            x.sin() * 1.125 + (x * 0.37).cos() * 0.3125
        })
        .collect()
}

fn signature(byte: u8) -> PackedBooleanSignature {
    let bits = (0..8).map(|bit| byte & (1 << bit) != 0).collect::<Vec<_>>();
    PackedBooleanSignature::from_bools(&bits).unwrap()
}

fn table_and_cache() -> (PagedKvTable, BooleanKvCache) {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 2,
        physical_pages: 4,
    })
    .unwrap();
    table.append(6).unwrap();

    let mut cache = BooleanKvCache::new(8).unwrap();
    cache.append(signature(0), None).unwrap();
    cache.append(signature(0xff), None).unwrap();
    cache.append(signature(0x01), None).unwrap();
    (table, cache)
}

fn selection(
    cache: &BooleanKvCache,
    table: &PagedKvTable,
    max_distance: usize,
) -> BooleanIndexedKvSelection {
    build_boolean_indexed_kv_selection(
        cache,
        table,
        &signature(0),
        max_distance,
        None,
        NumericalKvPageGeometry {
            kv_heads: 1,
            head_dim: 64,
            scalar_bytes: 4,
        },
    )
    .unwrap()
}

fn rotate_vector(row: &[f32], position: usize, theta: f32) -> Vec<f32> {
    let head_dim = row.len();
    let mut rotated = row.to_vec();
    for pair in 0..head_dim / 2 {
        let dim = 2 * pair;
        let exponent = -2.0 * pair as f32 / head_dim as f32;
        let angle = position as f32 * theta.powf(exponent);
        let (sin, cos) = angle.sin_cos();
        let even = row[dim];
        let odd = row[dim + 1];
        rotated[dim] = even * cos - odd * sin;
        rotated[dim + 1] = even * sin + odd * cos;
    }
    rotated
}

fn rotate_k_projection(
    raw: &[f32],
    kv_len: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
) -> Vec<f32> {
    let width = kv_heads * head_dim;
    let mut rotated = raw.to_vec();
    for position in 0..kv_len {
        for head in 0..kv_heads {
            let base = position * width + head * head_dim;
            let row = rotate_vector(&raw[base..base + head_dim], position, theta);
            rotated[base..base + head_dim].copy_from_slice(&row);
        }
    }
    rotated
}

#[derive(Clone, Copy)]
struct TestAttentionGeometry {
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    query_position: usize,
}

#[derive(Clone, Copy)]
struct DeviceInputs<'a> {
    q: &'a wgpu::Buffer,
    k: &'a wgpu::Buffer,
    v: &'a wgpu::Buffer,
}

fn restricted_oracle(
    q: &[f32],
    raw_k: &[f32],
    v: &[f32],
    geometry: TestAttentionGeometry,
    selected_positions: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let TestAttentionGeometry {
        q_heads,
        kv_heads,
        head_dim,
        theta,
        query_position,
    } = geometry;
    let mut output = vec![0.0f32; q_heads * head_dim];
    let mut lse = vec![0.0f32; q_heads];
    let group_size = q_heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_width = kv_heads * head_dim;

    for (q_head, lse_value) in lse.iter_mut().enumerate() {
        let kv_head = q_head / group_size;
        let q_base = q_head * head_dim;
        let rotated_q = rotate_vector(&q[q_base..q_base + head_dim], query_position, theta);
        let mut scores = Vec::with_capacity(selected_positions.len());
        for &position in selected_positions {
            let k_base = position * kv_width + kv_head * head_dim;
            let rotated_k = rotate_vector(&raw_k[k_base..k_base + head_dim], position, theta);
            let score = rotated_q
                .iter()
                .zip(&rotated_k)
                .map(|(left, right)| left * right)
                .sum::<f32>()
                * scale;
            scores.push(score);
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .sum::<f32>();
        *lse_value = maximum + denominator.ln();
        for (score, &position) in scores.iter().zip(selected_positions) {
            let weight = (*score - maximum).exp() / denominator;
            let v_base = position * kv_width + kv_head * head_dim;
            for dim in 0..head_dim {
                output[q_base + dim] += weight * v[v_base + dim];
            }
        }
    }
    (output, lse)
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
        label: Some("flat-bkv-k6-input"),
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
        label: Some("flat-bkv-k6-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-bkv-k6-readback"),
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

fn run_selected(
    harness: &DeviceHarness,
    pipeline: &WgpuBooleanSelectedPagedDecodePipeline,
    page_table: &BooleanSelectedPagedKvTable,
    authoritative_table: &PagedKvTable,
    inputs: DeviceInputs<'_>,
    geometry: TestAttentionGeometry,
) -> Vec<f32> {
    let TestAttentionGeometry {
        q_heads,
        kv_heads,
        head_dim,
        theta,
        query_position,
    } = geometry;
    let output = pipeline
        .create_output_buffer(&harness.device, q_heads, head_dim)
        .unwrap();
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-bkv-k6-selected-paged"),
        });
    let layout = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            BooleanSelectedDecodePass {
                q: inputs.q,
                k: inputs.k,
                v: inputs.v,
                page_table,
                authoritative_table,
                out_and_lse: &output,
                q_heads,
                kv_heads,
                head_dim,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                theta,
                q_rope_position: query_position,
                q_causal_position: query_position,
            },
        )
        .unwrap();
    harness.queue.submit(Some(encoder.finish()));
    read_f32(
        &harness.device,
        &harness.queue,
        &output,
        layout.combined_elements,
    )
}

#[test]
fn selected_table_preserves_original_page_identity_and_fails_closed() {
    let (table, cache) = table_and_cache();
    let sparse = selection(&cache, &table, 1);
    assert_eq!(sparse.selected_page_ids(), vec![0, 2]);
    let selected = BooleanSelectedPagedKvTable::from_selection(&sparse, &table).unwrap();
    assert_eq!(selected.full_live_tokens(), 6);
    assert_eq!(selected.selected_live_tokens(), 4);
    assert_eq!(selected.entries().len(), 2);
    assert_eq!(selected.entries()[0].logical_page, 0);
    assert_eq!(selected.entries()[1].logical_page, 2);

    let mut forged = sparse.clone();
    forged.selected_pages[0].physical_page = 3;
    assert!(matches!(
        BooleanSelectedPagedKvTable::from_selection(&forged, &table),
        Err(BooleanSelectedPagedDecodeError::PhysicalPageMismatch {
            logical_page: 0,
            ..
        })
    ));

    let mut stale = sparse;
    stale.generation += 1;
    assert!(matches!(
        BooleanSelectedPagedKvTable::from_selection(&stale, &table),
        Err(BooleanSelectedPagedDecodeError::GenerationMismatch { .. })
    ));

    let mut changed_table = table;
    changed_table.truncate(2).unwrap();
    assert!(matches!(
        selected.validate_against(&changed_table),
        Err(BooleanSelectedPagedDecodeError::SnapshotMismatch { .. })
    ));
    changed_table.reset().unwrap();
    assert!(matches!(
        selected.validate_against(&changed_table),
        Err(BooleanSelectedPagedDecodeError::GenerationMismatch { .. })
    ));
}

#[test]
fn selected_paged_shader_parses_and_validates() {
    let module = naga::front::wgsl::parse_str(BOOLEAN_SELECTED_PAGED_DECODE_WGSL)
        .expect("BKV-K6 selected paged WGSL must parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("BKV-K6 selected paged WGSL must validate");
}

#[test]
fn selected_decode_matches_all_accept_and_sparse_original_position_oracles() {
    let Some(harness) = harness() else {
        return;
    };
    let (table, cache) = table_and_cache();
    let (q_heads, kv_heads, kv_len, page_size, physical_pages, head_dim) =
        (2usize, 1usize, 6usize, 2usize, 4usize, 64usize);
    let theta = 10_000.0;
    let query_position = kv_len - 1;
    let q = fixture(q_heads * head_dim, 0.2);
    let raw_k = fixture(kv_len * kv_heads * head_dim, 0.8);
    let v = fixture(kv_len * kv_heads * head_dim, 1.4);

    let rotated_k = rotate_k_projection(&raw_k, kv_len, kv_heads, head_dim, theta);
    let physical_rows = page_size * physical_pages;
    let width = kv_heads * head_dim;
    let mut physical_k = vec![17.0f32; physical_rows * width];
    let mut physical_v = vec![-19.0f32; physical_rows * width];
    physical_k[..rotated_k.len()].copy_from_slice(&rotated_k);
    physical_v[..v.len()].copy_from_slice(&v);

    let q_gpu = input_buffer(&harness.device, &harness.queue, &q);
    let k_gpu = input_buffer(&harness.device, &harness.queue, &physical_k);
    let v_gpu = input_buffer(&harness.device, &harness.queue, &physical_v);
    let pipeline = WgpuBooleanSelectedPagedDecodePipeline::new(&harness.device).unwrap();
    let geometry = TestAttentionGeometry {
        q_heads,
        kv_heads,
        head_dim,
        theta,
        query_position,
    };
    let inputs = DeviceInputs {
        q: &q_gpu,
        k: &k_gpu,
        v: &v_gpu,
    };

    let all_selection = selection(&cache, &table, 8);
    assert_eq!(all_selection.selected_page_ids(), vec![0, 1, 2]);
    let all_table = BooleanSelectedPagedKvTable::from_selection(&all_selection, &table).unwrap();
    let all_actual = run_selected(&harness, &pipeline, &all_table, &table, inputs, geometry);
    let dense_expected = forward_reference_projection_grouped_rope_asymmetric(
        &q,
        &raw_k,
        &v,
        AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads,
            kv_heads,
            query_len: 1,
            kv_len,
            head_dim,
            query_position_offset: query_position,
        },
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        },
        AsymmetricRotaryEmbeddingConfig {
            theta,
            query_position_offset: query_position,
            kv_position_offset: 0,
        },
    )
    .unwrap();
    assert_close(
        "BKV-K6 all-accept O",
        &all_actual[..q_heads * head_dim],
        &dense_expected.output,
    );
    assert_close(
        "BKV-K6 all-accept LSE",
        &all_actual[q_heads * head_dim..],
        &dense_expected.lse,
    );

    let sparse_selection = selection(&cache, &table, 1);
    assert_eq!(sparse_selection.selected_page_ids(), vec![0, 2]);
    let sparse_table =
        BooleanSelectedPagedKvTable::from_selection(&sparse_selection, &table).unwrap();
    let sparse_actual = run_selected(&harness, &pipeline, &sparse_table, &table, inputs, geometry);
    let (sparse_output, sparse_lse) = restricted_oracle(&q, &raw_k, &v, geometry, &[0, 1, 4, 5]);
    assert_close(
        "BKV-K6 sparse O",
        &sparse_actual[..q_heads * head_dim],
        &sparse_output,
    );
    assert_close(
        "BKV-K6 sparse LSE",
        &sparse_actual[q_heads * head_dim..],
        &sparse_lse,
    );
}
