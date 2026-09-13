#![cfg(feature = "wgpu")]

use std::hint::black_box;
use std::process::Command;
use std::sync::mpsc;
use std::time::Instant;

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

#[path = "../src/research_bkv_qualification.rs"]
pub mod research_bkv_qualification;

use boolean_kv::{BooleanKvCache, PackedBooleanSignature};
use boolean_kv_paged_selection::{
    build_boolean_indexed_kv_selection, BooleanIndexedKvSelection, NumericalKvPageGeometry,
};
use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::{PagedDecodePass, WgpuPagedDecodePipeline, WgpuPagedKvTable};
use research_bkv_qualification::{
    BikvAccountingInput, BikvLatencyInput, BikvPromotionDecision, BikvQualificationRecord,
};
use wgpu_boolean_selected_paged_decode::{
    BooleanSelectedDecodePass, BooleanSelectedPagedKvTable, WgpuBooleanSelectedPagedDecodePipeline,
};

const ATOL: f32 = 2.0e-4;
const RTOL: f32 = 1.0e-3;

#[derive(Clone, Copy)]
struct Geometry {
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    physical_pages: usize,
    kv_len: usize,
    theta: f32,
}

#[derive(Clone, Copy)]
struct DeviceInputs<'a> {
    q: &'a wgpu::Buffer,
    k: &'a wgpu::Buffer,
    v: &'a wgpu::Buffer,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos().max(1)).unwrap_or(u64::MAX)
}

fn median(mut samples: Vec<u64>) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.019 + phase;
            x.sin() * 1.125 + (x * 0.37).cos() * 0.3125
        })
        .collect()
}

fn signature_from_query(q: &[f32], q_heads: usize, head_dim: usize) -> PackedBooleanSignature {
    let bits = (0..8)
        .map(|dim| {
            (0..q_heads)
                .map(|head| q[head * head_dim + dim])
                .sum::<f32>()
                >= 0.0
        })
        .collect::<Vec<_>>();
    PackedBooleanSignature::from_bools(&bits).unwrap()
}

fn signature_variant(query: &PackedBooleanSignature, page: usize) -> PackedBooleanSignature {
    let mut byte = query.words()[0] as u8;
    match page % 3 {
        0 => {}
        1 => byte ^= 0xff,
        _ => byte ^= 0x01,
    }
    let bits = (0..8).map(|bit| byte & (1 << bit) != 0).collect::<Vec<_>>();
    PackedBooleanSignature::from_bools(&bits).unwrap()
}

fn build_boolean_cache(query: &PackedBooleanSignature, pages: usize) -> BooleanKvCache {
    let mut cache = BooleanKvCache::new(8).unwrap();
    for page in 0..pages {
        cache.append(signature_variant(query, page), None).unwrap();
    }
    cache
}

fn rotate_vector(row: &[f32], position: usize, theta: f32) -> Vec<f32> {
    let mut rotated = row.to_vec();
    let head_dim = row.len();
    for pair in 0..head_dim / 2 {
        let dim = pair * 2;
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

fn rotate_k(raw: &[f32], geometry: Geometry) -> Vec<f32> {
    let width = geometry.kv_heads * geometry.head_dim;
    let mut rotated = raw.to_vec();
    for position in 0..geometry.kv_len {
        for head in 0..geometry.kv_heads {
            let base = position * width + head * geometry.head_dim;
            let row = rotate_vector(
                &raw[base..base + geometry.head_dim],
                position,
                geometry.theta,
            );
            rotated[base..base + geometry.head_dim].copy_from_slice(&row);
        }
    }
    rotated
}

fn physicalize(logical: &[f32], table: &PagedKvTable, geometry: Geometry, fill: f32) -> Vec<f32> {
    let width = geometry.kv_heads * geometry.head_dim;
    let mut physical = vec![fill; geometry.physical_pages * geometry.page_size * width];
    for token in 0..geometry.kv_len {
        let address = table.address(token).unwrap();
        let physical_row = address.physical_page * geometry.page_size + address.offset_in_page;
        let source = token * width;
        let destination = physical_row * width;
        physical[destination..destination + width]
            .copy_from_slice(&logical[source..source + width]);
    }
    physical
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
        label: Some("flat-bkv-k6-m16-qualification-input"),
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
        label: Some("flat-bkv-k6-m16-qualification-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-bkv-k6-m16-qualification-readback"),
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

fn selected_positions(selection: &BooleanIndexedKvSelection, page_size: usize) -> Vec<usize> {
    let mut positions = Vec::new();
    for page in &selection.selected_pages {
        let start = page.logical_page * page_size;
        positions.extend(start..start + page.live_tokens);
    }
    positions
}

fn restricted_oracle(
    q: &[f32],
    raw_k: &[f32],
    v: &[f32],
    geometry: Geometry,
    positions: &[usize],
) -> Vec<f32> {
    let width = geometry.kv_heads * geometry.head_dim;
    let group_size = geometry.q_heads / geometry.kv_heads;
    let scale = 1.0 / (geometry.head_dim as f32).sqrt();
    let output_len = geometry.q_heads * geometry.head_dim;
    let mut combined = vec![0.0f32; output_len + geometry.q_heads];
    for q_head in 0..geometry.q_heads {
        let kv_head = q_head / group_size;
        let q_base = q_head * geometry.head_dim;
        let rq = rotate_vector(
            &q[q_base..q_base + geometry.head_dim],
            geometry.kv_len - 1,
            geometry.theta,
        );
        let scores = positions
            .iter()
            .map(|&position| {
                let k_base = position * width + kv_head * geometry.head_dim;
                let rk = rotate_vector(
                    &raw_k[k_base..k_base + geometry.head_dim],
                    position,
                    geometry.theta,
                );
                rq.iter()
                    .zip(rk)
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    * scale
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .sum::<f32>();
        combined[output_len + q_head] = maximum + denominator.ln();
        for (&score, &position) in scores.iter().zip(positions) {
            let weight = (score - maximum).exp() / denominator;
            let v_base = position * width + kv_head * geometry.head_dim;
            for dim in 0..geometry.head_dim {
                combined[q_base + dim] += weight * v[v_base + dim];
            }
        }
    }
    combined
}

fn close(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&actual, &expected)| (actual - expected).abs() <= ATOL + RTOL * expected.abs())
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f32::max)
}

#[allow(clippy::too_many_arguments)]
fn encode_selected(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &WgpuBooleanSelectedPagedDecodePipeline,
    selected: &BooleanSelectedPagedKvTable,
    table: &PagedKvTable,
    output: &wgpu::Buffer,
    inputs: DeviceInputs<'_>,
    geometry: Geometry,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-bkv-k6-m16-selected"),
    });
    pipeline
        .encode(
            device,
            &mut encoder,
            BooleanSelectedDecodePass {
                q: inputs.q,
                k: inputs.k,
                v: inputs.v,
                page_table: selected,
                authoritative_table: table,
                out_and_lse: output,
                q_heads: geometry.q_heads,
                kv_heads: geometry.kv_heads,
                head_dim: geometry.head_dim,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                theta: geometry.theta,
                q_rope_position: geometry.kv_len - 1,
                q_causal_position: geometry.kv_len - 1,
            },
        )
        .unwrap();
    queue.submit(Some(encoder.finish()));
}

fn encode_dense(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &WgpuPagedDecodePipeline,
    page_table: &WgpuPagedKvTable,
    output: &wgpu::Buffer,
    inputs: DeviceInputs<'_>,
    geometry: Geometry,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-bkv-k6-m16-dense"),
    });
    pipeline
        .encode(
            device,
            &mut encoder,
            PagedDecodePass {
                q: inputs.q,
                k: inputs.k,
                v: inputs.v,
                page_table,
                out_and_lse: output,
                q_heads: geometry.q_heads,
                kv_heads: geometry.kv_heads,
                head_dim: geometry.head_dim,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                theta: geometry.theta,
                q_rope_position: geometry.kv_len - 1,
                q_causal_position: geometry.kv_len - 1,
            },
        )
        .unwrap();
    queue.submit(Some(encoder.finish()));
}

fn git_head() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

#[test]
fn bkv_k6_m16_same_buffer_qualification_harness() {
    let pages = env_usize("FLAT_BKV_QUAL_PAGES", 16);
    let page_size = env_usize("FLAT_BKV_QUAL_PAGE_SIZE", 8);
    let warmup = env_usize("FLAT_BKV_QUAL_WARMUP", 3);
    let iterations = env_usize("FLAT_BKV_QUAL_ITERS", 9);
    let max_distance = std::env::var("FLAT_BKV_QUAL_MAX_DISTANCE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .min(8);
    let geometry = Geometry {
        q_heads: 4,
        kv_heads: 1,
        head_dim: 64,
        page_size,
        physical_pages: pages.checked_add(2).expect("physical page overflow"),
        kv_len: pages.checked_mul(page_size).expect("KV length overflow"),
        theta: 10_000.0,
    };

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }));
    let Ok(adapter) = adapter else {
        if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
            panic!("BKV-K6/M16 qualification requires a WGPU adapter in the mandatory gate");
        }
        eprintln!("WGPU adapter unavailable; optional BKV-K6/M16 qualification skipped");
        return;
    };
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-bkv-k6-m16-qualification"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .expect("BKV-K6/M16 qualification request_device failed");

    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size,
        physical_pages: geometry.physical_pages,
    })
    .unwrap();
    table.append(geometry.kv_len).unwrap();
    let q = fixture(geometry.q_heads * geometry.head_dim, 0.2);
    let raw_k = fixture(geometry.kv_len * geometry.kv_heads * geometry.head_dim, 0.8);
    let v = fixture(geometry.kv_len * geometry.kv_heads * geometry.head_dim, 1.4);
    let rotated_k = rotate_k(&raw_k, geometry);
    let physical_k = physicalize(&rotated_k, &table, geometry, 17.0);
    let physical_v = physicalize(&v, &table, geometry, -19.0);
    let q_gpu = input_buffer(&device, &queue, &q);
    let k_gpu = input_buffer(&device, &queue, &physical_k);
    let v_gpu = input_buffer(&device, &queue, &physical_v);
    let inputs = DeviceInputs {
        q: &q_gpu,
        k: &k_gpu,
        v: &v_gpu,
    };

    let query = signature_from_query(&q, geometry.q_heads, geometry.head_dim);
    let cache = build_boolean_cache(&query, pages);
    let numerical_geometry = NumericalKvPageGeometry {
        kv_heads: geometry.kv_heads,
        head_dim: geometry.head_dim,
        scalar_bytes: std::mem::size_of::<f32>(),
    };
    let candidate = build_boolean_indexed_kv_selection(
        &cache,
        &table,
        &query,
        max_distance,
        None,
        numerical_geometry,
    )
    .unwrap();
    assert!(
        !candidate.selected_pages.is_empty(),
        "qualification threshold selected zero pages"
    );
    let selected_table = BooleanSelectedPagedKvTable::from_selection(&candidate, &table).unwrap();
    let all_accept =
        build_boolean_indexed_kv_selection(&cache, &table, &query, 8, None, numerical_geometry)
            .unwrap();
    let all_accept_table =
        BooleanSelectedPagedKvTable::from_selection(&all_accept, &table).unwrap();

    let selected_pipeline = WgpuBooleanSelectedPagedDecodePipeline::new(&device).unwrap();
    let dense_pipeline = WgpuPagedDecodePipeline::new(&device).unwrap();
    let dense_table = WgpuPagedKvTable::from_table(&table).unwrap();
    let selected_output = selected_pipeline
        .create_output_buffer(&device, geometry.q_heads, geometry.head_dim)
        .unwrap();
    let all_accept_output = selected_pipeline
        .create_output_buffer(&device, geometry.q_heads, geometry.head_dim)
        .unwrap();
    let dense_output = dense_pipeline
        .create_output_buffer(&device, geometry.q_heads, geometry.head_dim)
        .unwrap();

    encode_selected(
        &device,
        &queue,
        &selected_pipeline,
        &all_accept_table,
        &table,
        &all_accept_output,
        inputs,
        geometry,
    );
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    encode_dense(
        &device,
        &queue,
        &dense_pipeline,
        &dense_table,
        &dense_output,
        inputs,
        geometry,
    );
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let combined_elements = geometry.q_heads * geometry.head_dim + geometry.q_heads;
    let all_accept_values = read_f32(&device, &queue, &all_accept_output, combined_elements);
    let dense_values = read_f32(&device, &queue, &dense_output, combined_elements);
    let all_accept_parity = close(&all_accept_values, &dense_values);

    encode_selected(
        &device,
        &queue,
        &selected_pipeline,
        &selected_table,
        &table,
        &selected_output,
        inputs,
        geometry,
    );
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let selected_values = read_f32(&device, &queue, &selected_output, combined_elements);
    let sparse_oracle = restricted_oracle(
        &q,
        &raw_k,
        &v,
        geometry,
        &selected_positions(&candidate, page_size),
    );
    let sparse_correctness = close(&selected_values, &sparse_oracle);
    let quality_gate_passed = close(&selected_values, &dense_values);
    let correctness_gate_passed = all_accept_parity && sparse_correctness;
    assert!(
        all_accept_parity,
        "K6 all-accept must match M16 on identical buffers"
    );
    assert!(
        sparse_correctness,
        "K6 sparse path must match its restricted numerical oracle"
    );

    for _ in 0..warmup {
        let _ = black_box(signature_from_query(
            &q,
            geometry.q_heads,
            geometry.head_dim,
        ));
        let warm_selection = build_boolean_indexed_kv_selection(
            &cache,
            &table,
            &query,
            max_distance,
            None,
            numerical_geometry,
        )
        .unwrap();
        let _ = black_box(
            BooleanSelectedPagedKvTable::from_selection(&warm_selection, &table).unwrap(),
        );
        encode_selected(
            &device,
            &queue,
            &selected_pipeline,
            &selected_table,
            &table,
            &selected_output,
            inputs,
            geometry,
        );
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        encode_dense(
            &device,
            &queue,
            &dense_pipeline,
            &dense_table,
            &dense_output,
            inputs,
            geometry,
        );
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }

    let mut signature_samples = Vec::with_capacity(iterations);
    let mut search_samples = Vec::with_capacity(iterations);
    let mut selected_submit_samples = Vec::with_capacity(iterations);
    let mut selected_sync_samples = Vec::with_capacity(iterations);
    let mut candidate_total_samples = Vec::with_capacity(iterations);
    let mut dense_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let candidate_start = Instant::now();

        let start = Instant::now();
        let measured_query = black_box(signature_from_query(
            &q,
            geometry.q_heads,
            geometry.head_dim,
        ));
        signature_samples.push(elapsed_ns(start));

        let start = Instant::now();
        let measured_selection = build_boolean_indexed_kv_selection(
            &cache,
            &table,
            &measured_query,
            max_distance,
            None,
            numerical_geometry,
        )
        .unwrap();
        let measured_table =
            BooleanSelectedPagedKvTable::from_selection(&measured_selection, &table).unwrap();
        black_box(measured_table.selected_live_tokens());
        search_samples.push(elapsed_ns(start));

        let submit_start = Instant::now();
        encode_selected(
            &device,
            &queue,
            &selected_pipeline,
            &selected_table,
            &table,
            &selected_output,
            inputs,
            geometry,
        );
        selected_submit_samples.push(elapsed_ns(submit_start));
        let sync_start = Instant::now();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        selected_sync_samples.push(elapsed_ns(sync_start));
        candidate_total_samples.push(elapsed_ns(candidate_start));

        let dense_start = Instant::now();
        encode_dense(
            &device,
            &queue,
            &dense_pipeline,
            &dense_table,
            &dense_output,
            inputs,
            geometry,
        );
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        dense_samples.push(elapsed_ns(dense_start));
    }

    let signature_generation_ns = median(signature_samples);
    let boolean_search_ns = median(search_samples);
    let selected_attention_ns = median(selected_submit_samples);
    let synchronization_ns = median(selected_sync_samples);
    let candidate_end_to_end_ns = median(candidate_total_samples);
    let dense_attention_ns = median(dense_samples);
    let latency = BikvLatencyInput {
        signature_generation_ns,
        boolean_search_ns,
        synchronization_ns,
        selected_attention_ns,
        dense_attention_ns,
    };
    let telemetry = table.telemetry().unwrap();
    let accounting = BikvAccountingInput {
        live_tokens: telemetry.live_tokens,
        selected_live_tokens: selected_table.selected_live_tokens(),
        mapped_pages: telemetry.mapped_pages,
        selected_pages: selected_table.entries().len(),
        page_size,
        kv_heads: geometry.kv_heads,
        head_dim: geometry.head_dim,
        scalar_bytes: std::mem::size_of::<f32>(),
        boolean_index_bytes_read: u64::try_from(candidate.boolean_key_bytes_read).unwrap(),
    };
    let record = BikvQualificationRecord::new(accounting, latency).unwrap();
    assert_eq!(
        record.dense_numerical_kv_bytes(),
        u64::try_from(candidate.full_numerical_kv_bytes).unwrap()
    );
    assert_eq!(
        record.selected_numerical_kv_bytes(),
        u64::try_from(candidate.selected_numerical_kv_bytes).unwrap()
    );
    assert_eq!(
        record.avoided_numerical_kv_bytes(),
        u64::try_from(candidate.avoided_numerical_kv_bytes).unwrap()
    );
    let decision = if !correctness_gate_passed {
        BikvPromotionDecision::FallbackCorrectnessGate
    } else if !quality_gate_passed {
        BikvPromotionDecision::FallbackQualityGate
    } else if candidate_end_to_end_ns >= dense_attention_ns {
        BikvPromotionDecision::FallbackNoLatencyWin
    } else {
        BikvPromotionDecision::Promote
    };

    println!("schema=bkv-k6-m16-qualification@1");
    println!("commit={}", git_head());
    println!("adapter={info:?}");
    println!("timing_scope=host-observed host-mirrored-query; signature_generation=host-mirror; selected_attention=encode+submit; synchronization=poll; dense=encode+submit+poll; no GPU timestamp claim");
    println!("q_device_resident=true q_host_mirror_retained=true kv_device_resident=true uploads_readbacks_excluded=true resident-only-production-claim=false");
    println!(
        "geometry page_size={} mapped_pages={} kv_len={} q_heads={} kv_heads={} head_dim={} signature_bits=8 max_distance={}",
        page_size, pages, geometry.kv_len, geometry.q_heads, geometry.kv_heads, geometry.head_dim, max_distance
    );
    println!("warmup={warmup} iterations={iterations}");
    println!(
        "selection selected_pages={} selected_tokens={} page_density={:.6} token_density={:.6}",
        accounting.selected_pages,
        accounting.selected_live_tokens,
        record.selected_page_density(),
        record.selected_token_density()
    );
    println!(
        "bytes boolean_index_read={} dense_kv={} selected_kv={} avoided_kv={} avoided_per_boolean_byte={:.6}",
        accounting.boolean_index_bytes_read,
        record.dense_numerical_kv_bytes(),
        record.selected_numerical_kv_bytes(),
        record.avoided_numerical_kv_bytes(),
        record.avoided_bytes_per_boolean_byte_read()
    );
    println!(
        "latency_ns signature_generation={} boolean_search_and_materialization={} selected_encode_submit={} selected_poll_sync={} dense_encode_submit_poll={} candidate_end_to_end_median={} phase_median_sum_diagnostic={}",
        latency.signature_generation_ns,
        latency.boolean_search_ns,
        latency.selected_attention_ns,
        latency.synchronization_ns,
        latency.dense_attention_ns,
        candidate_end_to_end_ns,
        record.total_bikv_latency_ns()
    );
    println!(
        "correctness all_accept_k6_vs_m16={} sparse_k6_vs_restricted_oracle={} max_abs_all_accept_vs_dense={:.8} max_abs_sparse_vs_oracle={:.8}",
        all_accept_parity,
        sparse_correctness,
        max_abs_diff(&all_accept_values, &dense_values),
        max_abs_diff(&selected_values, &sparse_oracle)
    );
    println!(
        "fixture_quality_gate_passed={} max_abs_sparse_vs_dense={:.8}",
        quality_gate_passed,
        max_abs_diff(&selected_values, &dense_values)
    );
    println!("fixture_policy=matched-density synthetic Boolean signatures; quality result is not a model-quality claim");
    println!("promotion_scope=host-mirrored-query synthetic fixture; not sufficient for resident-only production or BKV-7");
    println!("promotion_decision={decision:?}");
}
