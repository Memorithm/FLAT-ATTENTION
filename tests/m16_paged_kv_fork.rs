#![cfg(feature = "wgpu")]

use flat_attention::paged_kv::{PagedKvConfig, PagedKvError, WgpuPagedKvCache, WgpuPagedKvCacheError};
use flat_attention::{FlatAttentionConfig, PagedDecodePass, WgpuPagedDecodePipeline, WgpuPagedKvTable};
use std::sync::mpsc;
use std::time::Duration;
use wgpu::util::DeviceExt;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn harness() -> Option<Harness> {
    let backends = match std::env::var("WGPU_BACKEND").as_deref() {
        Ok("vulkan") => wgpu::Backends::VULKAN,
        Ok("metal") => wgpu::Backends::METAL,
        Ok("dx12") => wgpu::Backends::DX12,
        _ => wgpu::Backends::all(),
    };
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }));
    let Ok(adapter) = adapter else {
        assert!(
            std::env::var_os("FLAT_REQUIRE_WGPU").is_none(),
            "physical KV fork qualification requires a WGPU adapter"
        );
        eprintln!("WGPU unavailable; optional physical KV fork tests skipped");
        return None;
    };
    eprintln!("KV fork adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-kv-physical-fork-tests"),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .expect("request fork test device");
    Some(Harness { device, queue })
}

fn upload(h: &Harness, words: &[u32], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
    h.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kv-fork-test-input"),
        contents: &bytes,
        usage,
    })
}

fn read_words(h: &Harness, source: &wgpu::Buffer) -> Vec<u32> {
    let staging = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kv-fork-test-readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = h.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    h.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    h.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let words = mapped
        .chunks_exact(4)
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    drop(mapped);
    staging.unmap();
    words
}

fn payload(rows: usize, width: usize, shift: f32) -> Vec<u32> {
    (0..rows * width)
        .map(|index| (shift + (index % 17) as f32 * 0.125).to_bits())
        .collect()
}

fn append(h: &Harness, cache: &mut WgpuPagedKvCache, keys: &[u32], values: &[u32]) {
    let width = cache.kv_heads() * cache.head_dim();
    assert_eq!(keys.len(), values.len());
    assert_eq!(keys.len() % width, 0);
    let k = upload(h, keys, wgpu::BufferUsages::COPY_SRC);
    let v = upload(h, values, wgpu::BufferUsages::COPY_SRC);
    cache
        .append_and_submit(&h.device, &h.queue, &k, &v, keys.len() / width)
        .unwrap();
}

fn seeded(h: &Harness, heads: usize, dim: usize) -> (WgpuPagedKvCache, Vec<u32>, Vec<u32>) {
    let mut cache = WgpuPagedKvCache::new(
        &h.device,
        PagedKvConfig { page_size: 4, physical_pages: 4 },
        heads,
        dim,
    )
    .unwrap();
    let keys = payload(10, heads * dim, -1.0);
    let values = payload(10, heads * dim, 2.0);
    append(h, &mut cache, &keys, &values);
    (cache, keys, values)
}

fn assert_payload(h: &Harness, cache: &WgpuPagedKvCache, keys: &[u32], values: &[u32]) {
    let width = cache.kv_heads() * cache.head_dim();
    assert_eq!(keys.len(), cache.len() * width);
    assert_eq!(values.len(), keys.len());
    let actual_k = read_words(h, cache.k_buffer());
    let actual_v = read_words(h, cache.v_buffer());
    for token in 0..cache.len() {
        let address = cache.table().address(token).unwrap();
        let start = (address.physical_page * cache.config().page_size + address.offset_in_page) * width;
        assert_eq!(&actual_k[start..start + width], &keys[token * width..(token + 1) * width]);
        assert_eq!(&actual_v[start..start + width], &values[token * width..(token + 1) * width]);
    }
}

#[test]
fn fork_copies_exact_live_prefix_across_page_sizes_and_head_geometries() {
    let Some(h) = harness() else { return; };
    for (heads, dim) in [(1, 1), (2, 3), (4, 32)] {
        let (source, keys, values) = seeded(&h, heads, dim);
        let observation = source.observation().unwrap();
        let checkpoint = source.checkpoint();
        for page_size in [1, 3, 4, 7] {
            for prefix in [0usize, 1, 3, 4, 5, 10] {
                let config = PagedKvConfig {
                    page_size,
                    physical_pages: prefix.div_ceil(page_size).max(1) + 1,
                };
                let (child, _) = source
                    .fork_prefix_and_submit(&h.device, &h.queue, prefix, config)
                    .unwrap();
                assert_eq!(child.len(), prefix);
                assert_eq!(child.config(), config);
                assert_eq!((child.kv_heads(), child.head_dim()), (heads, dim));
                let live = prefix * heads * dim;
                assert_payload(&h, &child, &keys[..live], &values[..live]);
                // Fresh buffers' unused capacity must not inherit the source tail.
                assert!(read_words(&h, child.k_buffer())[live..].iter().all(|word| *word == 0));
                assert!(read_words(&h, child.v_buffer())[live..].iter().all(|word| *word == 0));
                assert!(!child.has_unsubmitted_recorded_writes());
                assert_eq!(source.observation().unwrap(), observation);
                source.validate_checkpoint(&checkpoint).unwrap();
            }
        }
    }
}

#[test]
fn fork_preserves_bits_including_signed_zero_and_nonfinite_payloads() {
    let Some(h) = harness() else { return; };
    let mut source = WgpuPagedKvCache::new(
        &h.device,
        PagedKvConfig { page_size: 3, physical_pages: 3 },
        1,
        2,
    )
    .unwrap();
    let keys = vec![
        0x00000000, 0x80000000, 0x7f800000, 0xff800000,
        0x7fc01234, 0xffc04321, 0x00000001, 0x007fffff,
        0x3f800000, 0xbf800000, 0x41100000, 0x41200000,
    ];
    let values: Vec<u32> = keys.iter().rev().copied().collect();
    append(&h, &mut source, &keys, &values);
    source.truncate(5).unwrap();
    let (child, _) = source
        .fork_prefix_and_submit(
            &h.device,
            &h.queue,
            5,
            PagedKvConfig { page_size: 4, physical_pages: 2 },
        )
        .unwrap();
    assert_payload(&h, &child, &keys[..10], &values[..10]);
    assert!(read_words(&h, child.k_buffer())[10..].iter().all(|word| *word == 0));
    assert!(read_words(&h, child.v_buffer())[10..].iter().all(|word| *word == 0));
}

#[test]
fn parent_and_child_reuse_are_independent_without_intermediate_host_waits() {
    let Some(h) = harness() else { return; };
    let (mut source, keys, values) = seeded(&h, 2, 8);
    let (mut child, _) = source
        .fork_prefix_and_submit(
            &h.device,
            &h.queue,
            7,
            PagedKvConfig { page_size: 3, physical_pages: 4 },
        )
        .unwrap();
    // Deliberately do not poll or read back between fork and source overwrite.
    source.reset().unwrap();
    let new_keys = payload(10, 16, 15.0);
    let new_values = payload(10, 16, -12.0);
    append(&h, &mut source, &new_keys, &new_values);
    assert_payload(&h, &child, &keys[..7 * 16], &values[..7 * 16]);
    assert_payload(&h, &source, &new_keys, &new_values);

    child.truncate(2).unwrap();
    let tail_k = payload(5, 16, -30.0);
    let tail_v = payload(5, 16, 40.0);
    append(&h, &mut child, &tail_k, &tail_v);
    let mut child_keys = keys[..2 * 16].to_vec();
    let mut child_values = values[..2 * 16].to_vec();
    child_keys.extend_from_slice(&tail_k);
    child_values.extend_from_slice(&tail_v);
    assert_payload(&h, &child, &child_keys, &child_values);
    assert_payload(&h, &source, &new_keys, &new_values);
    let (grandchild, _) = child
        .fork_prefix_and_submit(&h.device, &h.queue, child.len(), source.config())
        .unwrap();
    child.reset().unwrap();
    drop(source);
    drop(child);
    assert_payload(&h, &grandchild, &child_keys, &child_values);
}

#[test]
fn fork_creates_new_checkpoint_identity_and_retains_source_lineage() {
    let Some(h) = harness() else { return; };
    let (mut source, keys, values) = seeded(&h, 1, 8);
    source.reset().unwrap();
    append(&h, &mut source, &keys, &values);
    source.truncate(8).unwrap();
    let before = source.checkpoint();
    let observation = source.observation().unwrap();
    let (mut child, _) = source
        .fork_prefix_and_submit(&h.device, &h.queue, 6, source.config())
        .unwrap();
    assert_eq!(child.generation(), 0);
    assert_eq!(child.branch_epoch(), 0);
    assert_eq!(source.observation().unwrap(), observation);
    source.validate_checkpoint(&before).unwrap();
    assert_eq!(child.validate_checkpoint(&before), Err(WgpuPagedKvCacheError::ForeignCheckpoint));
    let child_checkpoint = child.checkpoint();
    assert_eq!(source.validate_checkpoint(&child_checkpoint), Err(WgpuPagedKvCacheError::ForeignCheckpoint));
    append(&h, &mut child, &keys[..8], &values[..8]);
    child.restore(&child_checkpoint).unwrap();
    assert_eq!(child.len(), 6);
    source.validate_checkpoint(&before).unwrap();
    assert_payload(&h, &child, &keys[..6 * 8], &values[..6 * 8]);
}

#[test]
fn fork_rejects_invalid_bounds_capacity_geometry_and_external_recording() {
    let Some(h) = harness() else { return; };
    let (mut source, keys, values) = seeded(&h, 2, 8);
    let before = source.observation().unwrap();
    assert_eq!(
        source.fork_prefix_and_submit(&h.device, &h.queue, 11, source.config()).unwrap_err(),
        WgpuPagedKvCacheError::ForkPrefixOutOfBounds { requested_len: 11, current_len: 10 }
    );
    assert_eq!(
        source.fork_prefix_and_submit(
            &h.device, &h.queue, 5, PagedKvConfig { page_size: 2, physical_pages: 2 },
        ).unwrap_err(),
        WgpuPagedKvCacheError::Table(PagedKvError::CapacityExceeded { requested: 5, capacity: 4 })
    );
    for config in [
        PagedKvConfig { page_size: 0, physical_pages: 1 },
        PagedKvConfig { page_size: 1, physical_pages: 0 },
        PagedKvConfig { page_size: usize::MAX, physical_pages: 2 },
    ] {
        assert!(source.fork_prefix_and_submit(&h.device, &h.queue, 0, config).is_err());
    }
    let too_large = PagedKvConfig {
        page_size: 1,
        physical_pages: usize::try_from(h.device.limits().max_buffer_size / 64 + 1).unwrap(),
    };
    assert!(matches!(
        source.fork_prefix_and_submit(&h.device, &h.queue, 1, too_large),
        Err(WgpuPagedKvCacheError::DeviceBufferLimit { .. })
    ));
    assert_eq!(source.observation().unwrap(), before);
    assert_payload(&h, &source, &keys, &values);

    let k = upload(&h, &keys[..16], wgpu::BufferUsages::COPY_SRC);
    let v = upload(&h, &values[..16], wgpu::BufferUsages::COPY_SRC);
    let mut encoder = h.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    source.record_append(&mut encoder, &k, &v, 1).unwrap();
    let tainted = source.observation().unwrap();
    for prefix in [0, 10, 11] {
        assert_eq!(
            source.fork_prefix_and_submit(&h.device, &h.queue, prefix, source.config()).unwrap_err(),
            WgpuPagedKvCacheError::UnsubmittedRecordedWrites
        );
    }
    // An unrelated submission cannot clear the source's external-write taint.
    h.queue.submit(Some(encoder.finish()));
    assert_eq!(
        source.fork_prefix_and_submit(&h.device, &h.queue, 10, source.config()).unwrap_err(),
        WgpuPagedKvCacheError::UnsubmittedRecordedWrites
    );
    assert_eq!(source.observation().unwrap(), tainted);
}

fn scalar_decode(q: &[f32], keys: &[u32], values: &[u32], kv_heads: usize, dim: usize) -> Vec<f32> {
    let q_heads = q.len() / dim;
    let tokens = keys.len() / (kv_heads * dim);
    let scale = f64::from(FlatAttentionConfig::default().resolved_scale(dim).unwrap());
    let mut output = vec![0.0; q.len() + q_heads];
    for head in 0..q_heads {
        let kv_head = head / (q_heads / kv_heads);
        let scores: Vec<f64> = (0..tokens)
            .map(|token| {
                let start = (token * kv_heads + kv_head) * dim;
                (0..dim)
                    .map(|d| f64::from(q[head * dim + d]) * f64::from(f32::from_bits(keys[start + d])))
                    .sum::<f64>() * scale
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
        let denominator: f64 = weights.iter().sum();
        for d in 0..dim {
            output[head * dim + d] = ((0..tokens)
                .map(|token| weights[token] * f64::from(f32::from_bits(values[(token * kv_heads + kv_head) * dim + d])))
                .sum::<f64>() / denominator) as f32;
        }
        output[q.len() + head] = (max + denominator.ln()) as f32;
    }
    output
}

#[test]
fn forked_mha_gqa_mqa_decode_matches_oracle_after_source_overwrite() {
    let Some(h) = harness() else { return; };
    let pipeline = WgpuPagedDecodePipeline::new(&h.device).unwrap();
    let dim = 8;
    let q_heads = 4;
    let q: Vec<f32> = (0..q_heads * dim).map(|index| (index % 5) as f32 * 0.0625 - 0.125).collect();
    let q_words: Vec<u32> = q.iter().map(|value| value.to_bits()).collect();
    let q_buffer = upload(&h, &q_words, wgpu::BufferUsages::STORAGE);
    for kv_heads in [1, 2, 4] {
        let (mut source, keys, values) = seeded(&h, kv_heads, dim);
        let (child, _) = source.fork_prefix_and_submit(
            &h.device, &h.queue, 7, PagedKvConfig { page_size: 5, physical_pages: 2 },
        ).unwrap();
        source.reset().unwrap();
        append(&h, &mut source, &payload(10, kv_heads * dim, 50.0), &payload(10, kv_heads * dim, -80.0));
        let table = WgpuPagedKvTable::from_table(child.table()).unwrap();
        let destination = pipeline.create_output_buffer(&h.device, q_heads, dim).unwrap();
        let mut encoder = h.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        pipeline.encode(&h.device, &mut encoder, PagedDecodePass {
            q: &q_buffer,
            k: child.k_buffer(),
            v: child.v_buffer(),
            page_table: &table,
            out_and_lse: &destination,
            q_heads,
            kv_heads,
            head_dim: dim,
            config: FlatAttentionConfig { causal: true, ..Default::default() },
            theta: 10_000.0,
            // Q has identity RoPE here; keys are already the stored key vectors.
            q_rope_position: 0,
            q_causal_position: 6,
        }).unwrap();
        h.queue.submit(Some(encoder.finish()));
        let expected = scalar_decode(&q, &keys[..7 * kv_heads * dim], &values[..7 * kv_heads * dim], kv_heads, dim);
        let actual = read_words(&h, &destination);
        assert_eq!(actual.len(), expected.len());
        for (index, (bits, expected)) in actual.into_iter().zip(expected).enumerate() {
            let actual = f32::from_bits(bits);
            assert!(actual.is_finite());
            assert!(
                (actual - expected).abs() <= 2e-4 + 2e-4 * expected.abs(),
                "kv_heads={kv_heads}, component={index}, actual={actual}, expected={expected}"
            );
        }
    }
}
