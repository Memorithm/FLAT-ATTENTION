#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::api::boolean_attention_signature::{
    BooleanAttentionSignature, HammingAdmissionRule,
};
use flat_attention::api::wgpu_boolean_router::{
    BooleanKvWgpuParityEvidence, BooleanWgpuRouterPlan, WgpuBooleanRouterPipeline,
    BOOLEAN_KV_WGPU_PARITY_SCHEMA,
};

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    values: &[u32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let bytes = u32_bytes(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buffer, 0, &bytes);
    }
    buffer
}

fn read_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Vec<u32> {
    let bytes = (len * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-bkv7-router-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-bkv7-router-readback"),
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
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    drop(mapped);
    staging.unmap();
    values
}

#[test]
fn actual_wgpu_dispatch_matches_cpu_oracle_before_any_performance_comparison() {
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
            panic!("FLAT BKV-4 / KVLab BKV-K7 WGPU parity requires an adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional FLAT BKV-4 / KVLab BKV-K7 device parity test skipped");
        return;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-bkv7-wgpu-parity"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .expect("FLAT BKV-4 / KVLab BKV-K7 request_device failed");

    let query = BooleanAttentionSignature::new(65, vec![0b1011, 1]).unwrap();
    let keys = vec![
        BooleanAttentionSignature::new(65, vec![0b1011, 1]).unwrap(),
        BooleanAttentionSignature::new(65, vec![0b0011, 1]).unwrap(),
        BooleanAttentionSignature::new(65, vec![0b1111, 1]).unwrap(),
        BooleanAttentionSignature::new(65, vec![0, 0]).unwrap(),
    ];
    let plan = BooleanWgpuRouterPlan::new(&query, &keys, HammingAdmissionRule::new(1, 65).unwrap())
        .unwrap();
    let query_buffer = storage_buffer(
        &device,
        &queue,
        "flat-bkv7-query",
        plan.query_words(),
        wgpu::BufferUsages::STORAGE,
    );
    let key_buffer = storage_buffer(
        &device,
        &queue,
        "flat-bkv7-keys",
        plan.key_words(),
        wgpu::BufferUsages::STORAGE,
    );
    let admissions = storage_buffer(
        &device,
        &queue,
        "flat-bkv7-admissions",
        &vec![0; plan.key_count() as usize],
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let pipeline = WgpuBooleanRouterPipeline::new(&device).unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-bkv7-router-dispatch"),
    });
    pipeline
        .encode(
            &device,
            &mut encoder,
            &query_buffer,
            &key_buffer,
            &admissions,
            &plan,
        )
        .unwrap();
    queue.submit(Some(encoder.finish()));

    let observed = read_u32(&device, &queue, &admissions, plan.key_count() as usize);
    let evidence = BooleanKvWgpuParityEvidence::from_readback(&plan, &observed).unwrap();
    assert_eq!(evidence.cpu_admitted_blocks(), &[0, 1, 2]);
    assert_eq!(evidence.wgpu_admitted_blocks(), &[0, 1, 2]);
    assert!(evidence.exact_candidate_set_match());
    evidence.require_exact_match().unwrap();
    let json = evidence.canonical_json();
    assert!(json.contains(BOOLEAN_KV_WGPU_PARITY_SCHEMA));
    assert!(json.contains("\"exact_candidate_set_match\":true"));
}
