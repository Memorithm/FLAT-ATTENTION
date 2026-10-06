#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::pvp_tile8::WgpuPvpTile8Pipeline;
use flat_attention::pvp_vec4::{
    pvp_subset_zeta_vec4_host_in_place, FlatPvpVec4BitplanesV1, FlatPvpVec4LayoutV1,
};

fn fixture(layout: FlatPvpVec4LayoutV1) -> FlatPvpVec4BitplanesV1 {
    let words_per_gate = layout.addresses().div_ceil(64);
    let mut gate_major = vec![0_u64; layout.gates() * words_per_gate];
    for gate in 0..layout.gates() {
        for address in 0..layout.addresses() {
            if ((gate * 53 + address * 19 + (gate ^ address)) % 37) < 18 {
                gate_major[gate * words_per_gate + address / 64] |=
                    1_u64 << (address % 64);
            }
        }
    }
    FlatPvpVec4BitplanesV1::from_gate_major_u64(layout, &gate_major).unwrap()
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn storage_buffer(device: &wgpu::Device, queue: &wgpu::Queue, values: &[u32]) -> wgpu::Buffer {
    let bytes = encode_u32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-pvp-tile8-state"),
        size: bytes.len().max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
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
        label: Some("flat-pvp-tile8-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-pvp-tile8-readback"),
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
    let mapped = slice
        .get_mapped_range()
        .expect("valid PVP tile8 readback range");
    let values = mapped
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    drop(mapped);
    staging.unmap();
    values
}

#[test]
fn actual_wgpu_tile8_matches_qualified_vec4_reference_exactly() {
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
            panic!("FLAT PVP tile8 parity requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional FLAT PVP tile8 parity test skipped");
        return;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-pvp-tile8-parity"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .expect("FLAT PVP tile8 request_device failed");

    for (addresses, gates) in [
        (1, 31),
        (2, 65),
        (4, 129),
        (8, 257),
        (16, 513),
        (64, 129),
        (128, 257),
    ] {
        let layout = FlatPvpVec4LayoutV1::new(addresses, gates).unwrap();
        let initial = fixture(layout);
        let mut expected = initial.clone();
        pvp_subset_zeta_vec4_host_in_place(&mut expected).unwrap();

        let state = storage_buffer(&device, &queue, initial.words());
        let pipeline = WgpuPvpTile8Pipeline::new(&device).unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-pvp-tile8-parity"),
        });
        let accounting = pipeline
            .encode_all_stages(&device, &mut encoder, &state, layout)
            .unwrap();
        queue.submit(Some(encoder.finish()));

        let observed = read_u32(&device, &queue, &state, layout.storage_u32_words());
        assert_eq!(
            observed,
            expected.words(),
            "PVP tile8 mismatch for addresses={addresses} gates={gates}"
        );
        assert_eq!(accounting.workgroup_storage_bytes, 128);
        if addresses >= 8 {
            assert_eq!(
                accounting.logical_dispatches + 2,
                accounting.baseline_logical_dispatches
            );
        }
    }
}
