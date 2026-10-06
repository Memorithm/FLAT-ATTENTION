#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::pvp::{
    pvp_subset_zeta_u32_scalar_in_place, FlatPvpU32BitplanesV1, FlatPvpU32LayoutV1,
    WgpuPvpScalarPipeline,
};

fn fixture_gate_major(layout: FlatPvpU32LayoutV1) -> Vec<u64> {
    let words_per_gate = layout.addresses().div_ceil(64);
    let mut words = vec![0_u64; layout.gates() * words_per_gate];
    for gate in 0..layout.gates() {
        for address in 0..layout.addresses() {
            let bit = ((gate * 23 + address * 7 + (gate ^ address)) % 17) < 8;
            if bit {
                words[gate * words_per_gate + address / 64] |= 1_u64 << (address % 64);
            }
        }
    }
    words
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn storage_buffer(device: &wgpu::Device, queue: &wgpu::Queue, values: &[u32]) -> wgpu::Buffer {
    let bytes = u32_bytes(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-pvp-scalar-state"),
        size: bytes.len().max(4) as u64,
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
        label: Some("flat-pvp-scalar-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-pvp-scalar-readback"),
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
    let mapped = slice.get_mapped_range().expect("valid PVP readback range");
    let values = mapped
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    drop(mapped);
    staging.unmap();
    values
}

#[test]
fn actual_wgpu_scalar_butterfly_matches_host_oracle_exactly() {
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
            panic!("FLAT PVP scalar parity requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional FLAT PVP scalar parity test skipped");
        return;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-pvp-scalar-parity"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .expect("FLAT PVP request_device failed");

    for (addresses, gates) in [(16, 31), (32, 65), (64, 97), (128, 257)] {
        let layout = FlatPvpU32LayoutV1::new(addresses, gates).unwrap();
        let source = fixture_gate_major(layout);
        let initial = FlatPvpU32BitplanesV1::from_gate_major_u64(layout, &source).unwrap();
        let mut expected = initial.clone();
        pvp_subset_zeta_u32_scalar_in_place(&mut expected).unwrap();

        let state = storage_buffer(&device, &queue, initial.words());
        let pipeline = WgpuPvpScalarPipeline::new(&device).unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-pvp-scalar-parity"),
        });
        pipeline
            .encode_all_stages(&device, &mut encoder, &state, layout)
            .unwrap();
        queue.submit(Some(encoder.finish()));

        let observed = read_u32(&device, &queue, &state, layout.storage_words());
        assert_eq!(
            observed,
            expected.words(),
            "PVP scalar WGSL mismatch for addresses={addresses} gates={gates}"
        );
    }
}
