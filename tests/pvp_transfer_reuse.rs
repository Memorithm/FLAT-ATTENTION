#![cfg(feature = "wgpu")]
//! Equal-sized changing uploads: no compute shader and no timing claim.
use std::sync::mpsc;
#[test]
fn completed_uploads_and_reused_size_readbacks_preserve_every_word() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }));
    let Ok(adapter) = adapter else {
        assert!(
            std::env::var_os("FLAT_REQUIRE_WGPU").is_none(),
            "required Vulkan adapter unavailable"
        );
        eprintln!("No Vulkan adapter; transfer test not executed");
        return;
    };
    let info = adapter.get_info();
    println!(
        "PVP_TRANSFER_ADAPTER,name={},backend={:?},driver={}",
        info.name, info.backend, info.driver_info
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("request device");
    let words = 1_usize << 20;
    let bytes = (words * 4) as u64;
    let mut retained = Vec::new();
    for round in 0..40_u32 {
        let expected: Vec<u32> = (0..words as u32)
            .map(|i| {
                i.wrapping_mul(0x9e37_79b9).rotate_left(round % 32)
                    ^ 0xa5a5_a5a5_u32.wrapping_add(round)
            })
            .collect();
        let encoded: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
        let source = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pvp-transfer-source"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&source, 0, &encoded);
        queue.submit(None);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("upload completion");
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pvp-transfer-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&source, 0, &staging, 0, bytes);
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("copy completion");
        rx.recv().expect("map callback").expect("map");
        let mapped = slice.get_mapped_range().expect("mapped range");
        for (index, raw) in mapped.chunks_exact(4).enumerate() {
            let actual = u32::from_le_bytes(raw.try_into().expect("u32"));
            assert_eq!(
                actual, expected[index],
                "transfer mismatch round={round} word={index}"
            );
        }
        drop(mapped);
        staging.unmap();
        // Hold each readback allocation to exercise fresh allocation as well
        // as source reuse. The test owns this explicitly bounded 160 MiB.
        retained.push(staging);
    }
    println!("PVP_TRANSFER_COMPLETE,rounds=40,words_per_round=1048576,all_words_exact=true,performance_claim=none");
}
