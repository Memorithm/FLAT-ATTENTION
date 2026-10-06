#![cfg(feature = "wgpu")]

#[path = "support/pvp_packed_corpus.rs"]
mod corpus;

use flat_attention::pvp_packed::{
    FlatPvpPackedBitplanesV1, FlatPvpPackedError, FlatPvpPackedLayoutV1, WgpuPvpPackedPipeline,
};
use std::sync::mpsc;

fn read(device: &wgpu::Device, queue: &wgpu::Queue, state: &wgpu::Buffer) -> Vec<u32> {
    let bytes = state.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pvp-packed-exact-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(state, 0, &staging, 0, bytes);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("packed readback completion");
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().expect("packed mapped range");
    let words = mapped
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    drop(mapped);
    staging.unmap();
    words
}

#[test]
fn actual_wgpu_prepared_packed_plan_matches_direct_anf_and_inverse() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: if cfg!(target_os = "windows") {
            wgpu::Backends::DX12
        } else if cfg!(target_os = "macos") {
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::VULKAN
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: cfg!(target_os = "windows"),
        compatible_surface: None,
        apply_limit_buckets: false,
    }));
    let Ok(adapter) = adapter else {
        assert!(
            std::env::var_os("FLAT_REQUIRE_WGPU").is_none(),
            "mandatory packed PVP qualification requires an actual WGPU adapter"
        );
        eprintln!("PVP_PACKED,status=skipped_adapter_unavailable,performance_claim=none");
        return;
    };
    let info = adapter.get_info();
    println!(
        "PVP_PACKED_ADAPTER,backend={:?},device_type={:?},name={:?},performance_claim=none",
        info.backend, info.device_type, info.name
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("pvp-packed-qualification"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap();
    let pipeline = WgpuPvpPackedPipeline::new(&device).unwrap();

    let rejected_layout = FlatPvpPackedLayoutV1::new(128, 1).unwrap();
    let small = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pvp-packed-too-small"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert!(matches!(
        pipeline.prepare(&device, &small, rejected_layout),
        Err(FlatPvpPackedError::BufferTooSmall { .. })
    ));
    let wrong_usage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pvp-packed-no-storage"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    assert!(matches!(
        pipeline.prepare(&device, &wrong_usage, rejected_layout),
        Err(FlatPvpPackedError::MissingStorageUsage)
    ));

    let mut cases = 0;
    for (k, g) in corpus::GEOMETRIES {
        for kind in corpus::BANKS {
            let bank = corpus::AnfBank::frozen(k, g, kind);
            let layout = FlatPvpPackedLayoutV1::new(k, g).unwrap();
            let original =
                FlatPvpPackedBitplanesV1::from_gate_major_u64(layout, &bank.coefficients())
                    .unwrap();
            let bytes: Vec<_> = original
                .words()
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            let state = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pvp-packed-anf-state"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&state, 0, &bytes);
            let plan = pipeline.prepare(&device, &state, layout).unwrap();
            assert_eq!(
                plan.accounting().logical_dispatches,
                layout.logical_dispatches()
            );
            // Reuse exactly the same immutable plan for both transform directions.
            for inverse in [false, true] {
                let mut encoder = device.create_command_encoder(&Default::default());
                plan.encode(&mut encoder);
                queue.submit(Some(encoder.finish()));
                let observed = read(&device, &queue, &state);
                let expected = if inverse {
                    original.words().to_vec()
                } else {
                    bank.truth_u32_words()
                };
                assert_eq!(
                    observed, expected,
                    "packed ANF K={k} G={g} bank={kind} inverse={inverse}"
                );
                FlatPvpPackedBitplanesV1::from_words(layout, observed)
                    .unwrap()
                    .validate_padding_zero()
                    .unwrap();
            }
            println!("PVP_PACKED_CASE,K={k},G={g},bank={kind},truth=exact,inverse=exact,padding=exact,performance_claim=none");
            cases += 1;
        }
    }
    assert_eq!(cases, 39);
    println!("PVP_PACKED_COMPLETE,cases=39,inverse_checks=39,performance_claim=none");
}
