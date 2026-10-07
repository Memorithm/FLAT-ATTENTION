#![cfg(feature = "wgpu")]
#[path = "support/pvp_packed_corpus.rs"]
mod corpus;
use flat_attention::pvp_packed::{FlatPvpPackedBitplanesV1, FlatPvpPackedLayoutV1};
use flat_attention::pvp_vec4::{FlatPvpVec4BitplanesV1, FlatPvpVec4LayoutV1};
use flat_attention::pvp_prepared::{PvpPreparedArm as Arm, PvpPreparedError,
    PvpPreparedSpec, WgpuPvpPreparedPipeline};
use std::sync::mpsc;
const LARGE: [(usize, usize); 3] = [(16384, 2048), (65536, 512), (262144, 128)];
const GUARD: [u32; 4] = [0x13579bdf, 0x2468ace0, 0xdeadbeef, 0x76543210];

fn rounds(value: Option<&str>) -> Result<usize, &'static str> {
    match value { None | Some("1") => Ok(1), Some("2") => Ok(2), Some("3") => Ok(3),
        _ => Err("FLAT_PVP_PREPARED_ROUNDS must be exactly 1, 2 or 3") }
}
#[test]
fn rounds_are_bounded() {
    assert_eq!(rounds(None), Ok(1));
    for s in ["1", "2", "3"] { assert_eq!(rounds(Some(s)).unwrap().to_string(), s); }
    for s in ["", "0", "4", "01", " 1", "-1"] { assert!(rounds(Some(s)).is_err()); }
}
fn physical(arm: Arm, k: usize, g: usize, source: &[u64]) -> Vec<u32> {
    let mut words = if arm.is_packed() {
        FlatPvpPackedBitplanesV1::from_gate_major_u64(FlatPvpPackedLayoutV1::new(k, g).unwrap(), source)
            .unwrap().words().to_vec()
    } else {
        FlatPvpVec4BitplanesV1::from_gate_major_u64(FlatPvpVec4LayoutV1::new(k, g).unwrap(), source)
            .unwrap().words().to_vec()
    };
    words.extend_from_slice(&GUARD);
    words
}

fn paired_read(device: &wgpu::Device, queue: &wgpu::Queue, state: &wgpu::Buffer) -> [Vec<u32>; 2] {
    let staging = ["pvp-large-readback-A", "pvp-large-readback-B"].map(|label| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: state.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for target in &staging {
        encoder.copy_buffer_to_buffer(state, 0, target, 0, state.size());
    }
    // Both copies see the same unchanged source in one submission. Both
    // destinations receive the same explicit native mapping transition.
    encoder.transition_resources(
        staging.iter().map(|buffer| wgpu::BufferTransition {
            buffer,
            state: wgpu::BufferUses::MAP_READ,
        }),
        std::iter::empty(),
    );
    queue.submit(Some(encoder.finish()));
    let receivers: Vec<_> = staging
        .iter()
        .map(|buffer| {
            let (sender, receiver) = mpsc::channel();
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
            receiver
        })
        .collect();
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    for receiver in receivers {
        receiver.recv().unwrap().unwrap();
    }
    staging.map(|buffer| {
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        let words = mapped
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        drop(mapped);
        buffer.unmap();
        words
    })
}


#[test]
fn five_prepared_arms_match_direct_anf_with_paired_phase_and_guard_checks() {
    let rounds_env = std::env::var("FLAT_PVP_PREPARED_ROUNDS").ok();
    let count = rounds(rounds_env.as_deref()).unwrap();
    let source = option_env!("FLAT_SOURCE_REVISION").unwrap_or("unknown");
    if let Ok(runtime) = std::env::var("FLAT_SOURCE_REVISION") { assert_eq!(source, runtime); }
    println!("PVP_PREPARED_PROTOCOL,source={source},rounds={count},arms=5,geometries=16,mapping=explicit,performance_claim=none");
    // Cross-check the large-domain wordwise oracle against direct per-address
    // monomial evaluation for every small fixture and every coefficient round.
    for (k, g) in corpus::GEOMETRIES {
        for kind in corpus::BANKS {
            for r in 0..count {
                let bank = if r == 0 { corpus::AnfBank::frozen(k, g, kind) }
                    else { corpus::AnfBank::frozen_for_round(k, g, kind, r) };
                assert_eq!(bank.truth_u32_words(), bank.truth_u32_words_by_monomial_masks());
            }
        }
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: if cfg!(target_os = "windows") { wgpu::Backends::DX12 }
            else if cfg!(target_os = "macos") { wgpu::Backends::METAL }
            else { wgpu::Backends::VULKAN },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(), force_fallback_adapter: cfg!(target_os = "windows"),
        compatible_surface: None, apply_limit_buckets: false,
    }));
    let Ok(adapter) = adapter else {
        assert!(std::env::var_os("FLAT_REQUIRE_WGPU").is_none(), "mandatory prepared qualification needs an adapter");
        println!("PVP_PREPARED,status=skipped_adapter_unavailable,performance_claim=none"); return;
    };
    let info = adapter.get_info();
    println!("PVP_PREPARED_ADAPTER,backend={:?},type={:?},name={:?},driver={:?},performance_claim=none",
        info.backend, info.device_type, info.name, info.driver_info);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("pvp-prepared-qualification"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).unwrap();
    let pipelines: Vec<_> = Arm::ALL.into_iter().map(|arm|
        (arm, WgpuPvpPreparedPipeline::new(&device, arm).unwrap())).collect();
    for (arm, pipeline) in &pipelines {
        let spec = PvpPreparedSpec::new(*arm, 128, 1).unwrap();
        let buffer = |size, usage| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prepared-rejection-control"), size, usage, mapped_at_creation: false,
        });
        assert!(matches!(pipeline.prepare(&device, &buffer(spec.storage_bytes(), wgpu::BufferUsages::COPY_DST), &spec),
            Err(PvpPreparedError::MissingStorageUsage)));
        assert!(matches!(pipeline.prepare(&device, &buffer(4, wgpu::BufferUsages::STORAGE), &spec),
            Err(PvpPreparedError::BufferTooSmall { .. })));
        let other = if *arm == Arm::Vec4 { Arm::PackedOneStage } else { Arm::Vec4 };
        assert!(matches!(pipeline.prepare(&device, &buffer(16, wgpu::BufferUsages::STORAGE),
            &PvpPreparedSpec::new(other, 128, 1).unwrap()), Err(PvpPreparedError::ArmMismatch)));
    }
    let mut cases = 0;
    let mut comparisons = 0;
    let mut failed = 0;
    for (k, g) in corpus::GEOMETRIES.into_iter().chain(LARGE) {
        for kind in corpus::BANKS {
            let fixtures: Vec<_> = (0..count).map(|r| {
                let bank = corpus::AnfBank::frozen_for_round(k, g, kind, r);
                let packed_layout = FlatPvpPackedLayoutV1::new(k, g).unwrap();
                let truth = FlatPvpPackedBitplanesV1::from_words(packed_layout,
                    bank.truth_u32_words_by_monomial_masks()).unwrap().to_gate_major_u64().unwrap();
                (bank.coefficients(), truth)
            }).collect();
            for (arm, pipeline) in &pipelines {
                let spec = PvpPreparedSpec::new(*arm, k, g).unwrap();
                // No selective dropping: every arm must admit the fixed qualification grid.
                spec.validate_limits(&device.limits()).unwrap();
                let state = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("prepared-exact-state-with-guard"), size: spec.storage_bytes() + 16,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let plan = pipeline.prepare(&device, &state, &spec).unwrap();
                assert_eq!(plan.spec(), &spec);
                for (r, (coefficients, truth)) in fixtures.iter().enumerate() {
                    let original = physical(*arm, k, g, coefficients);
                    let transformed = physical(*arm, k, g, truth);
                    let bytes: Vec<_> = original.iter().flat_map(|w| w.to_le_bytes()).collect();
                    queue.write_buffer(&state, 0, &bytes);
                    queue.submit(std::iter::empty());
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    for phase in ["source", "transformed", "inverse"] {
                        if phase != "source" {
                            let mut encoder = device.create_command_encoder(&Default::default());
                            plan.encode(&mut encoder);
                            queue.submit(Some(encoder.finish()));
                        }
                        let expected = if phase == "transformed" { &transformed } else { &original };
                        let [a, b] = paired_read(&device, &queue, &state);
                        for (label, actual, wanted) in [("A_oracle", &a, expected),
                            ("B_oracle", &b, expected), ("A_B", &a, &b)] {
                            assert_eq!(actual.len(), wanted.len(), "complete readback");
                            let mismatches = actual.iter().zip(wanted).filter(|(a, b)| a != b).count();
                            let first: Vec<_> = actual.iter().zip(wanted).enumerate()
                                .filter(|(_, (a, b))| a != b).take(8)
                                .map(|(i, (a, b))| (i, *a, *b)).collect();
                            println!("PVP_PREPARED_COMPARE,arm={},K={k},G={g},bank={kind},round={r},phase={phase},comparison={label},words={},mismatched_words={mismatches},first={first:?},performance_claim=none",
                                arm.name(), actual.len());
                            comparisons += 1;
                            failed += usize::from(mismatches != 0);
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 240 * count);
    assert_eq!(comparisons, cases * 9);
    println!("PVP_PREPARED_COMPLETE,cases={cases},comparisons={comparisons},failed_comparisons={failed},performance_claim=none");
    assert_eq!(failed, 0, "retain every mismatch; no performance admission");
}
