#![cfg(feature = "wgpu")]
#[path = "support/pvp_packed_corpus.rs"]
mod corpus;
use flat_attention::pvp_packed::{FlatPvpPackedBitplanesV1, FlatPvpPackedLayoutV1};
use flat_attention::pvp_prepared::{
    PvpPreparedArm as Arm, PvpPreparedSpec, WgpuPvpPreparedPipeline,
};
use flat_attention::pvp_vec4::{FlatPvpVec4BitplanesV1, FlatPvpVec4LayoutV1};
use std::sync::mpsc;
use std::time::Instant;
const GUARD: [u32; 4] = [0x13579bdf, 0x2468ace0, 0xdeadbeef, 0x76543210];
const GRID: [(usize, usize); 6] = [
    (4096, 128),
    (16384, 2048),
    (65536, 512),
    (262144, 128),
    (16384, 129),
    (65536, 129),
];
const WARMUPS: usize = 5;
const REPEATS: usize = 20;
const ARMS: [Arm; 2] = [Arm::Vec4, Arm::PackedSevenStage];
fn physical(arm: Arm, k: usize, g: usize, source: &[u64]) -> Vec<u32> {
    let mut words = if arm.is_packed() {
        FlatPvpPackedBitplanesV1::from_gate_major_u64(
            FlatPvpPackedLayoutV1::new(k, g).unwrap(),
            source,
        )
        .unwrap()
        .words()
        .to_vec()
    } else {
        FlatPvpVec4BitplanesV1::from_gate_major_u64(FlatPvpVec4LayoutV1::new(k, g).unwrap(), source)
            .unwrap()
            .words()
            .to_vec()
    };
    words.extend_from_slice(&GUARD);
    words
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, state: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pvp-e2e-explicit-read"),
        size: state.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(state, 0, &staging, 0, state.size());
    encoder.transition_resources(
        [wgpu::BufferTransition {
            buffer: &staging,
            state: wgpu::BufferUses::MAP_READ,
        }]
        .into_iter(),
        std::iter::empty(),
    );
    queue.submit(Some(encoder.finish()));
    let (sender, receiver) = mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let words = mapped
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    drop(mapped);
    staging.unmap();
    words
}
fn canonical(arm: Arm, k: usize, g: usize, guarded: &[u32]) -> Vec<u64> {
    let words = guarded[..guarded.len() - 4].to_vec();
    if arm.is_packed() {
        FlatPvpPackedBitplanesV1::from_words(FlatPvpPackedLayoutV1::new(k, g).unwrap(), words)
            .unwrap()
            .to_gate_major_u64()
            .unwrap()
    } else {
        FlatPvpVec4BitplanesV1::from_words(FlatPvpVec4LayoutV1::new(k, g).unwrap(), words)
            .unwrap()
            .to_gate_major_u64()
            .unwrap()
    }
}
#[test]
fn canonical_round_trip_covers_padding_and_gate_boundaries() {
    for (k, g) in corpus::GEOMETRIES {
        for bank in corpus::BANKS {
            let fixture = corpus::AnfBank::frozen(k, g, bank);
            let coeff = fixture.coefficients();
            assert_eq!(
                fixture.truth_u32_words(),
                fixture.truth_u32_words_by_monomial_masks()
            );
            for arm in ARMS {
                assert_eq!(canonical(arm, k, g, &physical(arm, k, g, &coeff)), coeff);
            }
        }
    }
    let layout = FlatPvpVec4LayoutV1::new(8, 129).unwrap();
    let mut words = vec![0; layout.storage_u32_words()];
    words[4] = 2; // gate 129 is outside the 129 live gates.
    assert!(FlatPvpVec4BitplanesV1::from_words(layout, words).is_err());
}
#[test]
#[ignore = "native end-to-end diagnostic; run only through frozen scoped controller"]
fn canonical_end_to_end_diagnostic() {
    let revision = option_env!("FLAT_SOURCE_REVISION").unwrap_or("unknown");
    assert_ne!(revision, "unknown");
    assert_eq!(std::env::var("FLAT_SOURCE_REVISION").unwrap(), revision);
    println!("PVP_E2E_PROTOCOL,source={revision},grid=6,banks=3,arms=2,modes=2,warmups=5,repeats=20,input=gate_major_u64,output=gate_major_u64,timer=full_forward_wall,GPU_timestamps=not_requested,performance_admission=none");
    for (k, g) in corpus::GEOMETRIES.into_iter().take(5) {
        for bank in corpus::BANKS {
            let b = corpus::AnfBank::frozen(k, g, bank);
            assert_eq!(b.truth_u32_words(), b.truth_u32_words_by_monomial_masks());
        }
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        force_fallback_adapter: false,
        compatible_surface: None,
        power_preference: wgpu::PowerPreference::default(),
        apply_limit_buckets: false,
    }))
    .expect("hardware adapter required");
    let info = adapter.get_info();
    assert!(matches!(
        info.device_type,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
    ));
    println!(
        "PVP_E2E_ADAPTER,name={:?},backend={:?},driver={:?},type={:?}",
        info.name, info.backend, info.driver_info, info.device_type
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("canonical-e2e"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap();
    let mut samples = 0;
    let mut failures = 0;
    let mut rejected = 0;
    for (k, g) in GRID {
        let specs: Vec<_> = ARMS
            .into_iter()
            .map(|arm| PvpPreparedSpec::new(arm, k, g).unwrap())
            .collect();
        if specs
            .iter()
            .any(|s| s.validate_limits(&device.limits()).is_err())
        {
            println!("PVP_E2E_REJECT,K={k},G={g},common=true");
            rejected += 1;
            continue;
        }
        for bank in corpus::BANKS {
            let fixtures: Vec<_> = (0..3)
                .map(|round| {
                    let b = corpus::AnfBank::frozen_for_round(k, g, bank, round);
                    let truth = FlatPvpPackedBitplanesV1::from_words(
                        FlatPvpPackedLayoutV1::new(k, g).unwrap(),
                        b.truth_u32_words_by_monomial_masks(),
                    )
                    .unwrap()
                    .to_gate_major_u64()
                    .unwrap();
                    let coefficients = b.coefficients();
                    let expected: Vec<_> = ARMS
                        .into_iter()
                        .map(|arm| physical(arm, k, g, &truth))
                        .collect();
                    (coefficients, truth, expected)
                })
                .collect();
            for mode in ["prepared", "fresh_resources"] {
                let pipelines:Vec<_>=ARMS.into_iter().map(|arm|{
                    let t=Instant::now();let p=WgpuPvpPreparedPipeline::new(&device,arm).unwrap();
                    println!("PVP_E2E_SETUP,arm={},K={k},G={g},bank={bank},mode={mode},kind=pipeline,ns={}",arm.name(),t.elapsed().as_nanos());p
                }).collect();
                let states: Vec<_> = specs
                    .iter()
                    .map(|s| {
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("canonical-state"),
                            size: s.storage_bytes() + 16,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        })
                    })
                    .collect();
                let plans: Vec<_> = pipelines
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p.prepare(&device, &states[i], &specs[i]).unwrap())
                    .collect();
                for trial in 0..WARMUPS + REPEATS {
                    let measured = trial >= WARMUPS;
                    let repeat = if measured { trial - WARMUPS } else { trial };
                    for position in 0..2 {
                        let i = (position + repeat % 2) % 2;
                        let arm = ARMS[i];
                        let (coefficients, truth, expected) = &fixtures[trial % 3];
                        // Start from canonical host data for each complete invocation.
                        let start = Instant::now();
                        let phase = Instant::now();
                        let original = physical(arm, k, g, coefficients);
                        let bytes: Vec<_> = original.iter().flat_map(|w| w.to_le_bytes()).collect();
                        let conversion_in_ns = phase.elapsed().as_nanos();
                        let phase = Instant::now();
                        let fresh = if mode == "fresh_resources" {
                            let pipeline = WgpuPvpPreparedPipeline::new(&device, arm).unwrap();
                            let state = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("fresh-canonical-state"),
                                size: specs[i].storage_bytes() + 16,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_SRC
                                    | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            let plan = pipeline.prepare(&device, &state, &specs[i]).unwrap();
                            Some((pipeline, state, plan))
                        } else {
                            None
                        };
                        let (state, plan) = fresh
                            .as_ref()
                            .map_or((&states[i], &plans[i]), |(_, s, p)| (s, p));
                        let preparation_ns = phase.elapsed().as_nanos();
                        let phase = Instant::now();
                        queue.write_buffer(state, 0, &bytes);
                        queue.submit(std::iter::empty());
                        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                        let upload_ns = phase.elapsed().as_nanos();
                        let phase = Instant::now();
                        let mut encoder = device.create_command_encoder(&Default::default());
                        plan.encode(&mut encoder);
                        queue.submit(Some(encoder.finish()));
                        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                        let compute_wall_ns = phase.elapsed().as_nanos();
                        let phase = Instant::now();
                        let a = read(&device, &queue, state);
                        let readback_ns = phase.elapsed().as_nanos();
                        let phase = Instant::now();
                        let logical = canonical(arm, k, g, &a);
                        let conversion_out_ns = phase.elapsed().as_nanos();
                        let wall_ns = start.elapsed().as_nanos();
                        // Second independent explicit read and comparisons are outside the timer.
                        let b = read(&device, &queue, state);
                        let mut wrong = 0;
                        for (comparison, actual, wanted) in [
                            ("A_oracle", a.as_slice(), expected[i].as_slice()),
                            ("B_oracle", b.as_slice(), expected[i].as_slice()),
                            ("A_B", a.as_slice(), b.as_slice()),
                        ] {
                            let mismatches =
                                actual.iter().zip(wanted).filter(|(a, b)| a != b).count()
                                    + actual.len().abs_diff(wanted.len());
                            wrong += usize::from(mismatches != 0);
                            println!("PVP_E2E_CHECK,arm={},K={k},G={g},bank={bank},mode={mode},trial={trial},comparison={comparison},words={},wrong={mismatches}",arm.name(),actual.len());
                        }
                        let logical_wrong =
                            logical.iter().zip(truth).filter(|(a, b)| a != b).count()
                                + logical.len().abs_diff(truth.len());
                        wrong += usize::from(logical_wrong != 0);
                        failures += wrong;
                        println!("PVP_E2E_CHECK,arm={},K={k},G={g},bank={bank},mode={mode},trial={trial},comparison=canonical_oracle,words={},wrong={logical_wrong}",arm.name(),logical.len());
                        println!("PVP_E2E_SAMPLE,arm={},K={k},G={g},bank={bank},mode={mode},trial={trial},repeat={repeat},position={position},round={},measured={measured},wall_ns={wall_ns},conversion_in_ns={conversion_in_ns},preparation_ns={preparation_ns},upload_ns={upload_ns},compute_wall_ns={compute_wall_ns},readback_ns={readback_ns},conversion_out_ns={conversion_out_ns},exact={},admission=none",arm.name(),trial%3,wrong==0);
                        if measured {
                            samples += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(samples, (6 - rejected) * 3 * 2 * REPEATS * 2);
    println!("PVP_E2E_COMPLETE,samples={samples},rejected_geometries={rejected},failed_comparisons={failures},performance_admission=none");
    assert_eq!(failures, 0, "retain failures and reject cohort");
}
