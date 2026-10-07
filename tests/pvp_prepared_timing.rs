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

#[allow(clippy::too_many_arguments)]
fn check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    state: &wgpu::Buffer,
    expected: &[u32],
    arm: Arm,
    k: usize,
    g: usize,
    bank: &str,
    trial: usize,
    phase: &str,
) -> usize {
    let [a, b] = paired_read(device, queue, state);
    let mut failed = 0;
    for (label, actual, wanted) in [
        ("A_oracle", a.as_slice(), expected),
        ("B_oracle", b.as_slice(), expected),
        ("A_B", a.as_slice(), b.as_slice()),
    ] {
        assert_eq!(actual.len(), wanted.len());
        let wrong = actual.iter().zip(wanted).filter(|(a, b)| a != b).count();
        let first: Vec<_> = actual
            .iter()
            .zip(wanted)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .take(8)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect();
        println!("PVP_TIMING_CHECK,arm={},K={k},G={g},bank={bank},trial={trial},phase={phase},comparison={label},words={},wrong={wrong},first={first:?}",arm.name(),actual.len());
        failed += usize::from(wrong != 0);
    }
    failed
}
fn native_timing_adapter(kind: wgpu::DeviceType) -> bool {
    matches!(
        kind,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
    )
}

#[test]
fn timing_requires_explicit_hardware_adapter_identity() {
    for kind in [
        wgpu::DeviceType::Other,
        wgpu::DeviceType::VirtualGpu,
        wgpu::DeviceType::Cpu,
    ] {
        assert!(!native_timing_adapter(kind), "{kind:?}");
    }
    for kind in [
        wgpu::DeviceType::IntegratedGpu,
        wgpu::DeviceType::DiscreteGpu,
    ] {
        assert!(native_timing_adapter(kind), "{kind:?}");
    }
}

#[test]
#[ignore = "explicit native diagnostic timing only; not part of correctness CI"]
fn prepared_resident_wall_diagnostic() {
    let revision = option_env!("FLAT_SOURCE_REVISION").unwrap_or("unknown");
    assert_ne!(revision, "unknown", "freeze compile revision");
    assert_eq!(std::env::var("FLAT_SOURCE_REVISION").unwrap(), revision);
    println!("PVP_TIMING_PROTOCOL,source={revision},grid=6,banks=3,arms=5,warmups=5,repeats=20,mapping=explicit,timer=encode_submit_wait,GPU_timestamps=not_requested,performance_admission=none");
    for (k, g) in corpus::GEOMETRIES.into_iter().take(5) {
        for kind in corpus::BANKS {
            let bank = corpus::AnfBank::frozen(k, g, kind);
            assert_eq!(
                bank.truth_u32_words(),
                bank.truth_u32_words_by_monomial_masks()
            );
        }
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("native adapter required");
    let info = adapter.get_info();
    assert!(
        native_timing_adapter(info.device_type),
        "only explicitly identified integrated/discrete GPUs are admitted; got {:?}",
        info.device_type
    );
    println!(
        "PVP_TIMING_ADAPTER,name={:?},backend={:?},driver={:?},type={:?}",
        info.name, info.backend, info.driver_info, info.device_type
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("prepared-wall-diagnostic"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap();
    let pipelines: Vec<_> = Arm::ALL
        .into_iter()
        .map(|arm| {
            let t = Instant::now();
            let pipeline = WgpuPvpPreparedPipeline::new(&device, arm).unwrap();
            println!(
                "PVP_TIMING_SETUP,arm={},kind=pipeline,ns={}",
                arm.name(),
                t.elapsed().as_nanos()
            );
            (arm, pipeline)
        })
        .collect();
    let mut samples = 0;
    let mut failures = 0;
    let mut rejected = 0;
    for (k, g) in GRID {
        let specs: Vec<_> = Arm::ALL
            .into_iter()
            .map(|arm| PvpPreparedSpec::new(arm, k, g).unwrap())
            .collect();
        let rejected_arms: Vec<_> = specs
            .iter()
            .filter_map(|spec| {
                spec.validate_limits(&device.limits())
                    .err()
                    .map(|e| format!("{}:{e}", spec.arm().name()))
            })
            .collect();
        if !rejected_arms.is_empty() {
            println!("PVP_TIMING_REJECT,K={k},G={g},common=true,reasons={rejected_arms:?}");
            rejected += 1;
            continue;
        }
        for bank in corpus::BANKS {
            let fixtures: Vec<_> = (0..3).map(|round|{
                let b=corpus::AnfBank::frozen_for_round(k,g,bank,round);
                let coefficients=b.coefficients();
                let truth=FlatPvpPackedBitplanesV1::from_words(FlatPvpPackedLayoutV1::new(k,g).unwrap(),
                    b.truth_u32_words_by_monomial_masks()).unwrap().to_gate_major_u64().unwrap();
                let mut words=Vec::new();
                for (arm,role,data) in [(Arm::PackedOneStage,"source",&coefficients),
                    (Arm::PackedOneStage,"oracle",&truth),(Arm::Vec4,"source",&coefficients),(Arm::Vec4,"oracle",&truth)] {
                    let t=Instant::now(); let result=physical(arm,k,g,data);
                    println!("PVP_TIMING_CONVERSION,layout={},role={role},K={k},G={g},bank={bank},round={round},ns={}",arm.name(),t.elapsed().as_nanos());
                    words.push(result);
                }
                words
            }).collect();
            let states: Vec<_> = specs.iter().map(|spec| {
                let t=Instant::now();
                let state=device.create_buffer(&wgpu::BufferDescriptor{
                    label:Some("resident-timed-state"),size:spec.storage_bytes()+16,
                    usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation:false,
                });
                println!("PVP_TIMING_SETUP,arm={},K={k},G={g},bank={bank},kind=state_buffer,ns={},bytes={}",spec.arm().name(),t.elapsed().as_nanos(),spec.storage_bytes());
                state
            }).collect();
            let plans: Vec<_> = pipelines.iter().enumerate().map(|(i,(arm,pipeline))| {
                let t=Instant::now(); let plan=pipeline.prepare(&device,&states[i],&specs[i]).unwrap();
                println!("PVP_TIMING_SETUP,arm={},K={k},G={g},bank={bank},kind=plan,ns={},dispatches={}",arm.name(),t.elapsed().as_nanos(),specs[i].dispatches());
                plan
            }).collect();
            for trial in 0..WARMUPS + REPEATS {
                let measured = trial >= WARMUPS;
                let sequence = if measured { trial - WARMUPS } else { trial };
                for position in 0..5 {
                    // Ten mirrored cyclic orders, twice: equal arm positions
                    // and equal pair precedence over the 20 measured repeats.
                    let i = if (sequence / 5) % 2 == 0 {
                        (position + sequence % 5) % 5
                    } else {
                        (4 - position + sequence % 5) % 5
                    };
                    let arm = Arm::ALL[i];
                    let f = &fixtures[trial % 3];
                    let offset = if arm.is_packed() { 0 } else { 2 };
                    let original = &f[offset];
                    let transformed = &f[offset + 1];
                    let bytes: Vec<_> = original.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let t = Instant::now();
                    queue.write_buffer(&states[i], 0, &bytes);
                    queue.submit(std::iter::empty());
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    let upload_ns = t.elapsed().as_nanos();
                    failures += check(
                        &device, &queue, &states[i], original, arm, k, g, bank, trial, "source",
                    );
                    for (phase, expected) in [("forward", transformed), ("inverse", original)] {
                        let t = Instant::now();
                        let mut encoder = device.create_command_encoder(&Default::default());
                        plans[i].encode(&mut encoder);
                        queue.submit(Some(encoder.finish()));
                        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                        let wall_ns = t.elapsed().as_nanos();
                        let wrong = check(
                            &device, &queue, &states[i], expected, arm, k, g, bank, trial, phase,
                        );
                        failures += wrong;
                        println!("PVP_TIMING_SAMPLE,arm={},K={k},G={g},bank={bank},trial={trial},repeat={sequence},position={position},round={},phase={phase},measured={measured},wall_ns={wall_ns},upload_ns={upload_ns},exact={},admission=none",arm.name(),trial%3,wrong==0);
                        if measured {
                            samples += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(samples, (6 - rejected) * 3 * REPEATS * 5 * 2);
    println!("PVP_TIMING_COMPLETE,samples={samples},rejected_geometries={rejected},failed_comparisons={failures},performance_admission=none");
    assert_eq!(
        failures, 0,
        "retain all failed timings; reject affected cohort"
    );
}
