#![cfg(feature = "wgpu")]

#[path = "support/pvp_packed_corpus.rs"]
mod corpus;

use flat_attention::pvp_packed::{
    FlatPvpPackedBitplanesV1, FlatPvpPackedLayoutV1, WgpuPvpPackedPipeline,
};
use std::sync::mpsc;

const LARGE_GEOMETRIES: [(usize, usize); 5] = [
    (16384, 2048),
    (65536, 512),
    (262144, 128),
    (16384, 129),
    (65536, 129),
];

fn parse_rounds(value: Option<&str>) -> Result<usize, &'static str> {
    match value {
        None => Ok(1),
        Some("1") => Ok(1),
        Some("2") => Ok(2),
        Some("3") => Ok(3),
        Some(_) => Err("FLAT_PVP_PACKED_STRESS_ROUNDS must be exactly 1, 2 or 3"),
    }
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
            buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
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

struct Observation {
    k: usize,
    gates: usize,
    bank: &'static str,
    round: usize,
    phase: &'static str,
}

fn compare(context: &Observation, label: &str, actual: &[u32], expected: &[u32]) -> usize {
    assert_eq!(actual.len(), expected.len(), "complete readback length");
    let mut mismatches = 0;
    let mut first = Vec::new();
    for (index, (&observed, &wanted)) in actual.iter().zip(expected).enumerate() {
        if observed != wanted {
            mismatches += 1;
            if first.len() < 8 {
                first.push((index, observed, wanted));
            }
        }
    }
    println!(
        "PVP_LARGE_COMPARE,K={},G={},bank={},round={},phase={},comparison={label},words={},mismatched_words={mismatches},first={first:?},performance_claim=none",
        context.k, context.gates, context.bank, context.round, context.phase, actual.len()
    );
    mismatches
}

#[test]
fn stress_round_count_is_bounded_and_unambiguous() {
    assert_eq!(parse_rounds(None), Ok(1));
    for value in ["1", "2", "3"] {
        assert_eq!(parse_rounds(Some(value)).unwrap().to_string(), value);
    }
    for value in ["0", "4", "01", " 1", "1 ", "", "-1", "infinite"] {
        assert!(parse_rounds(Some(value)).is_err());
    }
}

#[test]
fn actual_wgpu_large_packed_plan_has_exact_paired_phase_readbacks() {
    let rounds_env = std::env::var("FLAT_PVP_PACKED_STRESS_ROUNDS").ok();
    let rounds = parse_rounds(rounds_env.as_deref()).unwrap();
    let source = option_env!("FLAT_SOURCE_REVISION").unwrap_or("unknown");
    if let Ok(runtime_source) = std::env::var("FLAT_SOURCE_REVISION") {
        assert_eq!(source, runtime_source, "compiled/runtime source identity");
    }
    println!("PVP_LARGE_PROTOCOL,source={source},rounds={rounds},mapping=explicit,performance_claim=none");

    // Keep the new wordwise oracle grounded in the original per-address
    // monomial definition before using it on the large fixed corpus.
    let mut oracle_controls = 0;
    for (k, gates) in corpus::GEOMETRIES {
        for bank_kind in corpus::BANKS {
            let bank = corpus::AnfBank::frozen(k, gates, bank_kind);
            assert_eq!(bank.truth_u32_words_by_monomial_masks(), bank.truth_u32_words());
            oracle_controls += 1;
        }
    }
    assert_eq!(oracle_controls, 39);

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
        assert!(std::env::var_os("FLAT_REQUIRE_WGPU").is_none(),
            "mandatory large packed qualification requires an actual WGPU adapter");
        eprintln!("PVP_LARGE,status=skipped_adapter_unavailable,performance_claim=none");
        return;
    };
    let info = adapter.get_info();
    println!("PVP_LARGE_ADAPTER,backend={:?},device_type={:?},name={:?},driver={:?},performance_claim=none",
        info.backend, info.device_type, info.name, info.driver_info);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("pvp-packed-large-phase-controls"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    })).unwrap();
    let pipeline = WgpuPvpPackedPipeline::new(&device).unwrap();
    let mut cases = 0;
    let mut observations = 0;
    let mut comparisons = 0;
    let mut failed_comparisons = 0;

    for (k, gates) in LARGE_GEOMETRIES {
        let layout = FlatPvpPackedLayoutV1::new(k, gates).unwrap();
        for bank_kind in corpus::BANKS {
            let state = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pvp-packed-large-state"),
                size: layout.storage_bytes().unwrap(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let plan = pipeline.prepare(&device, &state, layout).unwrap();
            assert_eq!(plan.accounting().logical_dispatches, layout.logical_dispatches());
            for round in 0..rounds {
                let bank = corpus::AnfBank::frozen_for_round(k, gates, bank_kind, round);
                let original = FlatPvpPackedBitplanesV1::from_gate_major_u64(
                    layout, &bank.coefficients()).unwrap();
                let truth = bank.truth_u32_words_by_monomial_masks();
                let bytes: Vec<_> = original.words().iter()
                    .flat_map(|word| word.to_le_bytes()).collect();
                queue.write_buffer(&state, 0, &bytes);
                queue.submit(std::iter::empty());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

                for phase in ["source", "transformed", "inverse"] {
                    if phase != "source" {
                        let mut encoder = device.create_command_encoder(&Default::default());
                        plan.encode(&mut encoder);
                        queue.submit(Some(encoder.finish()));
                    }
                    let expected = if phase == "transformed" {
                        truth.as_slice()
                    } else {
                        original.words()
                    };
                    let [a, b] = paired_read(&device, &queue, &state);
                    let context = Observation { k, gates, bank: bank_kind, round, phase };
                    for errors in [
                        compare(&context, "A_oracle", &a, expected),
                        compare(&context, "B_oracle", &b, expected),
                        compare(&context, "A_B", &a, &b),
                    ] {
                        comparisons += 1;
                        failed_comparisons += usize::from(errors != 0);
                    }
                    for words in [a, b] {
                        FlatPvpPackedBitplanesV1::from_words(layout, words).unwrap()
                            .validate_padding_zero().unwrap();
                    }
                    observations += 1;
                }
            }
            println!("PVP_LARGE_CASE,K={k},G={gates},bank={bank_kind},rounds={rounds},bytes={},dispatches={},performance_claim=none",
                layout.storage_bytes().unwrap(), layout.logical_dispatches());
            cases += 1;
        }
    }
    assert_eq!(cases, 15);
    assert_eq!(observations, cases * rounds * 3);
    assert_eq!(comparisons, observations * 3);
    println!("PVP_LARGE_COMPLETE,cases={cases},rounds={rounds},observations={observations},comparisons={comparisons},failed_comparisons={failed_comparisons},performance_claim=none");
    assert_eq!(failed_comparisons, 0, "large packed paired phase comparisons failed");
}
