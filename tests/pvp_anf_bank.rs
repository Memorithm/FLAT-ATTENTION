//! Explicit ANF coefficients -> physical bitplanes -> independently evaluated truth.
//! Frozen corpus: docs/research/PVP_ANF_BANK_QUALIFICATION_PROTOCOL.md.

use flat_attention::pvp::{
    pvp_subset_zeta_u32_scalar_in_place, FlatPvpU32BitplanesV1, FlatPvpU32LayoutV1,
};
use flat_attention::pvp_vec4::{
    pvp_subset_zeta_vec4_host_in_place, FlatPvpVec4BitplanesV1, FlatPvpVec4LayoutV1,
};

const GEOMETRIES: [(usize, usize); 8] = [
    (1, 1),
    (2, 31),
    (4, 65),
    (8, 129),
    (64, 257),
    (256, 513),
    (2048, 257),
    (256, 2048),
];
const BANKS: [&str; 3] = ["boundary", "sparse4", "dense32"];

struct AnfBank {
    k: usize,
    terms: Vec<Vec<usize>>,
}

impl AnfBank {
    fn frozen(k: usize, gates: usize, kind: &str) -> Self {
        let terms = (0..gates)
            .map(|g| match kind {
                "boundary" => match g % 6 {
                    0 => vec![],
                    1 => vec![0],
                    2 => vec![usize::from(k > 1)],
                    3 => vec![k - 1],
                    4 => vec![0, usize::from(k > 1)],
                    _ => (0..k.ilog2()).map(|bit| 1_usize << bit).collect(),
                },
                "sparse4" | "dense32" => {
                    let count = k.min(if kind == "sparse4" { 4 } else { 32 });
                    let masks: Vec<_> = (0..count).map(|t| (257 * g + 73 * t) % k).collect();
                    let mut seen = vec![false; k];
                    for &mask in &masks {
                        assert!(!seen[mask], "duplicate frozen ANF term");
                        seen[mask] = true;
                    }
                    masks
                }
                _ => panic!("unknown frozen bank"),
            })
            .collect();
        Self { k, terms }
    }

    fn coefficients_gate_major(&self) -> Vec<u64> {
        let words_per_gate = self.k.div_ceil(64);
        let mut words = vec![0_u64; words_per_gate * self.terms.len()];
        for (gate, terms) in self.terms.iter().enumerate() {
            for &mask in terms {
                // ANF is over GF(2): duplicate terms cancel, including K=1.
                words[gate * words_per_gate + mask / 64] ^= 1_u64 << (mask % 64);
            }
        }
        words
    }

    fn direct(&self, gate: usize, assignment: usize) -> bool {
        self.terms[gate]
            .iter()
            .fold(false, |value, &mask| value ^ ((assignment & mask) == mask))
    }

    fn truth_words(&self, words_per_address: usize) -> Vec<u32> {
        // Independent direct evaluation, without using any PVP layout adapter
        // or butterfly to construct expected values. Unused lanes stay zero.
        let mut words = vec![0_u32; self.k * words_per_address];
        for address in 0..self.k {
            for gate in 0..self.terms.len() {
                if self.direct(gate, address) {
                    words[address * words_per_address + gate / 32] |= 1_u32 << (gate % 32);
                }
            }
        }
        words
    }
}

#[test]
fn direct_anf_oracle_has_known_constant_linear_and_full_degree_truths() {
    let bank = AnfBank::frozen(8, 6, "boundary");
    assert_eq!(bank.truth_words(1), vec![18, 38, 50, 6, 50, 6, 18, 46]);
    // At K=1 the two constant monomials of boundary gate four cancel.
    let constant_bank = AnfBank::frozen(1, 6, "boundary");
    assert_eq!(constant_bank.truth_words(1), vec![14]);
    assert_eq!(
        constant_bank.coefficients_gate_major(),
        vec![0, 1, 1, 1, 0, 0]
    );
}

#[test]
fn all_frozen_anf_banks_match_host_scalar_and_vec4_and_round_trip() {
    let mut cases = 0;
    for (k, g) in GEOMETRIES {
        for kind in BANKS {
            let bank = AnfBank::frozen(k, g, kind);
            let coefficients = bank.coefficients_gate_major();
            let scalar_layout = FlatPvpU32LayoutV1::new(k, g).unwrap();
            let vec4_layout = FlatPvpVec4LayoutV1::new(k, g).unwrap();
            let mut scalar =
                FlatPvpU32BitplanesV1::from_gate_major_u64(scalar_layout, &coefficients).unwrap();
            let mut vector =
                FlatPvpVec4BitplanesV1::from_gate_major_u64(vec4_layout, &coefficients).unwrap();
            let original_scalar = scalar.words().to_vec();
            let original_vector = vector.words().to_vec();

            pvp_subset_zeta_u32_scalar_in_place(&mut scalar).unwrap();
            pvp_subset_zeta_vec4_host_in_place(&mut vector).unwrap();
            assert_eq!(
                scalar.words(),
                bank.truth_words(scalar_layout.words_per_address()),
                "scalar ANF semantics K={k} G={g} bank={kind}"
            );
            assert_eq!(
                vector.words(),
                bank.truth_words(vec4_layout.storage_u32_words() / k),
                "vec4 ANF semantics K={k} G={g} bank={kind}"
            );
            pvp_subset_zeta_u32_scalar_in_place(&mut scalar).unwrap();
            pvp_subset_zeta_vec4_host_in_place(&mut vector).unwrap();
            assert_eq!(scalar.words(), original_scalar);
            assert_eq!(vector.words(), original_vector);
            cases += 1;
        }
    }
    assert_eq!(cases, 24);
}

#[cfg(feature = "wgpu")]
mod gpu {
    use super::*;
    use flat_attention::pvp::WgpuPvpScalarPipeline;
    use flat_attention::pvp_fused2::WgpuPvpFused2Pipeline;
    use flat_attention::pvp_tile8::WgpuPvpTile8Pipeline;
    use flat_attention::pvp_vec4::WgpuPvpVec4Pipeline;
    use std::sync::mpsc;

    fn read(device: &wgpu::Device, queue: &wgpu::Queue, state: &wgpu::Buffer) -> Vec<u32> {
        let bytes = state.size();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pvp-anf-bank-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(state, 0, &staging, 0, bytes);
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        let words = mapped
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        drop(mapped);
        staging.unmap();
        words
    }

    #[test]
    fn actual_wgpu_all_four_kernels_preserve_explicit_anf_semantics() {
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
                "mandatory ANF-bank qualification requires an actual WGPU adapter"
            );
            eprintln!("PVP_ANF_BANK,status=skipped_adapter_unavailable,performance_claim=none");
            return;
        };
        let info = adapter.get_info();
        println!(
            "PVP_ANF_BANK_ADAPTER,backend={:?},device_type={:?},name={:?},driver={:?},performance_claim=none",
            info.backend, info.device_type, info.name, info.driver
        );
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pvp-explicit-anf-bank"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .unwrap();
        let scalar = WgpuPvpScalarPipeline::new(&device).unwrap();
        let vec4 = WgpuPvpVec4Pipeline::new(&device).unwrap();
        let fused2 = WgpuPvpFused2Pipeline::new(&device).unwrap();
        let tile8 = WgpuPvpTile8Pipeline::new(&device).unwrap();
        let mut checks = 0;
        for (k, g) in GEOMETRIES {
            for kind in BANKS {
                let bank = AnfBank::frozen(k, g, kind);
                let coefficients = bank.coefficients_gate_major();
                let scalar_layout = FlatPvpU32LayoutV1::new(k, g).unwrap();
                let vec4_layout = FlatPvpVec4LayoutV1::new(k, g).unwrap();
                let scalar_initial =
                    FlatPvpU32BitplanesV1::from_gate_major_u64(scalar_layout, &coefficients)
                        .unwrap();
                let vec4_initial =
                    FlatPvpVec4BitplanesV1::from_gate_major_u64(vec4_layout, &coefficients)
                        .unwrap();
                let scalar_truth = bank.truth_words(scalar_layout.words_per_address());
                let vec4_truth = bank.truth_words(vec4_layout.storage_u32_words() / k);
                for candidate in ["scalar", "vec4", "fused2", "tile8"] {
                    let (initial, truth) = if candidate == "scalar" {
                        (scalar_initial.words(), &scalar_truth)
                    } else {
                        (vec4_initial.words(), &vec4_truth)
                    };
                    let bytes: Vec<_> = initial.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let state = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("pvp-anf-bank-state"),
                        size: bytes.len() as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    queue.write_buffer(&state, 0, &bytes);
                    for pass in 0..2 {
                        let mut encoder = device.create_command_encoder(&Default::default());
                        match candidate {
                            "scalar" => scalar
                                .encode_all_stages(&device, &mut encoder, &state, scalar_layout)
                                .unwrap(),
                            "vec4" => vec4
                                .encode_all_stages(&device, &mut encoder, &state, vec4_layout)
                                .unwrap(),
                            "fused2" => {
                                fused2
                                    .encode_all_stages(&device, &mut encoder, &state, vec4_layout)
                                    .unwrap();
                            }
                            "tile8" => {
                                tile8
                                    .encode_all_stages(&device, &mut encoder, &state, vec4_layout)
                                    .unwrap();
                            }
                            _ => unreachable!(),
                        }
                        queue.submit(Some(encoder.finish()));
                        let expected = if pass == 0 { truth.as_slice() } else { initial };
                        assert_eq!(
                            read(&device, &queue, &state),
                            expected,
                            "ANF-bank mismatch K={k} G={g} bank={kind} candidate={candidate} pass={pass}"
                        );
                    }
                    println!(
                        "PVP_ANF_BANK_CASE,K={k},G={g},bank={kind},candidate={candidate},truth=exact,round_trip=exact,performance_claim=none"
                    );
                    checks += 1;
                }
            }
        }
        assert_eq!(checks, 96);
        println!("PVP_ANF_BANK_COMPLETE,cases=24,candidate_checks=96,performance_claim=none");
    }
}
