//! PVP3d resident comparison; see the prospectively frozen protocol.

#[path = "support/pvp3d_protocol.rs"]
mod protocol;

use std::sync::mpsc;
use std::time::Instant;

use flat_attention::pvp_fused2::WgpuPvpFused2Pipeline;
use flat_attention::pvp_tile8::WgpuPvpTile8Pipeline;
use flat_attention::pvp_vec4::{
    pvp_subset_zeta_vec4_host_in_place, FlatPvpVec4BitplanesV1, FlatPvpVec4LayoutV1,
    WgpuPvpVec4Pipeline,
};
use protocol::{candidate_order, checksum, percentile_ns, Candidate, Limits};

struct Harness {
    readbacks: std::cell::RefCell<Vec<wgpu::Buffer>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    vec4: WgpuPvpVec4Pipeline,
    fused2: WgpuPvpFused2Pipeline,
    tile8: WgpuPvpTile8Pipeline,
}

impl Harness {
    fn wait(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("PVP3d device completion failed; reject all timing evidence");
    }

    fn reset(&self, source: &wgpu::Buffer, state: &wgpu::Buffer, bytes: u64) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, state, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        self.wait();
    }

    fn dispatch(&self, candidate: Candidate, state: &wgpu::Buffer, layout: FlatPvpVec4LayoutV1) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        match candidate {
            Candidate::Vec4 => self
                .vec4
                .encode_all_stages(&self.device, &mut encoder, state, layout)
                .expect("PVP3d vec4 encode failed"),
            Candidate::Fused2 => {
                let stats = self
                    .fused2
                    .encode_all_stages(&self.device, &mut encoder, state, layout)
                    .expect("PVP3d fused2 encode failed");
                assert_eq!(
                    stats.logical_dispatches,
                    candidate.dispatches(layout.addresses())
                );
            }
            Candidate::Tile8 => {
                let stats = self
                    .tile8
                    .encode_all_stages(&self.device, &mut encoder, state, layout)
                    .expect("PVP3d tile8 encode failed");
                assert_eq!(
                    stats.logical_dispatches,
                    candidate.dispatches(layout.addresses())
                );
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.wait();
    }

    fn read(&self, state: &wgpu::Buffer, bytes: u64) -> Vec<u32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pvp3d-correctness-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(state, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.wait();
        receiver
            .recv()
            .expect("PVP3d map callback")
            .expect("PVP3d map failed");
        let mapped = slice.get_mapped_range().expect("PVP3d mapped range");
        let words = mapped
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 bytes")))
            .collect();
        drop(mapped);
        staging.unmap();
        // Retain completed readbacks across geometries so a newly mapped
        // allocation cannot alias a released previous oracle-readback buffer.
        self.readbacks.borrow_mut().push(staging);
        words
    }

    fn case(&self, k: usize, g: usize, smoke: bool) {
        let layout = FlatPvpVec4LayoutV1::new(k, g).expect("frozen PVP3d geometry");
        let bytes = u64::try_from(layout.storage_u32_words() * 4).expect("PVP3d state bytes");
        let device_limits = self.device.limits();
        let limits = Limits {
            buffer_bytes: device_limits.max_buffer_size,
            storage_binding_bytes: device_limits.max_storage_buffer_binding_size,
            workgroups: device_limits.max_compute_workgroups_per_dimension,
            invocations: device_limits.max_compute_invocations_per_workgroup,
            workgroup_x: device_limits.max_compute_workgroup_size_x,
            workgroup_storage_bytes: device_limits.max_compute_workgroup_storage_size,
        };
        if let Some(reason) = protocol::rejection(k, g, limits) {
            for candidate in Candidate::ALL {
                println!(
                    "skipped_limit,{k},{g},{},{bytes},{},NA,NA,NA,NA,NA,NA,NA,{reason},none",
                    candidate.name(),
                    bytes * 4
                );
            }
            return;
        }

        let initial = fixture(layout);
        let mut expected = initial.clone();
        pvp_subset_zeta_vec4_host_in_place(&mut expected).expect("PVP3d CPU oracle");
        let source = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pvp3d-resident-source"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let encoded: Vec<u8> = initial
            .words()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        self.queue.write_buffer(&source, 0, &encoded);
        // Make initial upload completion explicit before a resident reset.
        // This fence stays outside every warmup and measured wall interval.
        self.queue.submit(None);
        self.wait();
        drop(encoded);
        let states: [wgpu::Buffer; 3] = std::array::from_fn(|_| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pvp3d-resident-state"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        for candidate in Candidate::ALL {
            let state = &states[candidate as usize];
            self.reset(&source, state, bytes);
            self.dispatch(candidate, state, layout);
            assert_eq!(
                self.read(state, bytes),
                expected.words(),
                "PVP3d pre-timing oracle mismatch K={k} G={g} candidate={candidate:?}"
            );
        }
        let warmups = if smoke { 1 } else { 5 };
        let repeats = if smoke { 1 } else { 30 };
        let mut samples: [Vec<u128>; 3] = std::array::from_fn(|_| Vec::with_capacity(repeats));
        for iteration in 0..warmups + repeats {
            // Warmups and measurements each begin with ABC; the measured block
            // contains exactly five complete six-permutation cycles.
            let order_index = if iteration < warmups {
                iteration
            } else {
                iteration - warmups
            };
            for candidate in candidate_order(order_index) {
                let state = &states[candidate as usize];
                self.reset(&source, state, bytes);
                let start = Instant::now();
                self.dispatch(candidate, state, layout);
                let elapsed = start.elapsed().as_nanos();
                if iteration >= warmups {
                    samples[candidate as usize].push(elapsed);
                }
            }
        }
        // Validate the state produced by the last measured invocation, without
        // resetting it or running a replacement correctness dispatch.
        for candidate in Candidate::ALL {
            assert_eq!(
                self.read(&states[candidate as usize], bytes),
                expected.words(),
                "PVP3d post-timing oracle mismatch K={k} G={g} candidate={candidate:?}"
            );
        }
        let input_checksum = checksum(initial.words());
        let output_checksum = checksum(expected.words());
        for candidate in Candidate::ALL {
            let values = &samples[candidate as usize];
            let status = if smoke { "smoke" } else { "measured" };
            println!("{status},{k},{g},{},{bytes},{},{warmups},{repeats},{},{},{},{input_checksum:016x},{output_checksum:016x},none,none",
                candidate.name(), bytes * 4, candidate.dispatches(k),
                percentile_ns(values, 50), percentile_ns(values, 95));
            for (iteration, value) in values.iter().enumerate() {
                println!("sample,{k},{g},{},{iteration},{value}", candidate.name());
            }
        }
    }
}

fn fixture(layout: FlatPvpVec4LayoutV1) -> FlatPvpVec4BitplanesV1 {
    let mut words = vec![0_u32; layout.storage_u32_words()];
    let row_words = layout.vectors_per_address() * 4;
    let mut state = 0x9e37_79b9_u32 ^ layout.addresses() as u32 ^ layout.gates() as u32;
    for (index, word) in words.iter_mut().enumerate() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let remaining = layout.gates().saturating_sub(index % row_words * 32);
        *word = match remaining {
            0 => 0,
            1..=31 => state & ((1_u32 << remaining) - 1),
            _ => state,
        };
    }
    FlatPvpVec4BitplanesV1::from_words(layout, words).expect("canonical PVP3d fixture")
}

fn sanitize(value: &str) -> String {
    value.replace([',', '\n', '\r'], " ")
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let smoke = match args.as_slice() {
        [mode] if mode == "--smoke" => true,
        [mode] if mode == "--measure" => false,
        _ => panic!("use exactly --smoke or --measure; no implicit measurement"),
    };
    let revision = std::env::var("FLAT_SOURCE_REVISION")
        .expect("set FLAT_SOURCE_REVISION to the full source SHA");
    assert!(
        protocol::source_revision_valid(&revision),
        "full 40-hex source SHA required"
    );
    assert_eq!(
        Some(revision.as_str()),
        option_env!("FLAT_SOURCE_REVISION"),
        "runtime SHA must equal the compile-time source SHA; rebuild with FLAT_SOURCE_REVISION"
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: if smoke {
            wgpu::Backends::VULKAN
        } else {
            wgpu::Backends::all()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("PVP3d requires an actual WGPU adapter");
    let info = adapter.get_info();
    let hardware = matches!(
        info.device_type,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
    );
    assert!(
        smoke || hardware,
        "CPU/software/unknown adapters are ineligible for --measure"
    );
    println!("benchmark=pvp3d_three_candidate_same_device");
    println!("source_revision={revision}");
    println!("adapter={}", sanitize(&info.name));
    println!("backend={:?}", info.backend);
    println!("device_type={:?}", info.device_type);
    println!("vendor_id={}", info.vendor);
    println!("device_id={}", info.device);
    println!("driver={}", sanitize(&info.driver));
    println!("driver_info={}", sanitize(&info.driver_info));
    println!("hardware_measurement_eligible={}", !smoke && hardware);
    println!("physical_admission=external_remoteops_evidence_required");
    println!("performance_claim=none");
    println!("primary_timing_scope=resident_encode_plus_queue_submit_plus_device_wait");
    println!("state_reset_scope=completed_before_timer");
    println!("correctness_readback_scope=before_and_after_timing");
    println!("candidate_order=ABC_ACB_BAC_BCA_CAB_CBA");
    println!(
        "resident_payload_scope=source_plus_three_states_excludes_driver_uniforms_and_readback"
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("pvp3d-three-candidate-bench"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("PVP3d request_device failed");
    let harness = Harness {
        readbacks: Default::default(),
        vec4: WgpuPvpVec4Pipeline::new(&device).expect("PVP2 pipeline"),
        fused2: WgpuPvpFused2Pipeline::new(&device).expect("PVP3a pipeline"),
        tile8: WgpuPvpTile8Pipeline::new(&device).expect("PVP3c pipeline"),
        device,
        queue,
    };
    println!("status,k,g,candidate,state_bytes,resident_payload_bytes,warmups,repeats,logical_dispatches,p50_ns,p95_ns,input_checksum,output_checksum,skip_reason,performance_claim");
    if smoke {
        for (k, g) in protocol::SMOKE_GRID {
            harness.case(k, g, true);
        }
    } else {
        for k in protocol::K_GRID {
            for g in protocol::G_GRID {
                harness.case(k, g, false);
            }
        }
    }
    let readbacks = harness.readbacks.borrow();
    println!("retained_readbacks={}", readbacks.len());
    println!(
        "retained_readback_payload_bytes={}",
        readbacks.iter().map(wgpu::Buffer::size).sum::<u64>()
    );
    println!(
        "retained_readback_scope=oracle_buffers_excluded_from_four_state_payload_not_total_memory"
    );
    println!("qualification_status=complete");
}
