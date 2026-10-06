//! Untimed paired readbacks isolating PVP upload, reset and transformed state.
//!
//! A matching readback pair is not proof of device state correctness: both may
//! share a defect. This diagnostic does not modify or qualify the PVP3d banc.

#[path = "support/pvp3d_protocol.rs"]
#[allow(dead_code)] // Reuse the historical candidate, checksum and admission definitions.
mod bench_protocol;
#[path = "support/pvp_stage_diagnostic.rs"]
mod diagnostic;

use std::sync::mpsc;
use std::time::Duration;

use bench_protocol::{checksum, Candidate, Limits};
use flat_attention::pvp_fused2::WgpuPvpFused2Pipeline;
use flat_attention::pvp_tile8::WgpuPvpTile8Pipeline;
use flat_attention::pvp_vec4::{
    pvp_subset_zeta_vec4_host_in_place, FlatPvpVec4LayoutV1, WgpuPvpVec4Pipeline,
};

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    vec4: WgpuPvpVec4Pipeline,
    fused2: WgpuPvpFused2Pipeline,
    tile8: WgpuPvpTile8Pipeline,
    explicit_map_transition: bool,
}

impl Harness {
    fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .expect("diagnostic device completion");
    }

    /// Both copies use the same source and submission, with distinct readbacks.
    /// Each call allocates fresh readbacks and releases them after reading.
    fn read_pair(&self, source: &wgpu::Buffer, bytes: u64) -> [Vec<u32>; 2] {
        let staging: [wgpu::Buffer; 2] = std::array::from_fn(|_| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pvp-stage-paired-readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for readback in &staging {
            encoder.copy_buffer_to_buffer(source, 0, readback, 0, bytes);
        }
        if self.explicit_map_transition {
            // Explicitly test the transition omitted by map_async in wgpu
            // 30.0.1 (upstream issue #9306). Apply to BOTH destinations so the
            // two modes are separate experiments, not a mixed control pair.
            encoder.transition_resources(
                staging.iter().map(|buffer| wgpu::BufferTransition {
                    buffer,
                    state: wgpu::BufferUses::MAP_READ,
                }),
                std::iter::empty(),
            );
        }
        self.queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        for (index, readback) in staging.iter().enumerate() {
            let sender = sender.clone();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send((index, result));
                });
        }
        drop(sender);
        self.wait();
        for _ in 0..staging.len() {
            let (index, result) = receiver
                .recv_timeout(Duration::from_secs(5))
                .expect("diagnostic map callback deadline");
            result.unwrap_or_else(|error| panic!("diagnostic readback {index} map: {error}"));
        }
        std::array::from_fn(|index| {
            let mapped = staging[index]
                .slice(..)
                .get_mapped_range()
                .expect("diagnostic mapped range");
            let words = mapped
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 bytes")))
                .collect();
            drop(mapped);
            staging[index].unmap();
            words
        })
    }

    fn reset(&self, source: &wgpu::Buffer, state: &wgpu::Buffer, bytes: u64) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, state, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        self.wait();
    }

    fn transform(&self, candidate: Candidate, state: &wgpu::Buffer, layout: FlatPvpVec4LayoutV1) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        match candidate {
            Candidate::Vec4 => self
                .vec4
                .encode_all_stages(&self.device, &mut encoder, state, layout)
                .expect("diagnostic vec4 encode"),
            Candidate::Fused2 => {
                self.fused2
                    .encode_all_stages(&self.device, &mut encoder, state, layout)
                    .expect("diagnostic fused2 encode");
            }
            Candidate::Tile8 => {
                self.tile8
                    .encode_all_stages(&self.device, &mut encoder, state, layout)
                    .expect("diagnostic tile8 encode");
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.wait();
    }

    fn check_pair(
        &self,
        round: usize,
        layout: FlatPvpVec4LayoutV1,
        candidate_phase: (&str, &str),
        state: &wgpu::Buffer,
        expected: &[u32],
    ) -> usize {
        let (candidate, phase) = candidate_phase;
        let bytes = u64::try_from(layout.storage_u32_words() * 4).expect("state bytes");
        let [a, b] = self.read_pair(state, bytes);
        let comparisons = [
            ("a_vs_oracle", a.as_slice(), expected),
            ("b_vs_oracle", b.as_slice(), expected),
            ("a_vs_b", a.as_slice(), b.as_slice()),
        ];
        let mut failed = 0;
        for (comparison, actual, reference) in comparisons {
            let differences = diagnostic::differences(actual, reference);
            failed += usize::from(differences.count != 0);
            println!(
                "check,round={round},k={},g={},candidate={candidate},phase={phase},comparison={comparison},words={},mismatched_words={},actual_checksum={:016x},expected_checksum={:016x},first={:?}",
                layout.addresses(),
                layout.gates(),
                expected.len(),
                differences.count,
                checksum(actual),
                checksum(reference),
                differences.first,
            );
        }
        failed
    }

    fn geometry(&self, round: usize, k: usize, g: usize) -> usize {
        let layout = FlatPvpVec4LayoutV1::new(k, g).expect("fixed diagnostic geometry");
        let initial = diagnostic::fixture(layout);
        let mut expected = initial.clone();
        pvp_subset_zeta_vec4_host_in_place(&mut expected).expect("diagnostic CPU oracle");
        let bytes = u64::try_from(layout.storage_u32_words() * 4).expect("state bytes");
        let source = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pvp-stage-source"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let encoded: Vec<u8> = initial
            .words()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        self.queue.write_buffer(&source, 0, &encoded);
        drop(encoded);
        // Flush the queued upload through the first diagnostic copy submission.
        // The additional phase readbacks change scheduling versus PVP3d:
        // passing this diagnosis does not qualify or repair the historical banc.
        let mut failures =
            self.check_pair(round, layout, ("none", "source"), &source, initial.words());
        let states: [wgpu::Buffer; 3] = std::array::from_fn(|_| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pvp-stage-state"),
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
            failures += self.check_pair(
                round,
                layout,
                (candidate.name(), "reset"),
                state,
                initial.words(),
            );
            self.transform(candidate, state, layout);
            failures += self.check_pair(
                round,
                layout,
                (candidate.name(), "transformed"),
                state,
                expected.words(),
            );
        }
        failures
    }
}

fn sanitize(value: &str) -> String {
    value.replace([',', '\n', '\r'], " ")
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let explicit_map_transition = match args.as_slice() {
        [mode] if mode == "--diagnose" => false,
        [mode] if mode == "--diagnose-map-transition" => true,
        _ => panic!("use exactly --diagnose or --diagnose-map-transition"),
    };
    let revision = std::env::var("FLAT_SOURCE_REVISION").expect("full source SHA required");
    assert!(
        bench_protocol::source_revision_valid(&revision),
        "invalid source SHA"
    );
    assert_eq!(
        Some(revision.as_str()),
        option_env!("FLAT_SOURCE_REVISION"),
        "compile/runtime SHA mismatch; rebuild"
    );
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("diagnostic adapter unavailable; no silent skip");
    let info = adapter.get_info();
    println!("diagnostic=pvp_stage_paired_readback_v1");
    println!("source_revision={revision}");
    println!("explicit_map_transition={explicit_map_transition},device_wait_timeout_seconds=60,map_callback_timeout_seconds=5");
    println!(
        "adapter={},backend={:?},device_type={:?},vendor={},device={},driver={},driver_info={}",
        sanitize(&info.name),
        info.backend,
        info.device_type,
        info.vendor,
        info.device,
        sanitize(&info.driver),
        sanitize(&info.driver_info)
    );
    println!("rounds={},geometries=65536x512_then_262144x128,candidates=vec4_fused2_tile8,readbacks_per_check=2,fresh_readbacks=true", diagnostic::ROUNDS);
    println!("timing=none,performance_claim=none,paired_agreement_does_not_prove_device_correctness=true");
    println!("phase_reads_add_submissions_and_fences=true,coverage=source_reset_full_transform,historical_benchmark_reproduction=false");
    let limits = adapter.limits();
    let admission = Limits {
        buffer_bytes: limits.max_buffer_size,
        storage_binding_bytes: limits.max_storage_buffer_binding_size,
        workgroups: limits.max_compute_workgroups_per_dimension,
        invocations: limits.max_compute_invocations_per_workgroup,
        workgroup_x: limits.max_compute_workgroup_size_x,
        workgroup_storage_bytes: limits.max_compute_workgroup_storage_size,
    };
    println!("limits={limits:?}");
    for (k, g) in diagnostic::GEOMETRIES {
        assert_eq!(
            bench_protocol::rejection(k, g, admission),
            None,
            "fixed diagnostic geometry unsupported k={k} g={g}; do not shrink silently"
        );
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("pvp-stage-diagnostic"),
        required_limits: limits,
        ..Default::default()
    }))
    .expect("diagnostic request device");
    let harness = Harness {
        vec4: WgpuPvpVec4Pipeline::new(&device).expect("vec4 pipeline"),
        fused2: WgpuPvpFused2Pipeline::new(&device).expect("fused2 pipeline"),
        tile8: WgpuPvpTile8Pipeline::new(&device).expect("tile8 pipeline"),
        device,
        queue,
        explicit_map_transition,
    };
    let mut failed_comparisons = 0;
    for round in 0..diagnostic::ROUNDS {
        for (k, g) in diagnostic::GEOMETRIES {
            failed_comparisons += harness.geometry(round, k, g);
        }
    }
    println!(
        "diagnostic_status=complete,failed_comparisons={failed_comparisons},performance_claim=none"
    );
    if failed_comparisons != 0 {
        std::process::exit(1);
    }
}
