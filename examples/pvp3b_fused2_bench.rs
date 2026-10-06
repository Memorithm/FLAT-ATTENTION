#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("pvp3b_fused2_bench requires --features wgpu");
}

#[cfg(feature = "wgpu")]
fn main() {
    bench::run();
}

#[cfg(feature = "wgpu")]
mod bench {
    use std::cmp::Ordering;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use flat_attention::pvp_fused2::WgpuPvpFused2Pipeline;
    use flat_attention::pvp_vec4::{
        FlatPvpVec4BitplanesV1, FlatPvpVec4LayoutV1, WgpuPvpVec4Pipeline,
    };

    const DEFAULT_WARMUPS: usize = 5;
    const DEFAULT_REPEATS: usize = 30;
    const WORKGROUP_SIZE: u64 = 64;

    const K_GRID: [usize; 7] = [256, 1024, 4096, 16_384, 65_536, 262_144, 1_048_576];
    const G_GRID: [usize; 3] = [128, 512, 2048];

    #[derive(Clone, Copy)]
    enum Candidate {
        Vec4,
        Fused2,
    }

    struct DeviceHarness {
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_name: String,
        backend: wgpu::Backend,
        device_type: wgpu::DeviceType,
        driver: String,
        driver_info: String,
        timestamp_supported: bool,
        timestamp_period_ns: f32,
    }

    struct CaseBuffers {
        source: wgpu::Buffer,
        vec4_state: wgpu::Buffer,
        fused_state: wgpu::Buffer,
        bytes: u64,
    }

    struct TimestampHarness {
        query_set: wgpu::QuerySet,
        resolve: wgpu::Buffer,
        readback: wgpu::Buffer,
    }

    pub fn run() {
        let harness = harness().expect("PVP3b requires a WGPU adapter");
        let warmups = env_usize("FLAT_PVP3B_WARMUPS", DEFAULT_WARMUPS);
        let repeats = env_usize("FLAT_PVP3B_REPEATS", DEFAULT_REPEATS);
        assert_eq!(
            warmups, DEFAULT_WARMUPS,
            "PVP3b preregistration freezes five warmups"
        );
        assert_eq!(
            repeats, DEFAULT_REPEATS,
            "PVP3b preregistration freezes thirty measured repetitions"
        );

        println!("benchmark=pvp3b_fused2_same_device");
        println!("adapter={}", sanitize(&harness.adapter_name));
        println!("backend={:?}", harness.backend);
        println!("device_type={:?}", harness.device_type);
        println!("driver={}", sanitize(&harness.driver));
        println!("driver_info={}", sanitize(&harness.driver_info));
        println!(
            "source_revision={}",
            std::env::var("FLAT_SOURCE_REVISION").unwrap_or_else(|_| "unknown".into())
        );
        println!("primary_timing_scope=resident_encode_plus_queue_submit_plus_device_wait");
        println!("state_reset_scope=completed_before_timer");
        println!("correctness_readback_scope=before_timing");
        println!("pipeline_creation_in_timing=false");
        println!("timestamp_diagnostic_separate_from_primary_wall_samples=true");
        println!("timestamp_supported={}", harness.timestamp_supported);
        if harness.timestamp_supported {
            println!("timestamp_period_ns={:.9}", harness.timestamp_period_ns);
        }
        println!(
            "status,adapter,backend,k,g,state_bytes,warmups,repeats,vec4_dispatches,fused2_dispatches,vec4_median_ns,vec4_p95_ns,fused2_median_ns,fused2_p95_ns,vec4_over_fused2,fused2_wins,vec4_timestamp_median_ns,fused2_timestamp_median_ns,checksum,performance_claim"
        );

        let vec4 = WgpuPvpVec4Pipeline::new(&harness.device).expect("PVP2 pipeline");
        let fused = WgpuPvpFused2Pipeline::new(&harness.device).expect("PVP3a pipeline");

        for &k in &K_GRID {
            for &g in &G_GRID {
                run_case(&harness, &vec4, &fused, k, g, warmups, repeats);
            }
        }
    }

    fn harness() -> Option<DeviceHarness> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let info = adapter.get_info();
        let timestamp_features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let timestamp_supported = adapter.features().contains(timestamp_features);
        let required_features = if timestamp_supported {
            timestamp_features
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flat-pvp3b-fused2-bench"),
            required_features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        let timestamp_period_ns = queue.get_timestamp_period();

        Some(DeviceHarness {
            device,
            queue,
            adapter_name: info.name,
            backend: info.backend,
            device_type: info.device_type,
            driver: info.driver,
            driver_info: info.driver_info,
            timestamp_supported,
            timestamp_period_ns,
        })
    }

    fn run_case(
        harness: &DeviceHarness,
        vec4: &WgpuPvpVec4Pipeline,
        fused: &WgpuPvpFused2Pipeline,
        k: usize,
        g: usize,
        warmups: usize,
        repeats: usize,
    ) {
        let layout = FlatPvpVec4LayoutV1::new(k, g).expect("preregistered PVP3b geometry");
        let state_bytes = layout
            .storage_u32_words()
            .checked_mul(4)
            .expect("state bytes");
        let state_bytes_u64 = state_bytes as u64;
        let vectors_per_address = layout.vectors_per_address() as u64;
        let vec4_invocations = (k as u64 / 2)
            .checked_mul(vectors_per_address)
            .expect("vec4 invocation count");
        let fused_invocations = (k as u64 / 4)
            .checked_mul(vectors_per_address)
            .expect("fused invocation count");
        let vec4_workgroups = vec4_invocations.div_ceil(WORKGROUP_SIZE);
        let fused_workgroups = fused_invocations.div_ceil(WORKGROUP_SIZE);
        let limits = harness.device.limits();

        let rejected = state_bytes_u64 > limits.max_buffer_size
            || state_bytes_u64 > u64::from(limits.max_storage_buffer_binding_size)
            || vec4_workgroups > u64::from(limits.max_compute_workgroups_per_dimension)
            || fused_workgroups > u64::from(limits.max_compute_workgroups_per_dimension);

        if rejected {
            println!(
                "skipped_limit,{},{:?},{k},{g},{state_bytes},{warmups},{repeats},{},{},NA,NA,NA,NA,NA,0,NA,NA,NA,none",
                sanitize(&harness.adapter_name),
                harness.backend,
                layout.stages(),
                layout.stages().saturating_sub(1)
            );
            return;
        }

        let initial = fixture(layout);
        let buffers = buffers(harness, initial.words(), state_bytes_u64);

        reset(harness, &buffers.source, &buffers.vec4_state, buffers.bytes);
        dispatch_vec4(harness, vec4, &buffers.vec4_state, layout);
        let vec4_output = read_u32(harness, &buffers.vec4_state, layout.storage_u32_words());

        reset(
            harness,
            &buffers.source,
            &buffers.fused_state,
            buffers.bytes,
        );
        dispatch_fused(harness, fused, &buffers.fused_state, layout);
        let fused_output = read_u32(harness, &buffers.fused_state, layout.storage_u32_words());

        assert_eq!(
            fused_output, vec4_output,
            "PVP3b correctness gate failed for K={k} G={g}"
        );
        let checksum = checksum_u32(&vec4_output);
        drop(vec4_output);
        drop(fused_output);

        for iteration in 0..warmups {
            if iteration.is_multiple_of(2) {
                let _ = measure_wall(harness, Candidate::Vec4, vec4, fused, &buffers, layout);
                let _ = measure_wall(harness, Candidate::Fused2, vec4, fused, &buffers, layout);
            } else {
                let _ = measure_wall(harness, Candidate::Fused2, vec4, fused, &buffers, layout);
                let _ = measure_wall(harness, Candidate::Vec4, vec4, fused, &buffers, layout);
            }
        }

        let mut vec4_samples = Vec::with_capacity(repeats);
        let mut fused_samples = Vec::with_capacity(repeats);
        let mut fused_wins = 0usize;
        for iteration in 0..repeats {
            let (vec4_sample, fused_sample) = if iteration.is_multiple_of(2) {
                let a = measure_wall(harness, Candidate::Vec4, vec4, fused, &buffers, layout);
                let b = measure_wall(harness, Candidate::Fused2, vec4, fused, &buffers, layout);
                (a, b)
            } else {
                let b = measure_wall(harness, Candidate::Fused2, vec4, fused, &buffers, layout);
                let a = measure_wall(harness, Candidate::Vec4, vec4, fused, &buffers, layout);
                (a, b)
            };
            fused_wins += usize::from(fused_sample < vec4_sample);
            vec4_samples.push(vec4_sample);
            fused_samples.push(fused_sample);
        }

        let vec4_median = percentile_ns(&vec4_samples, 50);
        let fused_median = percentile_ns(&fused_samples, 50);
        let (vec4_timestamp, fused_timestamp) = if harness.timestamp_supported {
            let timestamps = TimestampHarness::new(&harness.device);
            let mut vec4_ns = Vec::with_capacity(repeats);
            let mut fused_ns = Vec::with_capacity(repeats);
            for iteration in 0..repeats {
                if iteration.is_multiple_of(2) {
                    vec4_ns.push(measure_timestamp(
                        harness,
                        &timestamps,
                        Candidate::Vec4,
                        vec4,
                        fused,
                        &buffers,
                        layout,
                    ));
                    fused_ns.push(measure_timestamp(
                        harness,
                        &timestamps,
                        Candidate::Fused2,
                        vec4,
                        fused,
                        &buffers,
                        layout,
                    ));
                } else {
                    fused_ns.push(measure_timestamp(
                        harness,
                        &timestamps,
                        Candidate::Fused2,
                        vec4,
                        fused,
                        &buffers,
                        layout,
                    ));
                    vec4_ns.push(measure_timestamp(
                        harness,
                        &timestamps,
                        Candidate::Vec4,
                        vec4,
                        fused,
                        &buffers,
                        layout,
                    ));
                }
            }
            (
                format!("{:.3}", percentile_f64(&vec4_ns, 50)),
                format!("{:.3}", percentile_f64(&fused_ns, 50)),
            )
        } else {
            ("NA".into(), "NA".into())
        };

        println!(
            "measured,{},{:?},{k},{g},{state_bytes},{warmups},{repeats},{},{},{vec4_median},{},{fused_median},{},{:.6},{fused_wins},{vec4_timestamp},{fused_timestamp},{checksum:016x},none",
            sanitize(&harness.adapter_name),
            harness.backend,
            layout.stages(),
            layout.stages().saturating_sub(1),
            percentile_ns(&vec4_samples, 95),
            percentile_ns(&fused_samples, 95),
            vec4_median as f64 / fused_median.max(1) as f64,
        );
    }

    impl TimestampHarness {
        fn new(device: &wgpu::Device) -> Self {
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("flat-pvp3b-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flat-pvp3b-timestamp-resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flat-pvp3b-timestamp-readback"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            Self {
                query_set,
                resolve,
                readback,
            }
        }
    }

    fn fixture(layout: FlatPvpVec4LayoutV1) -> FlatPvpVec4BitplanesV1 {
        let mut words = vec![0_u32; layout.storage_u32_words()];
        let mut state = 0x9e37_79b9_u32 ^ layout.addresses() as u32 ^ (layout.gates() as u32);
        for word in &mut words {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *word = state;
        }
        FlatPvpVec4BitplanesV1::from_words(layout, words)
            .expect("PVP3b grid is 128-bit aligned and canonical")
    }

    fn buffers(harness: &DeviceHarness, words: &[u32], bytes: u64) -> CaseBuffers {
        let source = harness.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-pvp3b-source"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        harness.queue.write_buffer(&source, 0, &encode_u32(words));

        let state = |label| {
            harness.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        CaseBuffers {
            source,
            vec4_state: state("flat-pvp3b-vec4-state"),
            fused_state: state("flat-pvp3b-fused-state"),
            bytes,
        }
    }

    fn reset(harness: &DeviceHarness, source: &wgpu::Buffer, state: &wgpu::Buffer, bytes: u64) {
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-pvp3b-reset"),
            });
        encoder.copy_buffer_to_buffer(source, 0, state, 0, bytes);
        harness.queue.submit(Some(encoder.finish()));
        let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
    }

    fn measure_wall(
        harness: &DeviceHarness,
        candidate: Candidate,
        vec4: &WgpuPvpVec4Pipeline,
        fused: &WgpuPvpFused2Pipeline,
        buffers: &CaseBuffers,
        layout: FlatPvpVec4LayoutV1,
    ) -> Duration {
        let state = match candidate {
            Candidate::Vec4 => &buffers.vec4_state,
            Candidate::Fused2 => &buffers.fused_state,
        };
        reset(harness, &buffers.source, state, buffers.bytes);
        let start = Instant::now();
        match candidate {
            Candidate::Vec4 => dispatch_vec4(harness, vec4, state, layout),
            Candidate::Fused2 => dispatch_fused(harness, fused, state, layout),
        }
        start.elapsed()
    }

    fn dispatch_vec4(
        harness: &DeviceHarness,
        pipeline: &WgpuPvpVec4Pipeline,
        state: &wgpu::Buffer,
        layout: FlatPvpVec4LayoutV1,
    ) {
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-pvp3b-vec4"),
            });
        pipeline
            .encode_all_stages(&harness.device, &mut encoder, state, layout)
            .expect("PVP3b vec4 encode");
        harness.queue.submit(Some(encoder.finish()));
        let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
    }

    fn dispatch_fused(
        harness: &DeviceHarness,
        pipeline: &WgpuPvpFused2Pipeline,
        state: &wgpu::Buffer,
        layout: FlatPvpVec4LayoutV1,
    ) {
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-pvp3b-fused2"),
            });
        pipeline
            .encode_all_stages(&harness.device, &mut encoder, state, layout)
            .expect("PVP3b fused2 encode");
        harness.queue.submit(Some(encoder.finish()));
        let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
    }

    fn measure_timestamp(
        harness: &DeviceHarness,
        timestamps: &TimestampHarness,
        candidate: Candidate,
        vec4: &WgpuPvpVec4Pipeline,
        fused: &WgpuPvpFused2Pipeline,
        buffers: &CaseBuffers,
        layout: FlatPvpVec4LayoutV1,
    ) -> f64 {
        let state = match candidate {
            Candidate::Vec4 => &buffers.vec4_state,
            Candidate::Fused2 => &buffers.fused_state,
        };
        reset(harness, &buffers.source, state, buffers.bytes);

        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-pvp3b-timestamp"),
            });
        encoder.write_timestamp(&timestamps.query_set, 0);
        match candidate {
            Candidate::Vec4 => vec4
                .encode_all_stages(&harness.device, &mut encoder, state, layout)
                .expect("PVP3b timestamp vec4 encode"),
            Candidate::Fused2 => {
                fused
                    .encode_all_stages(&harness.device, &mut encoder, state, layout)
                    .expect("PVP3b timestamp fused2 encode");
            }
        }
        encoder.write_timestamp(&timestamps.query_set, 1);
        encoder.resolve_query_set(&timestamps.query_set, 0..2, &timestamps.resolve, 0);
        encoder.copy_buffer_to_buffer(&timestamps.resolve, 0, &timestamps.readback, 0, 16);
        harness.queue.submit(Some(encoder.finish()));
        let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());

        let slice = timestamps.readback.slice(..16);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .expect("PVP3b timestamp map callback")
            .expect("PVP3b timestamp map");
        let mapped = slice
            .get_mapped_range()
            .expect("PVP3b timestamp mapped range");
        let start = u64::from_ne_bytes(mapped[0..8].try_into().expect("timestamp start bytes"));
        let end = u64::from_ne_bytes(mapped[8..16].try_into().expect("timestamp end bytes"));
        drop(mapped);
        timestamps.readback.unmap();
        end.wrapping_sub(start) as f64 * f64::from(harness.timestamp_period_ns)
    }

    fn read_u32(harness: &DeviceHarness, source: &wgpu::Buffer, len: usize) -> Vec<u32> {
        let bytes = (len * 4) as u64;
        let staging = harness.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-pvp3b-correctness-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-pvp3b-correctness-readback"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        harness.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .expect("PVP3b correctness map callback")
            .expect("PVP3b correctness map");
        let mapped = slice
            .get_mapped_range()
            .expect("PVP3b correctness mapped range");
        let values = mapped
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        drop(mapped);
        staging.unmap();
        values
    }

    fn encode_u32(values: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn checksum_u32(values: &[u32]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for &value in values {
            for byte in value.to_ne_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn percentile_ns(samples: &[Duration], percentile: usize) -> u128 {
        let mut values: Vec<u128> = samples.iter().map(Duration::as_nanos).collect();
        values.sort_unstable();
        let rank = percentile.saturating_mul(values.len()).div_ceil(100).max(1);
        values[rank - 1]
    }

    fn percentile_f64(samples: &[f64], percentile: usize) -> f64 {
        let mut values = samples.to_vec();
        values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        let rank = percentile.saturating_mul(values.len()).div_ceil(100).max(1);
        values[rank - 1]
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn sanitize(value: &str) -> String {
        value.replace(',', ";").replace('\n', " ")
    }
}
