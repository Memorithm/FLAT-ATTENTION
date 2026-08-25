use std::{cmp::Ordering, process::Command, sync::mpsc, time::Instant};

use epg_core::{EpgGeometryDescriptor, EpgGeometryKind, EpgPositionDomain, So4Geometry};
use flat_attention::{FlatAttentionConfig, GroupedAttentionShape};
use flat_epg_q4_candidate::{EpgQ4CandidatePipeline, PreparedEpgQ4Candidate};
use flat_epg_reference::{forward_reference_grouped_epg, EpgEmbeddingConfig};
use flat_epg_wgpu::{EpgQualificationPass, EpgVec4QualificationPipeline, PreparedEpgQualification};

const DEFAULT_WARMUP: usize = 8;
const DEFAULT_ITERATIONS: usize = 40;
const DEFAULT_SEQ_LENS: &[usize] = &[32, 128];
const ATOL: f32 = 1.2e-3;
const RTOL: f32 = 4.0e-3;

struct BenchContext<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    baseline: &'a EpgVec4QualificationPipeline,
    candidate: &'a EpgQ4CandidatePipeline,
    warmup: usize,
    iterations: usize,
    position_offset: u64,
    theta: f32,
}

#[derive(Clone, Copy)]
struct Case {
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    causal: bool,
    geometry: EpgGeometryKind,
}

struct CaseResult {
    baseline_median_us: f64,
    baseline_p95_us: f64,
    candidate_median_us: f64,
    candidate_p95_us: f64,
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.031 + phase;
            x.sin() * 0.63 + (x * 0.47).cos() * 0.27
        })
        .collect()
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn input_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    label: &'static str,
) -> wgpu::Buffer {
    let bytes = bytes_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn read_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
    label: &'static str,
) -> Vec<f32> {
    let bytes = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..bytes);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range().expect("valid mapped range");
    let values = mapped
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    drop(mapped);
    staging.unmap();
    values
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = ATOL + RTOL * actual.abs().max(expected.abs());
        let error = (actual - expected).abs();
        assert!(
            actual.is_finite() && expected.is_finite() && error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

fn summarize(mut samples_us: Vec<f64>) -> (f64, f64) {
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = if samples_us.len() % 2 == 0 {
        let upper = samples_us.len() / 2;
        (samples_us[upper - 1] + samples_us[upper]) * 0.5
    } else {
        samples_us[samples_us.len() / 2]
    };
    let p95_index = ((samples_us.len() * 95).div_ceil(100)).saturating_sub(1);
    (median, samples_us[p95_index])
}

fn execute_baseline(context: &BenchContext<'_>, prepared: &PreparedEpgQualification) -> f64 {
    let start = Instant::now();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-epg-hardware-baseline"),
        });
    context.baseline.encode_prepared(&mut encoder, prepared);
    context.queue.submit(Some(encoder.finish()));
    let _ = context.device.poll(wgpu::PollType::wait_indefinitely());
    start.elapsed().as_secs_f64() * 1.0e6
}

fn execute_candidate(context: &BenchContext<'_>, prepared: &PreparedEpgQ4Candidate) -> f64 {
    let start = Instant::now();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-epg-hardware-q4"),
        });
    context.candidate.encode_prepared(&mut encoder, prepared);
    context.queue.submit(Some(encoder.finish()));
    let _ = context.device.poll(wgpu::PollType::wait_indefinitely());
    start.elapsed().as_secs_f64() * 1.0e6
}

fn descriptor(theta: f32, head_dim: usize, kind: EpgGeometryKind) -> EpgGeometryDescriptor {
    match kind {
        EpgGeometryKind::So2 => EpgGeometryDescriptor::so2(theta).unwrap(),
        EpgGeometryKind::HybridSo4(geometry) => {
            EpgGeometryDescriptor::hybrid_so4(theta, u32::try_from(head_dim / 2).unwrap(), geometry)
                .unwrap()
        }
    }
}

fn geometry_name(kind: EpgGeometryKind) -> &'static str {
    match kind {
        EpgGeometryKind::So2 => "so2",
        EpgGeometryKind::HybridSo4(So4Geometry::Biplanar) => "so4_biplanar",
        EpgGeometryKind::HybridSo4(So4Geometry::Isoclinic) => "so4_isoclinic",
    }
}

fn run_case(context: &BenchContext<'_>, case: Case) -> CaseResult {
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: case.q_heads,
        kv_heads: case.kv_heads,
        seq_len: case.seq_len,
        head_dim: case.head_dim,
    };
    let config = FlatAttentionConfig {
        causal: case.causal,
        softmax_scale: None,
    };
    let geometry = descriptor(context.theta, case.head_dim, case.geometry);
    let position = EpgPositionDomain::new(context.position_offset);

    let q = fixture(shape.q_tensor_len().unwrap(), 0.13);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.73);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.37);
    let expected = forward_reference_grouped_epg(
        &q,
        &k,
        &v,
        shape,
        config,
        EpgEmbeddingConfig { geometry, position },
    )
    .unwrap();

    let q_gpu = input_buffer(context.device, context.queue, &q, "flat-epg-bench-q");
    let k_gpu = input_buffer(context.device, context.queue, &k, "flat-epg-bench-k");
    let v_gpu = input_buffer(context.device, context.queue, &v, "flat-epg-bench-v");
    let baseline_output = context
        .baseline
        .create_output_buffer(context.device, shape)
        .unwrap();
    let candidate_output = context
        .candidate
        .create_output_buffer(context.device, shape)
        .unwrap();

    let baseline_prepared = context
        .baseline
        .prepare(
            context.device,
            EpgQualificationPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &baseline_output,
                shape,
                config,
                geometry,
                position,
            },
        )
        .unwrap();
    let candidate_prepared = context
        .candidate
        .prepare(
            context.device,
            EpgQualificationPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &candidate_output,
                shape,
                config,
                geometry,
                position,
            },
        )
        .unwrap();
    let layout = candidate_prepared.layout();

    let _ = execute_baseline(context, &baseline_prepared);
    let _ = execute_candidate(context, &candidate_prepared);
    let baseline_actual = read_f32(
        context.device,
        context.queue,
        &baseline_output,
        layout.combined_elements,
        "flat-epg-bench-baseline-readback",
    );
    let candidate_actual = read_f32(
        context.device,
        context.queue,
        &candidate_output,
        layout.combined_elements,
        "flat-epg-bench-q4-readback",
    );
    assert_close(
        "baseline_O",
        &baseline_actual[..layout.lse_offset()],
        &expected.output,
    );
    assert_close(
        "baseline_LSE",
        &baseline_actual[layout.lse_offset()..],
        &expected.lse,
    );
    assert_close(
        "candidate_O",
        &candidate_actual[..layout.lse_offset()],
        &expected.output,
    );
    assert_close(
        "candidate_LSE",
        &candidate_actual[layout.lse_offset()..],
        &expected.lse,
    );
    assert_close("candidate_vs_baseline", &candidate_actual, &baseline_actual);

    for _ in 0..context.warmup {
        let _ = execute_baseline(context, &baseline_prepared);
        let _ = execute_candidate(context, &candidate_prepared);
    }

    let mut baseline_samples = Vec::with_capacity(context.iterations);
    let mut candidate_samples = Vec::with_capacity(context.iterations);
    for iteration in 0..context.iterations {
        if iteration % 2 == 0 {
            baseline_samples.push(execute_baseline(context, &baseline_prepared));
            candidate_samples.push(execute_candidate(context, &candidate_prepared));
        } else {
            candidate_samples.push(execute_candidate(context, &candidate_prepared));
            baseline_samples.push(execute_baseline(context, &baseline_prepared));
        }
    }
    let (baseline_median_us, baseline_p95_us) = summarize(baseline_samples);
    let (candidate_median_us, candidate_p95_us) = summarize(candidate_samples);
    CaseResult {
        baseline_median_us,
        baseline_p95_us,
        candidate_median_us,
        candidate_p95_us,
    }
}

fn parse_positive_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&parsed| parsed > 0)
            .unwrap_or_else(|| panic!("{name} must be a positive integer, got {value:?}")),
        Err(_) => default,
    }
}

fn parse_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .unwrap_or_else(|_| panic!("{name} must be a non-negative integer, got {value:?}")),
        Err(_) => default,
    }
}

fn parse_theta() -> f32 {
    match std::env::var("FLAT_EPG_BENCH_THETA") {
        Ok(value) => {
            let theta = value
                .parse::<f32>()
                .unwrap_or_else(|_| panic!("FLAT_EPG_BENCH_THETA must be f32, got {value:?}"));
            assert!(
                theta.is_finite() && theta > 0.0,
                "theta must be finite and positive"
            );
            theta
        }
        Err(_) => 10_000.0,
    }
}

fn parse_seq_lens() -> Vec<usize> {
    let Ok(raw) = std::env::var("FLAT_EPG_BENCH_SEQ_LENS") else {
        return DEFAULT_SEQ_LENS.to_vec();
    };
    let parsed: Vec<usize> = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .ok()
                .filter(|&seq_len| seq_len > 0)
                .unwrap_or_else(|| panic!("invalid sequence length {value:?}"))
        })
        .collect();
    assert!(
        !parsed.is_empty(),
        "FLAT_EPG_BENCH_SEQ_LENS must not be empty"
    );
    parsed
}

fn software_adapter(info: &wgpu::AdapterInfo) -> bool {
    let fingerprint = format!("{} {} {}", info.name, info.driver, info.driver_info).to_lowercase();
    matches!(info.device_type, wgpu::DeviceType::Cpu)
        || fingerprint.contains("llvmpipe")
        || fingerprint.contains("lavapipe")
        || fingerprint.contains("swiftshader")
}

fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|sha| sha.trim().to_owned())
        .filter(|sha| !sha.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn main() {
    let warmup = parse_positive_usize("FLAT_EPG_BENCH_WARMUP", DEFAULT_WARMUP);
    let iterations = parse_positive_usize("FLAT_EPG_BENCH_ITERATIONS", DEFAULT_ITERATIONS);
    let seq_lens = parse_seq_lens();
    let position_offset = parse_u64("FLAT_EPG_BENCH_POSITION_OFFSET", 0);
    let theta = parse_theta();
    let allow_software = std::env::var("FLAT_EPG_BENCH_ALLOW_SOFTWARE").as_deref() == Ok("1");

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
    .expect("EPG hardware benchmark requires a WGPU adapter");
    let info = adapter.get_info();
    let is_software = software_adapter(&info);
    assert!(
        !is_software || allow_software,
        "software/CPU adapter detected ({:?}, {}). Set FLAT_EPG_BENCH_ALLOW_SOFTWARE=1 only for harness debugging; do not use such results for a hardware performance claim",
        info.device_type,
        info.name
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-epg-q4-hardware-sweep"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .expect("EPG hardware benchmark request_device failed");
    let baseline = EpgVec4QualificationPipeline::new(&device)
        .expect("qualified EPG baseline pipeline creation failed");
    let candidate =
        EpgQ4CandidatePipeline::new(&device).expect("EPG Q4 candidate pipeline creation failed");
    let context = BenchContext {
        device: &device,
        queue: &queue,
        baseline: &baseline,
        candidate: &candidate,
        warmup,
        iterations,
        position_offset,
        theta,
    };

    println!("benchmark=epg_q4_hardware_sweep");
    println!("git_sha={}", git_sha());
    println!("device_name={}", info.name);
    println!("device_type={:?}", info.device_type);
    println!("backend={:?}", info.backend);
    println!("driver={}", info.driver);
    println!("driver_info={}", info.driver_info);
    println!("software_adapter={is_software}");
    println!("hardware_measurement_eligible={}", !is_software);
    println!("precision=f32");
    println!("warmup={warmup}");
    println!("iterations={iterations}");
    println!("theta={theta}");
    println!("position_offset={position_offset}");
    println!("seq_lens={seq_lens:?}");
    println!("timing_scope=command_encoder+encode_prepared+queue_submit+device_poll");
    println!("uploads_in_timing=false");
    println!("readback_in_timing=false");
    println!("prepare_bindings_in_timing=false");
    println!("sample_order=paired_interleaved_baseline_q4_then_q4_baseline");
    println!("correctness_gate=cpu_oracle_and_qualified_gpu_baseline_before_timing");
    println!("baseline=flat-epg-wgpu-qualification");
    println!("candidate=flat-epg-q4-candidate");
    println!("batch,q_heads,kv_heads,seq_len,head_dim,causal,geometry,so4_dims,baseline_median_us,baseline_p95_us,q4_median_us,q4_p95_us,speedup_baseline_over_q4,baseline_logical_query_tokens_per_s,q4_logical_query_tokens_per_s");

    let geometries = [
        EpgGeometryKind::So2,
        EpgGeometryKind::HybridSo4(So4Geometry::Biplanar),
        EpgGeometryKind::HybridSo4(So4Geometry::Isoclinic),
    ];
    for &(q_heads, kv_heads) in &[(4_usize, 4_usize), (4, 2), (4, 1)] {
        for &seq_len in &seq_lens {
            for &head_dim in &[64_usize, 128] {
                for &causal in &[false, true] {
                    for &geometry in &geometries {
                        let case = Case {
                            q_heads,
                            kv_heads,
                            seq_len,
                            head_dim,
                            causal,
                            geometry,
                        };
                        let result = run_case(&context, case);
                        let speedup = result.baseline_median_us / result.candidate_median_us;
                        let baseline_tokens_s = seq_len as f64 * 1.0e6 / result.baseline_median_us;
                        let candidate_tokens_s =
                            seq_len as f64 * 1.0e6 / result.candidate_median_us;
                        let so4_dims = if matches!(geometry, EpgGeometryKind::So2) {
                            0
                        } else {
                            head_dim / 2
                        };
                        println!(
                            "1,{q_heads},{kv_heads},{seq_len},{head_dim},{causal},{},{so4_dims},{:.3},{:.3},{:.3},{:.3},{speedup:.6},{baseline_tokens_s:.3},{candidate_tokens_s:.3}",
                            geometry_name(geometry),
                            result.baseline_median_us,
                            result.baseline_p95_us,
                            result.candidate_median_us,
                            result.candidate_p95_us,
                        );
                    }
                }
            }
        }
    }
    println!("performance_claim=measurement_only_no_production_routing_change");
}
