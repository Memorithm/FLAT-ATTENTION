use std::{borrow::Cow, process::Command, sync::mpsc};

use flat_ada_a1_candidate::{ada_a1_branchless_wgsl, ADA_A1_FWD_WGSL};
use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, FLAT_FWD_WGSL, WGSL_QUERY_ROWS,
};

const DEFAULT_WARMUP: usize = 5;
const DEFAULT_SAMPLES: usize = 21;
const DEFAULT_INNER_DISPATCHES: usize = 4;
const DEFAULT_SEQ_LENS: &[usize] = &[32, 128];
const DEFAULT_HEAD_DIMS: &[usize] = &[8, 64, 128];
const ORACLE_ATOL: f32 = 1.0e-3;
const ORACLE_RTOL: f32 = 4.0e-3;
const AB_ATOL: f32 = 5.0e-5;
const AB_RTOL: f32 = 5.0e-4;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    timestamp_period_ns: f32,
}

struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}

struct PreparedDispatch {
    bind_group: wgpu::BindGroup,
    output: wgpu::Buffer,
    _params: wgpu::Buffer,
    dispatch_x: u32,
    dispatch_y: u32,
    combined_len: usize,
}

#[derive(Clone, Copy)]
struct Case {
    seq_len: usize,
    head_dim: usize,
    causal: bool,
}

struct Summary {
    median_ns: f64,
    p95_ns: f64,
    mad_ns: f64,
}

fn software_adapter(info: &wgpu::AdapterInfo) -> bool {
    let fingerprint = format!("{} {} {}", info.name, info.driver, info.driver_info).to_lowercase();
    matches!(info.device_type, wgpu::DeviceType::Cpu)
        || fingerprint.contains("llvmpipe")
        || fingerprint.contains("lavapipe")
        || fingerprint.contains("swiftshader")
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

fn parse_list(name: &str, default: &[usize]) -> Vec<usize> {
    let Ok(raw) = std::env::var(name) else {
        return default.to_vec();
    };
    let parsed: Vec<usize> = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .ok()
                .filter(|&item| item > 0)
                .unwrap_or_else(|| panic!("invalid positive integer {value:?} in {name}"))
        })
        .collect();
    assert!(!parsed.is_empty(), "{name} must not be empty");
    parsed
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

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let bounded = u16::try_from(index).expect("benchmark fixture index fits u16");
            let x = f32::from(bounded) * 0.031 + phase;
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
        size: u64::try_from(bytes.len()).expect("input byte length fits u64"),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn create_pipeline(
    device: &wgpu::Device,
    source: &str,
    entry_point: Option<&'static str>,
    label: &'static str,
) -> wgpu::ComputePipeline {
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    if let Some(error) = pollster::block_on(error_scope.pop()) {
        panic!("{label} pipeline validation failed: {error}");
    }
    pipeline
}

fn prepare_dispatch(
    harness: &Harness,
    pipeline: &wgpu::ComputePipeline,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    shape: AttentionShape,
    config: FlatAttentionConfig,
) -> PreparedDispatch {
    let tensor_len = shape.tensor_len().unwrap();
    let lse_len = shape.lse_len().unwrap();
    let combined_len = tensor_len.checked_add(lse_len).unwrap();
    let combined_bytes = u64::try_from(combined_len).unwrap().checked_mul(4).unwrap();
    let output = harness.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-ada-a1-3way-output"),
        size: combined_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let batch_heads = shape.batch.checked_mul(shape.heads).unwrap();
    let scale = config.resolved_scale(shape.head_dim).unwrap();
    let params = [
        u32::try_from(shape.seq_len).unwrap(),
        u32::try_from(shape.head_dim).unwrap(),
        u32::try_from(batch_heads).unwrap(),
        u32::from(config.causal),
        scale.to_bits(),
        0,
        0,
        0,
    ];
    let mut param_bytes = Vec::with_capacity(std::mem::size_of_val(&params));
    for value in params {
        param_bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let params_buffer = harness.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-ada-a1-3way-params"),
        size: u64::try_from(param_bytes.len()).unwrap(),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    harness.queue.write_buffer(&params_buffer, 0, &param_bytes);

    let bind_group = harness
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-ada-a1-3way-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    PreparedDispatch {
        bind_group,
        output,
        _params: params_buffer,
        dispatch_x: u32::try_from(shape.seq_len.div_ceil(WGSL_QUERY_ROWS)).unwrap(),
        dispatch_y: u32::try_from(batch_heads).unwrap(),
        combined_len,
    }
}

fn execute_plain(
    harness: &Harness,
    pipeline: &wgpu::ComputePipeline,
    prepared: &PreparedDispatch,
    dispatches: usize,
) {
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-ada-a1-3way-plain"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-ada-a1-3way-plain"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &prepared.bind_group, &[]);
        for _ in 0..dispatches {
            pass.dispatch_workgroups(prepared.dispatch_x, prepared.dispatch_y, 1);
        }
    }
    harness.queue.submit(Some(encoder.finish()));
    let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
}

fn read_f32(harness: &Harness, source: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let bytes = u64::try_from(len)
        .expect("readback length fits u64")
        .checked_mul(4)
        .expect("readback byte length fits u64");
    let staging = harness.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-ada-a1-3way-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-ada-a1-3way-readback"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    harness.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
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

impl GpuTimer {
    fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("flat-ada-a1-3way-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-ada-a1-3way-timestamp-resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-ada-a1-3way-timestamp-readback"),
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

fn measure_gpu_ns(
    harness: &Harness,
    timer: &GpuTimer,
    pipeline: &wgpu::ComputePipeline,
    prepared: &PreparedDispatch,
    inner_dispatches: usize,
) -> f64 {
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-ada-a1-3way-gpu-timestamp"),
        });
    {
        let timestamp_writes = wgpu::ComputePassTimestampWrites {
            query_set: &timer.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-ada-a1-3way-gpu-timestamp"),
            timestamp_writes: Some(timestamp_writes),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &prepared.bind_group, &[]);
        for _ in 0..inner_dispatches {
            pass.dispatch_workgroups(prepared.dispatch_x, prepared.dispatch_y, 1);
        }
    }
    encoder.resolve_query_set(&timer.query_set, 0..2, &timer.resolve, 0);
    encoder.copy_buffer_to_buffer(&timer.resolve, 0, &timer.readback, 0, 16);
    harness.queue.submit(Some(encoder.finish()));

    let slice = timer.readback.slice(..16);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().expect("valid mapped range");
    let begin = u64::from_ne_bytes(mapped[0..8].try_into().unwrap());
    let end = u64::from_ne_bytes(mapped[8..16].try_into().unwrap());
    drop(mapped);
    timer.readback.unmap();

    let ticks = end.wrapping_sub(begin);
    let ticks32 = u32::try_from(ticks).expect("single timestamp interval fits u32 ticks");
    let inner32 = u32::try_from(inner_dispatches).expect("inner dispatch count fits u32");
    f64::from(ticks32) * f64::from(harness.timestamp_period_ns) / f64::from(inner32)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn summarize(mut values: Vec<f64>) -> Summary {
    values.sort_by(f64::total_cmp);
    let median_ns = if values.len().is_multiple_of(2) {
        let mid = values.len() / 2;
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[values.len() / 2]
    };
    let p95_index = ((values.len() * 95).div_ceil(100)).saturating_sub(1);
    let p95_ns = values[p95_index];
    let deviations = values
        .iter()
        .map(|value| (value - median_ns).abs())
        .collect();
    Summary {
        median_ns,
        p95_ns,
        mad_ns: median(deviations),
    }
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        let tolerance = atol + rtol * actual.abs().max(expected.abs());
        assert!(
            actual.is_finite() && expected.is_finite() && error <= tolerance,
            "{name}: index={index} actual={actual} expected={expected} error={error} tolerance={tolerance}"
        );
    }
}

fn record_sample(
    harness: &Harness,
    timer: &GpuTimer,
    pipeline: &wgpu::ComputePipeline,
    prepared: &PreparedDispatch,
    inner_dispatches: usize,
    samples: &mut Vec<f64>,
) {
    samples.push(measure_gpu_ns(
        harness,
        timer,
        pipeline,
        prepared,
        inner_dispatches,
    ));
}

fn main() {
    let warmup = parse_positive_usize("FLAT_ADA_A1_BENCH_WARMUP", DEFAULT_WARMUP);
    let samples = parse_positive_usize("FLAT_ADA_A1_BENCH_SAMPLES", DEFAULT_SAMPLES);
    let inner_dispatches = parse_positive_usize(
        "FLAT_ADA_A1_BENCH_INNER_DISPATCHES",
        DEFAULT_INNER_DISPATCHES,
    );
    let seq_lens = parse_list("FLAT_ADA_A1_BENCH_SEQ_LENS", DEFAULT_SEQ_LENS);
    let head_dims = parse_list("FLAT_ADA_A1_BENCH_HEAD_DIMS", DEFAULT_HEAD_DIMS);
    let require_thor = std::env::var("FLAT_ADA_A1_REQUIRE_THOR").as_deref() == Ok("1");

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
    .expect("ADA-A1 three-way benchmark requires a WGPU adapter");
    let info = adapter.get_info();
    let is_software = software_adapter(&info);
    assert!(
        !is_software,
        "software/CPU adapter detected ({:?}, {}); refusing hardware claim",
        info.device_type, info.name
    );
    if require_thor {
        assert!(
            info.name.contains("NVIDIA") && info.name.contains("Thor"),
            "FLAT_ADA_A1_REQUIRE_THOR=1 but adapter is {}",
            info.name
        );
        assert_eq!(
            info.backend,
            wgpu::Backend::Vulkan,
            "Thor qualification requires Vulkan"
        );
    }
    let features = adapter.features();
    assert!(
        features.contains(wgpu::Features::TIMESTAMP_QUERY),
        "adapter does not expose TIMESTAMP_QUERY; refusing CPU-wall-time substitute"
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-ada-a1-thor-3way"),
        required_features: wgpu::Features::TIMESTAMP_QUERY,
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .expect("ADA-A1 three-way timestamp-capable request_device failed");
    let timestamp_period_ns = queue.get_timestamp_period();
    assert!(
        timestamp_period_ns.is_finite() && timestamp_period_ns > 0.0,
        "invalid GPU timestamp period {timestamp_period_ns}"
    );
    let harness = Harness {
        device,
        queue,
        info,
        timestamp_period_ns,
    };

    let branchless_source = ada_a1_branchless_wgsl();
    let baseline = create_pipeline(
        &harness.device,
        FLAT_FWD_WGSL,
        Some("flat_attention_forward"),
        "flat-q4-qualified-baseline-3way",
    );
    let branched = create_pipeline(
        &harness.device,
        ADA_A1_FWD_WGSL,
        Some("flat_attention_forward_ada_a1"),
        "flat-ada-a1-branched-3way",
    );
    let branchless = create_pipeline(
        &harness.device,
        &branchless_source,
        Some("flat_attention_forward_ada_a1_branchless"),
        "flat-ada-a1b-branchless-3way",
    );
    let timer = GpuTimer::new(&harness.device);

    println!("benchmark=ada_a1_gpu_timestamp_3way");
    println!("git_sha={}", git_sha());
    println!("device_name={}", harness.info.name);
    println!("device_type={:?}", harness.info.device_type);
    println!("backend={:?}", harness.info.backend);
    println!("driver={}", harness.info.driver);
    println!("driver_info={}", harness.info.driver_info);
    println!("software_adapter=false");
    println!("timestamp_query=true");
    println!("timestamp_period_ns={}", harness.timestamp_period_ns);
    println!("warmup={warmup}");
    println!("samples={samples}");
    println!("inner_dispatches={inner_dispatches}");
    println!("seq_lens={seq_lens:?}");
    println!("head_dims={head_dims:?}");
    println!("timing_scope=gpu_compute_pass_timestamp_per_dispatch");
    println!("sample_order=cyclic_q4_a1_a1b_then_a1_a1b_q4_then_a1b_q4_a1");
    println!("uploads_in_timing=false");
    println!("readback_in_timing=false");
    println!("pipeline_compile_in_timing=false");
    println!("correctness_gate=cpu_oracle_and_three_way_gpu_parity_before_timing");
    println!("baseline=flat_fwd.wgsl");
    println!("branched=flat_fwd_ada_a1.wgsl");
    println!("branchless=ada_a1_branchless_wgsl()");
    println!("batch,heads,seq_len,head_dim,causal,baseline_median_ns,baseline_p95_ns,baseline_mad_ns,branched_median_ns,branched_p95_ns,branched_mad_ns,branchless_median_ns,branchless_p95_ns,branchless_mad_ns,speedup_baseline_over_branched,speedup_baseline_over_branchless,speedup_branched_over_branchless");

    for &seq_len in &seq_lens {
        for &head_dim in &head_dims {
            assert!(
                head_dim <= 128,
                "head_dim {head_dim} exceeds Q4 portable maximum"
            );
            for causal in [false, true] {
                let case = Case {
                    seq_len,
                    head_dim,
                    causal,
                };
                let shape = AttentionShape {
                    batch: 1,
                    heads: 1,
                    seq_len: case.seq_len,
                    head_dim: case.head_dim,
                };
                let config = FlatAttentionConfig {
                    causal: case.causal,
                    softmax_scale: None,
                };
                let len = shape.tensor_len().unwrap();
                let q = fixture(len, 0.13);
                let k = fixture(len, 0.73);
                let v = fixture(len, 1.37);
                let expected = forward_reference(&q, &k, &v, shape, config).unwrap();
                let mut expected_combined = expected.output;
                expected_combined.extend_from_slice(&expected.lse);

                let q_gpu = input_buffer(&harness.device, &harness.queue, &q, "flat-ada-a1-3way-q");
                let k_gpu = input_buffer(&harness.device, &harness.queue, &k, "flat-ada-a1-3way-k");
                let v_gpu = input_buffer(&harness.device, &harness.queue, &v, "flat-ada-a1-3way-v");
                let baseline_prepared =
                    prepare_dispatch(&harness, &baseline, &q_gpu, &k_gpu, &v_gpu, shape, config);
                let branched_prepared =
                    prepare_dispatch(&harness, &branched, &q_gpu, &k_gpu, &v_gpu, shape, config);
                let branchless_prepared =
                    prepare_dispatch(&harness, &branchless, &q_gpu, &k_gpu, &v_gpu, shape, config);

                execute_plain(&harness, &baseline, &baseline_prepared, 1);
                execute_plain(&harness, &branched, &branched_prepared, 1);
                execute_plain(&harness, &branchless, &branchless_prepared, 1);

                let baseline_actual = read_f32(
                    &harness,
                    &baseline_prepared.output,
                    baseline_prepared.combined_len,
                );
                let branched_actual = read_f32(
                    &harness,
                    &branched_prepared.output,
                    branched_prepared.combined_len,
                );
                let branchless_actual = read_f32(
                    &harness,
                    &branchless_prepared.output,
                    branchless_prepared.combined_len,
                );
                assert_close(
                    "qualified Q4 vs CPU oracle",
                    &baseline_actual,
                    &expected_combined,
                    ORACLE_ATOL,
                    ORACLE_RTOL,
                );
                assert_close(
                    "ADA-A1 branched vs CPU oracle",
                    &branched_actual,
                    &expected_combined,
                    ORACLE_ATOL,
                    ORACLE_RTOL,
                );
                assert_close(
                    "ADA-A1B branchless vs CPU oracle",
                    &branchless_actual,
                    &expected_combined,
                    ORACLE_ATOL,
                    ORACLE_RTOL,
                );
                assert_close(
                    "ADA-A1 branched vs Q4",
                    &branched_actual,
                    &baseline_actual,
                    AB_ATOL,
                    AB_RTOL,
                );
                assert_close(
                    "ADA-A1B branchless vs Q4",
                    &branchless_actual,
                    &baseline_actual,
                    AB_ATOL,
                    AB_RTOL,
                );
                assert_close(
                    "ADA-A1B branchless vs ADA-A1 branched",
                    &branchless_actual,
                    &branched_actual,
                    AB_ATOL,
                    AB_RTOL,
                );

                for _ in 0..warmup {
                    execute_plain(&harness, &baseline, &baseline_prepared, inner_dispatches);
                    execute_plain(&harness, &branched, &branched_prepared, inner_dispatches);
                    execute_plain(
                        &harness,
                        &branchless,
                        &branchless_prepared,
                        inner_dispatches,
                    );
                }

                let mut baseline_samples = Vec::with_capacity(samples);
                let mut branched_samples = Vec::with_capacity(samples);
                let mut branchless_samples = Vec::with_capacity(samples);
                for sample in 0..samples {
                    match sample % 3 {
                        0 => {
                            record_sample(
                                &harness,
                                &timer,
                                &baseline,
                                &baseline_prepared,
                                inner_dispatches,
                                &mut baseline_samples,
                            );
                            record_sample(
                                &harness,
                                &timer,
                                &branched,
                                &branched_prepared,
                                inner_dispatches,
                                &mut branched_samples,
                            );
                            record_sample(
                                &harness,
                                &timer,
                                &branchless,
                                &branchless_prepared,
                                inner_dispatches,
                                &mut branchless_samples,
                            );
                        }
                        1 => {
                            record_sample(
                                &harness,
                                &timer,
                                &branched,
                                &branched_prepared,
                                inner_dispatches,
                                &mut branched_samples,
                            );
                            record_sample(
                                &harness,
                                &timer,
                                &branchless,
                                &branchless_prepared,
                                inner_dispatches,
                                &mut branchless_samples,
                            );
                            record_sample(
                                &harness,
                                &timer,
                                &baseline,
                                &baseline_prepared,
                                inner_dispatches,
                                &mut baseline_samples,
                            );
                        }
                        _ => {
                            record_sample(
                                &harness,
                                &timer,
                                &branchless,
                                &branchless_prepared,
                                inner_dispatches,
                                &mut branchless_samples,
                            );
                            record_sample(
                                &harness,
                                &timer,
                                &baseline,
                                &baseline_prepared,
                                inner_dispatches,
                                &mut baseline_samples,
                            );
                            record_sample(
                                &harness,
                                &timer,
                                &branched,
                                &branched_prepared,
                                inner_dispatches,
                                &mut branched_samples,
                            );
                        }
                    }
                }

                let baseline_summary = summarize(baseline_samples);
                let branched_summary = summarize(branched_samples);
                let branchless_summary = summarize(branchless_samples);
                let speedup_baseline_over_branched =
                    baseline_summary.median_ns / branched_summary.median_ns;
                let speedup_baseline_over_branchless =
                    baseline_summary.median_ns / branchless_summary.median_ns;
                let speedup_branched_over_branchless =
                    branched_summary.median_ns / branchless_summary.median_ns;
                println!(
                    "1,1,{seq_len},{head_dim},{causal},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{speedup_baseline_over_branched:.6},{speedup_baseline_over_branchless:.6},{speedup_branched_over_branchless:.6}",
                    baseline_summary.median_ns,
                    baseline_summary.p95_ns,
                    baseline_summary.mad_ns,
                    branched_summary.median_ns,
                    branched_summary.p95_ns,
                    branched_summary.mad_ns,
                    branchless_summary.median_ns,
                    branchless_summary.p95_ns,
                    branchless_summary.mad_ns,
                );
            }
        }
    }
    println!("performance_claim=measurement_only_no_production_routing_change");
}
