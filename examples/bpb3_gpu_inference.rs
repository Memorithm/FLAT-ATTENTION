//! Model-independent BPB-3 GPU inference microkernel comparison.
//! The three GPU arms consume the SAME 64 signed ternary coefficients and
//! 64-bit bipolar input per block; no dense conversion during timed scans.
//! All-block scalar parity is mandatory before reporting descriptive timings.
//! This is NOT SML inference, a model-quality comparison, or a GPU speedup claim.
use std::borrow::Cow;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;

const BLOCKS: usize = 4096;
const REPEATS: usize = 15;
const WARMUPS: usize = 4;

const WGSL: &str = r#"
struct Words { data: array<u32>, }
struct Results { data: array<i32>, }
@group(0) @binding(0) var<storage, read> w: Words;
@group(0) @binding(1) var<storage, read> x: Words;
@group(0) @binding(2) var<storage, read_write> out: Results;

fn signed_count(signs: u32, plane: u32, input_sign: u32) -> i32 {
    let same = ~(signs ^ input_sign);
    return i32(2u * countOneBits(same & plane)) - i32(countOneBits(plane));
}

@compute @workgroup_size(64)
fn bpb3(@builtin(global_invocation_id) tid: vec3<u32>) {
    let id = tid.x;
    if id >= arrayLength(&out.data) { return; }
    let offset = id * 6u;
    let mask = id * 2u;
    let a = signed_count(w.data[offset], w.data[offset + 2u], x.data[mask]) +
            2 * signed_count(w.data[offset], w.data[offset + 4u], x.data[mask]);
    let b = signed_count(w.data[offset + 1u], w.data[offset + 3u], x.data[mask + 1u]) +
            2 * signed_count(w.data[offset + 1u], w.data[offset + 5u], x.data[mask + 1u]);
    out.data[id] = a + b;
}

@compute @workgroup_size(64)
fn sym3(@builtin(global_invocation_id) tid: vec3<u32>) {
    let id = tid.x;
    if id >= arrayLength(&out.data) { return; }
    var sum = 0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        let bit = i * 3u;
        let word_index = bit / 32u;
        let shift = bit % 32u;
        let part = w.data[id * 6u + word_index] >> shift;
        var payload = part;
        if shift > 29u {
            payload = payload | (w.data[id * 6u + word_index + 1u] << (32u - shift));
        }
        let raw = payload & 7u;
        let q = select(i32(raw), i32(raw) - 8, raw >= 4u);
        let neg = (x.data[id * 2u + i / 32u] >> (i % 32u)) & 1u;
        sum = sum + select(q, -q, neg == 1u);
    }
    out.data[id] = sum;
}

@compute @workgroup_size(64)
fn int8(@builtin(global_invocation_id) tid: vec3<u32>) {
    let id = tid.x;
    if id >= arrayLength(&out.data) { return; }
    var sum = 0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        let packed = w.data[id * 16u + i / 4u];
        let raw = (packed >> (8u * (i % 4u))) & 255u;
        let q = select(i32(raw), i32(raw) - 256, raw >= 128u);
        let neg = (x.data[id * 2u + i / 32u] >> (i % 32u)) & 1u;
        sum = sum + select(q, -q, neg == 1u);
    }
    out.data[id] = sum;
}
"#;

fn step(seed: &mut u64) -> u64 {
    *seed ^= *seed >> 12;
    *seed ^= *seed << 25;
    *seed ^= *seed >> 27;
    (*seed).wrapping_mul(2685821657736338717)
}

struct Corpus {
    bp: Vec<u32>,
    sym3: Vec<u32>,
    int8: Vec<u32>,
    masks: Vec<u32>,
    expected: Vec<i32>,
}

fn packed_data() -> Corpus {
    let mut rng = 0x95ab_cafe_2026_1107u64;
    let mut bp = Vec::with_capacity(BLOCKS * 6);
    let mut sym3 = Vec::with_capacity(BLOCKS * 6);
    let mut int8 = Vec::with_capacity(BLOCKS * 16);
    let mut inputs = Vec::with_capacity(BLOCKS * 2);
    let mut expected = Vec::with_capacity(BLOCKS);
    for _ in 0..BLOCKS {
        let mut signs = [0u32; 2];
        let mut low = [0u32; 2];
        let mut high = [0u32; 2];
        let mut packed3 = [0u32; 6];
        let mut packed8 = [0u32; 16];
        let mut q = [0i8; 64];
        for (i, slot) in q.iter_mut().enumerate() {
            let value = (step(&mut rng) % 7) as i8 - 3;
            *slot = value;
            let pos = i / 32;
            let bit = 1u32 << (i % 32);
            if value < 0 {
                signs[pos] |= bit;
            }
            if value.unsigned_abs() & 1 != 0 {
                low[pos] |= bit;
            }
            if value.unsigned_abs() & 2 != 0 {
                high[pos] |= bit;
            }
            let triple = (i32::from(value) as u32) & 7;
            let bit3 = i * 3;
            packed3[bit3 / 32] |= triple << (bit3 % 32);
            if bit3 % 32 > 29 {
                packed3[bit3 / 32 + 1] |= triple >> (32 - bit3 % 32);
            }
            packed8[i / 4] |= (value as u8 as u32) << (8 * (i % 4));
        }
        let mask = step(&mut rng);
        let scalar: i32 = q
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if (mask >> i) & 1 == 0 {
                    i32::from(v)
                } else {
                    -i32::from(v)
                }
            })
            .sum();
        expected.push(scalar);
        bp.extend_from_slice(&[signs[0], signs[1], low[0], low[1], high[0], high[1]]);
        sym3.extend_from_slice(&packed3);
        int8.extend_from_slice(&packed8);
        inputs.extend_from_slice(&[mask as u32, (mask >> 32) as u32]);
    }
    Corpus { bp, sym3, int8, masks: inputs, expected }
}

fn as_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn wait(device: &wgpu::Device) {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        })
        .expect("device wait failed");
}

fn readback(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<i32> {
    let size = (BLOCKS * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BPB3 scalar-oracle readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    encoder.transition_resources(
        std::iter::once(wgpu::BufferTransition {
            buffer: &staging,
            state: wgpu::BufferUses::MAP_READ,
        }),
        std::iter::empty(),
    );
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    wait(device);
    rx.recv_timeout(Duration::from_secs(5))
        .expect("readback callback timeout")
        .expect("GPU map error");
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .expect("GPU mapped data");
    let data = mapped
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect();
    drop(mapped);
    staging.unmap();
    data
}

fn device_buffer(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    let bytes = as_bytes(data);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

struct Arm {
    name: &'static str,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    output: wgpu::Buffer,
}

fn make_arm(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry: &'static str,
    weights: &wgpu::Buffer,
    inputs: &wgpu::Buffer,
) -> Arm {
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BPB3 independent readback buffer"),
        size: (BLOCKS * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(entry),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: inputs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });
    Arm {
        name: entry,
        pipeline,
        group,
        output,
    }
}

fn dispatch(device: &wgpu::Device, queue: &wgpu::Queue, arm: &Arm) -> u128 {
    let start = Instant::now();
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("BPB3 inference-only dot product"),
            ..Default::default()
        });
        pass.set_pipeline(&arm.pipeline);
        pass.set_bind_group(0, &arm.group, &[]);
        pass.dispatch_workgroups((BLOCKS as u32).div_ceil(64), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    wait(device);
    start.elapsed().as_nanos()
}

fn median(times: &mut [u128]) -> u128 {
    times.sort_unstable();
    times[times.len() / 2]
}

fn main() {
    let Corpus { bp, sym3, int8, masks, expected } = packed_data();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("STOP_NO_GPU_ADAPTER");
    let info = adapter.get_info();
    let physical =
        info.device_type != wgpu::DeviceType::Cpu && info.device_type != wgpu::DeviceType::Other;
    if std::env::var("BPB_GPU_REQUIRE_PHYSICAL").as_deref() == Ok("1") {
        assert!(physical, "STOP_SOFTWARE_ADAPTER_NO_PHYSICAL_RESULT");
    }
    let limits = adapter.limits();
    assert!(
        limits.max_storage_buffer_binding_size as usize >= BLOCKS * 64,
        "STOP_UNSUPPORTED_BUFFER_SIZE"
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("BPB3 standalone GPU inference study"),
        required_limits: limits,
        ..Default::default()
    }))
    .expect("STOP_DEVICE_UNAVAILABLE");
    let source = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("BPB3 / equal-bit INT3 / dense INT8 independent WGSL"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let buf_inputs = device_buffer(&device, "bipolar masks", &masks);
    let buf_bp = device_buffer(&device, "BPB3 three bitplanes", &bp);
    let buf_int3 = device_buffer(&device, "INT3 two's complement", &sym3);
    let buf_int8 = device_buffer(&device, "dense INT8", &int8);
    let arms = [
        make_arm(&device, &source, "bpb3", &buf_bp, &buf_inputs),
        make_arm(&device, &source, "sym3", &buf_int3, &buf_inputs),
        make_arm(&device, &source, "int8", &buf_int8, &buf_inputs),
    ];
    println!("protocol=bpb3-standalone-gpu-inference-v1,model_integrated=false,all_block_oracle=required");
    println!("adapter={:?},device_type={:?},backend={:?},driver={},physical_adapter={},blocks={},warmups={},repeats={}",
        info.name, info.device_type, info.backend, info.driver, physical, BLOCKS, WARMUPS, REPEATS);
    for arm in &arms {
        dispatch(&device, &queue, arm);
        let observed = readback(&device, &queue, &arm.output);
        assert_eq!(observed.len(), expected.len(), "STOP_READBACK_LENGTH");
        for (i, (&actual, &oracle)) in observed.iter().zip(expected.iter()).enumerate() {
            assert_eq!(actual, oracle, "STOP_GPU_ORACLE arm={} block={i}", arm.name);
        }
        println!(
            "PASS_GPU_BLOCK_ORACLE arm={} blocks={} mismatches=0",
            arm.name, BLOCKS
        );
    }
    for _ in 0..WARMUPS {
        for arm in &arms {
            dispatch(&device, &queue, arm);
        }
    }
    let mut timings: [Vec<u128>; 3] = std::array::from_fn(|_| Vec::with_capacity(REPEATS));
    for repeat in 0..REPEATS {
        for ordinal in 0..arms.len() {
            let index = (repeat + ordinal) % arms.len();
            timings[index].push(dispatch(&device, &queue, &arms[index]));
        }
    }
    for (i, arm) in arms.iter().enumerate() {
        let observed = readback(&device, &queue, &arm.output);
        assert_eq!(
            observed, expected,
            "STOP_POST_TIMING_PARITY arm={}",
            arm.name
        );
        println!("GPU_INFERENCE_OBSERVATION arm={},median_host_submit_wait_ns={},resident_weights_bytes={},complete=true,device_class={}",
            arm.name, median(&mut timings[i]), [bp.len(), sym3.len(), int8.len()][i] * 4,
            if physical { "physical" } else { "software" });
    }
    println!("PASS_GPU_INFERENCE_STANDALONE parity=exact timing=host_submission_plus_fence_no_energy_claim=true");
}
