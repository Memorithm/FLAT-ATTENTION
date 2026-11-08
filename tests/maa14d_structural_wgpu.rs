#![cfg(feature = "wgpu")]

use std::sync::mpsc;

use flat_attention::api::research_structural_device::StructuralDevicePlan;
use flat_attention::api::research_structural_routing::{
    forward_reference_structural_sparse, StructuralCandidateSet, StructuralRoutingError,
};
use flat_attention::api::research_structural_wgpu::{
    validate_structural_sparse_row_status, StructuralSparseWgpuError, StructuralSparseWgpuPass,
    StructuralSparseWgpuPipeline,
};
use flat_attention::{forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionOutput};

const O_ATOL: f32 = 2.0e-4;
const O_RTOL: f32 = 2.0e-3;
const LSE_ATOL: f32 = 3.0e-4;
const LSE_RTOL: f32 = 2.0e-3;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn harness() -> Option<Harness> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    })) {
        Ok(adapter) => adapter,
        Err(error) if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() => {
            eprintln!("WGPU adapter unavailable; Gate-A2 device test skipped: {error}");
            return None;
        }
        Err(error) => panic!("required WGPU adapter unavailable: {error}"),
    };
    let info = adapter.get_info();
    eprintln!(
        "MAA-14d Gate-A2 adapter={} backend={:?}",
        info.name, info.backend
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-maa14d-gate-a2-test"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("Gate-A2 request_device");
    Some(Harness { device, queue })
}

fn fixture(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.tensor_len().expect("tensor len");
    let q = (0..len)
        .map(|index| ((index as f32) * 0.071 - 0.3).sin() * 0.8)
        .collect();
    let k = (0..len)
        .map(|index| ((index as f32) * 0.113 + 0.4).cos() * 0.7)
        .collect();
    let v = (0..len)
        .map(|index| ((index as f32) * 0.047 - 0.2).sin() * 1.2)
        .collect();
    (q, k, v)
}

fn input_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    label: &'static str,
) -> wgpu::Buffer {
    let bytes = encode_f32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &bytes);
    buffer
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn read_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    bytes: u64,
) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-maa14d-gate-a2-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-maa14d-gate-a2-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver
        .recv()
        .expect("Gate-A2 map callback")
        .expect("Gate-A2 map");
    let mapped = slice.get_mapped_range().expect("Gate-A2 mapped range");
    let result = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    result
}

fn read_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    read_bytes(device, queue, source, (len * 4) as u64)
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Vec<u32> {
    read_bytes(device, queue, source, (len * 4) as u64)
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            actual.is_finite() && expected.is_finite() && error <= tolerance,
            "{label}[{index}] actual={actual} expected={expected} error={error} tolerance={tolerance}"
        );
    }
}

fn run_device(
    harness: &Harness,
    shape: AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    candidates: &StructuralCandidateSet,
    config: FlatAttentionConfig,
) -> (FlatAttentionOutput, Vec<u32>, Vec<u32>, Vec<u32>) {
    let plan = StructuralDevicePlan::from_candidates(candidates).expect("device plan");
    let pipeline = StructuralSparseWgpuPipeline::new(&harness.device).expect("pipeline");
    let gpu_candidates =
        StructuralSparseWgpuPipeline::upload_candidates(&harness.device, &harness.queue, &plan)
            .expect("candidate upload");

    let q_gpu = input_buffer(&harness.device, &harness.queue, q, "Gate-A2 Q");
    let k_gpu = input_buffer(&harness.device, &harness.queue, k, "Gate-A2 K");
    let v_gpu = input_buffer(&harness.device, &harness.queue, v, "Gate-A2 V");
    let output =
        StructuralSparseWgpuPipeline::create_output_buffer(&harness.device, shape).expect("output");
    let lse = StructuralSparseWgpuPipeline::create_lse_buffer(&harness.device, shape).expect("lse");
    let status =
        StructuralSparseWgpuPipeline::create_status_buffer(&harness.device, shape).expect("status");

    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gate-A2 structural sparse dispatch"),
        });
    let layout = pipeline
        .encode(
            &harness.device,
            &mut encoder,
            StructuralSparseWgpuPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                candidates: &gpu_candidates,
                output: &output,
                lse: &lse,
                row_status: &status,
                shape,
                config,
            },
        )
        .expect("encode");
    harness.queue.submit(Some(encoder.finish()));
    let _ = harness.device.poll(wgpu::PollType::wait_indefinitely());

    let actual = FlatAttentionOutput {
        output: read_f32(
            &harness.device,
            &harness.queue,
            &output,
            layout.tensor_elements,
        ),
        lse: read_f32(&harness.device, &harness.queue, &lse, layout.query_rows),
    };
    let statuses = read_u32(&harness.device, &harness.queue, &status, layout.query_rows);
    let offsets = read_u32(
        &harness.device,
        &harness.queue,
        gpu_candidates.offsets_buffer(),
        plan.offsets().len(),
    );
    let keys = read_u32(
        &harness.device,
        &harness.queue,
        gpu_candidates.key_positions_buffer(),
        plan.key_positions().len(),
    );
    (actual, statuses, offsets, keys)
}

#[test]
fn gate_a2_sparse_device_matches_host_oracle_and_candidate_identity() {
    let Some(harness) = harness() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 7,
        head_dim: 8,
    };
    let row_pattern = [
        vec![0],
        vec![0, 1],
        vec![0, 2],
        vec![1, 3],
        vec![0, 2, 4],
        vec![1, 4, 5],
        vec![0, 3, 6],
    ];
    let mut rows = Vec::new();
    for _head in 0..shape.heads {
        rows.extend(row_pattern.iter().cloned());
    }
    let candidates = StructuralCandidateSet::from_rows(shape, rows).unwrap();
    let plan = StructuralDevicePlan::from_candidates(&candidates).unwrap();
    let (q, k, v) = fixture(shape);

    for causal in [false, true] {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: Some(0.25),
        };
        let expected =
            forward_reference_structural_sparse(&q, &k, &v, shape, config, &candidates).unwrap();
        let (actual, status, offsets, keys) =
            run_device(&harness, shape, &q, &k, &v, &candidates, config);

        validate_structural_sparse_row_status(&status, shape.lse_len().unwrap()).unwrap();
        assert_eq!(offsets, plan.offsets());
        assert_eq!(keys, plan.key_positions());
        assert_close(
            "Gate-A2 sparse O",
            &actual.output,
            &expected.attention.output,
            O_ATOL,
            O_RTOL,
        );
        assert_close(
            "Gate-A2 sparse LSE",
            &actual.lse,
            &expected.attention.lse,
            LSE_ATOL,
            LSE_RTOL,
        );
    }
}

#[test]
fn gate_a2_all_accept_reproduces_dense_semantics() {
    let Some(harness) = harness() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 5,
        head_dim: 8,
    };
    let candidates = StructuralCandidateSet::all(shape).unwrap();
    let (q, k, v) = fixture(shape);
    let config = FlatAttentionConfig {
        causal: false,
        softmax_scale: None,
    };
    let dense = forward_reference(&q, &k, &v, shape, config).unwrap();
    let (actual, status, _, _) = run_device(&harness, shape, &q, &k, &v, &candidates, config);

    validate_structural_sparse_row_status(&status, shape.lse_len().unwrap()).unwrap();
    assert_close(
        "Gate-A2 all O",
        &actual.output,
        &dense.output,
        O_ATOL,
        O_RTOL,
    );
    assert_close(
        "Gate-A2 all LSE",
        &actual.lse,
        &dense.lse,
        LSE_ATOL,
        LSE_RTOL,
    );
}

#[test]
fn gate_a2_empty_effective_row_is_visible_and_rejected() {
    let Some(harness) = harness() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 4,
        head_dim: 4,
    };
    let candidates =
        StructuralCandidateSet::from_rows(shape, vec![vec![1], vec![0, 1], vec![0, 2], vec![0, 3]])
            .unwrap();
    let (q, k, v) = fixture(shape);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };

    assert!(matches!(
        forward_reference_structural_sparse(&q, &k, &v, shape, config, &candidates),
        Err(StructuralRoutingError::EmptyEffectiveCandidates { row: 0 })
    ));

    let (_, status, _, _) = run_device(&harness, shape, &q, &k, &v, &candidates, config);
    assert_eq!(status[0], 0);
    assert!(matches!(
        validate_structural_sparse_row_status(&status, shape.lse_len().unwrap()),
        Err(StructuralSparseWgpuError::EmptyEffectiveRow { row: 0 })
    ));
}

#[test]
fn row_status_validation_rejects_malformed_readback() {
    assert!(matches!(
        validate_structural_sparse_row_status(&[1, 1], 3),
        Err(StructuralSparseWgpuError::RowStatusCountMismatch {
            actual: 2,
            expected: 3
        })
    ));
    assert!(matches!(
        validate_structural_sparse_row_status(&[1, 2], 2),
        Err(StructuralSparseWgpuError::NonBinaryRowStatus { row: 1, value: 2 })
    ));
}

/// Research-only F2/Hamming GPU admission -> exact structural sparse GPU attention.
///
/// This is a small deterministic smoke, **not** the frozen MAA-13d projector,
/// not an end-to-end performance measurement and not a production route.
#[test]
fn maa_f2_gpu_admission_feeds_sparse_attention_with_exact_candidate_identity() {
    use flat_attention::api::boolean_attention_signature::{
        BooleanAttentionSignature, HammingAdmissionRule,
    };
    use flat_attention::api::wgpu_boolean_router::{
        BooleanWgpuRouterPlan, WgpuBooleanRouterPipeline,
    };

    fn signed_bits(row: &[f32]) -> BooleanAttentionSignature {
        assert_eq!(row.len(), 8);
        let mut bits = 0u64;
        for (index, value) in row.iter().enumerate() {
            if *value >= 0.0 {
                bits |= 1u64 << index;
            }
        }
        BooleanAttentionSignature::new(8, vec![bits]).expect("valid 8-bit F2 signature")
    }

    fn router_buffer(
        harness: &Harness,
        values: &[u32],
        label: &'static str,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        let buffer = harness.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        harness.queue.write_buffer(&buffer, 0, &bytes);
        buffer
    }

    let Some(harness) = harness() else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 9,
        head_dim: 8,
    };
    let (q, _, v) = fixture(shape);
    // The self-key guarantees one admissible candidate per query without
    // introducing a hidden rescue/fallback outside the F2 predicate.
    let k = q.clone();
    let config = FlatAttentionConfig {
        causal: false,
        softmax_scale: None,
    };
    let signatures: Vec<_> = q.chunks_exact(shape.head_dim).map(signed_bits).collect();
    let rule = HammingAdmissionRule::new(2, 8).expect("frozen local smoke threshold");
    let router = WgpuBooleanRouterPipeline::new(&harness.device).expect("GPU F2 router");
    let mut rows = Vec::with_capacity(shape.seq_len);

    for query_index in 0..shape.seq_len {
        let plan = BooleanWgpuRouterPlan::new(&signatures[query_index], &signatures, rule)
            .expect("F2/Hamming host plan");
        let query_buf = router_buffer(
            &harness,
            plan.query_words(),
            "F2 query signature",
            wgpu::BufferUsages::STORAGE,
        );
        let keys_buf = router_buffer(
            &harness,
            plan.key_words(),
            "F2 key signatures",
            wgpu::BufferUsages::STORAGE,
        );
        let admission_buf = router_buffer(
            &harness,
            &vec![0; shape.seq_len],
            "F2 GPU admission readback",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MAA F2 GPU signature filtering"),
            });
        router
            .encode(
                &harness.device,
                &mut encoder,
                &query_buf,
                &keys_buf,
                &admission_buf,
                &plan,
            )
            .expect("F2 GPU encode");
        harness.queue.submit(Some(encoder.finish()));

        let flags = read_u32(
            &harness.device,
            &harness.queue,
            &admission_buf,
            shape.seq_len,
        );
        assert!(
            flags.iter().all(|flag| *flag <= 1),
            "F2 router must emit only binary flags"
        );
        let selected: Vec<usize> = flags
            .iter()
            .enumerate()
            .filter_map(|(key_index, flag)| (*flag == 1).then_some(key_index))
            .collect();
        assert_eq!(
            selected,
            plan.oracle_mask().admitted_blocks(),
            "F2 WGPU route differs from bitwise Rust oracle"
        );
        assert!(selected.contains(&query_index), "self-key must survive");
        rows.push(selected);
    }

    let candidates = StructuralCandidateSet::from_rows(shape, rows).expect("canonical CSR");
    let dense = forward_reference(&q, &k, &v, shape, config).expect("dense reference");
    let host_sparse = forward_reference_structural_sparse(&q, &k, &v, shape, config, &candidates)
        .expect("masked sparse host oracle");
    let (gpu_sparse, statuses, offsets, keys) =
        run_device(&harness, shape, &q, &k, &v, &candidates, config);
    let plan = StructuralDevicePlan::from_candidates(&candidates).expect("CSR plan");
    assert_eq!(offsets, plan.offsets());
    assert_eq!(keys, plan.key_positions());
    validate_structural_sparse_row_status(&statuses, shape.seq_len).expect("no empty rows");

    assert_close(
        "F2 GPU->sparse GPU O",
        &gpu_sparse.output,
        &host_sparse.attention.output,
        O_ATOL,
        O_RTOL,
    );
    assert_close(
        "F2 GPU->sparse GPU LSE",
        &gpu_sparse.lse,
        &host_sparse.attention.lse,
        LSE_ATOL,
        LSE_RTOL,
    );
    let max_output_error_vs_dense = gpu_sparse
        .output
        .iter()
        .zip(dense.output.iter())
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0f32, f32::max);
    let max_lse_error_vs_dense = gpu_sparse
        .lse
        .iter()
        .zip(dense.lse.iter())
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0f32, f32::max);
    println!(
        "F2_GPU_SMOKE_VALIDATED selected={} total={} output_max_abs_error_vs_dense={} lse_max_abs_error_vs_dense={} physical_performance_claim=none",
        candidates.admitted_count(),
        shape.seq_len * shape.seq_len,
        max_output_error_vs_dense,
        max_lse_error_vs_dense
    );
    assert!(
        candidates.admitted_count() < shape.seq_len * shape.seq_len,
        "F2 smoke must actually filter at least one logical pair"
    );
    assert!(max_output_error_vs_dense.is_finite());
    assert!(max_lse_error_vs_dense.is_finite());
}
