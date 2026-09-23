//! MAA-14d Gate-A2 correctness-first WGPU structural sparse carrier.
//!
//! The shader consumes the exact u32 CSR layout produced by Gate A1. It is
//! deliberately simple: one invocation computes one output coordinate and
//! traverses the canonical candidate list in ascending key order. This is a
//! correctness carrier, not an optimized kernel or performance result.

use core::fmt;

use crate::{
    api::research_structural_device::{StructuralDevicePlan, StructuralDevicePlanError},
    wgpu_internal, AttentionShape, FlatAttentionConfig, FlatAttentionError,
};

const WORKGROUP_SIZE: usize = 64;

const STRUCTURAL_SPARSE_WGSL: &str = r#"
struct Params {
    seq_len: u32,
    head_dim: u32,
    query_rows: u32,
    causal: u32,
    scale_bits: u32,
    key_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> key_positions: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<storage, read_write> lse: array<f32>;
@group(0) @binding(7) var<storage, read_write> row_status: array<u32>;
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(64)
fn structural_sparse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let element = gid.x;
    let total = params.query_rows * params.head_dim;
    if (element >= total) {
        return;
    }

    let row = element / params.head_dim;
    let dim = element % params.head_dim;
    let query_pos = row % params.seq_len;
    let bh = row / params.seq_len;
    let head_base = bh * params.seq_len * params.head_dim;
    let q_base = head_base + query_pos * params.head_dim;

    let start = offsets[row];
    let end = offsets[row + 1u];
    let scale = bitcast<f32>(params.scale_bits);

    var has_value = false;
    var running_max = 0.0;
    var running_sum = 0.0;
    var numerator = 0.0;

    var cursor = start;
    loop {
        if (cursor >= end) {
            break;
        }
        let key_pos = key_positions[cursor];
        cursor = cursor + 1u;

        if (params.causal != 0u && key_pos > query_pos) {
            continue;
        }

        let kv_base = head_base + key_pos * params.head_dim;
        var dot = 0.0;
        var d = 0u;
        loop {
            if (d >= params.head_dim) {
                break;
            }
            dot = dot + q[q_base + d] * k[kv_base + d];
            d = d + 1u;
        }
        let score = dot * scale;
        let value = v[kv_base + dim];

        if (!has_value) {
            has_value = true;
            running_max = score;
            running_sum = 1.0;
            numerator = value;
        } else {
            let new_max = max(running_max, score);
            let alpha = exp(running_max - new_max);
            let beta = exp(score - new_max);
            numerator = numerator * alpha + value * beta;
            running_sum = running_sum * alpha + beta;
            running_max = new_max;
        }
    }

    if (has_value) {
        output[element] = numerator / running_sum;
        if (dim == 0u) {
            lse[row] = running_max + log(running_sum);
            row_status[row] = 1u;
        }
    } else {
        output[element] = 0.0;
        if (dim == 0u) {
            lse[row] = 0.0;
            row_status[row] = 0u;
        }
    }
}
"#;

/// Exact buffer geometry for one structural sparse WGPU dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuralSparseWgpuLayout {
    pub tensor_elements: usize,
    pub query_rows: usize,
    pub tensor_bytes: u64,
    pub lse_bytes: u64,
    pub status_bytes: u64,
    pub offsets_bytes: u64,
    pub key_positions_bytes: u64,
}

/// Uploaded immutable candidate buffers derived directly from Gate A1.
#[derive(Debug)]
pub struct StructuralSparseWgpuCandidates {
    offsets: wgpu::Buffer,
    key_positions: wgpu::Buffer,
    seq_len: u32,
    query_rows: u32,
    key_count: u32,
    fingerprint_fnv1a64: u64,
}

impl StructuralSparseWgpuCandidates {
    #[must_use]
    pub const fn fingerprint_fnv1a64(&self) -> u64 {
        self.fingerprint_fnv1a64
    }

    #[must_use]
    pub const fn key_count(&self) -> u32 {
        self.key_count
    }

    #[must_use]
    pub const fn query_rows(&self) -> u32 {
        self.query_rows
    }

    #[must_use]
    pub const fn seq_len(&self) -> u32 {
        self.seq_len
    }

    #[must_use]
    pub const fn offsets_buffer(&self) -> &wgpu::Buffer {
        &self.offsets
    }

    #[must_use]
    pub const fn key_positions_buffer(&self) -> &wgpu::Buffer {
        &self.key_positions
    }
}

/// Caller-owned buffers for one structural sparse pass.
pub struct StructuralSparseWgpuPass<'a> {
    pub q: &'a wgpu::Buffer,
    pub k: &'a wgpu::Buffer,
    pub v: &'a wgpu::Buffer,
    pub candidates: &'a StructuralSparseWgpuCandidates,
    pub output: &'a wgpu::Buffer,
    pub lse: &'a wgpu::Buffer,
    pub row_status: &'a wgpu::Buffer,
    pub shape: AttentionShape,
    pub config: FlatAttentionConfig,
}

/// Correctness-first sparse structural WGPU pipeline.
pub struct StructuralSparseWgpuPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for StructuralSparseWgpuPipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StructuralSparseWgpuPipeline")
            .finish_non_exhaustive()
    }
}

impl StructuralSparseWgpuPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, StructuralSparseWgpuError> {
        let pipeline = wgpu_internal::create_pipeline(
            device,
            STRUCTURAL_SPARSE_WGSL,
            "flat-maa14d-structural-sparse",
            "structural_sparse",
        )
        .map_err(StructuralSparseWgpuError::PipelineValidation)?;
        Ok(Self { pipeline })
    }

    pub fn layout(
        shape: AttentionShape,
        plan: &StructuralDevicePlan,
    ) -> Result<StructuralSparseWgpuLayout, StructuralSparseWgpuError> {
        validate_shape_plan(shape, plan)?;
        let tensor_elements = shape.tensor_len()?;
        let query_rows = shape.lse_len()?;
        let tensor_bytes = bytes_for_f32(tensor_elements)?;
        let lse_bytes = bytes_for_f32(query_rows)?;
        let status_bytes = bytes_for_u32(query_rows)?;
        Ok(StructuralSparseWgpuLayout {
            tensor_elements,
            query_rows,
            tensor_bytes,
            lse_bytes,
            status_bytes,
            offsets_bytes: plan.offsets_bytes(),
            key_positions_bytes: plan.key_positions_bytes(),
        })
    }

    /// Upload candidate metadata exactly from the validated Gate-A1 plan.
    pub fn upload_candidates(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &StructuralDevicePlan,
    ) -> Result<StructuralSparseWgpuCandidates, StructuralSparseWgpuError> {
        if plan.admitted_count() == 0 {
            return Err(StructuralSparseWgpuError::EmptyCandidateSet);
        }
        let offsets_bytes = wgpu_internal::encode_u32(plan.offsets());
        let key_bytes = wgpu_internal::encode_u32(plan.key_positions());
        let offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-maa14d-structural-offsets"),
            size: offsets_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let key_positions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-maa14d-structural-keys"),
            size: key_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&offsets, 0, &offsets_bytes);
        queue.write_buffer(&key_positions, 0, &key_bytes);
        Ok(StructuralSparseWgpuCandidates {
            offsets,
            key_positions,
            seq_len: plan.seq_len(),
            query_rows: plan.query_rows(),
            key_count: checked_u32(plan.admitted_count())?,
            fingerprint_fnv1a64: plan.fingerprint_fnv1a64(),
        })
    }

    pub fn create_output_buffer(
        device: &wgpu::Device,
        shape: AttentionShape,
    ) -> Result<wgpu::Buffer, StructuralSparseWgpuError> {
        let bytes = bytes_for_f32(shape.tensor_len()?)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-maa14d-structural-output"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    pub fn create_lse_buffer(
        device: &wgpu::Device,
        shape: AttentionShape,
    ) -> Result<wgpu::Buffer, StructuralSparseWgpuError> {
        let bytes = bytes_for_f32(shape.lse_len()?)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-maa14d-structural-lse"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    pub fn create_status_buffer(
        device: &wgpu::Device,
        shape: AttentionShape,
    ) -> Result<wgpu::Buffer, StructuralSparseWgpuError> {
        let bytes = bytes_for_u32(shape.lse_len()?)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-maa14d-structural-status"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: StructuralSparseWgpuPass<'_>,
    ) -> Result<StructuralSparseWgpuLayout, StructuralSparseWgpuError> {
        validate_candidate_geometry(pass.shape, pass.candidates)?;
        let tensor_elements = pass.shape.tensor_len()?;
        let query_rows = pass.shape.lse_len()?;
        let layout = StructuralSparseWgpuLayout {
            tensor_elements,
            query_rows,
            tensor_bytes: bytes_for_f32(tensor_elements)?,
            lse_bytes: bytes_for_f32(query_rows)?,
            status_bytes: bytes_for_u32(query_rows)?,
            offsets_bytes: pass.candidates.offsets.size(),
            key_positions_bytes: pass.candidates.key_positions.size(),
        };

        validate_buffer("Q", pass.q, layout.tensor_bytes)?;
        validate_buffer("K", pass.k, layout.tensor_bytes)?;
        validate_buffer("V", pass.v, layout.tensor_bytes)?;
        validate_buffer("O", pass.output, layout.tensor_bytes)?;
        validate_buffer("LSE", pass.lse, layout.lse_bytes)?;
        validate_buffer("row_status", pass.row_status, layout.status_bytes)?;

        let limits = device.limits();
        for (name, bytes) in [
            ("Q", layout.tensor_bytes),
            ("K", layout.tensor_bytes),
            ("V", layout.tensor_bytes),
            ("O", layout.tensor_bytes),
            ("offsets", layout.offsets_bytes),
            ("key_positions", layout.key_positions_bytes),
            ("LSE", layout.lse_bytes),
            ("row_status", layout.status_bytes),
        ] {
            if bytes > u64::from(limits.max_storage_buffer_binding_size) {
                return Err(StructuralSparseWgpuError::StorageBindingTooLarge {
                    tensor: name,
                    required_bytes: bytes,
                    maximum_bytes: u64::from(limits.max_storage_buffer_binding_size),
                });
            }
        }

        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;
        let params = [
            checked_u32(pass.shape.seq_len)?,
            checked_u32(pass.shape.head_dim)?,
            checked_u32(query_rows)?,
            u32::from(pass.config.causal),
            scale.to_bits(),
            pass.candidates.key_count,
            0,
            0,
        ];
        let params_bytes = wgpu_internal::encode_u32(&params);
        if params_bytes.len() as u64 > u64::from(limits.max_uniform_buffer_binding_size) {
            return Err(StructuralSparseWgpuError::UniformBindingTooLarge {
                required_bytes: params_bytes.len() as u64,
                maximum_bytes: u64::from(limits.max_uniform_buffer_binding_size),
            });
        }
        let params_buffer = wgpu_internal::create_uniform_buffer_init(
            device,
            "flat-maa14d-structural-params",
            &params_bytes,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-maa14d-structural-bind-group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pass.q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pass.k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pass.v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pass.candidates.offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pass.candidates.key_positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pass.output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: pass.lse.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: pass.row_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = tensor_elements.div_ceil(WORKGROUP_SIZE);
        if workgroups > limits.max_compute_workgroups_per_dimension as usize {
            return Err(StructuralSparseWgpuError::DispatchLimit {
                required: workgroups,
                maximum: limits.max_compute_workgroups_per_dimension,
            });
        }

        {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-maa14d-structural-sparse"),
                timestamp_writes: None,
            });
            compute.set_pipeline(&self.pipeline);
            compute.set_bind_group(0, &bind_group, &[]);
            compute.dispatch_workgroups(checked_u32(workgroups)?, 1, 1);
        }
        Ok(layout)
    }
}

fn validate_shape_plan(
    shape: AttentionShape,
    plan: &StructuralDevicePlan,
) -> Result<(), StructuralSparseWgpuError> {
    shape.tensor_len()?;
    let rows = shape.lse_len()?;
    if shape.seq_len != plan.seq_len() as usize || rows != plan.query_rows() as usize {
        return Err(StructuralSparseWgpuError::GeometryMismatch {
            shape_seq_len: shape.seq_len,
            plan_seq_len: plan.seq_len(),
            shape_query_rows: rows,
            plan_query_rows: plan.query_rows(),
        });
    }
    if plan.admitted_count() == 0 {
        return Err(StructuralSparseWgpuError::EmptyCandidateSet);
    }
    Ok(())
}

fn validate_candidate_geometry(
    shape: AttentionShape,
    candidates: &StructuralSparseWgpuCandidates,
) -> Result<(), StructuralSparseWgpuError> {
    shape.tensor_len()?;
    let rows = shape.lse_len()?;
    if shape.seq_len != candidates.seq_len as usize || rows != candidates.query_rows as usize {
        return Err(StructuralSparseWgpuError::GeometryMismatch {
            shape_seq_len: shape.seq_len,
            plan_seq_len: candidates.seq_len,
            shape_query_rows: rows,
            plan_query_rows: candidates.query_rows,
        });
    }
    if candidates.key_count == 0 {
        return Err(StructuralSparseWgpuError::EmptyCandidateSet);
    }
    Ok(())
}

fn validate_buffer(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), StructuralSparseWgpuError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(StructuralSparseWgpuError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, StructuralSparseWgpuError> {
    wgpu_internal::checked_u32(value)
        .ok_or(StructuralSparseWgpuError::IndexSpaceExceeded { value })
}

fn bytes_for_f32(elements: usize) -> Result<u64, StructuralSparseWgpuError> {
    wgpu_internal::f32_bytes(elements)
        .ok_or(StructuralSparseWgpuError::IndexSpaceExceeded { value: elements })
}

fn bytes_for_u32(elements: usize) -> Result<u64, StructuralSparseWgpuError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(StructuralSparseWgpuError::IndexSpaceExceeded { value: elements })?;
    u64::try_from(bytes).map_err(|_| StructuralSparseWgpuError::IndexSpaceExceeded { value: elements })
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StructuralSparseWgpuError {
    Core(FlatAttentionError),
    DevicePlan(StructuralDevicePlanError),
    EmptyCandidateSet,
    GeometryMismatch {
        shape_seq_len: usize,
        plan_seq_len: u32,
        shape_query_rows: usize,
        plan_query_rows: u32,
    },
    IndexSpaceExceeded {
        value: usize,
    },
    BufferTooSmall {
        tensor: &'static str,
        actual_bytes: u64,
        required_bytes: u64,
    },
    StorageBindingTooLarge {
        tensor: &'static str,
        required_bytes: u64,
        maximum_bytes: u64,
    },
    UniformBindingTooLarge {
        required_bytes: u64,
        maximum_bytes: u64,
    },
    DispatchLimit {
        required: usize,
        maximum: u32,
    },
    PipelineValidation(String),
}

impl From<FlatAttentionError> for StructuralSparseWgpuError {
    fn from(error: FlatAttentionError) -> Self {
        Self::Core(error)
    }
}

impl From<StructuralDevicePlanError> for StructuralSparseWgpuError {
    fn from(error: StructuralDevicePlanError) -> Self {
        Self::DevicePlan(error)
    }
}

impl fmt::Display for StructuralSparseWgpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StructuralSparseWgpuError {}
