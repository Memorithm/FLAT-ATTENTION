//! Caller-owned WGPU host path for native GQA/MQA forward execution.
//!
//! The caller owns data buffers, command encoders, queue submission,
//! synchronization and readback. Q uses query-head cardinality while K/V retain
//! native KV-head cardinality, including MQA.
//!
//! The portable grouped Q4 shader remains the baseline for every shape. Two
//! independent opt-in candidates exist for D64/D128: the already-qualified M6
//! vec4 shader for MHA, and a native grouped vec4 shader for GQA/MQA that keeps
//! physical K/V head cardinality unchanged. A third Phase O candidate reuses
//! each staged K/V tile across two query heads in the same native GQA group.
//! None of these candidates changes the default route.

use core::fmt;

use crate::{
    FlatAttentionConfig, FlatAttentionError, GroupedAttentionShape, FLAT_FWD_GROUPED_WGSL,
    WGSL_MAX_HEAD_DIM, WGSL_QUERY_ROWS,
};

const FLAT_FWD_VEC4_WGSL: &str = include_str!("../shaders/flat_fwd_vec4.wgsl");
const FLAT_FWD_GROUPED_VEC4_WGSL: &str = include_str!("../shaders/flat_fwd_grouped_vec4.wgsl");
const FLAT_FWD_GROUPED_KV_REUSE_WGSL: &str =
    include_str!("../shaders/flat_fwd_grouped_kv_reuse.wgsl");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedForwardLayout {
    pub q_elements: usize,
    pub kv_elements: usize,
    pub lse_elements: usize,
    pub output_elements: usize,
    pub q_bytes: u64,
    pub kv_bytes: u64,
    pub output_bytes: u64,
}

impl GroupedForwardLayout {
    pub const fn output_offset(self) -> usize {
        0
    }

    pub const fn lse_offset(self) -> usize {
        self.q_elements
    }
}

pub struct GroupedForwardPass<'a> {
    pub q: &'a wgpu::Buffer,
    pub k: &'a wgpu::Buffer,
    pub v: &'a wgpu::Buffer,
    /// Combined `[O | LSE]` storage buffer.
    pub output: &'a wgpu::Buffer,
    pub shape: GroupedAttentionShape,
    pub config: FlatAttentionConfig,
}

/// Concrete kernel selected for one prepared grouped-forward request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupedForwardKernelVariant {
    /// Portable native MHA/GQA/MQA grouped Q4 kernel.
    Q4PortableGrouped,
    /// Qualified M6 vec4 Q4 kernel reused for MHA with D=64/128.
    Q4Vec4Mha,
    /// Native grouped vec4 Q4 candidate for GQA/MQA with D=64/128.
    Q4Vec4Grouped,
    /// Native grouped vec4 candidate that shares each K/V tile across two Q heads.
    Q4Vec4GroupedKvReuse,
}

/// Reusable grouped-forward bindings for a fixed resident Q/K/V/output contract.
///
/// Preparing once moves uniform-buffer and bind-group creation out of repeated
/// dispatches. This is useful for benchmark and inference loops whose resident
/// buffers, shape, and configuration remain unchanged across submissions.
pub struct PreparedGroupedForward {
    layout: GroupedForwardLayout,
    bind_group: wgpu::BindGroup,
    query_workgroups: u32,
    dispatch_workgroups_y: u32,
    kernel_variant: GroupedForwardKernelVariant,
    // Kept explicitly so the uniform backing the bind group remains owned by the
    // prepared dispatch for as long as the bindings may be reused.
    _params_buffer: wgpu::Buffer,
}

impl fmt::Debug for PreparedGroupedForward {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedGroupedForward")
            .field("layout", &self.layout)
            .field("query_workgroups", &self.query_workgroups)
            .field("dispatch_workgroups_y", &self.dispatch_workgroups_y)
            .field("kernel_variant", &self.kernel_variant)
            .finish_non_exhaustive()
    }
}

impl PreparedGroupedForward {
    pub const fn layout(&self) -> GroupedForwardLayout {
        self.layout
    }

    pub const fn kernel_variant(&self) -> GroupedForwardKernelVariant {
        self.kernel_variant
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GroupedForwardError {
    Core(FlatAttentionError),
    UnsupportedHeadDim {
        actual: usize,
        maximum: usize,
    },
    IndexSpaceExceeded {
        elements: usize,
    },
    DispatchLimit {
        axis: &'static str,
        actual: usize,
        maximum: u32,
    },
    BufferTooSmall {
        tensor: &'static str,
        actual_bytes: u64,
        required_bytes: u64,
    },
    PipelineValidation(String),
}

impl fmt::Display for GroupedForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::UnsupportedHeadDim { actual, maximum } => write!(
                f,
                "grouped forward head_dim {actual} exceeds portable WGSL maximum {maximum}"
            ),
            Self::IndexSpaceExceeded { elements } => write!(
                f,
                "grouped forward exceeds WGPU u32 index space at {elements} elements"
            ),
            Self::DispatchLimit {
                axis,
                actual,
                maximum,
            } => write!(
                f,
                "grouped forward WGPU dispatch axis {axis} requires {actual} workgroups, device maximum is {maximum}"
            ),
            Self::BufferTooSmall {
                tensor,
                actual_bytes,
                required_bytes,
            } => write!(
                f,
                "buffer {tensor} contains {actual_bytes} bytes, requires at least {required_bytes}"
            ),
            Self::PipelineValidation(error) => {
                write!(f, "grouped forward pipeline validation failed: {error}")
            }
        }
    }
}

impl std::error::Error for GroupedForwardError {}

impl From<FlatAttentionError> for GroupedForwardError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

pub struct WgpuGroupedForwardPipeline {
    portable_pipeline: wgpu::ComputePipeline,
    mha_vec4_pipeline: Option<wgpu::ComputePipeline>,
    grouped_vec4_pipeline: Option<wgpu::ComputePipeline>,
    grouped_kv_reuse_pipeline: Option<wgpu::ComputePipeline>,
}

impl fmt::Debug for WgpuGroupedForwardPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuGroupedForwardPipeline")
            .field("mha_vec4_pipeline", &self.mha_vec4_pipeline.is_some())
            .field(
                "grouped_vec4_pipeline",
                &self.grouped_vec4_pipeline.is_some(),
            )
            .field(
                "grouped_kv_reuse_pipeline",
                &self.grouped_kv_reuse_pipeline.is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl WgpuGroupedForwardPipeline {
    /// Build the compatibility-preserving portable grouped pipeline.
    pub fn new(device: &wgpu::Device) -> Result<Self, GroupedForwardError> {
        Self::with_vectorization_modes(device, false, false, false)
    }

    /// Preserve the M44 opt-in contract: enable the M6 vec4 shader for MHA only.
    pub fn with_vectorization(
        device: &wgpu::Device,
        enabled: bool,
    ) -> Result<Self, GroupedForwardError> {
        Self::with_vectorization_modes(device, enabled, false, false)
    }

    /// Enable the native GQA/MQA vec4 candidate without changing the MHA route.
    ///
    /// This is intentionally independent from [`Self::with_vectorization`] so
    /// existing M44 callers keep exactly the same selection semantics.
    pub fn with_grouped_vectorization(
        device: &wgpu::Device,
        enabled: bool,
    ) -> Result<Self, GroupedForwardError> {
        Self::with_vectorization_modes(device, false, enabled, false)
    }

    /// Enable the Phase O GQA/MQA K/V-reuse candidate without changing any
    /// existing portable, MHA-vec4, or grouped-vec4 selection contract.
    pub fn with_grouped_kv_reuse(
        device: &wgpu::Device,
        enabled: bool,
    ) -> Result<Self, GroupedForwardError> {
        Self::with_vectorization_modes(device, false, false, enabled)
    }

    fn with_vectorization_modes(
        device: &wgpu::Device,
        mha_enabled: bool,
        grouped_enabled: bool,
        grouped_kv_reuse_enabled: bool,
    ) -> Result<Self, GroupedForwardError> {
        let portable_pipeline = create_pipeline(
            device,
            FLAT_FWD_GROUPED_WGSL,
            "flat-m24-grouped-forward-portable",
        )?;
        let mha_vec4_pipeline = if mha_enabled {
            Some(create_pipeline(
                device,
                FLAT_FWD_VEC4_WGSL,
                "flat-m44-grouped-forward-vec4-mha",
            )?)
        } else {
            None
        };
        let grouped_vec4_pipeline = if grouped_enabled {
            Some(create_pipeline(
                device,
                FLAT_FWD_GROUPED_VEC4_WGSL,
                "flat-m45-grouped-forward-vec4-native",
            )?)
        } else {
            None
        };
        let grouped_kv_reuse_pipeline = if grouped_kv_reuse_enabled {
            Some(create_pipeline(
                device,
                FLAT_FWD_GROUPED_KV_REUSE_WGSL,
                "flat-phase-o-grouped-kv-reuse-native",
            )?)
        } else {
            None
        };
        Ok(Self {
            portable_pipeline,
            mha_vec4_pipeline,
            grouped_vec4_pipeline,
            grouped_kv_reuse_pipeline,
        })
    }

    pub fn vectorization_enabled(&self) -> bool {
        self.mha_vec4_pipeline.is_some()
    }

    pub fn grouped_vectorization_enabled(&self) -> bool {
        self.grouped_vec4_pipeline.is_some()
    }

    pub fn grouped_kv_reuse_enabled(&self) -> bool {
        self.grouped_kv_reuse_pipeline.is_some()
    }

    pub fn kernel_variant_for_shape(
        &self,
        shape: GroupedAttentionShape,
    ) -> GroupedForwardKernelVariant {
        if self.grouped_kv_reuse_pipeline.is_some()
            && shape.q_heads != shape.kv_heads
            && shape.q_heads / shape.kv_heads >= 2
            && matches!(shape.head_dim, 64 | 128)
        {
            GroupedForwardKernelVariant::Q4Vec4GroupedKvReuse
        } else if self.grouped_vec4_pipeline.is_some()
            && shape.q_heads != shape.kv_heads
            && matches!(shape.head_dim, 64 | 128)
        {
            GroupedForwardKernelVariant::Q4Vec4Grouped
        } else if self.mha_vec4_pipeline.is_some()
            && shape.q_heads == shape.kv_heads
            && matches!(shape.head_dim, 64 | 128)
        {
            GroupedForwardKernelVariant::Q4Vec4Mha
        } else {
            GroupedForwardKernelVariant::Q4PortableGrouped
        }
    }

    pub fn layout(
        shape: GroupedAttentionShape,
    ) -> Result<GroupedForwardLayout, GroupedForwardError> {
        shape.validate()?;
        if shape.head_dim > WGSL_MAX_HEAD_DIM {
            return Err(GroupedForwardError::UnsupportedHeadDim {
                actual: shape.head_dim,
                maximum: WGSL_MAX_HEAD_DIM,
            });
        }
        let q_elements = shape.q_tensor_len()?;
        let kv_elements = shape.kv_tensor_len()?;
        let lse_elements = shape.lse_len()?;
        let output_elements = q_elements
            .checked_add(lse_elements)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        checked_u32(q_elements)?;
        checked_u32(kv_elements)?;
        checked_u32(output_elements)?;
        Ok(GroupedForwardLayout {
            q_elements,
            kv_elements,
            lse_elements,
            output_elements,
            q_bytes: bytes_for_f32(q_elements)?,
            kv_bytes: bytes_for_f32(kv_elements)?,
            output_bytes: bytes_for_f32(output_elements)?,
        })
    }

    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        shape: GroupedAttentionShape,
    ) -> Result<wgpu::Buffer, GroupedForwardError> {
        let layout = Self::layout(shape)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m24-grouped-forward-o-lse"),
            size: layout.output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Build reusable bind state for repeated dispatches over the same resident
    /// buffers, shape, and attention configuration.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        pass: GroupedForwardPass<'_>,
    ) -> Result<PreparedGroupedForward, GroupedForwardError> {
        let layout = Self::layout(pass.shape)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.output, layout.output_bytes)?;

        let query_workgroups = pass.shape.seq_len.div_ceil(WGSL_QUERY_ROWS);
        let q_batch_heads = pass
            .shape
            .batch
            .checked_mul(pass.shape.q_heads)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        if query_workgroups > maximum as usize {
            return Err(GroupedForwardError::DispatchLimit {
                axis: "x/query_tiles",
                actual: query_workgroups,
                maximum,
            });
        }
        let kernel_variant = self.kernel_variant_for_shape(pass.shape);
        let dispatch_workgroups_y = match kernel_variant {
            GroupedForwardKernelVariant::Q4Vec4GroupedKvReuse => {
                let group_size = pass.shape.q_heads / pass.shape.kv_heads;
                pass.shape
                    .batch
                    .checked_mul(pass.shape.kv_heads)
                    .and_then(|value| value.checked_mul(group_size.div_ceil(2)))
                    .ok_or(FlatAttentionError::ShapeOverflow)?
            }
            _ => q_batch_heads,
        };
        if dispatch_workgroups_y > maximum as usize {
            return Err(GroupedForwardError::DispatchLimit {
                axis: "y/grouped_heads",
                actual: dispatch_workgroups_y,
                maximum,
            });
        }

        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;
        let grouped_params = [
            checked_u32(pass.shape.seq_len)?,
            checked_u32(pass.shape.head_dim)?,
            checked_u32(pass.shape.q_heads)?,
            checked_u32(pass.shape.kv_heads)?,
            checked_u32(pass.shape.batch)?,
            u32::from(pass.config.causal),
            scale.to_bits(),
            0,
        ];
        let (selected_pipeline, params) = match kernel_variant {
            GroupedForwardKernelVariant::Q4PortableGrouped => {
                (&self.portable_pipeline, grouped_params)
            }
            GroupedForwardKernelVariant::Q4Vec4Grouped => (
                self.grouped_vec4_pipeline
                    .as_ref()
                    .expect("grouped vec4 variant is selected only when the pipeline exists"),
                grouped_params,
            ),
            GroupedForwardKernelVariant::Q4Vec4GroupedKvReuse => (
                self.grouped_kv_reuse_pipeline
                    .as_ref()
                    .expect("K/V-reuse variant is selected only when the pipeline exists"),
                grouped_params,
            ),
            GroupedForwardKernelVariant::Q4Vec4Mha => (
                self.mha_vec4_pipeline
                    .as_ref()
                    .expect("MHA vec4 variant is selected only when the pipeline exists"),
                [
                    checked_u32(pass.shape.seq_len)?,
                    checked_u32(pass.shape.head_dim)?,
                    checked_u32(q_batch_heads)?,
                    u32::from(pass.config.causal),
                    scale.to_bits(),
                    0,
                    0,
                    0,
                ],
            ),
        };
        let params_bytes = encode_u32(&params);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m24-grouped-forward-params"),
            size: params_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        {
            let mut mapped = params_buffer.slice(..).get_mapped_range_mut();
            mapped.copy_from_slice(&params_bytes);
        }
        params_buffer.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-m24-grouped-forward-bind-group"),
            layout: &selected_pipeline.get_bind_group_layout(0),
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
                    resource: pass.output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(PreparedGroupedForward {
            layout,
            bind_group,
            query_workgroups: checked_u32(query_workgroups)?,
            dispatch_workgroups_y: checked_u32(dispatch_workgroups_y)?,
            kernel_variant,
            _params_buffer: params_buffer,
        })
    }

    /// Encode a dispatch using previously prepared resident bindings.
    pub fn encode_prepared(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &PreparedGroupedForward,
    ) -> GroupedForwardLayout {
        let pipeline = match prepared.kernel_variant {
            GroupedForwardKernelVariant::Q4PortableGrouped => &self.portable_pipeline,
            GroupedForwardKernelVariant::Q4Vec4Mha => self
                .mha_vec4_pipeline
                .as_ref()
                .expect("prepared MHA vec4 pass retains a vectorized pipeline"),
            GroupedForwardKernelVariant::Q4Vec4Grouped => self
                .grouped_vec4_pipeline
                .as_ref()
                .expect("prepared grouped vec4 pass retains a vectorized pipeline"),
            GroupedForwardKernelVariant::Q4Vec4GroupedKvReuse => self
                .grouped_kv_reuse_pipeline
                .as_ref()
                .expect("prepared grouped K/V-reuse pass retains its pipeline"),
        };
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-m24-grouped-forward"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &prepared.bind_group, &[]);
        compute_pass.dispatch_workgroups(
            prepared.query_workgroups,
            prepared.dispatch_workgroups_y,
            1,
        );
        drop(compute_pass);
        prepared.layout
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: GroupedForwardPass<'_>,
    ) -> Result<GroupedForwardLayout, GroupedForwardError> {
        let prepared = self.prepare(device, pass)?;
        Ok(self.encode_prepared(encoder, &prepared))
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    source: &'static str,
    label: &'static str,
) -> Result<wgpu::ComputePipeline, GroupedForwardError> {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: "flat_attention_forward",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    match pollster::block_on(device.pop_error_scope()) {
        Some(error) => Err(GroupedForwardError::PipelineValidation(error.to_string())),
        None => Ok(pipeline),
    }
}

fn validate_buffer(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), GroupedForwardError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(GroupedForwardError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, GroupedForwardError> {
    u32::try_from(value).map_err(|_| GroupedForwardError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32(elements: usize) -> Result<u64, GroupedForwardError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| GroupedForwardError::IndexSpaceExceeded { elements })
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}
