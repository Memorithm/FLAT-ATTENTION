//! Correctness-first WGPU qualification path for Elastic Positional Geometry.
//!
//! This crate is intentionally isolated from FLAT's production routing. The
//! kernel is explicit opt-in and exists to establish CPU/GPU parity for the
//! qualified SO(2), biplanar SO(4), and isoclinic SO(4) controls.

#![forbid(unsafe_code)]

use core::fmt;
use std::borrow::Cow;

use epg_core::{EpgContractError, EpgGeometryDescriptor, EpgGeometryKind, EpgPositionDomain, So4Geometry};
use flat_attention::{FlatAttentionConfig, FlatAttentionError, GroupedAttentionShape};

/// Qualification shader source.
pub const EPG_GROUPED_VEC4_QUALIFY_WGSL: &str =
    include_str!("../shaders/epg_grouped_vec4_qualify.wgsl");

/// Maximum head dimension of the qualification shader.
pub const EPG_WGSL_MAX_HEAD_DIM: usize = 128;

/// Combined output layout `[O | LSE]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpgQualificationLayout {
    /// Number of Q/output scalar elements.
    pub q_elements: usize,
    /// Number of physical K or V scalar elements.
    pub kv_elements: usize,
    /// Number of LSE scalars.
    pub lse_elements: usize,
    /// Combined output scalar count.
    pub combined_elements: usize,
    /// Required Q bytes.
    pub q_bytes: u64,
    /// Required K/V bytes per buffer.
    pub kv_bytes: u64,
    /// Required combined output bytes.
    pub combined_bytes: u64,
}

impl EpgQualificationLayout {
    /// Scalar offset at which LSE begins in the combined output buffer.
    pub const fn lse_offset(self) -> usize {
        self.q_elements
    }
}

/// One explicit EPG qualification dispatch over caller-owned buffers.
pub struct EpgQualificationPass<'a> {
    /// Query storage in canonical grouped-head layout.
    pub q: &'a wgpu::Buffer,
    /// Physical key storage; GQA/MQA is not expanded.
    pub k: &'a wgpu::Buffer,
    /// Physical value storage; never geometrically transformed.
    pub v: &'a wgpu::Buffer,
    /// Combined `[O | LSE]` storage.
    pub output: &'a wgpu::Buffer,
    /// Equal-length grouped attention shape.
    pub shape: GroupedAttentionShape,
    /// Causal/scale configuration.
    pub config: FlatAttentionConfig,
    /// Runtime-neutral positional geometry.
    pub geometry: EpgGeometryDescriptor,
    /// Execution-local absolute position origin.
    pub position: EpgPositionDomain,
}

/// Validated reusable bindings for one fixed qualification dispatch.
pub struct PreparedEpgQualification {
    layout: EpgQualificationLayout,
    bind_group: wgpu::BindGroup,
    dispatch_x: u32,
    dispatch_y: u32,
    _params_buffer: wgpu::Buffer,
}

impl fmt::Debug for PreparedEpgQualification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedEpgQualification")
            .field("layout", &self.layout)
            .field("dispatch_x", &self.dispatch_x)
            .field("dispatch_y", &self.dispatch_y)
            .finish_non_exhaustive()
    }
}

impl PreparedEpgQualification {
    /// Validated output layout.
    pub const fn layout(&self) -> EpgQualificationLayout {
        self.layout
    }
}

/// Qualification host-path failure.
#[derive(Debug)]
pub enum EpgQualificationError {
    /// FLAT shape/configuration contract failed.
    Flat(FlatAttentionError),
    /// EPG representation contract failed.
    Epg(EpgContractError),
    /// Vec4 qualification requires a head dimension divisible by four and <=128.
    UnsupportedHeadDim(usize),
    /// Absolute position cannot be represented by the current WGSL u32 contract.
    PositionSpaceExceeded,
    /// A scalar element count cannot be represented safely by the host/GPU contract.
    IndexSpaceExceeded(usize),
    /// Device dispatch limit would be exceeded.
    DispatchLimit {
        /// Axis name.
        axis: &'static str,
        /// Required workgroups.
        actual: usize,
        /// Device limit.
        maximum: u32,
    },
    /// Caller-owned buffer is undersized.
    BufferTooSmall {
        /// Tensor/buffer label.
        tensor: &'static str,
        /// Actual bytes.
        actual_bytes: u64,
        /// Required bytes.
        required_bytes: u64,
    },
    /// WGPU rejected shader or pipeline creation.
    PipelineValidation(String),
}

impl fmt::Display for EpgQualificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Flat(error) => write!(f, "{error}"),
            Self::Epg(error) => write!(f, "EPG contract error: {error}"),
            Self::UnsupportedHeadDim(value) => write!(
                f,
                "EPG vec4 qualification requires head_dim divisible by four and <= {EPG_WGSL_MAX_HEAD_DIM}, got {value}"
            ),
            Self::PositionSpaceExceeded => {
                write!(f, "EPG qualification position exceeds WGSL u32 index space")
            }
            Self::IndexSpaceExceeded(elements) => {
                write!(f, "EPG qualification index space exceeded at {elements} elements")
            }
            Self::DispatchLimit {
                axis,
                actual,
                maximum,
            } => write!(
                f,
                "EPG qualification dispatch axis {axis} requires {actual} workgroups; device maximum is {maximum}"
            ),
            Self::BufferTooSmall {
                tensor,
                actual_bytes,
                required_bytes,
            } => write!(
                f,
                "buffer {tensor} contains {actual_bytes} bytes; requires at least {required_bytes}"
            ),
            Self::PipelineValidation(error) => {
                write!(f, "EPG qualification pipeline validation failed: {error}")
            }
        }
    }
}

impl std::error::Error for EpgQualificationError {}

impl From<FlatAttentionError> for EpgQualificationError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Flat(value)
    }
}

impl From<EpgContractError> for EpgQualificationError {
    fn from(value: EpgContractError) -> Self {
        Self::Epg(value)
    }
}

/// Explicit, correctness-first EPG vec4 qualification pipeline.
pub struct EpgVec4QualificationPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for EpgVec4QualificationPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EpgVec4QualificationPipeline")
            .finish_non_exhaustive()
    }
}

impl EpgVec4QualificationPipeline {
    /// Compile the isolated qualification shader.
    pub fn new(device: &wgpu::Device) -> Result<Self, EpgQualificationError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-epg-vec4-qualify"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EPG_GROUPED_VEC4_QUALIFY_WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-epg-vec4-qualify"),
            layout: None,
            module: &shader,
            entry_point: "epg_grouped_vec4_qualify",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        match pollster::block_on(device.pop_error_scope()) {
            Some(error) => Err(EpgQualificationError::PipelineValidation(error.to_string())),
            None => Ok(Self { pipeline }),
        }
    }

    /// Validate and compute the physical buffer layout.
    pub fn layout(
        shape: GroupedAttentionShape,
    ) -> Result<EpgQualificationLayout, EpgQualificationError> {
        shape.group_size()?;
        validate_head_dim(shape.head_dim)?;
        let q_elements = shape.q_tensor_len()?;
        let kv_elements = shape.kv_tensor_len()?;
        let lse_elements = shape.lse_len()?;
        let combined_elements = q_elements
            .checked_add(lse_elements)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        checked_u32(q_elements)?;
        checked_u32(kv_elements)?;
        checked_u32(combined_elements)?;
        Ok(EpgQualificationLayout {
            q_elements,
            kv_elements,
            lse_elements,
            combined_elements,
            q_bytes: bytes_for_f32(q_elements)?,
            kv_bytes: bytes_for_f32(kv_elements)?,
            combined_bytes: bytes_for_f32(combined_elements)?,
        })
    }

    /// Allocate the combined `[O | LSE]` buffer.
    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        shape: GroupedAttentionShape,
    ) -> Result<wgpu::Buffer, EpgQualificationError> {
        let layout = Self::layout(shape)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-epg-vec4-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Validate caller buffers and build reusable bind state.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        pass: EpgQualificationPass<'_>,
    ) -> Result<PreparedEpgQualification, EpgQualificationError> {
        let layout = Self::layout(pass.shape)?;
        pass.geometry
            .validate_head_dim(checked_u32(pass.shape.head_dim)?)?;
        validate_geometry(pass.geometry)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.output, layout.combined_bytes)?;

        let final_local = pass.shape.seq_len.saturating_sub(1);
        let final_position = pass
            .position
            .resolve(u64::try_from(final_local).map_err(|_| EpgQualificationError::PositionSpaceExceeded)?)?;
        if final_position > u32::MAX as u64 {
            return Err(EpgQualificationError::PositionSpaceExceeded);
        }

        let dispatch_x = pass.shape.seq_len;
        let dispatch_y = pass
            .shape
            .batch
            .checked_mul(pass.shape.q_heads)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        validate_dispatch("x/query_rows", dispatch_x, maximum)?;
        validate_dispatch("y/batch_q_heads", dispatch_y, maximum)?;

        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;
        let params = [
            checked_u32(pass.shape.seq_len)?,
            checked_u32(pass.shape.head_dim)?,
            checked_u32(pass.shape.q_heads)?,
            checked_u32(pass.shape.kv_heads)?,
            checked_u32(pass.shape.batch)?,
            u32::from(pass.config.causal),
            scale.to_bits(),
            pass.geometry.theta_bits(),
            u32::try_from(pass.position.offset())
                .map_err(|_| EpgQualificationError::PositionSpaceExceeded)?,
            pass.geometry.so4_dims(),
            geometry_mode(pass.geometry),
            0,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-epg-vec4-params"),
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
            label: Some("flat-epg-vec4-bind-group"),
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
                    resource: pass.output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(PreparedEpgQualification {
            layout,
            bind_group,
            dispatch_x: checked_u32(dispatch_x)?,
            dispatch_y: checked_u32(dispatch_y)?,
            _params_buffer: params_buffer,
        })
    }

    /// Encode a previously prepared qualification dispatch.
    pub fn encode_prepared(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &PreparedEpgQualification,
    ) -> EpgQualificationLayout {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-epg-vec4-qualify"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &prepared.bind_group, &[]);
        compute_pass.dispatch_workgroups(prepared.dispatch_x, prepared.dispatch_y, 1);
        drop(compute_pass);
        prepared.layout
    }

    /// Validate, bind, and encode one qualification dispatch.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: EpgQualificationPass<'_>,
    ) -> Result<EpgQualificationLayout, EpgQualificationError> {
        let prepared = self.prepare(device, pass)?;
        Ok(self.encode_prepared(encoder, &prepared))
    }
}

fn validate_geometry(geometry: EpgGeometryDescriptor) -> Result<(), EpgQualificationError> {
    match geometry.kind() {
        EpgGeometryKind::So2 => {
            if geometry.so4_dims() != 0 {
                return Err(EpgQualificationError::Epg(
                    EpgContractError::InvalidSo4Dims(geometry.so4_dims()),
                ));
            }
        }
        EpgGeometryKind::HybridSo4(_) => {
            if geometry.so4_dims() == 0 || !geometry.so4_dims().is_multiple_of(4) {
                return Err(EpgQualificationError::Epg(
                    EpgContractError::InvalidSo4Dims(geometry.so4_dims()),
                ));
            }
        }
    }
    Ok(())
}

fn geometry_mode(geometry: EpgGeometryDescriptor) -> u32 {
    match geometry.kind() {
        EpgGeometryKind::So2 => 0,
        EpgGeometryKind::HybridSo4(So4Geometry::Biplanar) => 1,
        EpgGeometryKind::HybridSo4(So4Geometry::Isoclinic) => 2,
    }
}

fn validate_head_dim(head_dim: usize) -> Result<(), EpgQualificationError> {
    if head_dim == 0 || !head_dim.is_multiple_of(4) || head_dim > EPG_WGSL_MAX_HEAD_DIM {
        return Err(EpgQualificationError::UnsupportedHeadDim(head_dim));
    }
    Ok(())
}

fn validate_dispatch(
    axis: &'static str,
    actual: usize,
    maximum: u32,
) -> Result<(), EpgQualificationError> {
    if actual > maximum as usize {
        return Err(EpgQualificationError::DispatchLimit {
            axis,
            actual,
            maximum,
        });
    }
    Ok(())
}

fn validate_buffer(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), EpgQualificationError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(EpgQualificationError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, EpgQualificationError> {
    u32::try_from(value).map_err(|_| EpgQualificationError::IndexSpaceExceeded(value))
}

fn bytes_for_f32(elements: usize) -> Result<u64, EpgQualificationError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| EpgQualificationError::IndexSpaceExceeded(elements))
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}
