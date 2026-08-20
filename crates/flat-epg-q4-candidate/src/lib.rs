//! Opt-in Q4 WGPU performance candidate for Elastic Positional Geometry.
//!
//! This crate intentionally remains separate from both FLAT's production router
//! and the already-qualified `flat-epg-wgpu` baseline. It reuses FLAT's native
//! grouped Q4 execution shape while fusing EPG into Q/K staging.

#![forbid(unsafe_code)]

use core::fmt;
use std::borrow::Cow;

use epg_core::{EpgGeometryDescriptor, EpgGeometryKind, EpgPositionDomain, So4Geometry};
use flat_attention::{FlatAttentionConfig, FlatAttentionError, GroupedAttentionShape};
use flat_epg_wgpu::{
    EpgQualificationError, EpgQualificationLayout, EpgQualificationPass,
    EpgVec4QualificationPipeline,
};

/// Q4 candidate shader source.
pub const EPG_GROUPED_VEC4_Q4_WGSL: &str =
    include_str!("../shaders/epg_grouped_vec4_q4.wgsl");

/// Number of query rows computed by one Q4 workgroup.
pub const EPG_Q4_QUERY_ROWS: usize = 4;

/// Validated reusable state for one fixed Q4 candidate dispatch.
pub struct PreparedEpgQ4Candidate {
    layout: EpgQualificationLayout,
    bind_group: wgpu::BindGroup,
    dispatch_x: u32,
    dispatch_y: u32,
    _params_buffer: wgpu::Buffer,
}

impl fmt::Debug for PreparedEpgQ4Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedEpgQ4Candidate")
            .field("layout", &self.layout)
            .field("dispatch_x", &self.dispatch_x)
            .field("dispatch_y", &self.dispatch_y)
            .finish_non_exhaustive()
    }
}

impl PreparedEpgQ4Candidate {
    /// Validated output layout.
    pub const fn layout(&self) -> EpgQualificationLayout {
        self.layout
    }

    /// Number of query-axis workgroups for this prepared dispatch.
    pub const fn dispatch_x(&self) -> u32 {
        self.dispatch_x
    }
}

/// Explicit Q4 EPG performance candidate.
///
/// Construction and use are always opt-in. No FLAT production route selects
/// this pipeline automatically.
pub struct EpgQ4CandidatePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for EpgQ4CandidatePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EpgQ4CandidatePipeline")
            .finish_non_exhaustive()
    }
}

impl EpgQ4CandidatePipeline {
    /// Compile the Q4 performance candidate.
    pub fn new(device: &wgpu::Device) -> Result<Self, EpgQualificationError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-epg-q4-candidate"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EPG_GROUPED_VEC4_Q4_WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-epg-q4-candidate"),
            layout: None,
            module: &shader,
            entry_point: "epg_grouped_vec4_q4",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        match pollster::block_on(device.pop_error_scope()) {
            Some(error) => Err(EpgQualificationError::PipelineValidation(error.to_string())),
            None => Ok(Self { pipeline }),
        }
    }

    /// Validate the shape and compute the shared `[O | LSE]` physical layout.
    pub fn layout(
        shape: GroupedAttentionShape,
    ) -> Result<EpgQualificationLayout, EpgQualificationError> {
        shape.validate()?;
        if !matches!(shape.head_dim, 64 | 128) {
            return Err(EpgQualificationError::UnsupportedHeadDim(shape.head_dim));
        }
        EpgVec4QualificationPipeline::layout(shape)
    }

    /// Allocate the combined `[O | LSE]` output buffer.
    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        shape: GroupedAttentionShape,
    ) -> Result<wgpu::Buffer, EpgQualificationError> {
        let layout = Self::layout(shape)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-epg-q4-candidate-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Validate caller-owned buffers and prepare reusable Q4 bind state.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        pass: EpgQualificationPass<'_>,
    ) -> Result<PreparedEpgQ4Candidate, EpgQualificationError> {
        let layout = Self::layout(pass.shape)?;
        pass.geometry
            .validate_head_dim(checked_u32(pass.shape.head_dim)?)?;
        validate_geometry(pass.geometry)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.output, layout.combined_bytes)?;

        let final_local = pass.shape.seq_len.saturating_sub(1);
        let final_position = pass.position.resolve(
            u64::try_from(final_local).map_err(|_| EpgQualificationError::PositionSpaceExceeded)?,
        )?;
        if final_position > u32::MAX as u64 {
            return Err(EpgQualificationError::PositionSpaceExceeded);
        }

        let dispatch_x = pass.shape.seq_len.div_ceil(EPG_Q4_QUERY_ROWS);
        let dispatch_y = pass
            .shape
            .batch
            .checked_mul(pass.shape.q_heads)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        validate_dispatch("x/query_tiles", dispatch_x, maximum)?;
        validate_dispatch("y/batch_q_heads", dispatch_y, maximum)?;

        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;
        let params = encode_params(&pass, scale)?;
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-epg-q4-candidate-params"),
            size: params.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        {
            let mut mapped = params_buffer.slice(..).get_mapped_range_mut();
            mapped.copy_from_slice(&params);
        }
        params_buffer.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-epg-q4-candidate-bind-group"),
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

        Ok(PreparedEpgQ4Candidate {
            layout,
            bind_group,
            dispatch_x: checked_u32(dispatch_x)?,
            dispatch_y: checked_u32(dispatch_y)?,
            _params_buffer: params_buffer,
        })
    }

    /// Encode a previously prepared candidate dispatch.
    pub fn encode_prepared(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &PreparedEpgQ4Candidate,
    ) -> EpgQualificationLayout {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-epg-q4-candidate"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &prepared.bind_group, &[]);
        compute_pass.dispatch_workgroups(prepared.dispatch_x, prepared.dispatch_y, 1);
        drop(compute_pass);
        prepared.layout
    }

    /// Validate, bind, and encode one candidate dispatch.
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

fn encode_params(
    pass: &EpgQualificationPass<'_>,
    scale: f32,
) -> Result<Vec<u8>, EpgQualificationError> {
    let values = [
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
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(&values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(bytes)
}

fn validate_geometry(geometry: EpgGeometryDescriptor) -> Result<(), EpgQualificationError> {
    match geometry.kind() {
        EpgGeometryKind::So2 => {
            if geometry.so4_dims() != 0 {
                return Err(EpgQualificationError::Epg(
                    epg_core::EpgContractError::InvalidSo4Dims(geometry.so4_dims()),
                ));
            }
        }
        EpgGeometryKind::HybridSo4(_) => {
            if geometry.so4_dims() == 0 || !geometry.so4_dims().is_multiple_of(4) {
                return Err(EpgQualificationError::Epg(
                    epg_core::EpgContractError::InvalidSo4Dims(geometry.so4_dims()),
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

/// Construct a candidate pass without changing the qualified baseline pass type.
pub const fn pass<'a>(
    q: &'a wgpu::Buffer,
    k: &'a wgpu::Buffer,
    v: &'a wgpu::Buffer,
    output: &'a wgpu::Buffer,
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
    geometry: EpgGeometryDescriptor,
    position: EpgPositionDomain,
) -> EpgQualificationPass<'a> {
    EpgQualificationPass {
        q,
        k,
        v,
        output,
        shape,
        config,
        geometry,
        position,
    }
}
