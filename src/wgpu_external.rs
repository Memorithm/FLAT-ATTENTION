//! Caller-owned WGPU encoding for direct framework integration.
//!
//! This module owns only the compiled FLAT compute pipeline. It does not own a
//! `wgpu::Device`, `wgpu::Queue`, input buffer, output buffer, encoder or command
//! submission. The host framework controls all resource lifetime and submission.

use super::wgpu_internal;

use core::fmt;

use super::{
    FlatAttentionConfig, FlatAttentionError, GroupedAttentionShape, RotaryEmbeddingConfig,
    FLAT_FWD_PROJECTION_ROPE_WGSL, WGSL_MAX_HEAD_DIM, WGSL_QUERY_ROWS,
};

/// Byte/element geometry expected by [`ExternalProjectionRotaryGroupedPipeline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalProjectionLayout {
    pub q_elements: usize,
    pub kv_elements: usize,
    pub output_elements: usize,
    pub lse_elements: usize,
    pub combined_elements: usize,
    pub q_bytes: u64,
    pub kv_bytes: u64,
    pub combined_bytes: u64,
}

/// One caller-owned FLAT-R2 dispatch description.
///
/// Buffers are borrowed and remain owned by the framework. The output buffer
/// stores projection-layout O first and LSE in its tail.
pub struct ExternalProjectionPass<'a> {
    pub q: &'a wgpu::Buffer,
    pub k: &'a wgpu::Buffer,
    pub v: &'a wgpu::Buffer,
    pub out_and_lse: &'a wgpu::Buffer,
    pub shape: GroupedAttentionShape,
    pub config: FlatAttentionConfig,
    pub rotary: RotaryEmbeddingConfig,
}

/// Errors specific to caller-owned WGPU encoding.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ExternalWgpuError {
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
    DeviceBufferLimit {
        required_bytes: u64,
        maximum_bytes: u64,
    },
    CandidateNotEnabled {
        candidate: &'static str,
    },
    UnsupportedCandidateShape {
        candidate: &'static str,
        reason: &'static str,
    },
    PipelineValidation(String),
}

impl fmt::Display for ExternalWgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::UnsupportedHeadDim { actual, maximum } => {
                write!(f, "head_dim {actual} exceeds portable maximum {maximum}")
            }
            Self::IndexSpaceExceeded { elements } => {
                write!(f, "WGPU u32 index space exceeded by {elements} elements")
            }
            Self::DispatchLimit {
                axis,
                actual,
                maximum,
            } => write!(
                f,
                "WGPU dispatch axis {axis} requires {actual} workgroups, device maximum is {maximum}"
            ),
            Self::BufferTooSmall {
                tensor,
                actual_bytes,
                required_bytes,
            } => write!(
                f,
                "buffer {tensor} contains {actual_bytes} bytes, requires at least {required_bytes}"
            ),
            Self::DeviceBufferLimit {
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "external FLAT pass requires {required_bytes} bytes per buffer, device maximum is {maximum_bytes}"
            ),
            Self::CandidateNotEnabled { candidate } => {
                write!(f, "external FLAT candidate {candidate} was not enabled")
            }
            Self::UnsupportedCandidateShape { candidate, reason } => {
                write!(f, "external FLAT candidate {candidate} does not support this shape: {reason}")
            }
            Self::PipelineValidation(error) => {
                write!(f, "external FLAT pipeline validation failed: {error}")
            }
        }
    }
}

impl std::error::Error for ExternalWgpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FlatAttentionError> for ExternalWgpuError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

/// R2 pipeline for fused projection-layout RoPE + GQA/MQA.
///
/// The pipeline is compiled once from the caller's device and can then record
/// arbitrarily many passes into caller-owned encoders. `encode` never submits,
/// polls or reads back data.
pub struct ExternalProjectionRotaryGroupedPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for ExternalProjectionRotaryGroupedPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalProjectionRotaryGroupedPipeline")
            .finish_non_exhaustive()
    }
}

impl ExternalProjectionRotaryGroupedPipeline {
    /// Compile the FLAT-R2 pipeline on an externally-owned device.
    pub fn new(device: &wgpu::Device) -> Result<Self, ExternalWgpuError> {
        let pipeline = wgpu_internal::create_pipeline(
            device,
            FLAT_FWD_PROJECTION_ROPE_WGSL,
            "flat-r2-projection-rope-gqa",
            "flat_attention_forward",
        )
        .map_err(ExternalWgpuError::PipelineValidation)?;
        Ok(Self { pipeline })
    }

    /// Compute exact logical/byte requirements without allocating a GPU buffer.
    pub fn layout(
        shape: GroupedAttentionShape,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        shape.validate()?;
        let q_elements = shape.q_tensor_len()?;
        let kv_elements = shape.kv_tensor_len()?;
        let lse_elements = shape.lse_len()?;
        let combined_elements = q_elements
            .checked_add(lse_elements)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        Ok(ExternalProjectionLayout {
            q_elements,
            kv_elements,
            output_elements: q_elements,
            lse_elements,
            combined_elements,
            q_bytes: bytes_for_f32_len(q_elements)?,
            kv_bytes: bytes_for_f32_len(kv_elements)?,
            combined_bytes: bytes_for_f32_len(combined_elements)?,
        })
    }

    /// Convenience allocator for the combined `[O | LSE]` storage buffer.
    ///
    /// Frameworks may instead allocate their own buffer and call [`Self::encode`].
    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        shape: GroupedAttentionShape,
    ) -> Result<wgpu::Buffer, ExternalWgpuError> {
        let layout = Self::layout(shape)?;
        let maximum_bytes = u64::from(device.limits().max_storage_buffer_binding_size);
        if layout.combined_bytes > maximum_bytes {
            return Err(ExternalWgpuError::DeviceBufferLimit {
                required_bytes: layout.combined_bytes,
                maximum_bytes,
            });
        }
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-r2-external-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Record one fused R2 pass into an externally-owned command encoder.
    ///
    /// The pass buffers remain owned by the caller. Q/K/V must use the
    /// sequence-major projection layout documented by
    /// [`crate::forward_reference_projection_grouped_rope`]. The first
    /// `output_elements` f32 values of `out_and_lse` are immediately a row-major
    /// `(batch * seq_len) × (q_heads * head_dim)` matrix suitable for an output
    /// projection GEMM; LSE follows in the tail.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalProjectionPass<'_>,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        let dispatch = validate_dispatch(device, pass.shape, pass.rotary)?;
        let layout = Self::layout(pass.shape)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;
        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;

        let params = [
            dispatch.seq_len,
            checked_u32(pass.shape.head_dim)?,
            checked_u32(pass.shape.q_heads)?,
            checked_u32(pass.shape.kv_heads)?,
            checked_u32(pass.shape.batch)?,
            u32::from(pass.config.causal),
            scale.to_bits(),
            pass.rotary.theta.to_bits(),
            dispatch.position_offset,
            0,
            0,
            0,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-r2-external-params"),
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
            label: Some("flat-r2-external-bind-group"),
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
                    resource: pass.out_and_lse.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-r2-external-projection-rope-gqa"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch.query_workgroups, dispatch.q_batch_heads, 1);
        }

        Ok(layout)
    }
}

struct ExternalDispatchGeometry {
    q_batch_heads: u32,
    seq_len: u32,
    query_workgroups: u32,
    position_offset: u32,
}

fn validate_dispatch(
    device: &wgpu::Device,
    shape: GroupedAttentionShape,
    rotary: RotaryEmbeddingConfig,
) -> Result<ExternalDispatchGeometry, ExternalWgpuError> {
    shape.validate()?;
    rotary.validate(shape.head_dim, shape.seq_len)?;
    if shape.head_dim > WGSL_MAX_HEAD_DIM {
        return Err(ExternalWgpuError::UnsupportedHeadDim {
            actual: shape.head_dim,
            maximum: WGSL_MAX_HEAD_DIM,
        });
    }

    let final_position = rotary
        .position_offset
        .checked_add(shape.seq_len.saturating_sub(1))
        .ok_or(FlatAttentionError::PositionOverflow)?;
    if final_position > u32::MAX as usize {
        return Err(FlatAttentionError::PositionOverflow.into());
    }

    let layout = ExternalProjectionRotaryGroupedPipeline::layout(shape)?;
    if layout.combined_elements > u32::MAX as usize || layout.kv_elements > u32::MAX as usize {
        return Err(ExternalWgpuError::IndexSpaceExceeded {
            elements: layout.combined_elements.max(layout.kv_elements),
        });
    }

    let q_batch_heads = shape
        .batch
        .checked_mul(shape.q_heads)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let query_workgroups = shape.seq_len.div_ceil(WGSL_QUERY_ROWS);
    let maximum = device.limits().max_compute_workgroups_per_dimension;
    if query_workgroups > maximum as usize {
        return Err(ExternalWgpuError::DispatchLimit {
            axis: "x/query_tiles",
            actual: query_workgroups,
            maximum,
        });
    }
    if q_batch_heads > maximum as usize {
        return Err(ExternalWgpuError::DispatchLimit {
            axis: "y/batch_q_heads",
            actual: q_batch_heads,
            maximum,
        });
    }

    Ok(ExternalDispatchGeometry {
        q_batch_heads: checked_u32(q_batch_heads)?,
        seq_len: checked_u32(shape.seq_len)?,
        query_workgroups: checked_u32(query_workgroups)?,
        position_offset: checked_u32(rotary.position_offset)?,
    })
}

fn validate_buffer(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), ExternalWgpuError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(ExternalWgpuError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, ExternalWgpuError> {
    wgpu_internal::checked_u32(value)
        .ok_or(ExternalWgpuError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32_len(len: usize) -> Result<u64, ExternalWgpuError> {
    let bytes = len
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| ExternalWgpuError::IndexSpaceExceeded { elements: len })
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    wgpu_internal::encode_u32(values)
}
