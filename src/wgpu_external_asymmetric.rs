//! Caller-owned rectangular WGPU encoding for M11 decode/cross-attention.

use core::fmt;

use super::{
    AsymmetricGroupedAttentionShape, FlatAttentionConfig, FlatAttentionError,
    FLAT_FWD_PROJECTION_ROPE_RECT_WGSL, WGSL_MAX_HEAD_DIM,
};
use crate::RotaryEmbeddingConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalAsymmetricProjectionLayout {
    pub q_elements: usize,
    pub kv_elements: usize,
    pub output_elements: usize,
    pub lse_elements: usize,
    pub combined_elements: usize,
    pub q_bytes: u64,
    pub kv_bytes: u64,
    pub combined_bytes: u64,
}

pub struct ExternalAsymmetricProjectionPass<'a> {
    pub q: &'a wgpu::Buffer,
    pub k: &'a wgpu::Buffer,
    pub v: &'a wgpu::Buffer,
    pub out_and_lse: &'a wgpu::Buffer,
    pub shape: AsymmetricGroupedAttentionShape,
    pub config: FlatAttentionConfig,
    /// RoPE base theta and absolute offset for K/V-cache key positions.
    pub rotary: RotaryEmbeddingConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExternalAsymmetricWgpuError {
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

impl fmt::Display for ExternalAsymmetricWgpuError {
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
            Self::PipelineValidation(error) => {
                write!(f, "external asymmetric FLAT pipeline validation failed: {error}")
            }
        }
    }
}

impl std::error::Error for ExternalAsymmetricWgpuError {}

impl From<FlatAttentionError> for ExternalAsymmetricWgpuError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

pub struct ExternalAsymmetricProjectionRotaryGroupedPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for ExternalAsymmetricProjectionRotaryGroupedPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalAsymmetricProjectionRotaryGroupedPipeline")
            .finish_non_exhaustive()
    }
}

impl ExternalAsymmetricProjectionRotaryGroupedPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ExternalAsymmetricWgpuError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-r3-projection-rope-rectangular-gqa"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                FLAT_FWD_PROJECTION_ROPE_RECT_WGSL,
            )),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-r3-projection-rope-rectangular-gqa"),
            layout: None,
            module: &shader,
            entry_point: "flat_attention_forward",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        match pollster::block_on(device.pop_error_scope()) {
            Some(error) => Err(ExternalAsymmetricWgpuError::PipelineValidation(
                error.to_string(),
            )),
            None => Ok(Self { pipeline }),
        }
    }

    pub fn layout(
        shape: AsymmetricGroupedAttentionShape,
    ) -> Result<ExternalAsymmetricProjectionLayout, ExternalAsymmetricWgpuError> {
        shape.validate()?;
        let q_elements = shape.q_tensor_len()?;
        let kv_elements = shape.kv_tensor_len()?;
        let lse_elements = shape.lse_len()?;
        let combined_elements = q_elements
            .checked_add(lse_elements)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        Ok(ExternalAsymmetricProjectionLayout {
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

    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        shape: AsymmetricGroupedAttentionShape,
    ) -> Result<wgpu::Buffer, ExternalAsymmetricWgpuError> {
        let layout = Self::layout(shape)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-r3-external-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
    ) -> Result<ExternalAsymmetricProjectionLayout, ExternalAsymmetricWgpuError> {
        let dispatch = validate_dispatch(device, pass.shape, pass.rotary)?;
        let layout = Self::layout(pass.shape)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;
        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;

        let params = [
            dispatch.query_len,
            dispatch.kv_len,
            checked_u32(pass.shape.head_dim)?,
            checked_u32(pass.shape.q_heads)?,
            checked_u32(pass.shape.kv_heads)?,
            checked_u32(pass.shape.batch)?,
            u32::from(pass.config.causal),
            scale.to_bits(),
            pass.rotary.theta.to_bits(),
            dispatch.query_position_offset,
            dispatch.key_position_offset,
            0,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-r3-external-params"),
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
            label: Some("flat-r3-external-bind-group"),
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
                label: Some("flat-r3-external-projection-rope-rectangular-gqa"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch.query_len, dispatch.q_batch_heads, 1);
        }

        Ok(layout)
    }
}

struct ExternalAsymmetricDispatchGeometry {
    q_batch_heads: u32,
    query_len: u32,
    kv_len: u32,
    query_position_offset: u32,
    key_position_offset: u32,
}

fn validate_dispatch(
    device: &wgpu::Device,
    shape: AsymmetricGroupedAttentionShape,
    rotary: RotaryEmbeddingConfig,
) -> Result<ExternalAsymmetricDispatchGeometry, ExternalAsymmetricWgpuError> {
    shape.validate()?;
    rotary.validate(shape.head_dim, shape.kv_len)?;
    if shape.head_dim > WGSL_MAX_HEAD_DIM {
        return Err(ExternalAsymmetricWgpuError::UnsupportedHeadDim {
            actual: shape.head_dim,
            maximum: WGSL_MAX_HEAD_DIM,
        });
    }

    let final_query_position = shape
        .query_position_offset
        .checked_add(shape.query_len.saturating_sub(1))
        .ok_or(FlatAttentionError::PositionOverflow)?;
    let final_key_position = rotary
        .position_offset
        .checked_add(shape.kv_len.saturating_sub(1))
        .ok_or(FlatAttentionError::PositionOverflow)?;
    if final_query_position > u32::MAX as usize || final_key_position > u32::MAX as usize {
        return Err(FlatAttentionError::PositionOverflow.into());
    }

    let layout = ExternalAsymmetricProjectionRotaryGroupedPipeline::layout(shape)?;
    if layout.combined_elements > u32::MAX as usize || layout.kv_elements > u32::MAX as usize {
        return Err(ExternalAsymmetricWgpuError::IndexSpaceExceeded {
            elements: layout.combined_elements.max(layout.kv_elements),
        });
    }

    let q_batch_heads = shape
        .batch
        .checked_mul(shape.q_heads)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let maximum = device.limits().max_compute_workgroups_per_dimension;
    if shape.query_len > maximum as usize {
        return Err(ExternalAsymmetricWgpuError::DispatchLimit {
            axis: "x/query_rows",
            actual: shape.query_len,
            maximum,
        });
    }
    if q_batch_heads > maximum as usize {
        return Err(ExternalAsymmetricWgpuError::DispatchLimit {
            axis: "y/batch_q_heads",
            actual: q_batch_heads,
            maximum,
        });
    }

    Ok(ExternalAsymmetricDispatchGeometry {
        q_batch_heads: checked_u32(q_batch_heads)?,
        query_len: checked_u32(shape.query_len)?,
        kv_len: checked_u32(shape.kv_len)?,
        query_position_offset: checked_u32(shape.query_position_offset)?,
        key_position_offset: checked_u32(rotary.position_offset)?,
    })
}

fn validate_buffer(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), ExternalAsymmetricWgpuError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(ExternalAsymmetricWgpuError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, ExternalAsymmetricWgpuError> {
    u32::try_from(value)
        .map_err(|_| ExternalAsymmetricWgpuError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32_len(len: usize) -> Result<u64, ExternalAsymmetricWgpuError> {
    let bytes = len
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes)
        .map_err(|_| ExternalAsymmetricWgpuError::IndexSpaceExceeded { elements: len })
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_layout_keeps_q_and_kv_lengths_independent() {
        let shape = AsymmetricGroupedAttentionShape {
            batch: 2,
            q_heads: 8,
            kv_heads: 2,
            query_len: 1,
            kv_len: 4096,
            head_dim: 64,
            query_position_offset: 4095,
        };
        let layout = ExternalAsymmetricProjectionRotaryGroupedPipeline::layout(shape).unwrap();
        assert_eq!(layout.q_elements, 2 * 8 * 64);
        assert_eq!(layout.kv_elements, 2 * 2 * 4096 * 64);
        assert_eq!(layout.output_elements, layout.q_elements);
        assert_eq!(layout.lse_elements, 2 * 8);
    }
}
