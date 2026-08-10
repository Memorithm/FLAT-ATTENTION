//! Specialized caller-owned decode for resident caches that already store RoPE K.
//!
//! This path intentionally does not alter the general M11/M12 contracts. It is
//! a single-query (`q_len = 1`, batch = 1) inference specialization for framework
//! KV caches such as SciRust's resident cache, where K has already been rotated
//! at append time and V remains raw.

use core::fmt;

use super::{
    ExternalProjectionLayout, ExternalWgpuError, FlatAttentionConfig, FlatAttentionError,
    RotaryEmbeddingConfig, FLAT_DECODE_PREROTATED_K_WGSL, WGSL_MAX_HEAD_DIM,
};

/// Logical geometry for one single-query decode over a resident K/V cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreRotatedKDecodeShape {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub kv_len: usize,
    pub head_dim: usize,
}

impl PreRotatedKDecodeShape {
    pub fn q_elements(self) -> Result<usize, FlatAttentionError> {
        self.q_heads
            .checked_mul(self.head_dim)
            .ok_or(FlatAttentionError::ShapeOverflow)
    }

    pub fn kv_elements(self) -> Result<usize, FlatAttentionError> {
        self.kv_len
            .checked_mul(self.kv_heads)
            .and_then(|n| n.checked_mul(self.head_dim))
            .ok_or(FlatAttentionError::ShapeOverflow)
    }

    pub fn lse_elements(self) -> usize {
        self.q_heads
    }

    fn validate(self) -> Result<(), FlatAttentionError> {
        if self.q_heads == 0 || self.kv_heads == 0 || self.kv_len == 0 || self.head_dim == 0 {
            return Err(FlatAttentionError::ZeroDimension);
        }
        if self.q_heads % self.kv_heads != 0 {
            return Err(FlatAttentionError::InvalidHeadGrouping {
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
            });
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(FlatAttentionError::InvalidRotaryHeadDim {
                head_dim: self.head_dim,
            });
        }
        self.q_elements()?;
        self.kv_elements()?;
        Ok(())
    }
}

/// One caller-owned decode dispatch.
pub struct ExternalPreRotatedKDecodePass<'a> {
    pub q: &'a wgpu::Buffer,
    pub k: &'a wgpu::Buffer,
    pub v: &'a wgpu::Buffer,
    pub out_and_lse: &'a wgpu::Buffer,
    pub shape: PreRotatedKDecodeShape,
    pub config: FlatAttentionConfig,
    /// Q is raw and rotated in the kernel at this absolute position.
    pub rotary: RotaryEmbeddingConfig,
}

/// Reusable M15 single-query decode pipeline for pre-rotated resident K caches.
pub struct ExternalPreRotatedKDecodePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for ExternalPreRotatedKDecodePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalPreRotatedKDecodePipeline")
            .finish_non_exhaustive()
    }
}

impl ExternalPreRotatedKDecodePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ExternalWgpuError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m15-prerotated-k-decode"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                FLAT_DECODE_PREROTATED_K_WGSL,
            )),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m15-prerotated-k-decode"),
            layout: None,
            module: &shader,
            entry_point: "flat_attention_decode",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        match pollster::block_on(device.pop_error_scope()) {
            Some(error) => Err(ExternalWgpuError::PipelineValidation(error.to_string())),
            None => Ok(Self { pipeline }),
        }
    }

    pub fn layout(shape: PreRotatedKDecodeShape) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        shape.validate()?;
        let q_elements = shape.q_elements()?;
        let kv_elements = shape.kv_elements()?;
        let lse_elements = shape.lse_elements();
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

    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        shape: PreRotatedKDecodeShape,
    ) -> Result<wgpu::Buffer, ExternalWgpuError> {
        let layout = Self::layout(shape)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m15-prerotated-k-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Record one decode pass. The caller owns submission and synchronization.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalPreRotatedKDecodePass<'_>,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        validate_dispatch(device, pass.shape, pass.rotary)?;
        let layout = Self::layout(pass.shape)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;
        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;

        let params = [
            checked_u32(pass.shape.kv_len)?,
            checked_u32(pass.shape.head_dim)?,
            checked_u32(pass.shape.q_heads)?,
            checked_u32(pass.shape.kv_heads)?,
            scale.to_bits(),
            pass.rotary.theta.to_bits(),
            checked_u32(pass.rotary.position_offset)?,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m15-prerotated-k-params"),
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
            label: Some("flat-m15-prerotated-k-bind-group"),
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
                label: Some("flat-m15-prerotated-k-decode"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(checked_u32(pass.shape.q_heads)?, 1, 1);
        }

        Ok(layout)
    }
}

fn validate_dispatch(
    device: &wgpu::Device,
    shape: PreRotatedKDecodeShape,
    rotary: RotaryEmbeddingConfig,
) -> Result<(), ExternalWgpuError> {
    shape.validate()?;
    rotary.validate(shape.head_dim, 1)?;
    if shape.head_dim > WGSL_MAX_HEAD_DIM {
        return Err(ExternalWgpuError::UnsupportedHeadDim {
            actual: shape.head_dim,
            maximum: WGSL_MAX_HEAD_DIM,
        });
    }
    if rotary.position_offset > u32::MAX as usize {
        return Err(FlatAttentionError::PositionOverflow.into());
    }
    let layout = ExternalPreRotatedKDecodePipeline::layout(shape)?;
    if layout.combined_elements > u32::MAX as usize || layout.kv_elements > u32::MAX as usize {
        return Err(ExternalWgpuError::IndexSpaceExceeded {
            elements: layout.combined_elements.max(layout.kv_elements),
        });
    }
    let maximum = device.limits().max_compute_workgroups_per_dimension;
    if shape.q_heads > maximum as usize {
        return Err(ExternalWgpuError::DispatchLimit {
            axis: "x/q_heads",
            actual: shape.q_heads,
            maximum,
        });
    }
    Ok(())
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
    u32::try_from(value).map_err(|_| ExternalWgpuError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32_len(len: usize) -> Result<u64, ExternalWgpuError> {
    let bytes = len
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| ExternalWgpuError::IndexSpaceExceeded { elements: len })
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}
