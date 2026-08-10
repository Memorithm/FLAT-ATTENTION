//! M15 specialized caller-owned decode pipeline over [`WgpuResidentKvCache`].
//!
//! The pipeline records one `q_len = 1` attention dispatch into a caller-owned
//! command encoder. K/V remain in the fixed-capacity M14 cache and are indexed
//! with the cache capacity as the physical batch stride. No cache compaction,
//! host round-trip, submission, polling or mapping occurs here.

use core::fmt;

use super::{
    FlatAttentionConfig, FlatAttentionError, WgpuResidentKvCache, FLAT_DECODE_RESIDENT_WGSL,
    WGSL_MAX_HEAD_DIM,
};

/// Output geometry for one resident decode dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentDecodeLayout {
    pub q_elements: usize,
    pub output_elements: usize,
    pub lse_elements: usize,
    pub combined_elements: usize,
    pub q_bytes: u64,
    pub combined_bytes: u64,
}

/// One caller-owned decode dispatch.
pub struct ResidentDecodePass<'a> {
    pub q: &'a wgpu::Buffer,
    pub out_and_lse: &'a wgpu::Buffer,
    pub cache: &'a WgpuResidentKvCache,
    pub q_heads: usize,
    pub config: FlatAttentionConfig,
    pub theta: f32,
    /// Absolute RoPE position of the single query row.
    pub q_rope_position: usize,
}

/// Explicit M15 decode failures.
#[derive(Debug, Clone, PartialEq)]
pub enum ResidentDecodeError {
    Core(FlatAttentionError),
    EmptyCache,
    UnsupportedHeadDim {
        actual: usize,
        maximum: usize,
    },
    IndexSpaceExceeded {
        elements: usize,
    },
    DispatchLimit {
        actual: usize,
        maximum: u32,
    },
    BufferTooSmall {
        tensor: &'static str,
        actual_bytes: u64,
        required_bytes: u64,
    },
    InvalidTheta(f32),
    PipelineValidation(String),
}

impl fmt::Display for ResidentDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::EmptyCache => write!(f, "resident decode requires at least one live KV row"),
            Self::UnsupportedHeadDim { actual, maximum } => {
                write!(f, "head_dim {actual} exceeds portable maximum {maximum}")
            }
            Self::IndexSpaceExceeded { elements } => {
                write!(f, "resident decode exceeds WGPU u32 index space at {elements} elements")
            }
            Self::DispatchLimit { actual, maximum } => write!(
                f,
                "resident decode requires {actual} workgroups, device maximum is {maximum}"
            ),
            Self::BufferTooSmall {
                tensor,
                actual_bytes,
                required_bytes,
            } => write!(
                f,
                "buffer {tensor} contains {actual_bytes} bytes, requires at least {required_bytes}"
            ),
            Self::InvalidTheta(theta) => {
                write!(f, "resident decode RoPE theta must be finite and positive, got {theta}")
            }
            Self::PipelineValidation(error) => {
                write!(f, "resident decode pipeline validation failed: {error}")
            }
        }
    }
}

impl std::error::Error for ResidentDecodeError {}

impl From<FlatAttentionError> for ResidentDecodeError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

/// Reusable M15 compute pipeline. It owns no framework data buffers or queue.
pub struct WgpuResidentDecodePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for WgpuResidentDecodePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuResidentDecodePipeline")
            .finish_non_exhaustive()
    }
}

impl WgpuResidentDecodePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ResidentDecodeError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m15-resident-decode"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                FLAT_DECODE_RESIDENT_WGSL,
            )),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m15-resident-decode"),
            layout: None,
            module: &shader,
            entry_point: "flat_attention_decode",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        match pollster::block_on(device.pop_error_scope()) {
            Some(error) => Err(ResidentDecodeError::PipelineValidation(error.to_string())),
            None => Ok(Self { pipeline }),
        }
    }

    pub fn layout(
        cache: &WgpuResidentKvCache,
        q_heads: usize,
    ) -> Result<ResidentDecodeLayout, ResidentDecodeError> {
        validate_shape(cache, q_heads)?;
        let q_elements = checked_mul(checked_mul(cache.batch(), q_heads)?, cache.head_dim())?;
        let lse_elements = checked_mul(cache.batch(), q_heads)?;
        let combined_elements = q_elements
            .checked_add(lse_elements)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        Ok(ResidentDecodeLayout {
            q_elements,
            output_elements: q_elements,
            lse_elements,
            combined_elements,
            q_bytes: bytes_for_f32(q_elements)?,
            combined_bytes: bytes_for_f32(combined_elements)?,
        })
    }

    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        cache: &WgpuResidentKvCache,
        q_heads: usize,
    ) -> Result<wgpu::Buffer, ResidentDecodeError> {
        let layout = Self::layout(cache, q_heads)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m15-resident-decode-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Record one q_len=1 decode dispatch over the live cache prefix.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ResidentDecodePass<'_>,
    ) -> Result<ResidentDecodeLayout, ResidentDecodeError> {
        let layout = Self::layout(pass.cache, pass.q_heads)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;
        if !pass.theta.is_finite() || pass.theta <= 0.0 {
            return Err(ResidentDecodeError::InvalidTheta(pass.theta));
        }
        let scale = pass.config.resolved_scale(pass.cache.head_dim())?;
        let q_batch_heads = checked_mul(pass.cache.batch(), pass.q_heads)?;
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        if q_batch_heads > maximum as usize {
            return Err(ResidentDecodeError::DispatchLimit {
                actual: q_batch_heads,
                maximum,
            });
        }

        let params = [
            checked_u32(pass.cache.len())?,
            checked_u32(pass.cache.capacity())?,
            checked_u32(pass.cache.head_dim())?,
            checked_u32(pass.q_heads)?,
            checked_u32(pass.cache.kv_heads())?,
            checked_u32(pass.cache.batch())?,
            scale.to_bits(),
            pass.theta.to_bits(),
            checked_u32(pass.q_rope_position)?,
            0,
            0,
            0,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m15-resident-decode-params"),
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
            label: Some("flat-m15-resident-decode-bind-group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pass.q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pass.cache.k_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pass.cache.v_buffer().as_entire_binding(),
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

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-m15-resident-decode"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(checked_u32(q_batch_heads)?, 1, 1);
        drop(compute_pass);
        Ok(layout)
    }
}

fn validate_shape(
    cache: &WgpuResidentKvCache,
    q_heads: usize,
) -> Result<(), ResidentDecodeError> {
    if cache.is_empty() {
        return Err(ResidentDecodeError::EmptyCache);
    }
    if q_heads == 0 || q_heads % cache.kv_heads() != 0 {
        return Err(FlatAttentionError::InvalidHeadGrouping {
            q_heads,
            kv_heads: cache.kv_heads(),
        }
        .into());
    }
    if cache.head_dim() == 0 || cache.head_dim() % 2 != 0 {
        return Err(FlatAttentionError::InvalidRotaryHeadDim {
            head_dim: cache.head_dim(),
        }
        .into());
    }
    if cache.head_dim() > WGSL_MAX_HEAD_DIM {
        return Err(ResidentDecodeError::UnsupportedHeadDim {
            actual: cache.head_dim(),
            maximum: WGSL_MAX_HEAD_DIM,
        });
    }
    let kv_elements = checked_mul(
        checked_mul(cache.batch(), cache.capacity())?,
        checked_mul(cache.kv_heads(), cache.head_dim())?,
    )?;
    if kv_elements > u32::MAX as usize {
        return Err(ResidentDecodeError::IndexSpaceExceeded {
            elements: kv_elements,
        });
    }
    Ok(())
}

fn validate_buffer(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), ResidentDecodeError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(ResidentDecodeError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    Ok(())
}

fn checked_mul(a: usize, b: usize) -> Result<usize, ResidentDecodeError> {
    a.checked_mul(b)
        .ok_or_else(|| FlatAttentionError::ShapeOverflow.into())
}

fn checked_u32(value: usize) -> Result<u32, ResidentDecodeError> {
    u32::try_from(value).map_err(|_| ResidentDecodeError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32(elements: usize) -> Result<u64, ResidentDecodeError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| ResidentDecodeError::IndexSpaceExceeded { elements })
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}
