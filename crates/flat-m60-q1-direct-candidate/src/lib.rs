//! M60 opt-in Q1 vec4 MHA candidate with direct K/V storage loads.
//!
//! M58 proved that one-workgroup-per-query-row is substantially better than the
//! older Q4 MHA route on physical Thor. M60 isolates the next memory hypothesis:
//! a Q1 workgroup consumes every K/V vec4 exactly once, so staging K/V through
//! workgroup memory adds traffic and synchronization without cross-query reuse.
//! This crate removes only that staging step and otherwise preserves M58's
//! 64-lane reduction and online-softmax synchronization structure.

#![forbid(unsafe_code)]

use core::fmt;
use std::borrow::Cow;

use flat_attention::{
    FlatAttentionError, GroupedAttentionShape, GroupedForwardError, GroupedForwardLayout,
    GroupedForwardPass, WgpuGroupedForwardPipeline,
};

/// M60 direct-load Q1 vec4 shader source.
pub const Q1_DIRECT_VEC4_WGSL: &str = include_str!("../shaders/q1_direct_vec4.wgsl");

/// Validated reusable bind state for one M60 dispatch.
pub struct PreparedQ1DirectMha {
    layout: GroupedForwardLayout,
    bind_group: wgpu::BindGroup,
    dispatch_x: u32,
    dispatch_y: u32,
    _params_buffer: wgpu::Buffer,
}

impl fmt::Debug for PreparedQ1DirectMha {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedQ1DirectMha")
            .field("layout", &self.layout)
            .field("dispatch_x", &self.dispatch_x)
            .field("dispatch_y", &self.dispatch_y)
            .finish_non_exhaustive()
    }
}

impl PreparedQ1DirectMha {
    /// Validated packed `[O | LSE]` layout.
    #[must_use]
    pub const fn layout(&self) -> GroupedForwardLayout {
        self.layout
    }

    /// Query-axis workgroups. M60 dispatches one workgroup per query row.
    #[must_use]
    pub const fn dispatch_x(&self) -> u32 {
        self.dispatch_x
    }
}

/// Errors specific to the isolated M60 candidate.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Q1DirectError {
    /// Shared FLAT grouped-forward validation failed.
    Grouped(GroupedForwardError),
    /// M60 is MHA-only and requires identical query/KV head cardinality.
    RequiresMha { q_heads: usize, kv_heads: usize },
    /// Only vec4-qualified D64/D128 are admitted.
    UnsupportedHeadDim(usize),
    /// One dispatch dimension exceeds the selected device limit.
    DispatchLimit {
        axis: &'static str,
        actual: usize,
        maximum: u32,
    },
    /// A caller-owned buffer is smaller than the validated physical layout.
    BufferTooSmall {
        tensor: &'static str,
        actual_bytes: u64,
        required_bytes: u64,
    },
    /// A required storage buffer exceeds the device binding limit.
    DeviceBufferLimit {
        required_bytes: u64,
        maximum_bytes: u64,
    },
    /// WGSL/pipeline validation failed.
    PipelineValidation(String),
    /// A mapped parameter buffer could not be accessed.
    BufferMapping(String),
    /// A host-side value cannot be represented in the WGSL u32 index space.
    IndexSpaceExceeded(usize),
}

impl fmt::Display for Q1DirectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grouped(error) => write!(f, "{error}"),
            Self::RequiresMha { q_heads, kv_heads } => write!(
                f,
                "M60 Q1 direct candidate requires MHA q_heads == kv_heads, got {q_heads} and {kv_heads}"
            ),
            Self::UnsupportedHeadDim(actual) => write!(
                f,
                "M60 Q1 direct candidate supports only head_dim 64 or 128, got {actual}"
            ),
            Self::DispatchLimit {
                axis,
                actual,
                maximum,
            } => write!(
                f,
                "M60 dispatch axis {axis} requires {actual} workgroups, device maximum is {maximum}"
            ),
            Self::BufferTooSmall {
                tensor,
                actual_bytes,
                required_bytes,
            } => write!(
                f,
                "M60 buffer {tensor} contains {actual_bytes} bytes, requires at least {required_bytes}"
            ),
            Self::DeviceBufferLimit {
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "M60 output requires {required_bytes} bytes, device maximum storage binding is {maximum_bytes} bytes"
            ),
            Self::PipelineValidation(error) => {
                write!(f, "M60 pipeline validation failed: {error}")
            }
            Self::BufferMapping(error) => write!(f, "M60 mapped-buffer access failed: {error}"),
            Self::IndexSpaceExceeded(value) => {
                write!(f, "M60 value {value} exceeds WGSL u32 index space")
            }
        }
    }
}

impl std::error::Error for Q1DirectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Grouped(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GroupedForwardError> for Q1DirectError {
    fn from(value: GroupedForwardError) -> Self {
        Self::Grouped(value)
    }
}

impl From<FlatAttentionError> for Q1DirectError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Grouped(GroupedForwardError::Core(value))
    }
}

/// Explicit M60 performance candidate.
///
/// No production FLAT router selects this type automatically.
pub struct Q1DirectMhaPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for Q1DirectMhaPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Q1DirectMhaPipeline")
            .finish_non_exhaustive()
    }
}

impl Q1DirectMhaPipeline {
    /// Compile the isolated direct-load candidate.
    pub fn new(device: &wgpu::Device) -> Result<Self, Q1DirectError> {
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m60-q1-direct-vec4"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(Q1_DIRECT_VEC4_WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m60-q1-direct-vec4"),
            layout: None,
            module: &shader,
            entry_point: Some("flat_attention_forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        match pollster::block_on(error_scope.pop()) {
            Some(error) => Err(Q1DirectError::PipelineValidation(error.to_string())),
            None => Ok(Self { pipeline }),
        }
    }

    /// Validate the MHA/D64-D128 contract and return FLAT's canonical layout.
    pub fn layout(shape: GroupedAttentionShape) -> Result<GroupedForwardLayout, Q1DirectError> {
        if shape.q_heads != shape.kv_heads {
            return Err(Q1DirectError::RequiresMha {
                q_heads: shape.q_heads,
                kv_heads: shape.kv_heads,
            });
        }
        if !matches!(shape.head_dim, 64 | 128) {
            return Err(Q1DirectError::UnsupportedHeadDim(shape.head_dim));
        }
        Ok(WgpuGroupedForwardPipeline::layout(shape)?)
    }

    /// Allocate the canonical packed `[O | LSE]` output buffer.
    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        shape: GroupedAttentionShape,
    ) -> Result<wgpu::Buffer, Q1DirectError> {
        let layout = Self::layout(shape)?;
        let maximum_bytes = device.limits().max_storage_buffer_binding_size;
        if layout.output_bytes > maximum_bytes {
            return Err(Q1DirectError::DeviceBufferLimit {
                required_bytes: layout.output_bytes,
                maximum_bytes,
            });
        }
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m60-q1-direct-o-lse"),
            size: layout.output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Validate caller-owned buffers and prepare reusable bind state.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        pass: GroupedForwardPass<'_>,
    ) -> Result<PreparedQ1DirectMha, Q1DirectError> {
        let layout = Self::layout(pass.shape)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.output, layout.output_bytes)?;

        let dispatch_x = pass.shape.seq_len;
        let dispatch_y = pass
            .shape
            .batch
            .checked_mul(pass.shape.q_heads)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        validate_dispatch("x/query_rows", dispatch_x, maximum)?;
        validate_dispatch("y/batch_heads", dispatch_y, maximum)?;

        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;
        let values = [
            checked_u32(pass.shape.seq_len)?,
            checked_u32(pass.shape.head_dim)?,
            checked_u32(dispatch_y)?,
            u32::from(pass.config.causal),
            scale.to_bits(),
            0,
            0,
            0,
        ];
        let params = encode_u32(&values);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m60-q1-direct-params"),
            size: params.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        {
            let mut mapped = params_buffer
                .slice(..)
                .get_mapped_range_mut()
                .map_err(|error| Q1DirectError::BufferMapping(error.to_string()))?;
            mapped.copy_from_slice(&params);
        }
        params_buffer.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-m60-q1-direct-bind-group"),
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

        Ok(PreparedQ1DirectMha {
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
        prepared: &PreparedQ1DirectMha,
    ) -> GroupedForwardLayout {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-m60-q1-direct"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &prepared.bind_group, &[]);
        compute_pass.dispatch_workgroups(prepared.dispatch_x, prepared.dispatch_y, 1);
        drop(compute_pass);
        prepared.layout
    }

    /// Validate, prepare, and encode one candidate dispatch.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: GroupedForwardPass<'_>,
    ) -> Result<GroupedForwardLayout, Q1DirectError> {
        let prepared = self.prepare(device, pass)?;
        Ok(self.encode_prepared(encoder, &prepared))
    }
}

fn validate_dispatch(axis: &'static str, actual: usize, maximum: u32) -> Result<(), Q1DirectError> {
    if actual > maximum as usize {
        return Err(Q1DirectError::DispatchLimit {
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
) -> Result<(), Q1DirectError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(Q1DirectError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, Q1DirectError> {
    u32::try_from(value).map_err(|_| Q1DirectError::IndexSpaceExceeded(value))
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}
