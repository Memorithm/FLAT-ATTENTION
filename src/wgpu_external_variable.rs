//! Caller-owned WGPU encoding for M12 padded variable-length batches.
//!
//! The physical Q/K/V buffers remain dense sequence-major projection outputs,
//! while each batch element supplies independent active query/KV lengths and
//! causal/query-RoPE origins. The kernel never stages padded K/V rows, and
//! padded query rows deterministically become O=0 and LSE=-∞.

use core::fmt;

use super::{
    AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig, ExternalProjectionLayout,
    ExternalWgpuError, FlatAttentionConfig, FlatAttentionError,
    FLAT_FWD_PROJECTION_ROPE_VARIABLE_WGSL, WGSL_MAX_HEAD_DIM, WGSL_QUERY_ROWS,
};

/// Maximum batch cardinality encoded in the portable M12 uniform block.
///
/// Keeping length metadata in a uniform preserves the four-storage-buffer
/// portable contract used by M11 (Q, K, V, O|LSE) instead of requiring a fifth
/// storage binding on downlevel adapters.
pub const WGSL_VARIABLE_MAX_BATCH: usize = 256;

/// Active logical extents and query position domains for one padded sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariableLengthSequenceMetadata {
    pub active_query_len: usize,
    pub active_kv_len: usize,
    /// Absolute query position used by causal masking for local query row zero.
    pub query_position_offset: usize,
    /// RoPE position used by local query row zero.
    pub query_rope_position_offset: usize,
}

/// Rotary configuration shared by one variable-length batch dispatch.
///
/// Q origins are per sequence in [`VariableLengthSequenceMetadata`]. K/V share
/// one origin because all padded cache rows use the same physical key index
/// domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariableLengthRotaryEmbeddingConfig {
    pub theta: f32,
    pub kv_position_offset: usize,
}

/// One caller-owned padded variable-length projection-layout dispatch.
pub struct ExternalVariableProjectionPass<'a> {
    pub q: &'a wgpu::Buffer,
    pub k: &'a wgpu::Buffer,
    pub v: &'a wgpu::Buffer,
    pub out_and_lse: &'a wgpu::Buffer,
    /// Physical padded extents. `shape.query_position_offset` is ignored; the
    /// causal offset is supplied per sequence in `metadata`.
    pub shape: AsymmetricGroupedAttentionShape,
    pub metadata: &'a [VariableLengthSequenceMetadata],
    pub config: FlatAttentionConfig,
    pub rotary: VariableLengthRotaryEmbeddingConfig,
}

/// M12 padded variable-length projection-layout + RoPE + GQA/MQA pipeline.
///
/// The pipeline owns only compiled GPU state. Q/K/V/O buffers, the command
/// encoder and submission remain caller-owned.
pub struct ExternalVariableProjectionRotaryGroupedPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for ExternalVariableProjectionRotaryGroupedPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalVariableProjectionRotaryGroupedPipeline")
            .finish_non_exhaustive()
    }
}

impl ExternalVariableProjectionRotaryGroupedPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ExternalWgpuError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m12-variable-projection-rope-gqa"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                FLAT_FWD_PROJECTION_ROPE_VARIABLE_WGSL,
            )),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m12-variable-projection-rope-gqa"),
            layout: None,
            module: &shader,
            entry_point: "flat_attention_forward",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        match pollster::block_on(device.pop_error_scope()) {
            Some(error) => Err(ExternalWgpuError::PipelineValidation(error.to_string())),
            None => Ok(Self { pipeline }),
        }
    }

    pub fn layout(
        shape: AsymmetricGroupedAttentionShape,
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

    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        shape: AsymmetricGroupedAttentionShape,
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
            label: Some("flat-m12-variable-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Record one padded variable-length pass. No submission or synchronization
    /// occurs. Only the small length/position uniform is materialised by FLAT.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalVariableProjectionPass<'_>,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        let dispatch = validate_dispatch(device, pass.shape, pass.metadata, pass.rotary)?;
        let layout = Self::layout(pass.shape)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;
        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;

        let mut params = Vec::with_capacity(12 + WGSL_VARIABLE_MAX_BATCH * 4);
        params.extend_from_slice(&[
            checked_u32(pass.shape.query_len)?,
            checked_u32(pass.shape.kv_len)?,
            checked_u32(pass.shape.head_dim)?,
            checked_u32(pass.shape.q_heads)?,
            checked_u32(pass.shape.kv_heads)?,
            checked_u32(pass.shape.batch)?,
            u32::from(pass.config.causal),
            scale.to_bits(),
            pass.rotary.theta.to_bits(),
            checked_u32(pass.rotary.kv_position_offset)?,
            0,
            0,
        ]);
        for metadata in pass.metadata {
            params.extend_from_slice(&[
                checked_u32(metadata.active_query_len)?,
                checked_u32(metadata.active_kv_len)?,
                checked_u32(metadata.query_position_offset)?,
                checked_u32(metadata.query_rope_position_offset)?,
            ]);
        }
        params.resize(12 + WGSL_VARIABLE_MAX_BATCH * 4, 0);
        let params_bytes = encode_u32(&params);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m12-variable-params"),
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
            label: Some("flat-m12-variable-bind-group"),
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
                label: Some("flat-m12-variable-projection-rope-gqa"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch.query_workgroups, dispatch.q_batch_heads, 1);
        }

        Ok(layout)
    }
}

struct VariableDispatchGeometry {
    q_batch_heads: u32,
    query_workgroups: u32,
}

fn validate_dispatch(
    device: &wgpu::Device,
    shape: AsymmetricGroupedAttentionShape,
    metadata: &[VariableLengthSequenceMetadata],
    rotary: VariableLengthRotaryEmbeddingConfig,
) -> Result<VariableDispatchGeometry, ExternalWgpuError> {
    shape.validate()?;
    if shape.head_dim > WGSL_MAX_HEAD_DIM {
        return Err(ExternalWgpuError::UnsupportedHeadDim {
            actual: shape.head_dim,
            maximum: WGSL_MAX_HEAD_DIM,
        });
    }
    if shape.batch > WGSL_VARIABLE_MAX_BATCH {
        return Err(ExternalWgpuError::DispatchLimit {
            axis: "batch_metadata",
            actual: shape.batch,
            maximum: WGSL_VARIABLE_MAX_BATCH as u32,
        });
    }
    if metadata.len() != shape.batch {
        return Err(FlatAttentionError::LengthMismatch {
            tensor: "active sequence metadata",
            actual: metadata.len(),
            expected: shape.batch,
        }
        .into());
    }

    for entry in metadata {
        if entry.active_query_len == 0 || entry.active_kv_len == 0 {
            return Err(FlatAttentionError::ZeroDimension.into());
        }
        if entry.active_query_len > shape.query_len {
            return Err(FlatAttentionError::LengthMismatch {
                tensor: "active query length",
                actual: entry.active_query_len,
                expected: shape.query_len,
            }
            .into());
        }
        if entry.active_kv_len > shape.kv_len {
            return Err(FlatAttentionError::LengthMismatch {
                tensor: "active KV length",
                actual: entry.active_kv_len,
                expected: shape.kv_len,
            }
            .into());
        }
        let causal_exclusive = entry
            .query_position_offset
            .checked_add(entry.active_query_len)
            .ok_or(FlatAttentionError::PositionOverflow)?;
        let q_rope_last = entry
            .query_rope_position_offset
            .checked_add(entry.active_query_len - 1)
            .ok_or(FlatAttentionError::PositionOverflow)?;
        let kv_rope_last = rotary
            .kv_position_offset
            .checked_add(entry.active_kv_len - 1)
            .ok_or(FlatAttentionError::PositionOverflow)?;
        if causal_exclusive > u32::MAX as usize
            || q_rope_last > u32::MAX as usize
            || kv_rope_last > u32::MAX as usize
        {
            return Err(FlatAttentionError::PositionOverflow.into());
        }
        AsymmetricRotaryEmbeddingConfig {
            theta: rotary.theta,
            query_position_offset: entry.query_rope_position_offset,
            kv_position_offset: rotary.kv_position_offset,
        }
        .validate(shape.head_dim, entry.active_query_len, entry.active_kv_len)?;
    }

    let layout = ExternalVariableProjectionRotaryGroupedPipeline::layout(shape)?;
    if layout.combined_elements > u32::MAX as usize || layout.kv_elements > u32::MAX as usize {
        return Err(ExternalWgpuError::IndexSpaceExceeded {
            elements: layout.combined_elements.max(layout.kv_elements),
        });
    }

    let q_batch_heads = shape
        .batch
        .checked_mul(shape.q_heads)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let query_workgroups = shape.query_len.div_ceil(WGSL_QUERY_ROWS);
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

    Ok(VariableDispatchGeometry {
        q_batch_heads: checked_u32(q_batch_heads)?,
        query_workgroups: checked_u32(query_workgroups)?,
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
