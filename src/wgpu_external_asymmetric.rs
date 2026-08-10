//! Caller-owned WGPU encoding for rectangular projection-layout attention.
//!
//! This is the M11 companion to the equal-length FLAT-R2 external pipeline. It
//! records Q/K/V attention with independent query/KV lengths into a caller-owned
//! command encoder and never submits, polls, maps or copies framework buffers.
//! M12 extends the same pipeline with padded variable-length batch metadata while
//! preserving the four-storage-buffer portable contract.

use core::fmt;

use super::{
    AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig, ExternalProjectionLayout,
    ExternalWgpuError, FlatAttentionConfig, FlatAttentionError,
    FLAT_FWD_PROJECTION_ROPE_ASYMMETRIC_WGSL, WGSL_MAX_HEAD_DIM, WGSL_QUERY_ROWS,
};

const FLAT_FWD_PROJECTION_ROPE_VARIABLE_WGSL: &str =
    include_str!("../shaders/flat_fwd_projection_rope_variable.wgsl");
const VARIABLE_MAX_BATCH: usize = 256;

/// One caller-owned rectangular projection-layout dispatch.
pub struct ExternalAsymmetricProjectionPass<'a> {
    pub q: &'a wgpu::Buffer,
    pub k: &'a wgpu::Buffer,
    pub v: &'a wgpu::Buffer,
    pub out_and_lse: &'a wgpu::Buffer,
    pub shape: AsymmetricGroupedAttentionShape,
    pub config: FlatAttentionConfig,
    pub rotary: AsymmetricRotaryEmbeddingConfig,
}

/// M11 rectangular projection-layout + RoPE + GQA/MQA pipeline.
///
/// The pipeline is reusable and owns only compiled compute pipelines. Every data
/// buffer, command encoder and submission remains caller-owned. M12's
/// [`encode_variable`](Self::encode_variable) reuses the same public pass type:
/// `shape.query_len` / `shape.kv_len` are physical padded extents and `active`
/// supplies one `(active_q_len, active_kv_len, causal_query_offset,
/// query_rope_offset)` tuple per batch element. `pass.rotary.kv_position_offset`
/// remains the shared K/V RoPE origin; `pass.rotary.query_position_offset` and
/// `shape.query_position_offset` are ignored by that variable-length method.
pub struct ExternalAsymmetricProjectionRotaryGroupedPipeline {
    pipeline: wgpu::ComputePipeline,
    variable_pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for ExternalAsymmetricProjectionRotaryGroupedPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalAsymmetricProjectionRotaryGroupedPipeline")
            .finish_non_exhaustive()
    }
}

impl ExternalAsymmetricProjectionRotaryGroupedPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ExternalWgpuError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m11-asymmetric-projection-rope-gqa"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                FLAT_FWD_PROJECTION_ROPE_ASYMMETRIC_WGSL,
            )),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m11-asymmetric-projection-rope-gqa"),
            layout: None,
            module: &shader,
            entry_point: "flat_attention_forward",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let variable_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m12-variable-projection-rope-gqa"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                FLAT_FWD_PROJECTION_ROPE_VARIABLE_WGSL,
            )),
        });
        let variable_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flat-m12-variable-projection-rope-gqa"),
                layout: None,
                module: &variable_shader,
                entry_point: "flat_attention_forward",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        match pollster::block_on(device.pop_error_scope()) {
            Some(error) => Err(ExternalWgpuError::PipelineValidation(error.to_string())),
            None => Ok(Self {
                pipeline,
                variable_pipeline,
            }),
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
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m11-asymmetric-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Record one rectangular pass. No submission or synchronization occurs.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        let dispatch = validate_dispatch(device, pass.shape, pass.rotary)?;
        let layout = Self::layout(pass.shape)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;
        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;

        let params = [
            dispatch.q_len,
            dispatch.kv_len,
            checked_u32(pass.shape.head_dim)?,
            checked_u32(pass.shape.q_heads)?,
            checked_u32(pass.shape.kv_heads)?,
            checked_u32(pass.shape.batch)?,
            u32::from(pass.config.causal),
            scale.to_bits(),
            pass.rotary.theta.to_bits(),
            dispatch.causal_query_offset,
            dispatch.q_rope_offset,
            dispatch.kv_rope_offset,
            0,
            0,
            0,
            0,
        ];
        let params_buffer = uniform_buffer(device, "flat-m11-asymmetric-params", &params);
        let bind_group = projection_bind_group(
            device,
            &self.pipeline,
            "flat-m11-asymmetric-bind-group",
            &pass,
            &params_buffer,
        );

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-m11-asymmetric-projection-rope-gqa"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch.query_workgroups, dispatch.q_batch_heads, 1);
        }

        Ok(layout)
    }

    /// Record one M12 padded variable-length pass.
    ///
    /// `active` must contain exactly one tuple per physical batch element:
    /// `(active_query_len, active_kv_len, causal_query_offset, query_rope_offset)`.
    /// Only active K/V prefixes are staged. Padded query rows are written as
    /// zero O with `LSE = -∞`. The small metadata block is uniform memory, so
    /// the shader still consumes only four storage buffers (Q/K/V/O|LSE).
    /// No submission or synchronization occurs.
    pub fn encode_variable(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
        active: &[(usize, usize, usize, usize)],
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        let dispatch = validate_variable_dispatch(device, pass.shape, pass.rotary, active)?;
        let layout = Self::layout(pass.shape)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;
        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;

        let mut params = Vec::with_capacity(12 + VARIABLE_MAX_BATCH * 4);
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
        for &(active_q_len, active_kv_len, causal_offset, q_rope_offset) in active {
            params.extend_from_slice(&[
                checked_u32(active_q_len)?,
                checked_u32(active_kv_len)?,
                checked_u32(causal_offset)?,
                checked_u32(q_rope_offset)?,
            ]);
        }
        params.resize(12 + VARIABLE_MAX_BATCH * 4, 0);

        let params_buffer = uniform_buffer(device, "flat-m12-variable-params", &params);
        let bind_group = projection_bind_group(
            device,
            &self.variable_pipeline,
            "flat-m12-variable-bind-group",
            &pass,
            &params_buffer,
        );
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-m12-variable-projection-rope-gqa"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.variable_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch.query_workgroups, dispatch.q_batch_heads, 1);
        }
        Ok(layout)
    }
}

fn uniform_buffer(device: &wgpu::Device, label: &str, values: &[u32]) -> wgpu::Buffer {
    let bytes = encode_u32(values);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped.copy_from_slice(&bytes);
    }
    buffer.unmap();
    buffer
}

fn projection_bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    pass: &ExternalAsymmetricProjectionPass<'_>,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
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
    })
}

struct ExternalAsymmetricDispatchGeometry {
    q_batch_heads: u32,
    q_len: u32,
    kv_len: u32,
    query_workgroups: u32,
    causal_query_offset: u32,
    q_rope_offset: u32,
    kv_rope_offset: u32,
}

struct VariableDispatchGeometry {
    q_batch_heads: u32,
    query_workgroups: u32,
}

fn validate_dispatch(
    device: &wgpu::Device,
    shape: AsymmetricGroupedAttentionShape,
    rotary: AsymmetricRotaryEmbeddingConfig,
) -> Result<ExternalAsymmetricDispatchGeometry, ExternalWgpuError> {
    shape.validate()?;
    rotary.validate(shape.head_dim, shape.query_len, shape.kv_len)?;
    validate_common_device_shape(device, shape)?;

    let causal_exclusive = shape
        .query_position_offset
        .checked_add(shape.query_len)
        .ok_or(FlatAttentionError::PositionOverflow)?;
    if causal_exclusive > u32::MAX as usize {
        return Err(FlatAttentionError::PositionOverflow.into());
    }
    let q_rotary_final = rotary
        .query_position_offset
        .checked_add(shape.query_len.saturating_sub(1))
        .ok_or(FlatAttentionError::PositionOverflow)?;
    let kv_rotary_final = rotary
        .kv_position_offset
        .checked_add(shape.kv_len.saturating_sub(1))
        .ok_or(FlatAttentionError::PositionOverflow)?;
    if q_rotary_final > u32::MAX as usize || kv_rotary_final > u32::MAX as usize {
        return Err(FlatAttentionError::PositionOverflow.into());
    }

    let (q_batch_heads, query_workgroups) = dispatch_geometry(device, shape)?;
    Ok(ExternalAsymmetricDispatchGeometry {
        q_batch_heads,
        q_len: checked_u32(shape.query_len)?,
        kv_len: checked_u32(shape.kv_len)?,
        query_workgroups,
        causal_query_offset: checked_u32(shape.query_position_offset)?,
        q_rope_offset: checked_u32(rotary.query_position_offset)?,
        kv_rope_offset: checked_u32(rotary.kv_position_offset)?,
    })
}

fn validate_variable_dispatch(
    device: &wgpu::Device,
    shape: AsymmetricGroupedAttentionShape,
    rotary: AsymmetricRotaryEmbeddingConfig,
    active: &[(usize, usize, usize, usize)],
) -> Result<VariableDispatchGeometry, ExternalWgpuError> {
    shape.validate()?;
    validate_common_device_shape(device, shape)?;
    if shape.batch > VARIABLE_MAX_BATCH {
        return Err(ExternalWgpuError::DispatchLimit {
            axis: "batch_metadata",
            actual: shape.batch,
            maximum: VARIABLE_MAX_BATCH as u32,
        });
    }
    if active.len() != shape.batch {
        return Err(FlatAttentionError::LengthMismatch {
            tensor: "active sequence metadata",
            actual: active.len(),
            expected: shape.batch,
        }
        .into());
    }
    for &(active_q_len, active_kv_len, causal_offset, q_rope_offset) in active {
        if active_q_len == 0 || active_kv_len == 0 {
            return Err(FlatAttentionError::ZeroDimension.into());
        }
        if active_q_len > shape.query_len {
            return Err(FlatAttentionError::LengthMismatch {
                tensor: "active query length",
                actual: active_q_len,
                expected: shape.query_len,
            }
            .into());
        }
        if active_kv_len > shape.kv_len {
            return Err(FlatAttentionError::LengthMismatch {
                tensor: "active KV length",
                actual: active_kv_len,
                expected: shape.kv_len,
            }
            .into());
        }
        let causal_last = causal_offset
            .checked_add(active_q_len - 1)
            .ok_or(FlatAttentionError::PositionOverflow)?;
        let q_rope_last = q_rope_offset
            .checked_add(active_q_len - 1)
            .ok_or(FlatAttentionError::PositionOverflow)?;
        let kv_rope_last = rotary
            .kv_position_offset
            .checked_add(active_kv_len - 1)
            .ok_or(FlatAttentionError::PositionOverflow)?;
        if causal_last > u32::MAX as usize
            || q_rope_last > u32::MAX as usize
            || kv_rope_last > u32::MAX as usize
        {
            return Err(FlatAttentionError::PositionOverflow.into());
        }
        AsymmetricRotaryEmbeddingConfig {
            theta: rotary.theta,
            query_position_offset: q_rope_offset,
            kv_position_offset: rotary.kv_position_offset,
        }
        .validate(shape.head_dim, active_q_len, active_kv_len)?;
    }
    let (q_batch_heads, query_workgroups) = dispatch_geometry(device, shape)?;
    Ok(VariableDispatchGeometry {
        q_batch_heads,
        query_workgroups,
    })
}

fn validate_common_device_shape(
    _device: &wgpu::Device,
    shape: AsymmetricGroupedAttentionShape,
) -> Result<(), ExternalWgpuError> {
    if shape.head_dim > WGSL_MAX_HEAD_DIM {
        return Err(ExternalWgpuError::UnsupportedHeadDim {
            actual: shape.head_dim,
            maximum: WGSL_MAX_HEAD_DIM,
        });
    }
    let layout = ExternalAsymmetricProjectionRotaryGroupedPipeline::layout(shape)?;
    if layout.combined_elements > u32::MAX as usize || layout.kv_elements > u32::MAX as usize {
        return Err(ExternalWgpuError::IndexSpaceExceeded {
            elements: layout.combined_elements.max(layout.kv_elements),
        });
    }
    Ok(())
}

fn dispatch_geometry(
    device: &wgpu::Device,
    shape: AsymmetricGroupedAttentionShape,
) -> Result<(u32, u32), ExternalWgpuError> {
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
    Ok((checked_u32(q_batch_heads)?, checked_u32(query_workgroups)?))
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
