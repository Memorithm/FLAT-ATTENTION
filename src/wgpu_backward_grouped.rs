//! Public caller-owned WGPU host path for the qualified M19 GQA/MQA backward shader.
//!
//! The pipeline owns compiled GPU state only. Callers retain ownership of data
//! buffers, command encoders, queue submission, synchronization and readback.
//! Q/dQ use query-head cardinality; K/V/dK/dV remain at native KV-head
//! cardinality, including MQA.

use core::fmt;

use crate::{
    validate_input, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
    GroupedAttentionShape, FLAT_BACKWARD_GROUPED_RECOMPUTE_WGSL,
};

const BACKWARD_WORKGROUP_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedBackwardRecomputeLayout {
    pub q_elements: usize,
    pub kv_elements: usize,
    pub lse_elements: usize,
    pub packed_forward_elements: usize,
    pub gradient_elements: usize,
    pub packed_forward_bytes: u64,
    pub gradient_bytes: u64,
}

impl GroupedBackwardRecomputeLayout {
    pub const fn q_offset(self) -> usize {
        0
    }
    pub const fn k_offset(self) -> usize {
        self.q_elements
    }
    pub const fn v_offset(self) -> usize {
        self.q_elements + self.kv_elements
    }
    pub const fn d_out_offset(self) -> usize {
        self.q_elements + 2 * self.kv_elements
    }
    pub const fn output_offset(self) -> usize {
        2 * self.q_elements + 2 * self.kv_elements
    }
    pub const fn lse_offset(self) -> usize {
        3 * self.q_elements + 2 * self.kv_elements
    }
    pub const fn dq_offset(self) -> usize {
        0
    }
    pub const fn dk_offset(self) -> usize {
        self.q_elements
    }
    pub const fn dv_offset(self) -> usize {
        self.q_elements + self.kv_elements
    }
}

pub struct GroupedBackwardRecomputePass<'a> {
    pub packed_forward: &'a wgpu::Buffer,
    pub packed_grads: &'a wgpu::Buffer,
    pub shape: GroupedAttentionShape,
    pub config: FlatAttentionConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GroupedBackwardRecomputeError {
    Core(FlatAttentionError),
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
    PipelineValidation(String),
}

impl fmt::Display for GroupedBackwardRecomputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::IndexSpaceExceeded { elements } => write!(
                f,
                "grouped backward recomputation exceeds WGPU u32 index space at {elements} elements"
            ),
            Self::DispatchLimit { actual, maximum } => write!(
                f,
                "grouped backward recomputation requires {actual} workgroups, device maximum is {maximum}"
            ),
            Self::BufferTooSmall {
                tensor,
                actual_bytes,
                required_bytes,
            } => write!(
                f,
                "buffer {tensor} contains {actual_bytes} bytes, requires at least {required_bytes}"
            ),
            Self::PipelineValidation(error) => write!(
                f,
                "grouped backward recomputation pipeline validation failed: {error}"
            ),
        }
    }
}

impl std::error::Error for GroupedBackwardRecomputeError {}

impl From<FlatAttentionError> for GroupedBackwardRecomputeError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

pub struct WgpuGroupedBackwardRecomputePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for WgpuGroupedBackwardRecomputePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuGroupedBackwardRecomputePipeline")
            .finish_non_exhaustive()
    }
}

impl WgpuGroupedBackwardRecomputePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, GroupedBackwardRecomputeError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m19-grouped-backward-recompute"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                FLAT_BACKWARD_GROUPED_RECOMPUTE_WGSL,
            )),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m19-grouped-backward-recompute"),
            layout: None,
            module: &shader,
            entry_point: "flat_attention_backward_grouped",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        match pollster::block_on(device.pop_error_scope()) {
            Some(error) => Err(GroupedBackwardRecomputeError::PipelineValidation(
                error.to_string(),
            )),
            None => Ok(Self { pipeline }),
        }
    }

    pub fn layout(
        shape: GroupedAttentionShape,
    ) -> Result<GroupedBackwardRecomputeLayout, GroupedBackwardRecomputeError> {
        shape.validate()?;
        let q_elements = shape.q_tensor_len()?;
        let kv_elements = shape.kv_tensor_len()?;
        let lse_elements = shape.lse_len()?;
        let packed_forward_elements = q_elements
            .checked_mul(3)
            .and_then(|value| {
                kv_elements
                    .checked_mul(2)
                    .and_then(|kv| value.checked_add(kv))
            })
            .and_then(|value| value.checked_add(lse_elements))
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let gradient_elements = q_elements
            .checked_add(
                kv_elements
                    .checked_mul(2)
                    .ok_or(FlatAttentionError::ShapeOverflow)?,
            )
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        checked_u32(packed_forward_elements)?;
        checked_u32(gradient_elements)?;
        Ok(GroupedBackwardRecomputeLayout {
            q_elements,
            kv_elements,
            lse_elements,
            packed_forward_elements,
            gradient_elements,
            packed_forward_bytes: bytes_for_f32(packed_forward_elements)?,
            gradient_bytes: bytes_for_f32(gradient_elements)?,
        })
    }

    pub fn create_gradient_buffer(
        &self,
        device: &wgpu::Device,
        shape: GroupedAttentionShape,
    ) -> Result<wgpu::Buffer, GroupedBackwardRecomputeError> {
        let layout = Self::layout(shape)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m19-grouped-backward-gradients"),
            size: layout.gradient_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: GroupedBackwardRecomputePass<'_>,
    ) -> Result<GroupedBackwardRecomputeLayout, GroupedBackwardRecomputeError> {
        let layout = Self::layout(pass.shape)?;
        validate_buffer(
            "Q|K|V|dO|O|LSE",
            pass.packed_forward,
            layout.packed_forward_bytes,
        )?;
        validate_buffer("dQ|dK|dV", pass.packed_grads, layout.gradient_bytes)?;
        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;
        let workgroups = layout.gradient_elements.div_ceil(BACKWARD_WORKGROUP_SIZE);
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        if workgroups > maximum as usize {
            return Err(GroupedBackwardRecomputeError::DispatchLimit {
                actual: workgroups,
                maximum,
            });
        }
        let params = [
            checked_u32(pass.shape.batch)?,
            checked_u32(pass.shape.q_heads)?,
            checked_u32(pass.shape.kv_heads)?,
            checked_u32(pass.shape.seq_len)?,
            checked_u32(pass.shape.head_dim)?,
            u32::from(pass.config.causal),
            scale.to_bits(),
            checked_u32(layout.q_elements)?,
            checked_u32(layout.kv_elements)?,
            checked_u32(layout.lse_elements)?,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m19-grouped-backward-params"),
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
            label: Some("flat-m19-grouped-backward-bind-group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pass.packed_forward.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pass.packed_grads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-m19-grouped-backward-recompute"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(checked_u32(workgroups)?, 1, 1);
        drop(compute_pass);
        Ok(layout)
    }
}

/// Convenience packer for qualification and simple callers. Performance-sensitive
/// resident integrations may populate the equivalent GPU buffer directly.
pub fn pack_grouped_backward_recompute_inputs(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    forward: &FlatAttentionOutput,
    shape: GroupedAttentionShape,
) -> Result<Vec<f32>, GroupedBackwardRecomputeError> {
    shape.validate()?;
    let q_elements = shape.q_tensor_len()?;
    let kv_elements = shape.kv_tensor_len()?;
    let lse_elements = shape.lse_len()?;
    validate_input("Q", q, q_elements)?;
    validate_input("K", k, kv_elements)?;
    validate_input("V", v, kv_elements)?;
    validate_input("dO", d_out, q_elements)?;
    validate_input("O", &forward.output, q_elements)?;
    validate_input("LSE", &forward.lse, lse_elements)?;
    let layout = WgpuGroupedBackwardRecomputePipeline::layout(shape)?;
    let mut packed = Vec::with_capacity(layout.packed_forward_elements);
    packed.extend_from_slice(q);
    packed.extend_from_slice(k);
    packed.extend_from_slice(v);
    packed.extend_from_slice(d_out);
    packed.extend_from_slice(&forward.output);
    packed.extend_from_slice(&forward.lse);
    Ok(packed)
}

fn validate_buffer(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), GroupedBackwardRecomputeError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(GroupedBackwardRecomputeError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, GroupedBackwardRecomputeError> {
    u32::try_from(value)
        .map_err(|_| GroupedBackwardRecomputeError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32(elements: usize) -> Result<u64, GroupedBackwardRecomputeError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| GroupedBackwardRecomputeError::IndexSpaceExceeded { elements })
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}
