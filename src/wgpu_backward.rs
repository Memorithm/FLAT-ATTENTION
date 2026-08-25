//! Public caller-owned WGPU host path for the qualified M18 recomputation shader.
//!
//! The pipeline owns only compiled GPU state. Callers own the packed forward
//! buffer, gradient buffer, command encoder, queue, submission, synchronization,
//! and readback policy. No hot-path allocation or submission is hidden here.

use core::fmt;

use super::wgpu_internal;

use crate::{
    validate_input, AttentionShape, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
    FLAT_BACKWARD_RECOMPUTE_WGSL,
};

const BACKWARD_WORKGROUP_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackwardRecomputeLayout {
    /// Element count of one forward tensor (Q or O).
    pub tensor_elements: usize,
    /// Element count of the LSE statistic vector.
    pub lse_elements: usize,
    /// Element count of the packed [Q|K|V|O|LSE] record.
    pub packed_forward_elements: usize,
    /// Element count of the packed [dQ|dK|dV] record.
    pub gradient_elements: usize,
    /// Byte size of the packed forward record.
    pub packed_forward_bytes: u64,
    /// Byte size of the packed gradient record.
    pub gradient_bytes: u64,
}

impl BackwardRecomputeLayout {
    pub const fn q_offset(self) -> usize {
        0
    }

    pub const fn k_offset(self) -> usize {
        self.tensor_elements
    }

    pub const fn v_offset(self) -> usize {
        2 * self.tensor_elements
    }

    pub const fn d_out_offset(self) -> usize {
        3 * self.tensor_elements
    }

    pub const fn output_offset(self) -> usize {
        4 * self.tensor_elements
    }

    pub const fn lse_offset(self) -> usize {
        5 * self.tensor_elements
    }

    pub const fn dq_offset(self) -> usize {
        0
    }

    pub const fn dk_offset(self) -> usize {
        self.tensor_elements
    }

    pub const fn dv_offset(self) -> usize {
        2 * self.tensor_elements
    }
}

pub struct BackwardRecomputePass<'a> {
    /// Packed [Q|K|V|O|LSE] forward record produced by the forward pass.
    pub packed_forward: &'a wgpu::Buffer,
    /// Destination for packed [dQ|dK|dV]; must declare STORAGE usage.
    pub packed_grads: &'a wgpu::Buffer,
    /// Canonical geometry of the recorded forward pass.
    pub shape: AttentionShape,
    /// Attention configuration used by the forward pass.
    pub config: FlatAttentionConfig,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BackwardRecomputeError {
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

impl fmt::Display for BackwardRecomputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::IndexSpaceExceeded { elements } => write!(
                f,
                "backward recomputation exceeds WGPU u32 index space at {elements} elements"
            ),
            Self::DispatchLimit { actual, maximum } => write!(
                f,
                "backward recomputation requires {actual} workgroups, device maximum is {maximum}"
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
                write!(
                    f,
                    "backward recomputation pipeline validation failed: {error}"
                )
            }
        }
    }
}

impl std::error::Error for BackwardRecomputeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FlatAttentionError> for BackwardRecomputeError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

pub struct WgpuBackwardRecomputePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for WgpuBackwardRecomputePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuBackwardRecomputePipeline")
            .finish_non_exhaustive()
    }
}

impl WgpuBackwardRecomputePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, BackwardRecomputeError> {
        let pipeline = wgpu_internal::create_pipeline(
            device,
            FLAT_BACKWARD_RECOMPUTE_WGSL,
            "flat-m18-backward-recompute",
            "flat_attention_backward",
        )
        .map_err(BackwardRecomputeError::PipelineValidation)?;
        Ok(Self { pipeline })
    }

    pub fn layout(
        shape: AttentionShape,
    ) -> Result<BackwardRecomputeLayout, BackwardRecomputeError> {
        shape.validate()?;
        let tensor_elements = shape.tensor_len()?;
        let lse_elements = shape.lse_len()?;
        let packed_forward_elements = tensor_elements
            .checked_mul(5)
            .and_then(|value| value.checked_add(lse_elements))
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let gradient_elements = tensor_elements
            .checked_mul(3)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        checked_u32(packed_forward_elements)?;
        checked_u32(gradient_elements)?;
        Ok(BackwardRecomputeLayout {
            tensor_elements,
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
        shape: AttentionShape,
    ) -> Result<wgpu::Buffer, BackwardRecomputeError> {
        let layout = Self::layout(shape)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m18-backward-gradients"),
            size: layout.gradient_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: BackwardRecomputePass<'_>,
    ) -> Result<BackwardRecomputeLayout, BackwardRecomputeError> {
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
            return Err(BackwardRecomputeError::DispatchLimit {
                actual: workgroups,
                maximum,
            });
        }

        let params = [
            checked_u32(pass.shape.batch)?,
            checked_u32(pass.shape.heads)?,
            checked_u32(pass.shape.seq_len)?,
            checked_u32(pass.shape.head_dim)?,
            u32::from(pass.config.causal),
            scale.to_bits(),
            checked_u32(layout.tensor_elements)?,
            checked_u32(layout.lse_elements)?,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = wgpu_internal::create_uniform_buffer_init(
            device,
            "flat-m18-backward-params",
            &params_bytes,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-m18-backward-bind-group"),
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
            label: Some("flat-m18-backward-recompute"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(checked_u32(workgroups)?, 1, 1);
        drop(compute_pass);
        Ok(layout)
    }
}

/// Convenience packer for qualification and simple callers.
///
/// Performance-sensitive integrations may populate an equivalent resident GPU
/// buffer directly and avoid this host allocation/copy.
pub fn pack_backward_recompute_inputs(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    forward: &FlatAttentionOutput,
    shape: AttentionShape,
) -> Result<Vec<f32>, BackwardRecomputeError> {
    shape.validate()?;
    let tensor_elements = shape.tensor_len()?;
    let lse_elements = shape.lse_len()?;
    validate_input("Q", q, tensor_elements)?;
    validate_input("K", k, tensor_elements)?;
    validate_input("V", v, tensor_elements)?;
    validate_input("dO", d_out, tensor_elements)?;
    validate_input("O", &forward.output, tensor_elements)?;
    validate_input("LSE", &forward.lse, lse_elements)?;
    let layout = WgpuBackwardRecomputePipeline::layout(shape)?;
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
) -> Result<(), BackwardRecomputeError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(BackwardRecomputeError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, BackwardRecomputeError> {
    u32::try_from(value).map_err(|_| BackwardRecomputeError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32(elements: usize) -> Result<u64, BackwardRecomputeError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| BackwardRecomputeError::IndexSpaceExceeded { elements })
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}
