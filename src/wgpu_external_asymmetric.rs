//! Caller-owned WGPU encoding for rectangular projection-layout attention.
//!
//! This is the M11 companion to the equal-length FLAT-R2 external pipeline. It
//! records Q/K/V attention with independent query/KV lengths into a caller-owned
//! command encoder and never submits, polls, maps or copies framework buffers.

use super::wgpu_internal;

use core::fmt;

use super::{
    AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig, ExternalProjectionLayout,
    ExternalWgpuError, FlatAttentionConfig, FlatAttentionError,
    FLAT_FWD_PROJECTION_ROPE_ASYMMETRIC_VEC4_WGSL, FLAT_FWD_PROJECTION_ROPE_ASYMMETRIC_WGSL,
    WGSL_MAX_HEAD_DIM, WGSL_QUERY_ROWS,
};

const FLAT_DECODE_PROJECTION_KV_REUSE_WGSL: &str =
    include_str!("../shaders/flat_decode_projection_kv_reuse.wgsl");

/// Maximum query-head count carried by the portable M13 ALiBi uniform block.
pub const WGSL_ALIBI_MAX_HEADS: usize = 256;

/// One caller-owned rectangular projection-layout dispatch.
pub struct ExternalAsymmetricProjectionPass<'a> {
    /// Sequence-major projected query tensor [batch, query_len, q_heads * head_dim].
    pub q: &'a wgpu::Buffer,
    /// Sequence-major projected key tensor (raw; rotation is fused).
    pub k: &'a wgpu::Buffer,
    /// Sequence-major projected value tensor.
    pub v: &'a wgpu::Buffer,
    /// Destination for packed [O | LSE]; must declare STORAGE usage.
    pub out_and_lse: &'a wgpu::Buffer,
    /// Rectangular grouped geometry with the causal position domain.
    pub shape: AsymmetricGroupedAttentionShape,
    /// Attention configuration (causality and softmax scale).
    pub config: FlatAttentionConfig,
    /// RoPE parameters with independent Q and KV rotation domains.
    pub rotary: AsymmetricRotaryEmbeddingConfig,
}

/// Kernel selected for one rectangular external projection pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalAsymmetricKernelVariant {
    /// Existing scalar-storage portable M11/M13/M15 kernel.
    Portable,
    /// M53 vec4 Q/K/V storage loads for head dimensions 64 and 128.
    Vec4,
}

/// M11 rectangular projection-layout + RoPE + GQA/MQA pipeline.
///
/// The pipeline is reusable and owns only the compiled compute pipelines. Every
/// data buffer, command encoder and submission remains caller-owned.
pub struct ExternalAsymmetricProjectionRotaryGroupedPipeline {
    portable_pipeline: wgpu::ComputePipeline,
    vec4_pipeline: Option<wgpu::ComputePipeline>,
    decode_kv_reuse_pipeline: Option<wgpu::ComputePipeline>,
}

impl fmt::Debug for ExternalAsymmetricProjectionRotaryGroupedPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalAsymmetricProjectionRotaryGroupedPipeline")
            .field("vec4_pipeline", &self.vec4_pipeline.is_some())
            .field(
                "decode_kv_reuse_pipeline",
                &self.decode_kv_reuse_pipeline.is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl ExternalAsymmetricProjectionRotaryGroupedPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ExternalWgpuError> {
        Self::with_candidates(device, false, false)
    }

    /// Compile the M53 vec4 candidate in addition to the unchanged portable
    /// kernel. Unsupported head dimensions continue to use the portable path.
    pub fn with_vectorization(
        device: &wgpu::Device,
        enabled: bool,
    ) -> Result<Self, ExternalWgpuError> {
        Self::with_candidates(device, enabled, false)
    }

    /// Build the baseline pipeline and optionally compile the isolated M48
    /// q_len=1 GQA K/V tile-reuse candidate. Existing encode methods retain
    /// their baseline routing when the candidate is compiled.
    pub fn with_decode_kv_reuse(
        device: &wgpu::Device,
        enable_candidate: bool,
    ) -> Result<Self, ExternalWgpuError> {
        Self::with_candidates(device, false, enable_candidate)
    }

    fn with_candidates(
        device: &wgpu::Device,
        enable_vectorization: bool,
        enable_decode_kv_reuse: bool,
    ) -> Result<Self, ExternalWgpuError> {
        let portable_pipeline = create_pipeline(
            device,
            FLAT_FWD_PROJECTION_ROPE_ASYMMETRIC_WGSL,
            "flat-m11-asymmetric-projection-rope-gqa",
            "flat_attention_forward",
        )?;
        let vec4_pipeline = if enable_vectorization {
            Some(create_pipeline(
                device,
                FLAT_FWD_PROJECTION_ROPE_ASYMMETRIC_VEC4_WGSL,
                "flat-m53-asymmetric-projection-rope-gqa-vec4",
                "flat_attention_forward",
            )?)
        } else {
            None
        };
        let decode_kv_reuse_pipeline = if enable_decode_kv_reuse {
            Some(create_pipeline(
                device,
                FLAT_DECODE_PROJECTION_KV_REUSE_WGSL,
                "flat-m48-decode-projection-kv-reuse",
                "flat_attention_decode_kv_reuse",
            )?)
        } else {
            None
        };
        Ok(Self {
            portable_pipeline,
            vec4_pipeline,
            decode_kv_reuse_pipeline,
        })
    }

    #[must_use]
    pub fn vectorization_enabled(&self) -> bool {
        self.vec4_pipeline.is_some()
    }

    #[must_use]
    pub fn kernel_variant_for_shape(
        &self,
        shape: AsymmetricGroupedAttentionShape,
    ) -> ExternalAsymmetricKernelVariant {
        if self.vec4_pipeline.is_some() && matches!(shape.head_dim, 64 | 128) {
            ExternalAsymmetricKernelVariant::Vec4
        } else {
            ExternalAsymmetricKernelVariant::Portable
        }
    }

    fn selected_pipeline(&self, shape: AsymmetricGroupedAttentionShape) -> &wgpu::ComputePipeline {
        match self.kernel_variant_for_shape(shape) {
            ExternalAsymmetricKernelVariant::Portable => &self.portable_pipeline,
            ExternalAsymmetricKernelVariant::Vec4 => self
                .vec4_pipeline
                .as_ref()
                .expect("M53 vec4 variant is selected only when compiled"),
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
        let maximum_bytes = device.limits().max_storage_buffer_binding_size;
        if layout.combined_bytes > maximum_bytes {
            return Err(ExternalWgpuError::DeviceBufferLimit {
                required_bytes: layout.combined_bytes,
                maximum_bytes,
            });
        }
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m11-asymmetric-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Record one rectangular pass using raw projected K. Q and K are both
    /// RoPE-rotated inside the fused kernel. No submission or synchronization
    /// occurs.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        self.encode_with_options(device, encoder, pass, true, None, false)
    }

    /// Record one rectangular pass where K is **already RoPE-rotated** by the
    /// resident cache owner. Q RoPE remains fused; K is consumed as-is.
    pub fn encode_pre_rotated_k(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        self.encode_with_options(device, encoder, pass, false, None, false)
    }

    /// Record the opt-in M48 q_len=1 decode candidate using pre-rotated K.
    /// Each workgroup reuses one physical K/V tile across up to four query
    /// heads from the same GQA group. This records only; it never submits,
    /// polls, maps, copies or changes the baseline route.
    pub fn encode_pre_rotated_k_decode_kv_reuse(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        if self.decode_kv_reuse_pipeline.is_none() {
            return Err(ExternalWgpuError::CandidateNotEnabled { candidate: "M48" });
        }
        pass.shape.validate()?;
        if pass.shape.query_len != 1 {
            return Err(ExternalWgpuError::UnsupportedCandidateShape {
                candidate: "M48",
                reason: "query_len must equal 1",
            });
        }
        let group_size = pass.shape.q_heads / pass.shape.kv_heads;
        if group_size < 2 {
            return Err(ExternalWgpuError::UnsupportedCandidateShape {
                candidate: "M48",
                reason: "GQA group size must be at least 2",
            });
        }
        self.encode_with_options(device, encoder, pass, false, None, true)
    }

    /// Record one M13 ALiBi pass while preserving the four-storage-buffer
    /// portable contract. Per-query-head slopes are packed into the existing
    /// uniform binding; Q/K/V/O remain caller-owned resident buffers.
    pub fn encode_alibi(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
        slopes: &[f32],
        query_position_offset: usize,
        kv_position_offset: usize,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        self.encode_with_options(
            device,
            encoder,
            pass,
            true,
            Some((slopes, query_position_offset, kv_position_offset)),
            false,
        )
    }

    /// ALiBi variant for a resident cache whose K rows are already RoPE-rotated.
    pub fn encode_pre_rotated_k_alibi(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
        slopes: &[f32],
        query_position_offset: usize,
        kv_position_offset: usize,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        self.encode_with_options(
            device,
            encoder,
            pass,
            false,
            Some((slopes, query_position_offset, kv_position_offset)),
            false,
        )
    }

    fn encode_with_options(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
        rotate_k: bool,
        alibi: Option<(&[f32], usize, usize)>,
        decode_kv_reuse: bool,
    ) -> Result<ExternalProjectionLayout, ExternalWgpuError> {
        let dispatch = validate_dispatch(device, pass.shape, pass.rotary)?;
        let layout = Self::layout(pass.shape)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, layout.kv_bytes)?;
        validate_buffer("V", pass.v, layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;
        let scale = pass.config.resolved_scale(pass.shape.head_dim)?;

        let (bias_mode, bias_q_offset, bias_kv_offset, slopes) = match alibi {
            Some((slopes, query_position_offset, kv_position_offset)) => {
                validate_alibi(
                    pass.shape,
                    slopes,
                    query_position_offset,
                    kv_position_offset,
                )?;
                (
                    1u32,
                    checked_u32(query_position_offset)?,
                    checked_u32(kv_position_offset)?,
                    slopes,
                )
            }
            None => (0u32, 0u32, 0u32, &[][..]),
        };

        let mut params = Vec::with_capacity(16 + WGSL_ALIBI_MAX_HEADS);
        params.extend_from_slice(&[
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
            u32::from(rotate_k),
            bias_mode,
            bias_q_offset,
            bias_kv_offset,
        ]);
        params.extend(slopes.iter().map(|slope| slope.to_bits()));
        params.resize(16 + WGSL_ALIBI_MAX_HEADS, 0);

        let params_bytes = encode_u32(&params);
        let params_buffer = wgpu_internal::create_uniform_buffer_init(
            device,
            "flat-m13-asymmetric-params",
            &params_bytes,
        );

        let pipeline = if decode_kv_reuse {
            self.decode_kv_reuse_pipeline
                .as_ref()
                .ok_or(ExternalWgpuError::CandidateNotEnabled { candidate: "M48" })?
        } else {
            self.selected_pipeline(pass.shape)
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-m13-asymmetric-bind-group"),
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
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-m13-asymmetric-projection-rope-gqa"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let dispatch_y = if decode_kv_reuse {
                m48_dispatch_workgroups_y(device, pass.shape)?
            } else {
                dispatch.q_batch_heads
            };
            compute_pass.dispatch_workgroups(dispatch.query_workgroups, dispatch_y, 1);
        }

        Ok(layout)
    }
}

fn m48_dispatch_workgroups_y(
    device: &wgpu::Device,
    shape: AsymmetricGroupedAttentionShape,
) -> Result<u32, ExternalWgpuError> {
    const Q_HEADS_PER_WORKGROUP: usize = 4;
    let group_size = shape.q_heads / shape.kv_heads;
    let tiles_per_kv_head = group_size.div_ceil(Q_HEADS_PER_WORKGROUP);
    let workgroups = shape
        .batch
        .checked_mul(shape.kv_heads)
        .and_then(|value| value.checked_mul(tiles_per_kv_head))
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let maximum = device.limits().max_compute_workgroups_per_dimension;
    if workgroups > maximum as usize {
        return Err(ExternalWgpuError::DispatchLimit {
            axis: "y/batch_kv_head_tiles",
            actual: workgroups,
            maximum,
        });
    }
    checked_u32(workgroups)
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

fn validate_dispatch(
    device: &wgpu::Device,
    shape: AsymmetricGroupedAttentionShape,
    rotary: AsymmetricRotaryEmbeddingConfig,
) -> Result<ExternalAsymmetricDispatchGeometry, ExternalWgpuError> {
    shape.validate()?;
    rotary.validate(shape.head_dim, shape.query_len, shape.kv_len)?;
    if shape.head_dim > WGSL_MAX_HEAD_DIM {
        return Err(ExternalWgpuError::UnsupportedHeadDim {
            actual: shape.head_dim,
            maximum: WGSL_MAX_HEAD_DIM,
        });
    }

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

    let layout = ExternalAsymmetricProjectionRotaryGroupedPipeline::layout(shape)?;
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

    Ok(ExternalAsymmetricDispatchGeometry {
        q_batch_heads: checked_u32(q_batch_heads)?,
        q_len: checked_u32(shape.query_len)?,
        kv_len: checked_u32(shape.kv_len)?,
        query_workgroups: checked_u32(query_workgroups)?,
        causal_query_offset: checked_u32(shape.query_position_offset)?,
        q_rope_offset: checked_u32(rotary.query_position_offset)?,
        kv_rope_offset: checked_u32(rotary.kv_position_offset)?,
    })
}

fn validate_alibi(
    shape: AsymmetricGroupedAttentionShape,
    slopes: &[f32],
    query_position_offset: usize,
    kv_position_offset: usize,
) -> Result<(), ExternalWgpuError> {
    if slopes.len() != shape.q_heads {
        return Err(FlatAttentionError::LengthMismatch {
            tensor: "ALiBi slopes",
            actual: slopes.len(),
            expected: shape.q_heads,
        }
        .into());
    }
    if let Some(index) = slopes.iter().position(|slope| !slope.is_finite()) {
        return Err(FlatAttentionError::NonFiniteInput {
            tensor: "ALiBi slopes",
            index,
        }
        .into());
    }
    if shape.q_heads > WGSL_ALIBI_MAX_HEADS {
        return Err(ExternalWgpuError::IndexSpaceExceeded {
            elements: shape.q_heads,
        });
    }
    let q_final = query_position_offset
        .checked_add(shape.query_len.saturating_sub(1))
        .ok_or(FlatAttentionError::PositionOverflow)?;
    let kv_final = kv_position_offset
        .checked_add(shape.kv_len.saturating_sub(1))
        .ok_or(FlatAttentionError::PositionOverflow)?;
    if q_final > u32::MAX as usize || kv_final > u32::MAX as usize {
        return Err(FlatAttentionError::PositionOverflow.into());
    }
    Ok(())
}

fn create_pipeline(
    device: &wgpu::Device,
    source: &'static str,
    label: &'static str,
    entry_point: &'static str,
) -> Result<wgpu::ComputePipeline, ExternalWgpuError> {
    wgpu_internal::create_pipeline(device, source, label, entry_point)
        .map_err(ExternalWgpuError::PipelineValidation)
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
