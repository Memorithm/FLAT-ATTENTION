//! M15 specialized caller-owned decode pipeline over resident K/V storage.
//!
//! The pipeline records one `q_len = 1` attention dispatch into a caller-owned
//! command encoder. K/V may come from FLAT's [`WgpuResidentKvCache`] or directly
//! from framework-owned fixed-capacity buffers through
//! [`ExternalAsymmetricProjectionPass`]. Physical capacity remains the batch
//! stride in both cases. No cache compaction, host round-trip, submission,
//! polling or mapping occurs here.

use super::wgpu_internal;

use core::fmt;

use super::{
    AsymmetricGroupedAttentionShape, ExternalAsymmetricProjectionPass, FlatAttentionConfig,
    FlatAttentionError, WgpuResidentKvCache, FLAT_DECODE_RESIDENT_WGSL, WGSL_MAX_HEAD_DIM,
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

/// Internal non-owning description of fixed-capacity resident K/V storage.
///
/// Physical layout is `[batch, capacity, kv_heads * head_dim]`. Only rows
/// `[0, len)` in each batch are logically live. K is already RoPE-rotated and V
/// remains raw.
#[derive(Clone, Copy)]
struct ResidentKvView<'a> {
    k: &'a wgpu::Buffer,
    v: &'a wgpu::Buffer,
    batch: usize,
    kv_heads: usize,
    capacity: usize,
    head_dim: usize,
    len: usize,
}

impl ResidentKvView<'_> {
    #[allow(clippy::too_many_arguments)]
    fn new<'a>(
        k: &'a wgpu::Buffer,
        v: &'a wgpu::Buffer,
        batch: usize,
        kv_heads: usize,
        capacity: usize,
        head_dim: usize,
        len: usize,
    ) -> Result<ResidentKvView<'a>, ResidentDecodeError> {
        let view = ResidentKvView {
            k,
            v,
            batch,
            kv_heads,
            capacity,
            head_dim,
            len,
        };
        validate_kv_view(view)?;
        Ok(view)
    }

    fn from_cache(cache: &WgpuResidentKvCache) -> ResidentKvView<'_> {
        ResidentKvView {
            k: cache.k_buffer(),
            v: cache.v_buffer(),
            batch: cache.batch(),
            kv_heads: cache.kv_heads(),
            capacity: cache.capacity(),
            head_dim: cache.head_dim(),
            len: cache.len(),
        }
    }
}

/// One decode dispatch backed by FLAT's M14 cache owner.
pub struct ResidentDecodePass<'a> {
    pub q: &'a wgpu::Buffer,
    pub out_and_lse: &'a wgpu::Buffer,
    pub cache: &'a WgpuResidentKvCache,
    pub q_heads: usize,
    pub config: FlatAttentionConfig,
    pub theta: f32,
    /// Absolute RoPE position of the single query row.
    ///
    /// This drives only the fused query rotation. Causal visibility is judged
    /// from [`Self::q_causal_position`], keeping the rotation domain
    /// independent of the causal domain exactly like the asymmetric oracle.
    pub q_rope_position: usize,
    /// Absolute causal position of the single query row.
    ///
    /// Under `config.causal`, the kernel requires
    /// `q_causal_position + 1 >= live_tokens`; RoPE and causal origins may
    /// differ (cross-attention, offset rope schedules).
    pub q_causal_position: usize,
}

struct ExternalResidentDecodePass<'a> {
    q: &'a wgpu::Buffer,
    out_and_lse: &'a wgpu::Buffer,
    kv: ResidentKvView<'a>,
    q_heads: usize,
    config: FlatAttentionConfig,
    theta: f32,
    q_rope_position: usize,
    q_causal_position: usize,
}

/// Explicit M15 decode failures.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ResidentDecodeError {
    Core(FlatAttentionError),
    EmptyCache,
    InvalidCacheLength {
        len: usize,
        capacity: usize,
    },
    InvalidQueryLen {
        actual: usize,
    },
    CausalVisibilityMismatch {
        query_position: usize,
        kv_len: usize,
    },
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
    StorageBindingTooLarge {
        tensor: &'static str,
        required_bytes: u64,
        maximum_bytes: u64,
    },
    InvalidTheta(f32),
    PipelineValidation(String),
}

impl fmt::Display for ResidentDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::EmptyCache => write!(f, "resident decode requires at least one live KV row"),
            Self::InvalidCacheLength { len, capacity } => write!(
                f,
                "resident decode live KV length {len} exceeds physical capacity {capacity}"
            ),
            Self::InvalidQueryLen { actual } => {
                write!(f, "specialized resident decode requires query_len=1, got {actual}")
            }
            Self::CausalVisibilityMismatch {
                query_position,
                kv_len,
            } => write!(
                f,
                "resident causal decode query position {query_position} cannot see all {kv_len} live KV rows"
            ),
            Self::UnsupportedHeadDim { actual, maximum } => {
                write!(f, "head_dim {actual} exceeds portable maximum {maximum}")
            }
            Self::IndexSpaceExceeded { elements } => {
                write!(
                    f,
                    "resident decode exceeds WGPU u32 index space at {elements} elements"
                )
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
            Self::StorageBindingTooLarge {
                tensor,
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "storage binding {tensor} requires {required_bytes} bytes, device maximum is {maximum_bytes}"
            ),
            Self::InvalidTheta(theta) => {
                write!(
                    f,
                    "resident decode RoPE theta must be finite and positive, got {theta}"
                )
            }
            Self::PipelineValidation(error) => {
                write!(f, "resident decode pipeline validation failed: {error}")
            }
        }
    }
}

impl std::error::Error for ResidentDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            _ => None,
        }
    }
}

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
        let pipeline = wgpu_internal::create_pipeline(
            device,
            FLAT_DECODE_RESIDENT_WGSL,
            "flat-m15-resident-decode",
            "flat_attention_decode",
        )
        .map_err(ResidentDecodeError::PipelineValidation)?;
        Ok(Self { pipeline })
    }

    pub fn layout(
        cache: &WgpuResidentKvCache,
        q_heads: usize,
    ) -> Result<ResidentDecodeLayout, ResidentDecodeError> {
        layout_for_view(ResidentKvView::from_cache(cache), q_heads)
    }

    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        cache: &WgpuResidentKvCache,
        q_heads: usize,
    ) -> Result<wgpu::Buffer, ResidentDecodeError> {
        let layout = Self::layout(cache, q_heads)?;
        Ok(create_output_buffer(device, layout))
    }

    /// Record one q_len=1 decode dispatch over FLAT's live cache prefix.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ResidentDecodePass<'_>,
    ) -> Result<ResidentDecodeLayout, ResidentDecodeError> {
        self.encode_view(
            device,
            encoder,
            ExternalResidentDecodePass {
                q: pass.q,
                out_and_lse: pass.out_and_lse,
                kv: ResidentKvView::from_cache(pass.cache),
                q_heads: pass.q_heads,
                config: pass.config,
                theta: pass.theta,
                q_rope_position: pass.q_rope_position,
                q_causal_position: pass.q_causal_position,
            },
        )
    }

    /// Record the specialized q_len=1 kernel directly over framework-owned
    /// fixed-capacity K/V buffers.
    ///
    /// `pass.k` must already contain RoPE-rotated K in physical layout
    /// `[batch, kv_capacity, kv_heads * head_dim]`; `pass.v` remains raw. The
    /// logical live length is `pass.shape.kv_len`. Q RoPE is fused from
    /// `pass.rotary.query_position_offset`. No K/V allocation, prefix copy,
    /// submission, polling, mapping or synchronization occurs here.
    pub fn encode_external_pre_rotated_k(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
        kv_capacity: usize,
    ) -> Result<ResidentDecodeLayout, ResidentDecodeError> {
        validate_external_semantics(pass.shape, pass.config)?;
        let kv = ResidentKvView::new(
            pass.k,
            pass.v,
            pass.shape.batch,
            pass.shape.kv_heads,
            kv_capacity,
            pass.shape.head_dim,
            pass.shape.kv_len,
        )?;
        self.encode_view(
            device,
            encoder,
            ExternalResidentDecodePass {
                q: pass.q,
                out_and_lse: pass.out_and_lse,
                kv,
                q_heads: pass.shape.q_heads,
                config: pass.config,
                theta: pass.rotary.theta,
                q_rope_position: pass.rotary.query_position_offset,
                // The external contract keeps the causal domain on the shape.
                q_causal_position: pass.shape.query_position_offset,
            },
        )
    }

    fn encode_view(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalResidentDecodePass<'_>,
    ) -> Result<ResidentDecodeLayout, ResidentDecodeError> {
        let layout = layout_for_view(pass.kv, pass.q_heads)?;
        let kv_bytes = kv_required_bytes(pass.kv)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;
        if !pass.theta.is_finite() || pass.theta <= 0.0 {
            return Err(ResidentDecodeError::InvalidTheta(pass.theta));
        }
        // Causal visibility is judged from the dedicated causal domain, not
        // from the RoPE rotation position.
        if pass.config.causal
            && pass
                .q_causal_position
                .checked_add(1)
                .ok_or(FlatAttentionError::PositionOverflow)?
                < pass.kv.len
        {
            return Err(ResidentDecodeError::CausalVisibilityMismatch {
                query_position: pass.q_causal_position,
                kv_len: pass.kv.len,
            });
        }
        let scale = pass.config.resolved_scale(pass.kv.head_dim)?;
        let q_batch_heads = checked_mul(pass.kv.batch, pass.q_heads)?;
        let limits = device.limits();
        let maximum = limits.max_compute_workgroups_per_dimension;
        if q_batch_heads > maximum as usize {
            return Err(ResidentDecodeError::DispatchLimit {
                actual: q_batch_heads,
                maximum,
            });
        }
        let maximum_storage_bytes = u64::from(limits.max_storage_buffer_binding_size);
        validate_storage_binding_size("Q", layout.q_bytes, maximum_storage_bytes)?;
        validate_storage_binding_size("K", kv_bytes, maximum_storage_bytes)?;
        validate_storage_binding_size("V", kv_bytes, maximum_storage_bytes)?;
        validate_storage_binding_size("O|LSE", layout.combined_bytes, maximum_storage_bytes)?;

        let params = [
            checked_u32(pass.kv.len)?,
            checked_u32(pass.kv.capacity)?,
            checked_u32(pass.kv.head_dim)?,
            checked_u32(pass.q_heads)?,
            checked_u32(pass.kv.kv_heads)?,
            checked_u32(pass.kv.batch)?,
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
                    resource: storage_binding(pass.q, layout.q_bytes),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage_binding(pass.kv.k, kv_bytes),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: storage_binding(pass.kv.v, kv_bytes),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: storage_binding(pass.out_and_lse, layout.combined_bytes),
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

fn validate_external_semantics(
    shape: AsymmetricGroupedAttentionShape,
    config: FlatAttentionConfig,
) -> Result<(), ResidentDecodeError> {
    if shape.query_len != 1 {
        return Err(ResidentDecodeError::InvalidQueryLen {
            actual: shape.query_len,
        });
    }
    if config.causal && shape.query_position_offset.saturating_add(1) < shape.kv_len {
        return Err(ResidentDecodeError::CausalVisibilityMismatch {
            query_position: shape.query_position_offset,
            kv_len: shape.kv_len,
        });
    }
    Ok(())
}

fn layout_for_view(
    kv: ResidentKvView<'_>,
    q_heads: usize,
) -> Result<ResidentDecodeLayout, ResidentDecodeError> {
    validate_shape(kv, q_heads)?;
    let q_elements = checked_mul(checked_mul(kv.batch, q_heads)?, kv.head_dim)?;
    let lse_elements = checked_mul(kv.batch, q_heads)?;
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

fn create_output_buffer(device: &wgpu::Device, layout: ResidentDecodeLayout) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-m15-resident-decode-o-lse"),
        size: layout.combined_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn validate_kv_view(kv: ResidentKvView<'_>) -> Result<(), ResidentDecodeError> {
    if kv.batch == 0 || kv.kv_heads == 0 || kv.capacity == 0 || kv.head_dim == 0 {
        return Err(FlatAttentionError::ZeroDimension.into());
    }
    if kv.len == 0 {
        return Err(ResidentDecodeError::EmptyCache);
    }
    if kv.len > kv.capacity {
        return Err(ResidentDecodeError::InvalidCacheLength {
            len: kv.len,
            capacity: kv.capacity,
        });
    }
    if kv.head_dim % 2 != 0 {
        return Err(FlatAttentionError::InvalidRotaryHeadDim {
            head_dim: kv.head_dim,
        }
        .into());
    }
    if kv.head_dim > WGSL_MAX_HEAD_DIM {
        return Err(ResidentDecodeError::UnsupportedHeadDim {
            actual: kv.head_dim,
            maximum: WGSL_MAX_HEAD_DIM,
        });
    }
    let kv_elements = checked_mul(
        checked_mul(kv.batch, kv.capacity)?,
        checked_mul(kv.kv_heads, kv.head_dim)?,
    )?;
    if kv_elements > u32::MAX as usize {
        return Err(ResidentDecodeError::IndexSpaceExceeded {
            elements: kv_elements,
        });
    }
    let required_bytes = bytes_for_f32(kv_elements)?;
    validate_buffer("K", kv.k, required_bytes)?;
    validate_buffer("V", kv.v, required_bytes)?;
    Ok(())
}

fn validate_shape(kv: ResidentKvView<'_>, q_heads: usize) -> Result<(), ResidentDecodeError> {
    validate_kv_view(kv)?;
    if q_heads == 0 || q_heads % kv.kv_heads != 0 {
        return Err(FlatAttentionError::InvalidHeadGrouping {
            q_heads,
            kv_heads: kv.kv_heads,
        }
        .into());
    }
    Ok(())
}

fn kv_required_bytes(kv: ResidentKvView<'_>) -> Result<u64, ResidentDecodeError> {
    let elements = checked_mul(
        checked_mul(kv.batch, kv.capacity)?,
        checked_mul(kv.kv_heads, kv.head_dim)?,
    )?;
    bytes_for_f32(elements)
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

fn validate_storage_binding_size(
    tensor: &'static str,
    required_bytes: u64,
    maximum_bytes: u64,
) -> Result<(), ResidentDecodeError> {
    if required_bytes > maximum_bytes {
        return Err(ResidentDecodeError::StorageBindingTooLarge {
            tensor,
            required_bytes,
            maximum_bytes,
        });
    }
    Ok(())
}

fn storage_binding(buffer: &wgpu::Buffer, required_bytes: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: core::num::NonZeroU64::new(required_bytes),
    })
}

fn checked_mul(a: usize, b: usize) -> Result<usize, ResidentDecodeError> {
    a.checked_mul(b)
        .ok_or_else(|| FlatAttentionError::ShapeOverflow.into())
}

fn checked_u32(value: usize) -> Result<u32, ResidentDecodeError> {
    wgpu_internal::checked_u32(value)
        .ok_or(ResidentDecodeError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32(len: usize) -> Result<u64, ResidentDecodeError> {
    wgpu_internal::f32_bytes(len)
        .ok_or_else(|| ResidentDecodeError::from(FlatAttentionError::ShapeOverflow))
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    wgpu_internal::encode_u32(values)
}
