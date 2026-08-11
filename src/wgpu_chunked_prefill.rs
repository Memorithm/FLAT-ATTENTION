//! M16 WGPU chunked-prefill orchestration for projection-layout GQA/MQA.
//!
//! This correctness-first path reuses the already-qualified M11 asymmetric
//! projection/RoPE kernel for each query chunk. Q rows are compacted into a
//! device-local scratch buffer with device-to-device copies, K/V remain in the
//! caller-owned resident projection buffers, and compact O/LSE results are
//! scattered back into the caller-owned full output buffer. No host mapping,
//! queue submission, polling or synchronization occurs here.
//!
//! The scratch-copy architecture is deliberately a qualification baseline. It
//! makes no latency, bandwidth or memory-efficiency claim; a direct-offset
//! kernel may only replace it after paired target-adapter benchmarks.

use core::fmt;

use super::{
    AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig,
    ExternalAsymmetricProjectionPass, ExternalAsymmetricProjectionRotaryGroupedPipeline,
    ExternalProjectionLayout, ExternalWgpuError, FlatAttentionConfig, FlatAttentionError,
    GroupedAttentionShape, RotaryEmbeddingConfig,
};

/// One caller-owned full-sequence chunked-prefill dispatch.
pub struct WgpuChunkedProjectionPrefillPass<'a> {
    /// Full projection-layout Q: `[batch, seq_len, q_heads * head_dim]`.
    pub q: &'a wgpu::Buffer,
    /// Full projection-layout K: `[batch, seq_len, kv_heads * head_dim]`.
    pub k: &'a wgpu::Buffer,
    /// Full projection-layout V: `[batch, seq_len, kv_heads * head_dim]`.
    pub v: &'a wgpu::Buffer,
    /// Full combined O|LSE destination.
    pub out_and_lse: &'a wgpu::Buffer,
    pub shape: GroupedAttentionShape,
    pub config: FlatAttentionConfig,
    pub rotary: RotaryEmbeddingConfig,
    pub query_chunk_size: usize,
}

/// Explicit M16 chunked-prefill host-side failures.
#[derive(Debug, Clone, PartialEq)]
pub enum WgpuChunkedProjectionPrefillError {
    Core(FlatAttentionError),
    External(ExternalWgpuError),
    ZeroQueryChunkSize,
    BufferTooSmall {
        tensor: &'static str,
        actual_bytes: u64,
        required_bytes: u64,
    },
    MissingBufferUsage {
        tensor: &'static str,
        required: &'static str,
    },
}

impl fmt::Display for WgpuChunkedProjectionPrefillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::External(error) => write!(f, "{error}"),
            Self::ZeroQueryChunkSize => write!(
                f,
                "WGPU chunked projection prefill requires a non-zero query chunk size"
            ),
            Self::BufferTooSmall {
                tensor,
                actual_bytes,
                required_bytes,
            } => write!(
                f,
                "buffer {tensor} contains {actual_bytes} bytes, requires at least {required_bytes}"
            ),
            Self::MissingBufferUsage { tensor, required } => {
                write!(f, "buffer {tensor} requires WGPU usage {required}")
            }
        }
    }
}

impl std::error::Error for WgpuChunkedProjectionPrefillError {}

impl From<FlatAttentionError> for WgpuChunkedProjectionPrefillError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

impl From<ExternalWgpuError> for WgpuChunkedProjectionPrefillError {
    fn from(value: ExternalWgpuError) -> Self {
        Self::External(value)
    }
}

/// Reusable M16 orchestration pipeline.
///
/// It owns only the qualified M11 compute pipeline. Per-chunk scratch buffers
/// are device-local and live for the recorded command buffer. Framework Q/K/V
/// and final O|LSE remain caller-owned.
pub struct WgpuChunkedProjectionPrefillPipeline {
    inner: ExternalAsymmetricProjectionRotaryGroupedPipeline,
}

impl fmt::Debug for WgpuChunkedProjectionPrefillPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuChunkedProjectionPrefillPipeline")
            .finish_non_exhaustive()
    }
}

impl WgpuChunkedProjectionPrefillPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, WgpuChunkedProjectionPrefillError> {
        Ok(Self {
            inner: ExternalAsymmetricProjectionRotaryGroupedPipeline::new(device)?,
        })
    }

    /// Full-sequence O|LSE geometry used by [`Self::encode`].
    pub fn layout(
        shape: GroupedAttentionShape,
    ) -> Result<ExternalProjectionLayout, WgpuChunkedProjectionPrefillError> {
        shape.validate()?;
        let full_shape = AsymmetricGroupedAttentionShape {
            batch: shape.batch,
            q_heads: shape.q_heads,
            kv_heads: shape.kv_heads,
            query_len: shape.seq_len,
            kv_len: shape.seq_len,
            head_dim: shape.head_dim,
            query_position_offset: 0,
        };
        Ok(ExternalAsymmetricProjectionRotaryGroupedPipeline::layout(
            full_shape,
        )?)
    }

    /// Create a full O|LSE destination suitable for the scatter phase.
    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        shape: GroupedAttentionShape,
    ) -> Result<wgpu::Buffer, WgpuChunkedProjectionPrefillError> {
        let layout = Self::layout(shape)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m16-chunked-prefill-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Record the complete chunked prefill into a caller-owned encoder.
    ///
    /// This method never submits, polls, maps or synchronizes. K/V are never
    /// copied or expanded. Only Q chunk compaction and O/LSE scattering use
    /// device-to-device copies.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: WgpuChunkedProjectionPrefillPass<'_>,
    ) -> Result<ExternalProjectionLayout, WgpuChunkedProjectionPrefillError> {
        if pass.query_chunk_size == 0 {
            return Err(WgpuChunkedProjectionPrefillError::ZeroQueryChunkSize);
        }
        pass.shape.validate()?;
        pass.rotary
            .validate(pass.shape.head_dim, pass.shape.seq_len)?;

        let full_layout = Self::layout(pass.shape)?;
        validate_buffer("Q", pass.q, full_layout.q_bytes)?;
        validate_buffer("K", pass.k, full_layout.kv_bytes)?;
        validate_buffer("V", pass.v, full_layout.kv_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, full_layout.combined_bytes)?;
        validate_usage("Q", pass.q, wgpu::BufferUsages::COPY_SRC, "COPY_SRC")?;
        validate_usage(
            "O|LSE",
            pass.out_and_lse,
            wgpu::BufferUsages::COPY_DST,
            "COPY_DST",
        )?;

        let q_width = checked_mul(pass.shape.q_heads, pass.shape.head_dim)?;
        let f32_bytes = core::mem::size_of::<f32>();
        let full_output_bytes = bytes_for_f32(full_layout.output_elements)?;

        let mut query_start = 0usize;
        while query_start < pass.shape.seq_len {
            let chunk_len = pass.query_chunk_size.min(pass.shape.seq_len - query_start);
            let chunk_shape = AsymmetricGroupedAttentionShape {
                batch: pass.shape.batch,
                q_heads: pass.shape.q_heads,
                kv_heads: pass.shape.kv_heads,
                query_len: chunk_len,
                kv_len: pass.shape.seq_len,
                head_dim: pass.shape.head_dim,
                query_position_offset: query_start,
            };
            let chunk_rotary = AsymmetricRotaryEmbeddingConfig {
                theta: pass.rotary.theta,
                query_position_offset: pass
                    .rotary
                    .position_offset
                    .checked_add(query_start)
                    .ok_or(FlatAttentionError::PositionOverflow)?,
                kv_position_offset: pass.rotary.position_offset,
            };
            let chunk_layout =
                ExternalAsymmetricProjectionRotaryGroupedPipeline::layout(chunk_shape)?;

            let compact_q = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flat-m16-chunked-prefill-q-scratch"),
                size: chunk_layout.q_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let compact_out = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flat-m16-chunked-prefill-o-lse-scratch"),
                size: chunk_layout.combined_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let row_bytes = bytes_for_f32(checked_mul(chunk_len, q_width)?)?;
            for batch in 0..pass.shape.batch {
                let source_row = checked_add(checked_mul(batch, pass.shape.seq_len)?, query_start)?;
                let source_offset = bytes_for_f32(checked_mul(source_row, q_width)?)?;
                let destination_offset =
                    bytes_for_f32(checked_mul(checked_mul(batch, chunk_len)?, q_width)?)?;
                encoder.copy_buffer_to_buffer(
                    pass.q,
                    source_offset,
                    &compact_q,
                    destination_offset,
                    row_bytes,
                );
            }

            self.inner.encode(
                device,
                encoder,
                ExternalAsymmetricProjectionPass {
                    q: &compact_q,
                    k: pass.k,
                    v: pass.v,
                    out_and_lse: &compact_out,
                    shape: chunk_shape,
                    config: pass.config,
                    rotary: chunk_rotary,
                },
            )?;

            for batch in 0..pass.shape.batch {
                let compact_output_row = checked_mul(batch, chunk_len)?;
                let compact_output_offset =
                    bytes_for_f32(checked_mul(compact_output_row, q_width)?)?;
                let full_output_row =
                    checked_add(checked_mul(batch, pass.shape.seq_len)?, query_start)?;
                let full_output_offset = bytes_for_f32(checked_mul(full_output_row, q_width)?)?;
                encoder.copy_buffer_to_buffer(
                    &compact_out,
                    compact_output_offset,
                    pass.out_and_lse,
                    full_output_offset,
                    row_bytes,
                );

                for q_head in 0..pass.shape.q_heads {
                    let compact_lse_index = checked_mul(
                        checked_add(checked_mul(batch, pass.shape.q_heads)?, q_head)?,
                        chunk_len,
                    )?;
                    let compact_lse_offset = checked_add(
                        bytes_for_f32(chunk_layout.output_elements)?,
                        bytes_for_f32(compact_lse_index)?,
                    )?;
                    let full_lse_index = checked_add(
                        checked_mul(
                            checked_add(checked_mul(batch, pass.shape.q_heads)?, q_head)?,
                            pass.shape.seq_len,
                        )?,
                        query_start,
                    )?;
                    let full_lse_offset =
                        checked_add(full_output_bytes, bytes_for_f32(full_lse_index)?)?;
                    encoder.copy_buffer_to_buffer(
                        &compact_out,
                        compact_lse_offset,
                        pass.out_and_lse,
                        full_lse_offset,
                        (chunk_len * f32_bytes) as u64,
                    );
                }
            }

            query_start = checked_add(query_start, chunk_len)?;
        }

        Ok(full_layout)
    }
}

fn validate_buffer(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), WgpuChunkedProjectionPrefillError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(WgpuChunkedProjectionPrefillError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    Ok(())
}

fn validate_usage(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_usage: wgpu::BufferUsages,
    required: &'static str,
) -> Result<(), WgpuChunkedProjectionPrefillError> {
    if !buffer.usage().contains(required_usage) {
        return Err(WgpuChunkedProjectionPrefillError::MissingBufferUsage { tensor, required });
    }
    Ok(())
}

fn checked_add(a: usize, b: usize) -> Result<usize, WgpuChunkedProjectionPrefillError> {
    a.checked_add(b)
        .ok_or_else(|| FlatAttentionError::ShapeOverflow.into())
}

fn checked_mul(a: usize, b: usize) -> Result<usize, WgpuChunkedProjectionPrefillError> {
    a.checked_mul(b)
        .ok_or_else(|| FlatAttentionError::ShapeOverflow.into())
}

fn bytes_for_f32(elements: usize) -> Result<u64, WgpuChunkedProjectionPrefillError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| FlatAttentionError::ShapeOverflow.into())
}
