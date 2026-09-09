//! M16 correctness-first chunked prefill over resident paged K/V.
//!
//! This baseline composes the already-qualified q_len=1 paged decode kernel for
//! every query row while reading K/V directly from [`WgpuPagedKvCache`]. Query
//! rows are copied into one reusable device-local scratch buffer and O/LSE are
//! scattered into the caller-owned full output. K/V are never materialized,
//! compacted, copied, mapped, or read back by this orchestration layer.
//!
//! For causal prefill, each query dispatch receives a temporary logical page
//! table truncated to exactly the visible prefix. The physical K/V buffers stay
//! unchanged. Non-causal prefill reuses the full page table for every query.
//!
//! This q1-composed path is a qualification baseline, not a performance claim.
//! A direct multi-row paged prefill kernel may replace it only after equivalent
//! correctness coverage and paired target-adapter measurement.

use core::fmt;

use super::WgpuPagedKvCache;
use crate::{
    FlatAttentionConfig, FlatAttentionError, PagedDecodeError, PagedDecodePass,
    WgpuPagedDecodePipeline, WgpuPagedKvTable, WGSL_MAX_HEAD_DIM,
};

/// Full-sequence packed O|LSE geometry for paged chunked prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedChunkedPrefillLayout {
    pub query_len: usize,
    pub q_elements: usize,
    pub output_elements: usize,
    pub lse_elements: usize,
    pub combined_elements: usize,
    pub q_bytes: u64,
    pub output_bytes: u64,
    pub lse_bytes: u64,
    pub combined_bytes: u64,
}

/// One single-sequence chunked-prefill pass over resident paged K/V.
pub struct PagedChunkedPrefillPass<'a> {
    /// Full sequence-major Q: `[query_len, q_heads * head_dim]`.
    pub q: &'a wgpu::Buffer,
    /// Resident physical K/V plus authoritative logical page table.
    pub cache: &'a WgpuPagedKvCache,
    /// Full packed O|LSE destination: O then head-major LSE.
    pub out_and_lse: &'a wgpu::Buffer,
    pub q_heads: usize,
    pub config: FlatAttentionConfig,
    /// Positive finite RoPE base frequency used to rotate Q.
    ///
    /// K in `cache` must already have been rotated with the compatible RoPE
    /// convention before append; this orchestration cannot infer or verify that
    /// history from resident bytes.
    pub theta: f32,
    /// Absolute RoPE position assigned to query row zero.
    pub query_position_offset: usize,
    /// Host orchestration chunk size. Must be non-zero.
    pub query_chunk_size: usize,
}

/// Explicit M16 paged chunked-prefill failures.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PagedChunkedPrefillError {
    Decode(PagedDecodeError),
    ZeroQueryChunkSize,
    CacheHasUnsubmittedRecordedWrites,
    BufferTooSmall {
        tensor: &'static str,
        actual_bytes: u64,
        required_bytes: u64,
    },
    MissingBufferUsage {
        tensor: &'static str,
        required: &'static str,
    },
    ShapeOverflow,
    PositionOverflow,
}

impl fmt::Display for PagedChunkedPrefillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => write!(f, "{error}"),
            Self::ZeroQueryChunkSize => {
                write!(f, "paged chunked prefill requires a non-zero query chunk size")
            }
            Self::CacheHasUnsubmittedRecordedWrites => write!(
                f,
                "paged chunked prefill refuses cache metadata whose externally recorded K/V writes may still be unsubmitted"
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
            Self::ShapeOverflow => write!(f, "paged chunked prefill shape overflows address space"),
            Self::PositionOverflow => write!(f, "paged chunked prefill query position overflows usize"),
        }
    }
}

impl std::error::Error for PagedChunkedPrefillError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PagedDecodeError> for PagedChunkedPrefillError {
    fn from(value: PagedDecodeError) -> Self {
        Self::Decode(value)
    }
}

/// Reusable q1-composed paged prefill baseline.
pub struct WgpuPagedChunkedPrefillPipeline {
    decode: WgpuPagedDecodePipeline,
}

impl fmt::Debug for WgpuPagedChunkedPrefillPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuPagedChunkedPrefillPipeline")
            .finish_non_exhaustive()
    }
}

impl WgpuPagedChunkedPrefillPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, PagedChunkedPrefillError> {
        Ok(Self {
            decode: WgpuPagedDecodePipeline::new(device)?,
        })
    }

    /// Full packed O|LSE geometry implied by the live cache length.
    pub fn layout(
        cache: &WgpuPagedKvCache,
        q_heads: usize,
    ) -> Result<PagedChunkedPrefillLayout, PagedChunkedPrefillError> {
        validate_geometry(cache, q_heads)?;
        let query_len = cache.len();
        let q_row_elements = checked_mul(q_heads, cache.head_dim())?;
        let q_elements = checked_mul(query_len, q_row_elements)?;
        let output_elements = q_elements;
        let lse_elements = checked_mul(q_heads, query_len)?;
        let combined_elements = checked_add(output_elements, lse_elements)?;
        Ok(PagedChunkedPrefillLayout {
            query_len,
            q_elements,
            output_elements,
            lse_elements,
            combined_elements,
            q_bytes: bytes_for_f32(q_elements)?,
            output_bytes: bytes_for_f32(output_elements)?,
            lse_bytes: bytes_for_f32(lse_elements)?,
            combined_bytes: bytes_for_f32(combined_elements)?,
        })
    }

    /// Create a caller-readable O|LSE destination for [`Self::encode`].
    pub fn create_output_buffer(
        &self,
        device: &wgpu::Device,
        cache: &WgpuPagedKvCache,
        q_heads: usize,
    ) -> Result<wgpu::Buffer, PagedChunkedPrefillError> {
        let layout = Self::layout(cache, q_heads)?;
        if layout.combined_bytes > device.limits().max_buffer_size {
            return Err(PagedDecodeError::BufferTooSmall {
                tensor: "device max buffer size",
                actual_bytes: device.limits().max_buffer_size,
                required_bytes: layout.combined_bytes,
            }
            .into());
        }
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m16-paged-chunked-prefill-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Record a complete single-sequence paged prefill into `encoder`.
    ///
    /// This method never submits, polls, maps, synchronizes, compacts, or copies
    /// K/V. It fails closed when the cache reports externally recorded writes
    /// that may not yet have reached the resident buffers.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: PagedChunkedPrefillPass<'_>,
    ) -> Result<PagedChunkedPrefillLayout, PagedChunkedPrefillError> {
        if pass.query_chunk_size == 0 {
            return Err(PagedChunkedPrefillError::ZeroQueryChunkSize);
        }
        if pass.cache.has_unsubmitted_recorded_writes() {
            return Err(PagedChunkedPrefillError::CacheHasUnsubmittedRecordedWrites);
        }
        preflight(device, &pass)?;
        let full_layout = Self::layout(pass.cache, pass.q_heads)?;
        validate_buffer("Q", pass.q, full_layout.q_bytes)?;
        validate_usage("Q", pass.q, wgpu::BufferUsages::COPY_SRC, "COPY_SRC")?;
        validate_buffer("O|LSE", pass.out_and_lse, full_layout.combined_bytes)?;
        validate_usage(
            "O|LSE",
            pass.out_and_lse,
            wgpu::BufferUsages::COPY_DST,
            "COPY_DST",
        )?;

        let full_page_table = WgpuPagedKvTable::from_table(pass.cache.table())?;
        let q1_layout = WgpuPagedDecodePipeline::layout(pass.q_heads, pass.cache.head_dim())?;
        let q_scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m16-paged-chunked-prefill-q-scratch"),
            size: q1_layout.q_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let out_scratch =
            self.decode
                .create_output_buffer(device, pass.q_heads, pass.cache.head_dim())?;

        let q_row_bytes = q1_layout.q_bytes;
        let lse_scalar_bytes = core::mem::size_of::<f32>() as u64;
        let mut chunk_start = 0usize;
        while chunk_start < full_layout.query_len {
            let chunk_len = pass
                .query_chunk_size
                .min(full_layout.query_len - chunk_start);
            let chunk_end = checked_add(chunk_start, chunk_len)?;

            for query_index in chunk_start..chunk_end {
                let q_source_offset = checked_u64_mul(query_index, q_row_bytes)?;
                encoder.copy_buffer_to_buffer(pass.q, q_source_offset, &q_scratch, 0, q_row_bytes);

                let prefix_page_table;
                let page_table = if pass.config.causal {
                    let visible_tokens = query_index
                        .checked_add(1)
                        .ok_or(PagedChunkedPrefillError::PositionOverflow)?;
                    let mut prefix = pass.cache.table().clone();
                    prefix
                        .truncate(visible_tokens)
                        .map_err(PagedDecodeError::from)?;
                    prefix_page_table = WgpuPagedKvTable::from_table(&prefix)?;
                    &prefix_page_table
                } else {
                    &full_page_table
                };

                let q_rope_position = pass
                    .query_position_offset
                    .checked_add(query_index)
                    .ok_or(PagedChunkedPrefillError::PositionOverflow)?;
                self.decode.encode(
                    device,
                    encoder,
                    PagedDecodePass {
                        q: &q_scratch,
                        k: pass.cache.k_buffer(),
                        v: pass.cache.v_buffer(),
                        page_table,
                        out_and_lse: &out_scratch,
                        q_heads: pass.q_heads,
                        kv_heads: pass.cache.kv_heads(),
                        head_dim: pass.cache.head_dim(),
                        config: pass.config,
                        theta: pass.theta,
                        q_rope_position,
                        q_causal_position: query_index,
                    },
                )?;

                let output_destination_offset = checked_u64_mul(query_index, q_row_bytes)?;
                encoder.copy_buffer_to_buffer(
                    &out_scratch,
                    0,
                    pass.out_and_lse,
                    output_destination_offset,
                    q_row_bytes,
                );

                for q_head in 0..pass.q_heads {
                    let scratch_lse_index = checked_add(q1_layout.output_elements, q_head)?;
                    let scratch_lse_offset = bytes_for_f32(scratch_lse_index)?;
                    let full_lse_index =
                        checked_add(checked_mul(q_head, full_layout.query_len)?, query_index)?;
                    let full_lse_offset =
                        checked_add_u64(full_layout.output_bytes, bytes_for_f32(full_lse_index)?)?;
                    encoder.copy_buffer_to_buffer(
                        &out_scratch,
                        scratch_lse_offset,
                        pass.out_and_lse,
                        full_lse_offset,
                        lse_scalar_bytes,
                    );
                }
            }

            chunk_start = chunk_end;
        }

        Ok(full_layout)
    }
}

fn preflight(
    device: &wgpu::Device,
    pass: &PagedChunkedPrefillPass<'_>,
) -> Result<(), PagedChunkedPrefillError> {
    validate_geometry(pass.cache, pass.q_heads)?;
    if !pass.theta.is_finite() || pass.theta <= 0.0 {
        return Err(PagedDecodeError::InvalidTheta(pass.theta).into());
    }
    pass.config
        .resolved_scale(pass.cache.head_dim())
        .map_err(PagedDecodeError::Core)?;
    let final_query_position = pass
        .query_position_offset
        .checked_add(pass.cache.len() - 1)
        .ok_or(PagedChunkedPrefillError::PositionOverflow)?;
    if final_query_position > u32::MAX as usize {
        return Err(PagedDecodeError::IndexSpaceExceeded {
            elements: final_query_position,
        }
        .into());
    }
    if pass.q_heads > device.limits().max_compute_workgroups_per_dimension as usize {
        return Err(PagedDecodeError::DispatchLimit {
            actual: pass.q_heads,
            maximum: device.limits().max_compute_workgroups_per_dimension,
        }
        .into());
    }
    Ok(())
}

fn validate_geometry(
    cache: &WgpuPagedKvCache,
    q_heads: usize,
) -> Result<(), PagedChunkedPrefillError> {
    if cache.is_empty() {
        return Err(PagedDecodeError::EmptyTable.into());
    }
    if q_heads == 0 || q_heads % cache.kv_heads() != 0 {
        return Err(PagedDecodeError::InvalidHeadGrouping {
            q_heads,
            kv_heads: cache.kv_heads(),
        }
        .into());
    }
    if cache.head_dim() == 0 || cache.head_dim() % 2 != 0 {
        return Err(PagedDecodeError::Core(FlatAttentionError::InvalidRotaryHeadDim {
            head_dim: cache.head_dim(),
        })
        .into());
    }
    if cache.head_dim() > WGSL_MAX_HEAD_DIM {
        return Err(PagedDecodeError::UnsupportedHeadDim {
            actual: cache.head_dim(),
            maximum: WGSL_MAX_HEAD_DIM,
        }
        .into());
    }
    Ok(())
}

fn validate_buffer(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), PagedChunkedPrefillError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(PagedChunkedPrefillError::BufferTooSmall {
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
) -> Result<(), PagedChunkedPrefillError> {
    if !buffer.usage().contains(required_usage) {
        return Err(PagedChunkedPrefillError::MissingBufferUsage { tensor, required });
    }
    Ok(())
}

fn checked_add(a: usize, b: usize) -> Result<usize, PagedChunkedPrefillError> {
    a.checked_add(b)
        .ok_or(PagedChunkedPrefillError::ShapeOverflow)
}

fn checked_mul(a: usize, b: usize) -> Result<usize, PagedChunkedPrefillError> {
    a.checked_mul(b)
        .ok_or(PagedChunkedPrefillError::ShapeOverflow)
}

fn checked_u64_mul(a: usize, b: u64) -> Result<u64, PagedChunkedPrefillError> {
    let a = u64::try_from(a).map_err(|_| PagedChunkedPrefillError::ShapeOverflow)?;
    a.checked_mul(b)
        .ok_or(PagedChunkedPrefillError::ShapeOverflow)
}

fn checked_add_u64(a: u64, b: u64) -> Result<u64, PagedChunkedPrefillError> {
    a.checked_add(b)
        .ok_or(PagedChunkedPrefillError::ShapeOverflow)
}

fn bytes_for_f32(elements: usize) -> Result<u64, PagedChunkedPrefillError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(PagedChunkedPrefillError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| PagedChunkedPrefillError::ShapeOverflow)
}
