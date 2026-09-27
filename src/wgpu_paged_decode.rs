//! M16 portable q_len=1 decode over caller-owned paged resident K/V storage.
//!
//! This first device consumer deliberately qualifies the single-sequence page-table
//! contract before adding batched page tables. K/V data buffers remain caller-owned;
//! the compact page table is packed into the uniform block so the shader retains the
//! portable four-storage-binding contract. The encode path does not submit, poll,
//! map, synchronize, compact, or copy K/V.

use super::wgpu_internal;

use core::fmt;

use crate::paged_kv::{PagedKvError, PagedKvTable};
use crate::{FlatAttentionConfig, FlatAttentionError, FLAT_DECODE_PAGED_WGSL, WGSL_MAX_HEAD_DIM};

/// Maximum logical pages carried by the portable M16 uniform block.
pub const WGSL_PAGED_MAX_LOGICAL_PAGES: usize = 256;
/// Fixed scalar/header words preceding the page-map array in the WGSL uniform.
pub const WGSL_PAGED_UNIFORM_HEADER_U32: usize = 12;
/// Total encoded u32 words in the fixed-size portable page-table uniform.
pub const WGSL_PAGED_UNIFORM_U32: usize =
    WGSL_PAGED_UNIFORM_HEADER_U32 + WGSL_PAGED_MAX_LOGICAL_PAGES;
/// Maximum physical pages addressable by a packed u16 page index.
pub const WGSL_PAGED_U16_MAX_PHYSICAL_PAGES: usize = (u16::MAX as usize) + 1;
/// Two u16 physical-page indices are packed in each u32 word.
pub const WGSL_PAGED_U16_PER_U32: usize = 2;
/// Fixed packed page-map words needed for 256 logical pages.
pub const WGSL_PAGED_U16_PACKED_WORDS: usize =
    WGSL_PAGED_MAX_LOGICAL_PAGES / WGSL_PAGED_U16_PER_U32;
/// Theoretical fixed uniform words if the WGSL contract adopts packed u16 pages.
pub const WGSL_PAGED_U16_UNIFORM_U32: usize =
    WGSL_PAGED_UNIFORM_HEADER_U32 + WGSL_PAGED_U16_PACKED_WORDS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedDecodeLayout {
    /// Query element count (batch * q_heads * head_dim).
    pub q_elements: usize,
    /// Output context element count for this decode pass.
    pub output_elements: usize,
    /// LSE element count for this decode pass.
    pub lse_elements: usize,
    /// Total packed O|LSE element count.
    pub combined_elements: usize,
    /// Query buffer size in bytes.
    pub q_bytes: u64,
    /// Packed O|LSE destination size in bytes.
    pub combined_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PagedDecodeError {
    Core(FlatAttentionError),
    Table(PagedKvError),
    EmptyTable,
    TooManyMappedPages {
        actual: usize,
        maximum: usize,
    },
    InvalidHeadGrouping {
        q_heads: usize,
        kv_heads: usize,
    },
    UnsupportedHeadDim {
        actual: usize,
        maximum: usize,
    },
    InvalidTheta(f32),
    CausalVisibilityMismatch {
        query_position: usize,
        kv_len: usize,
    },
    IndexSpaceExceeded {
        elements: usize,
    },
    PhysicalPageIndexExceedsU32 {
        physical_page: u64,
    },
    PhysicalPageIndexExceedsU16 {
        physical_page: u64,
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
    UniformBindingTooLarge {
        required_bytes: u64,
        maximum_bytes: u64,
    },
    PipelineValidation(String),
}

impl fmt::Display for PagedDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::Table(error) => write!(f, "{error}"),
            Self::EmptyTable => write!(f, "paged decode requires at least one live KV token"),
            Self::TooManyMappedPages { actual, maximum } => write!(
                f,
                "paged decode maps {actual} logical pages, portable maximum is {maximum}"
            ),
            Self::InvalidHeadGrouping { q_heads, kv_heads } => write!(
                f,
                "q_heads ({q_heads}) must be exactly divisible by kv_heads ({kv_heads})"
            ),
            Self::UnsupportedHeadDim { actual, maximum } => {
                write!(f, "head_dim {actual} exceeds portable maximum {maximum}")
            }
            Self::InvalidTheta(theta) => {
                write!(f, "paged decode RoPE theta must be finite and positive, got {theta}")
            }
            Self::CausalVisibilityMismatch {
                query_position,
                kv_len,
            } => write!(
                f,
                "paged causal decode query position {query_position} cannot see all {kv_len} live KV tokens"
            ),
            Self::IndexSpaceExceeded { elements } => {
                write!(f, "paged decode exceeds WGPU u32 index space at {elements} elements")
            }
            Self::PhysicalPageIndexExceedsU32 { physical_page } => write!(
                f,
                "paged decode physical page index {physical_page} exceeds WGPU u32 page-table space"
            ),
            Self::PhysicalPageIndexExceedsU16 { physical_page } => write!(
                f,
                "paged decode physical page index {physical_page} exceeds packed u16 page-table space"
            ),
            Self::DispatchLimit { actual, maximum } => write!(
                f,
                "paged decode requires {actual} workgroups, device maximum is {maximum}"
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
            Self::UniformBindingTooLarge {
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "paged decode uniform requires {required_bytes} bytes, device maximum is {maximum_bytes}"
            ),
            Self::PipelineValidation(error) => {
                write!(f, "paged decode pipeline validation failed: {error}")
            }
        }
    }
}

impl std::error::Error for PagedDecodeError {}

impl From<FlatAttentionError> for PagedDecodeError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

impl From<PagedKvError> for PagedDecodeError {
    fn from(value: PagedKvError) -> Self {
        Self::Table(value)
    }
}

/// Compact host-side page-table descriptor ready for the portable uniform block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuPagedKvTable {
    entries: Vec<u32>,
    live_tokens: usize,
    page_size: usize,
    physical_pages: usize,
    generation: u64,
}

impl WgpuPagedKvTable {
    pub fn from_table(table: &PagedKvTable) -> Result<Self, PagedDecodeError> {
        let live_tokens = table.len();
        let page_lanes = table.mapped_page_lanes();
        if live_tokens == 0 || page_lanes.is_empty() {
            return Err(PagedDecodeError::EmptyTable);
        }
        if page_lanes.len() > WGSL_PAGED_MAX_LOGICAL_PAGES {
            return Err(PagedDecodeError::TooManyMappedPages {
                actual: page_lanes.len(),
                maximum: WGSL_PAGED_MAX_LOGICAL_PAGES,
            });
        }

        let config = table.config();
        let _ = checked_u32(config.physical_pages)?;
        let physical_pages_u64 = u64::try_from(config.physical_pages).map_err(|_| {
            PagedDecodeError::IndexSpaceExceeded {
                elements: config.physical_pages,
            }
        })?;
        let mut entries = Vec::with_capacity(page_lanes.len());
        for &physical_page in page_lanes {
            if physical_page >= physical_pages_u64 {
                return Err(PagedDecodeError::PhysicalPageIndexExceedsU32 { physical_page });
            }
            let device_page = u32::try_from(physical_page)
                .map_err(|_| PagedDecodeError::PhysicalPageIndexExceedsU32 { physical_page })?;
            entries.push(device_page);
        }

        Ok(Self {
            entries,
            live_tokens,
            page_size: config.page_size,
            physical_pages: config.physical_pages,
            generation: table.generation(),
        })
    }

    /// Exact W32 device page-map entries packed into the portable uniform.
    #[must_use]
    pub fn entries(&self) -> &[u32] {
        &self.entries
    }

    /// Exact meaningful page-map payload bytes before uniform padding.
    ///
    /// This excludes the fixed uniform header and zero-padded unused page slots.
    #[must_use]
    pub fn page_map_payload_bytes(&self) -> usize {
        core::mem::size_of_val(self.entries.as_slice())
    }

    /// Exact bytes encoded for the fixed-size portable paged-decode uniform.
    ///
    /// This includes the twelve header words and all 256 page-map slots,
    /// including zero padding. It describes encoded uniform bytes only, not
    /// allocator/device physical residency.
    #[must_use]
    pub const fn encoded_uniform_bytes(&self) -> usize {
        WGSL_PAGED_UNIFORM_U32 * core::mem::size_of::<u32>()
    }

    #[must_use]
    pub fn live_tokens(&self) -> usize {
        self.live_tokens
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn physical_pages(&self) -> usize {
        self.physical_pages
    }

    pub fn mapped_pages(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Planning-only eligibility report for the packed-u16 page-map shadow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuPackedU16Preflight {
    mapped_pages: usize,
    physical_pages: usize,
    packed_words: usize,
}

impl WgpuPackedU16Preflight {
    #[must_use]
    pub const fn mapped_pages(self) -> usize {
        self.mapped_pages
    }

    #[must_use]
    pub const fn physical_pages(self) -> usize {
        self.physical_pages
    }

    #[must_use]
    pub const fn packed_words(self) -> usize {
        self.packed_words
    }
}

/// Validate packed-u16 page-map eligibility without allocating packed storage.
///
/// This function preserves the same fail-closed ordering used by packed
/// materialization: non-empty table, portable logical page limit, physical
/// u16 domain, then individual lane conversion.
///
/// # Errors
///
/// Returns the exact PagedDecodeError that blocks packed-u16 materialization.
pub fn preflight_packed_u16_page_map(
    table: &PagedKvTable,
) -> Result<WgpuPackedU16Preflight, PagedDecodeError> {
    let live_tokens = table.len();
    let page_lanes = table.mapped_page_lanes();
    if live_tokens == 0 || page_lanes.is_empty() {
        return Err(PagedDecodeError::EmptyTable);
    }
    if page_lanes.len() > WGSL_PAGED_MAX_LOGICAL_PAGES {
        return Err(PagedDecodeError::TooManyMappedPages {
            actual: page_lanes.len(),
            maximum: WGSL_PAGED_MAX_LOGICAL_PAGES,
        });
    }

    let config = table.config();
    if config.physical_pages > WGSL_PAGED_U16_MAX_PHYSICAL_PAGES {
        return Err(PagedDecodeError::PhysicalPageIndexExceedsU16 {
            physical_page: u64::try_from(config.physical_pages - 1).unwrap_or(u64::MAX),
        });
    }

    for &physical_page in page_lanes {
        u16::try_from(physical_page)
            .map_err(|_| PagedDecodeError::PhysicalPageIndexExceedsU16 { physical_page })?;
    }

    Ok(WgpuPackedU16Preflight {
        mapped_pages: page_lanes.len(),
        physical_pages: config.physical_pages,
        packed_words: page_lanes.len().div_ceil(WGSL_PAGED_U16_PER_U32),
    })
}

/// Experimental host-only packed-u16 page-map projection.
///
/// This is not consumed by the production WGSL shader. It qualifies the exact
/// reversible packing contract before any shader/uniform migration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuPackedPagedKvTable16 {
    packed_entries: Vec<u32>,
    mapped_pages: usize,
    live_tokens: usize,
    page_size: usize,
    physical_pages: usize,
    generation: u64,
}

impl WgpuPackedPagedKvTable16 {
    /// Build the packed shadow projection from the authoritative host W64 map.
    pub fn from_table(table: &PagedKvTable) -> Result<Self, PagedDecodeError> {
        let preflight = preflight_packed_u16_page_map(table)?;
        let live_tokens = table.len();
        let page_lanes = table.mapped_page_lanes();
        let config = table.config();

        let mut packed_entries = Vec::with_capacity(preflight.packed_words());
        for pair in page_lanes.chunks(WGSL_PAGED_U16_PER_U32) {
            let low = u16::try_from(pair[0]).map_err(|_| {
                PagedDecodeError::PhysicalPageIndexExceedsU16 {
                    physical_page: pair[0],
                }
            })?;
            let high = match pair.get(1).copied() {
                Some(value) => u16::try_from(value).map_err(|_| {
                    PagedDecodeError::PhysicalPageIndexExceedsU16 {
                        physical_page: value,
                    }
                })?,
                None => 0,
            };
            packed_entries.push(u32::from(low) | (u32::from(high) << 16));
        }

        Ok(Self {
            packed_entries,
            mapped_pages: page_lanes.len(),
            live_tokens,
            page_size: config.page_size,
            physical_pages: config.physical_pages,
            generation: table.generation(),
        })
    }

    /// Packed u32 words, two logical-page indices per word.
    #[must_use]
    pub fn packed_entries(&self) -> &[u32] {
        &self.packed_entries
    }

    /// Decode one logical-page mapping exactly.
    #[must_use]
    pub fn physical_page(&self, logical_page: usize) -> Option<u16> {
        if logical_page >= self.mapped_pages {
            return None;
        }
        let word = self.packed_entries[logical_page / WGSL_PAGED_U16_PER_U32];
        let shift = (logical_page % WGSL_PAGED_U16_PER_U32) * 16;
        Some(((word >> shift) & u32::from(u16::MAX)) as u16)
    }

    #[must_use]
    pub const fn mapped_pages(&self) -> usize {
        self.mapped_pages
    }

    #[must_use]
    pub const fn live_tokens(&self) -> usize {
        self.live_tokens
    }

    #[must_use]
    pub const fn page_size(&self) -> usize {
        self.page_size
    }

    #[must_use]
    pub const fn physical_pages(&self) -> usize {
        self.physical_pages
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Useful packed page-map bytes before fixed-uniform padding.
    #[must_use]
    pub fn packed_page_map_bytes(&self) -> usize {
        core::mem::size_of_val(self.packed_entries.as_slice())
    }

    /// Theoretical fixed encoded bytes if the portable shader adopts this
    /// packing while keeping the same twelve-u32 header.
    #[must_use]
    pub const fn theoretical_encoded_uniform_bytes(&self) -> usize {
        WGSL_PAGED_U16_UNIFORM_U32 * core::mem::size_of::<u32>()
    }

    /// Materialize the exact fixed-size packed-u16 shadow uniform words.
    ///
    /// The production shader does not consume this representation. The first
    /// four words are derived from this table: live_tokens, page_size,
    /// physical_pages and mapped_pages. The caller supplies the remaining
    /// eight decode-specific header words. Packed page-map words start at
    /// index 12 and unused fixed capacity is zero-padded.
    pub fn shadow_uniform_words(
        &self,
        decode_header_tail: [u32; 8],
    ) -> Result<Vec<u32>, PagedDecodeError> {
        let mut words = Vec::with_capacity(WGSL_PAGED_U16_UNIFORM_U32);
        words.extend_from_slice(&[
            checked_u32(self.live_tokens)?,
            checked_u32(self.page_size)?,
            checked_u32(self.physical_pages)?,
            checked_u32(self.mapped_pages)?,
        ]);
        words.extend_from_slice(&decode_header_tail);
        words.extend_from_slice(&self.packed_entries);
        words.resize(WGSL_PAGED_U16_UNIFORM_U32, 0);
        Ok(words)
    }
}

pub struct PagedDecodePass<'a> {
    pub q: &'a wgpu::Buffer,
    /// Pre-rotated K in physical layout `[physical_pages, page_size, kv_heads * head_dim]`.
    pub k: &'a wgpu::Buffer,
    /// Raw V in the same physical layout as K.
    pub v: &'a wgpu::Buffer,
    pub page_table: &'a WgpuPagedKvTable,
    pub out_and_lse: &'a wgpu::Buffer,
    /// Query heads for this decode pass.
    pub q_heads: usize,
    /// Physical K/V heads backing the cache.
    pub kv_heads: usize,
    /// Feature width of every head row.
    pub head_dim: usize,
    /// Attention configuration (causality and softmax scale).
    pub config: FlatAttentionConfig,
    /// Positive finite RoPE base frequency.
    pub theta: f32,
    /// Absolute RoPE position of the single query row (rotation domain only).
    pub q_rope_position: usize,
    /// Absolute causal position of the single query row.
    ///
    /// Under `config.causal` the kernel requires
    /// `q_causal_position + 1 >= live_tokens`; RoPE and causal origins may
    /// differ exactly like the asymmetric oracle contract.
    pub q_causal_position: usize,
}

pub struct WgpuPagedDecodePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for WgpuPagedDecodePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuPagedDecodePipeline")
            .finish_non_exhaustive()
    }
}

impl WgpuPagedDecodePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, PagedDecodeError> {
        let pipeline = wgpu_internal::create_pipeline(
            device,
            FLAT_DECODE_PAGED_WGSL,
            "flat-m16-paged-decode",
            "flat_attention_decode_paged",
        )
        .map_err(PagedDecodeError::PipelineValidation)?;
        Ok(Self { pipeline })
    }

    pub fn layout(q_heads: usize, head_dim: usize) -> Result<PagedDecodeLayout, PagedDecodeError> {
        validate_geometry(q_heads, 1, head_dim)?;
        let q_elements = checked_mul(q_heads, head_dim)?;
        let lse_elements = q_heads;
        let combined_elements = q_elements
            .checked_add(lse_elements)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        Ok(PagedDecodeLayout {
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
        q_heads: usize,
        head_dim: usize,
    ) -> Result<wgpu::Buffer, PagedDecodeError> {
        let layout = Self::layout(q_heads, head_dim)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m16-paged-decode-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: PagedDecodePass<'_>,
    ) -> Result<PagedDecodeLayout, PagedDecodeError> {
        if pass.page_table.live_tokens == 0 || pass.page_table.entries.is_empty() {
            return Err(PagedDecodeError::EmptyTable);
        }
        if pass.page_table.entries.len() > WGSL_PAGED_MAX_LOGICAL_PAGES {
            return Err(PagedDecodeError::TooManyMappedPages {
                actual: pass.page_table.entries.len(),
                maximum: WGSL_PAGED_MAX_LOGICAL_PAGES,
            });
        }
        validate_geometry(pass.q_heads, pass.kv_heads, pass.head_dim)?;
        if !pass.theta.is_finite() || pass.theta <= 0.0 {
            return Err(PagedDecodeError::InvalidTheta(pass.theta));
        }
        if pass.config.causal
            && pass
                .q_causal_position
                .checked_add(1)
                .ok_or(FlatAttentionError::PositionOverflow)?
                < pass.page_table.live_tokens
        {
            return Err(PagedDecodeError::CausalVisibilityMismatch {
                query_position: pass.q_causal_position,
                kv_len: pass.page_table.live_tokens,
            });
        }
        let covered_tokens = checked_mul(pass.page_table.entries.len(), pass.page_table.page_size)?;
        if covered_tokens < pass.page_table.live_tokens {
            return Err(PagedDecodeError::IndexSpaceExceeded {
                elements: pass.page_table.live_tokens,
            });
        }

        let layout = Self::layout(pass.q_heads, pass.head_dim)?;
        let physical_rows = checked_mul(pass.page_table.physical_pages, pass.page_table.page_size)?;
        let kv_width = checked_mul(pass.kv_heads, pass.head_dim)?;
        let kv_elements = checked_mul(physical_rows, kv_width)?;
        let kv_bytes = bytes_for_f32(kv_elements)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, kv_bytes)?;
        validate_buffer("V", pass.v, kv_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;

        let limits = device.limits();
        if pass.q_heads > limits.max_compute_workgroups_per_dimension as usize {
            return Err(PagedDecodeError::DispatchLimit {
                actual: pass.q_heads,
                maximum: limits.max_compute_workgroups_per_dimension,
            });
        }
        let maximum_storage_bytes = limits.max_storage_buffer_binding_size;
        validate_storage_binding_size("Q", layout.q_bytes, maximum_storage_bytes)?;
        validate_storage_binding_size("K", kv_bytes, maximum_storage_bytes)?;
        validate_storage_binding_size("V", kv_bytes, maximum_storage_bytes)?;
        validate_storage_binding_size("O|LSE", layout.combined_bytes, maximum_storage_bytes)?;

        let scale = pass.config.resolved_scale(pass.head_dim)?;
        let mut params = Vec::with_capacity(WGSL_PAGED_UNIFORM_U32);
        params.extend_from_slice(&[
            checked_u32(pass.page_table.live_tokens)?,
            checked_u32(pass.page_table.page_size)?,
            checked_u32(pass.page_table.physical_pages)?,
            checked_u32(pass.page_table.entries.len())?,
            checked_u32(pass.head_dim)?,
            checked_u32(pass.q_heads)?,
            checked_u32(pass.kv_heads)?,
            scale.to_bits(),
            pass.theta.to_bits(),
            checked_u32(pass.q_rope_position)?,
            0,
            0,
        ]);
        params.extend_from_slice(&pass.page_table.entries);
        params.resize(WGSL_PAGED_UNIFORM_U32, 0);
        let params_bytes = encode_u32(&params);
        let params_len = params_bytes.len() as u64;
        let maximum_uniform_bytes = limits.max_uniform_buffer_binding_size;
        if params_len > maximum_uniform_bytes {
            return Err(PagedDecodeError::UniformBindingTooLarge {
                required_bytes: params_len,
                maximum_bytes: maximum_uniform_bytes,
            });
        }
        let params_buffer = wgpu_internal::create_uniform_buffer_init(
            device,
            "flat-m16-paged-decode-params",
            &params_bytes,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-m16-paged-decode-bind-group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_binding(pass.q, layout.q_bytes),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage_binding(pass.k, kv_bytes),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: storage_binding(pass.v, kv_bytes),
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
            label: Some("flat-m16-paged-decode"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(checked_u32(pass.q_heads)?, 1, 1);
        drop(compute_pass);
        Ok(layout)
    }
}

fn validate_geometry(
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<(), PagedDecodeError> {
    if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
        return Err(PagedDecodeError::InvalidHeadGrouping { q_heads, kv_heads });
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(FlatAttentionError::InvalidRotaryHeadDim { head_dim }.into());
    }
    if head_dim > WGSL_MAX_HEAD_DIM {
        return Err(PagedDecodeError::UnsupportedHeadDim {
            actual: head_dim,
            maximum: WGSL_MAX_HEAD_DIM,
        });
    }
    Ok(())
}

fn validate_buffer(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), PagedDecodeError> {
    if buffer.size() < required_bytes {
        return Err(PagedDecodeError::BufferTooSmall {
            tensor,
            actual_bytes: buffer.size(),
            required_bytes,
        });
    }
    Ok(())
}

fn validate_storage_binding_size(
    tensor: &'static str,
    required_bytes: u64,
    maximum_bytes: u64,
) -> Result<(), PagedDecodeError> {
    if required_bytes > maximum_bytes {
        return Err(PagedDecodeError::StorageBindingTooLarge {
            tensor,
            required_bytes,
            maximum_bytes,
        });
    }
    Ok(())
}

fn storage_binding(buffer: &wgpu::Buffer, size: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: core::num::NonZeroU64::new(size),
    })
}

fn checked_mul(a: usize, b: usize) -> Result<usize, PagedDecodeError> {
    a.checked_mul(b)
        .ok_or_else(|| FlatAttentionError::ShapeOverflow.into())
}

fn checked_u32(value: usize) -> Result<u32, PagedDecodeError> {
    wgpu_internal::checked_u32(value)
        .ok_or(PagedDecodeError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32(len: usize) -> Result<u64, PagedDecodeError> {
    wgpu_internal::f32_bytes(len)
        .ok_or_else(|| PagedDecodeError::from(FlatAttentionError::ShapeOverflow))
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    wgpu_internal::encode_u32(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_kv::PagedKvConfig;

    #[test]
    fn production_w64_maps_directly_to_w32_device_entries() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 8,
        })
        .unwrap();
        table.append(17).unwrap();

        let device = WgpuPagedKvTable::from_table(&table).unwrap();
        assert_eq!(table.mapped_page_lanes(), &[0, 1, 2, 3, 4]);
        assert_eq!(device.entries(), &[0_u32, 1, 2, 3, 4]);
        assert_eq!(device.live_tokens(), 17);
        assert_eq!(device.page_size(), 4);
        assert_eq!(device.physical_pages(), 8);
        assert_eq!(device.generation(), 0);
    }

    #[test]
    fn device_page_map_payload_and_fixed_uniform_bytes_are_distinct() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 8,
        })
        .unwrap();
        table.append(17).unwrap();

        let device = WgpuPagedKvTable::from_table(&table).unwrap();
        assert_eq!(device.entries().len(), 5);
        assert_eq!(device.page_map_payload_bytes(), 5 * 4);
        assert_eq!(
            device.encoded_uniform_bytes(),
            (WGSL_PAGED_UNIFORM_HEADER_U32 + WGSL_PAGED_MAX_LOGICAL_PAGES) * 4
        );
        assert_eq!(device.encoded_uniform_bytes(), 1072);
    }

    #[test]
    fn device_projection_tracks_epoch_generation_without_page_entry_rewrite() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 2,
            physical_pages: 4,
        })
        .unwrap();
        table.append(5).unwrap();
        let first = WgpuPagedKvTable::from_table(&table).unwrap();

        table.reset().unwrap();
        table.append(5).unwrap();
        let second = WgpuPagedKvTable::from_table(&table).unwrap();

        assert_eq!(first.entries(), second.entries());
        assert_eq!(first.generation(), 0);
        assert_eq!(second.generation(), 1);
    }

    #[test]
    fn device_projection_preserves_truncate_and_reuse_mapping() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 6,
        })
        .unwrap();
        table.append(21).unwrap();
        table.truncate(5).unwrap();
        table.append(8).unwrap();

        let device = WgpuPagedKvTable::from_table(&table).unwrap();
        assert_eq!(device.entries(), &[0_u32, 1, 2, 3]);
        assert_eq!(device.generation(), table.generation());
    }
}

#[cfg(test)]
mod packed_u16_shadow_tests {
    use super::*;
    use crate::paged_kv::PagedKvConfig;

    #[test]
    fn packed_u16_preflight_reports_exact_shape_without_allocation() {
        for mapped_pages in [1_usize, 2, 3, 5, 16, 255, 256] {
            let mut table = PagedKvTable::new(PagedKvConfig {
                page_size: 1,
                physical_pages: mapped_pages,
            })
            .unwrap();
            table.append(mapped_pages).unwrap();

            let preflight = preflight_packed_u16_page_map(&table).unwrap();
            assert_eq!(preflight.mapped_pages(), mapped_pages);
            assert_eq!(preflight.physical_pages(), mapped_pages);
            assert_eq!(
                preflight.packed_words(),
                mapped_pages.div_ceil(WGSL_PAGED_U16_PER_U32)
            );
        }
    }

    #[test]
    fn packed_u16_preflight_preserves_fail_closed_boundaries() {
        let empty = PagedKvTable::new(PagedKvConfig {
            page_size: 1,
            physical_pages: 1,
        })
        .unwrap();
        assert_eq!(
            preflight_packed_u16_page_map(&empty),
            Err(PagedDecodeError::EmptyTable)
        );

        let mut too_many_logical = PagedKvTable::new(PagedKvConfig {
            page_size: 1,
            physical_pages: WGSL_PAGED_MAX_LOGICAL_PAGES + 1,
        })
        .unwrap();
        too_many_logical
            .append(WGSL_PAGED_MAX_LOGICAL_PAGES + 1)
            .unwrap();
        assert_eq!(
            preflight_packed_u16_page_map(&too_many_logical),
            Err(PagedDecodeError::TooManyMappedPages {
                actual: WGSL_PAGED_MAX_LOGICAL_PAGES + 1,
                maximum: WGSL_PAGED_MAX_LOGICAL_PAGES,
            })
        );

        let mut too_wide_physical = PagedKvTable::new(PagedKvConfig {
            page_size: 1,
            physical_pages: WGSL_PAGED_U16_MAX_PHYSICAL_PAGES + 1,
        })
        .unwrap();
        too_wide_physical.append(1).unwrap();
        assert!(matches!(
            preflight_packed_u16_page_map(&too_wide_physical),
            Err(PagedDecodeError::PhysicalPageIndexExceedsU16 { .. })
        ));
    }
    #[test]
    fn packed_u16_round_trips_even_and_odd_page_counts() {
        for mapped_pages in [1_usize, 2, 3, 5, 16, 255, 256] {
            let mut table = PagedKvTable::new(PagedKvConfig {
                page_size: 1,
                physical_pages: mapped_pages,
            })
            .unwrap();
            table.append(mapped_pages).unwrap();

            let packed = WgpuPackedPagedKvTable16::from_table(&table).unwrap();
            assert_eq!(packed.mapped_pages(), mapped_pages);
            assert_eq!(
                packed.packed_entries().len(),
                mapped_pages.div_ceil(WGSL_PAGED_U16_PER_U32)
            );
            for logical_page in 0..mapped_pages {
                assert_eq!(
                    packed.physical_page(logical_page),
                    Some(u16::try_from(logical_page).unwrap())
                );
            }
            assert_eq!(packed.physical_page(mapped_pages), None);
        }
    }

    #[test]
    fn packed_u16_halves_fixed_page_map_words() {
        assert_eq!(WGSL_PAGED_U16_PACKED_WORDS, 128);
        assert_eq!(WGSL_PAGED_U16_UNIFORM_U32, 140);
        assert_eq!(WGSL_PAGED_U16_UNIFORM_U32 * 4, 560);
        assert_eq!(WGSL_PAGED_UNIFORM_U32 * 4, 1072);
    }

    #[test]
    fn packed_u16_shadow_uniform_has_exact_header_map_and_zero_padding() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 8,
        })
        .unwrap();
        table.append(17).unwrap();

        let packed = WgpuPackedPagedKvTable16::from_table(&table).unwrap();
        let tail = [64, 8, 2, 0x3f00_0000, 0x447a_0000, 16, 0, 0];
        let words = packed.shadow_uniform_words(tail).unwrap();

        assert_eq!(words.len(), WGSL_PAGED_U16_UNIFORM_U32);
        assert_eq!(&words[..4], &[17, 4, 8, 5]);
        assert_eq!(&words[4..12], &tail);
        assert_eq!(
            &words[12..12 + packed.packed_entries().len()],
            packed.packed_entries()
        );
        assert!(words[12 + packed.packed_entries().len()..]
            .iter()
            .all(|word| *word == 0));
        assert_eq!(
            core::mem::size_of_val(words.as_slice()),
            packed.theoretical_encoded_uniform_bytes()
        );
    }

    #[test]
    fn packed_u16_shadow_uniform_preserves_odd_page_padding() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 1,
            physical_pages: 3,
        })
        .unwrap();
        table.append(3).unwrap();

        let packed = WgpuPackedPagedKvTable16::from_table(&table).unwrap();
        let words = packed.shadow_uniform_words([0; 8]).unwrap();
        assert_eq!(packed.packed_entries().len(), 2);
        assert_eq!(words[12], 1 << 16);
        assert_eq!(words[13], 2);
    }
    #[test]
    fn packed_u16_preserves_epoch_generation_outside_page_entries() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 2,
            physical_pages: 4,
        })
        .unwrap();
        table.append(5).unwrap();
        let before = WgpuPackedPagedKvTable16::from_table(&table).unwrap();

        table.reset().unwrap();
        table.append(5).unwrap();
        let after = WgpuPackedPagedKvTable16::from_table(&table).unwrap();

        assert_eq!(before.packed_entries(), after.packed_entries());
        assert_eq!(before.generation(), 0);
        assert_eq!(after.generation(), 1);
    }

    #[test]
    fn physical_page_domain_above_u16_fails_closed() {
        let table = PagedKvTable::new(PagedKvConfig {
            page_size: 1,
            physical_pages: WGSL_PAGED_U16_MAX_PHYSICAL_PAGES + 1,
        })
        .unwrap();
        let mut table = table;
        table.append(1).unwrap();

        assert!(matches!(
            WgpuPackedPagedKvTable16::from_table(&table),
            Err(PagedDecodeError::PhysicalPageIndexExceedsU16 { .. })
        ));
    }
}
