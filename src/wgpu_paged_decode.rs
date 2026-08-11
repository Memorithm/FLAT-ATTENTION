//! M16 portable q_len=1 decode over caller-owned paged resident K/V storage.
//!
//! This first device consumer deliberately qualifies the single-sequence page-table
//! contract before adding batched page tables. K/V data buffers remain caller-owned;
//! FLAT owns only the compact uploaded page table and compiled compute pipeline.
//! The encode path does not submit, poll, map, synchronize, compact, or copy K/V.

use core::fmt;

use crate::paged_kv::{PagedKvError, PagedKvTable};
use crate::{FlatAttentionConfig, FlatAttentionError, FLAT_DECODE_PAGED_WGSL, WGSL_MAX_HEAD_DIM};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedDecodeLayout {
    pub q_elements: usize,
    pub output_elements: usize,
    pub lse_elements: usize,
    pub combined_elements: usize,
    pub q_bytes: u64,
    pub combined_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PagedDecodeError {
    Core(FlatAttentionError),
    Table(PagedKvError),
    EmptyTable,
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
    PipelineValidation(String),
}

impl fmt::Display for PagedDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::Table(error) => write!(f, "{error}"),
            Self::EmptyTable => write!(f, "paged decode requires at least one live KV token"),
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

/// Device-resident compact logical-page -> physical-page table.
#[derive(Debug)]
pub struct WgpuPagedKvTable {
    buffer: wgpu::Buffer,
    live_tokens: usize,
    page_size: usize,
    physical_pages: usize,
    mapped_pages: usize,
    generation: u64,
}

impl WgpuPagedKvTable {
    pub fn from_table(
        device: &wgpu::Device,
        table: &PagedKvTable,
    ) -> Result<Self, PagedDecodeError> {
        let telemetry = table.telemetry()?;
        let config = table.config();
        let mut entries = Vec::with_capacity(telemetry.mapped_pages);
        for logical_page in 0..telemetry.mapped_pages {
            let logical_token = logical_page.checked_mul(config.page_size).ok_or(
                PagedDecodeError::IndexSpaceExceeded {
                    elements: logical_page,
                },
            )?;
            let address = table
                .address(logical_token)
                .ok_or(PagedDecodeError::EmptyTable)?;
            entries.push(checked_u32(address.physical_page)?);
        }
        let bytes = encode_u32(&entries);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m16-paged-kv-table"),
            size: bytes.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        if !bytes.is_empty() {
            let mut mapped = buffer.slice(..bytes.len() as u64).get_mapped_range_mut();
            mapped.copy_from_slice(&bytes);
        }
        buffer.unmap();
        Ok(Self {
            buffer,
            live_tokens: telemetry.live_tokens,
            page_size: config.page_size,
            physical_pages: config.physical_pages,
            mapped_pages: telemetry.mapped_pages,
            generation: telemetry.generation,
        })
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

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
        self.mapped_pages
    }

    pub fn generation(&self) -> u64 {
        self.generation
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
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub config: FlatAttentionConfig,
    pub theta: f32,
    pub q_rope_position: usize,
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
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m16-paged-decode"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(FLAT_DECODE_PAGED_WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m16-paged-decode"),
            layout: None,
            module: &shader,
            entry_point: "flat_attention_decode_paged",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        match pollster::block_on(device.pop_error_scope()) {
            Some(error) => Err(PagedDecodeError::PipelineValidation(error.to_string())),
            None => Ok(Self { pipeline }),
        }
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
        if pass.page_table.live_tokens == 0 || pass.page_table.mapped_pages == 0 {
            return Err(PagedDecodeError::EmptyTable);
        }
        validate_geometry(pass.q_heads, pass.kv_heads, pass.head_dim)?;
        if !pass.theta.is_finite() || pass.theta <= 0.0 {
            return Err(PagedDecodeError::InvalidTheta(pass.theta));
        }
        if pass.config.causal
            && pass
                .q_rope_position
                .checked_add(1)
                .ok_or(FlatAttentionError::PositionOverflow)?
                < pass.page_table.live_tokens
        {
            return Err(PagedDecodeError::CausalVisibilityMismatch {
                query_position: pass.q_rope_position,
                kv_len: pass.page_table.live_tokens,
            });
        }

        let layout = Self::layout(pass.q_heads, pass.head_dim)?;
        let physical_rows = checked_mul(pass.page_table.physical_pages, pass.page_table.page_size)?;
        let kv_width = checked_mul(pass.kv_heads, pass.head_dim)?;
        let kv_elements = checked_mul(physical_rows, kv_width)?;
        let kv_bytes = bytes_for_f32(kv_elements)?;
        let page_table_bytes = checked_bytes_u32(pass.page_table.mapped_pages)?;
        validate_buffer("Q", pass.q, layout.q_bytes)?;
        validate_buffer("K", pass.k, kv_bytes)?;
        validate_buffer("V", pass.v, kv_bytes)?;
        validate_buffer("page_table", pass.page_table.buffer(), page_table_bytes)?;
        validate_buffer("O|LSE", pass.out_and_lse, layout.combined_bytes)?;

        let limits = device.limits();
        if pass.q_heads > limits.max_compute_workgroups_per_dimension as usize {
            return Err(PagedDecodeError::DispatchLimit {
                actual: pass.q_heads,
                maximum: limits.max_compute_workgroups_per_dimension,
            });
        }
        let maximum_storage_bytes = u64::from(limits.max_storage_buffer_binding_size);
        validate_storage_binding_size("Q", layout.q_bytes, maximum_storage_bytes)?;
        validate_storage_binding_size("K", kv_bytes, maximum_storage_bytes)?;
        validate_storage_binding_size("V", kv_bytes, maximum_storage_bytes)?;
        validate_storage_binding_size("page_table", page_table_bytes, maximum_storage_bytes)?;
        validate_storage_binding_size("O|LSE", layout.combined_bytes, maximum_storage_bytes)?;

        let scale = pass.config.resolved_scale(pass.head_dim)?;
        let params = [
            checked_u32(pass.page_table.live_tokens)?,
            checked_u32(pass.page_table.page_size)?,
            checked_u32(pass.page_table.physical_pages)?,
            checked_u32(pass.page_table.mapped_pages)?,
            checked_u32(pass.head_dim)?,
            checked_u32(pass.q_heads)?,
            checked_u32(pass.kv_heads)?,
            scale.to_bits(),
            pass.theta.to_bits(),
            checked_u32(pass.q_rope_position)?,
            0,
            0,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m16-paged-decode-params"),
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
                    resource: storage_binding(pass.page_table.buffer(), page_table_bytes),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: storage_binding(pass.out_and_lse, layout.combined_bytes),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
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
    u32::try_from(value).map_err(|_| PagedDecodeError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32(elements: usize) -> Result<u64, PagedDecodeError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| PagedDecodeError::IndexSpaceExceeded { elements })
}

fn checked_bytes_u32(elements: usize) -> Result<u64, PagedDecodeError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| PagedDecodeError::IndexSpaceExceeded { elements })
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}
