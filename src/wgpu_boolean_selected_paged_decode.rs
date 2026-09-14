//! BKV-K6 correctness-first numerical consumer for Boolean-selected paged KV.
//!
//! This module is research-only and is intentionally not wired into the stable
//! reusable API yet. It consumes a BKV-K5 [`BooleanIndexedKvSelection`] while
//! keeping the numerical [`PagedKvTable`] authoritative. Sparse page selection
//! preserves each original logical page; selected pages are never densely
//! renumbered before the numerical attention pass.

use core::fmt;
use std::borrow::Cow;

use wgpu::util::DeviceExt;

use crate::api::boolean_kv_paged_selection::BooleanIndexedKvSelection;
use crate::paged_kv::{PagedKvError, PagedKvTable};
use crate::{FlatAttentionConfig, FlatAttentionError, WGSL_MAX_HEAD_DIM};

/// WGSL source for the BKV-K6 sparse numerical consumer.
pub const BOOLEAN_SELECTED_PAGED_DECODE_WGSL: &str =
    include_str!("../shaders/flat_decode_boolean_selected_paged.wgsl");

/// Portable descriptor capacity. This mirrors M16's bounded uniform table.
pub const BOOLEAN_SELECTED_PAGED_MAX_PAGES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanSelectedPagedDescriptor {
    pub physical_page: usize,
    pub logical_page: usize,
    pub live_tokens: usize,
}

/// Frozen, validated sparse page table derived from BKV-K5 selection metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanSelectedPagedKvTable {
    entries: Vec<BooleanSelectedPagedDescriptor>,
    full_live_tokens: usize,
    selected_live_tokens: usize,
    page_size: usize,
    physical_pages: usize,
    generation: u64,
}

impl BooleanSelectedPagedKvTable {
    /// Revalidate a BKV-K5 selection against the current authoritative numerical
    /// paged table. Stale generation, forged physical mappings, reordered or
    /// duplicated logical pages, and incorrect partial-page occupancy fail closed.
    pub fn from_selection(
        selection: &BooleanIndexedKvSelection,
        table: &PagedKvTable,
    ) -> Result<Self, BooleanSelectedPagedDecodeError> {
        let telemetry = table.telemetry()?;
        let config = table.config();

        if selection.generation != telemetry.generation {
            return Err(BooleanSelectedPagedDecodeError::GenerationMismatch {
                selection_generation: selection.generation,
                table_generation: telemetry.generation,
            });
        }
        if selection.live_tokens != telemetry.live_tokens
            || selection.mapped_pages != telemetry.mapped_pages
        {
            return Err(BooleanSelectedPagedDecodeError::SnapshotMismatch {
                selection_live_tokens: selection.live_tokens,
                table_live_tokens: telemetry.live_tokens,
                selection_mapped_pages: selection.mapped_pages,
                table_mapped_pages: telemetry.mapped_pages,
            });
        }
        if selection.selected_pages.is_empty() {
            return Err(BooleanSelectedPagedDecodeError::EmptySelection);
        }
        if selection.selected_pages.len() > BOOLEAN_SELECTED_PAGED_MAX_PAGES {
            return Err(BooleanSelectedPagedDecodeError::TooManySelectedPages {
                actual: selection.selected_pages.len(),
                maximum: BOOLEAN_SELECTED_PAGED_MAX_PAGES,
            });
        }

        let mut entries = Vec::with_capacity(selection.selected_pages.len());
        let mut selected_live_tokens = 0usize;
        let mut previous_logical_page = None;

        for selected in &selection.selected_pages {
            if selected.logical_page >= telemetry.mapped_pages {
                return Err(BooleanSelectedPagedDecodeError::LogicalPageOutOfRange {
                    logical_page: selected.logical_page,
                    mapped_pages: telemetry.mapped_pages,
                });
            }
            if let Some(previous) = previous_logical_page {
                if selected.logical_page <= previous {
                    return Err(BooleanSelectedPagedDecodeError::NonIncreasingLogicalPages {
                        previous,
                        current: selected.logical_page,
                    });
                }
            }
            previous_logical_page = Some(selected.logical_page);

            let first_token = selected.logical_page.checked_mul(config.page_size).ok_or(
                BooleanSelectedPagedDecodeError::IndexSpaceExceeded {
                    elements: selected.logical_page,
                },
            )?;
            let address = table.address(first_token).ok_or(
                BooleanSelectedPagedDecodeError::MissingAuthoritativePage {
                    logical_page: selected.logical_page,
                },
            )?;
            if selected.physical_page != address.physical_page {
                return Err(BooleanSelectedPagedDecodeError::PhysicalPageMismatch {
                    logical_page: selected.logical_page,
                    selected_physical_page: selected.physical_page,
                    authoritative_physical_page: address.physical_page,
                });
            }

            let remaining = telemetry.live_tokens.checked_sub(first_token).ok_or(
                BooleanSelectedPagedDecodeError::IndexSpaceExceeded {
                    elements: first_token,
                },
            )?;
            let expected_live_tokens = remaining.min(config.page_size);
            if selected.live_tokens != expected_live_tokens {
                return Err(BooleanSelectedPagedDecodeError::LiveTokenMismatch {
                    logical_page: selected.logical_page,
                    selected_live_tokens: selected.live_tokens,
                    authoritative_live_tokens: expected_live_tokens,
                });
            }
            if selected.physical_page >= config.physical_pages {
                return Err(BooleanSelectedPagedDecodeError::PhysicalPageOutOfRange {
                    physical_page: selected.physical_page,
                    physical_pages: config.physical_pages,
                });
            }

            selected_live_tokens = selected_live_tokens
                .checked_add(selected.live_tokens)
                .ok_or(BooleanSelectedPagedDecodeError::IndexSpaceExceeded {
                    elements: selected_live_tokens,
                })?;
            entries.push(BooleanSelectedPagedDescriptor {
                physical_page: selected.physical_page,
                logical_page: selected.logical_page,
                live_tokens: selected.live_tokens,
            });
        }

        Ok(Self {
            entries,
            full_live_tokens: telemetry.live_tokens,
            selected_live_tokens,
            page_size: config.page_size,
            physical_pages: config.physical_pages,
            generation: telemetry.generation,
        })
    }

    #[must_use]
    pub fn entries(&self) -> &[BooleanSelectedPagedDescriptor] {
        &self.entries
    }

    #[must_use]
    pub fn full_live_tokens(&self) -> usize {
        self.full_live_tokens
    }

    #[must_use]
    pub fn selected_live_tokens(&self) -> usize {
        self.selected_live_tokens
    }

    #[must_use]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    #[must_use]
    pub fn physical_pages(&self) -> usize {
        self.physical_pages
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Revalidate frozen sparse descriptors against the current authoritative
    /// numerical page table immediately before GPU submission.
    pub fn validate_against(
        &self,
        table: &PagedKvTable,
    ) -> Result<(), BooleanSelectedPagedDecodeError> {
        let telemetry = table.telemetry()?;
        let config = table.config();
        let frozen_mapped_pages = self.full_live_tokens.div_ceil(self.page_size);

        if self.generation != telemetry.generation {
            return Err(BooleanSelectedPagedDecodeError::GenerationMismatch {
                selection_generation: self.generation,
                table_generation: telemetry.generation,
            });
        }
        if self.full_live_tokens != telemetry.live_tokens
            || frozen_mapped_pages != telemetry.mapped_pages
            || self.page_size != config.page_size
            || self.physical_pages != config.physical_pages
        {
            return Err(BooleanSelectedPagedDecodeError::SnapshotMismatch {
                selection_live_tokens: self.full_live_tokens,
                table_live_tokens: telemetry.live_tokens,
                selection_mapped_pages: frozen_mapped_pages,
                table_mapped_pages: telemetry.mapped_pages,
            });
        }

        for entry in &self.entries {
            let first_token = entry.logical_page.checked_mul(config.page_size).ok_or(
                BooleanSelectedPagedDecodeError::IndexSpaceExceeded {
                    elements: entry.logical_page,
                },
            )?;
            let address = table.address(first_token).ok_or(
                BooleanSelectedPagedDecodeError::MissingAuthoritativePage {
                    logical_page: entry.logical_page,
                },
            )?;
            if entry.physical_page != address.physical_page {
                return Err(BooleanSelectedPagedDecodeError::PhysicalPageMismatch {
                    logical_page: entry.logical_page,
                    selected_physical_page: entry.physical_page,
                    authoritative_physical_page: address.physical_page,
                });
            }
            let remaining = telemetry.live_tokens.checked_sub(first_token).ok_or(
                BooleanSelectedPagedDecodeError::IndexSpaceExceeded {
                    elements: first_token,
                },
            )?;
            let expected_live_tokens = remaining.min(config.page_size);
            if entry.live_tokens != expected_live_tokens {
                return Err(BooleanSelectedPagedDecodeError::LiveTokenMismatch {
                    logical_page: entry.logical_page,
                    selected_live_tokens: entry.live_tokens,
                    authoritative_live_tokens: expected_live_tokens,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanSelectedDecodeLayout {
    pub q_elements: usize,
    pub output_elements: usize,
    pub lse_elements: usize,
    pub combined_elements: usize,
    pub q_bytes: u64,
    pub combined_bytes: u64,
}

pub struct BooleanSelectedDecodePass<'a> {
    pub q: &'a wgpu::Buffer,
    /// Pre-rotated K in physical layout `[physical_pages, page_size, kv_heads * head_dim]`.
    pub k: &'a wgpu::Buffer,
    /// Raw V in the same physical layout as K.
    pub v: &'a wgpu::Buffer,
    pub page_table: &'a BooleanSelectedPagedKvTable,
    /// Current numerical table, revalidated immediately before encoding.
    pub authoritative_table: &'a PagedKvTable,
    pub out_and_lse: &'a wgpu::Buffer,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub config: FlatAttentionConfig,
    pub theta: f32,
    pub q_rope_position: usize,
    pub q_causal_position: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BooleanSelectedPagedDecodeError {
    Core(FlatAttentionError),
    Table(PagedKvError),
    EmptySelection,
    TooManySelectedPages {
        actual: usize,
        maximum: usize,
    },
    GenerationMismatch {
        selection_generation: u64,
        table_generation: u64,
    },
    SnapshotMismatch {
        selection_live_tokens: usize,
        table_live_tokens: usize,
        selection_mapped_pages: usize,
        table_mapped_pages: usize,
    },
    LogicalPageOutOfRange {
        logical_page: usize,
        mapped_pages: usize,
    },
    NonIncreasingLogicalPages {
        previous: usize,
        current: usize,
    },
    MissingAuthoritativePage {
        logical_page: usize,
    },
    PhysicalPageMismatch {
        logical_page: usize,
        selected_physical_page: usize,
        authoritative_physical_page: usize,
    },
    PhysicalPageOutOfRange {
        physical_page: usize,
        physical_pages: usize,
    },
    LiveTokenMismatch {
        logical_page: usize,
        selected_live_tokens: usize,
        authoritative_live_tokens: usize,
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
        full_kv_len: usize,
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
    UniformBindingTooLarge {
        required_bytes: u64,
        maximum_bytes: u64,
    },
    PipelineValidation(String),
}

impl fmt::Display for BooleanSelectedPagedDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::Table(error) => write!(f, "{error}"),
            Self::EmptySelection => write!(f, "BIKV selected decode requires at least one numerical survivor page"),
            Self::TooManySelectedPages { actual, maximum } => write!(f, "BIKV selected decode has {actual} pages, portable maximum is {maximum}"),
            Self::GenerationMismatch { selection_generation, table_generation } => write!(f, "BIKV selection generation {selection_generation} does not match numerical table generation {table_generation}"),
            Self::SnapshotMismatch { selection_live_tokens, table_live_tokens, selection_mapped_pages, table_mapped_pages } => write!(f, "BIKV selection snapshot ({selection_live_tokens} live tokens, {selection_mapped_pages} pages) does not match numerical table ({table_live_tokens} live tokens, {table_mapped_pages} pages)"),
            Self::LogicalPageOutOfRange { logical_page, mapped_pages } => write!(f, "BIKV logical page {logical_page} is outside {mapped_pages} mapped numerical pages"),
            Self::NonIncreasingLogicalPages { previous, current } => write!(f, "BIKV selected logical pages must be strictly increasing: {previous} then {current}"),
            Self::MissingAuthoritativePage { logical_page } => write!(f, "numerical KV has no authoritative mapping for selected logical page {logical_page}"),
            Self::PhysicalPageMismatch { logical_page, selected_physical_page, authoritative_physical_page } => write!(f, "BIKV logical page {logical_page} selects physical page {selected_physical_page}, authoritative mapping is {authoritative_physical_page}"),
            Self::PhysicalPageOutOfRange { physical_page, physical_pages } => write!(f, "BIKV physical page {physical_page} is outside {physical_pages} numerical pages"),
            Self::LiveTokenMismatch { logical_page, selected_live_tokens, authoritative_live_tokens } => write!(f, "BIKV logical page {logical_page} carries {selected_live_tokens} live tokens, authoritative occupancy is {authoritative_live_tokens}"),
            Self::InvalidHeadGrouping { q_heads, kv_heads } => write!(f, "q_heads ({q_heads}) must be exactly divisible by kv_heads ({kv_heads})"),
            Self::UnsupportedHeadDim { actual, maximum } => write!(f, "head_dim {actual} exceeds portable maximum {maximum}"),
            Self::InvalidTheta(theta) => write!(f, "BIKV selected decode RoPE theta must be finite and positive, got {theta}"),
            Self::CausalVisibilityMismatch { query_position, full_kv_len } => write!(f, "BIKV causal decode query position {query_position} cannot see all {full_kv_len} live KV tokens from which survivors were selected"),
            Self::IndexSpaceExceeded { elements } => write!(f, "BIKV selected decode exceeds WGPU u32 index space at {elements} elements"),
            Self::DispatchLimit { actual, maximum } => write!(f, "BIKV selected decode requires {actual} workgroups, device maximum is {maximum}"),
            Self::BufferTooSmall { tensor, actual_bytes, required_bytes } => write!(f, "buffer {tensor} contains {actual_bytes} bytes, requires at least {required_bytes}"),
            Self::StorageBindingTooLarge { tensor, required_bytes, maximum_bytes } => write!(f, "storage binding {tensor} requires {required_bytes} bytes, device maximum is {maximum_bytes}"),
            Self::UniformBindingTooLarge { required_bytes, maximum_bytes } => write!(f, "BIKV selected decode uniform requires {required_bytes} bytes, device maximum is {maximum_bytes}"),
            Self::PipelineValidation(error) => write!(f, "BIKV selected decode pipeline validation failed: {error}"),
        }
    }
}

impl std::error::Error for BooleanSelectedPagedDecodeError {}

impl From<FlatAttentionError> for BooleanSelectedPagedDecodeError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

impl From<PagedKvError> for BooleanSelectedPagedDecodeError {
    fn from(value: PagedKvError) -> Self {
        Self::Table(value)
    }
}

pub struct WgpuBooleanSelectedPagedDecodePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for WgpuBooleanSelectedPagedDecodePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuBooleanSelectedPagedDecodePipeline")
            .finish_non_exhaustive()
    }
}

impl WgpuBooleanSelectedPagedDecodePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, BooleanSelectedPagedDecodeError> {
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-bkv-k6-selected-paged-decode"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BOOLEAN_SELECTED_PAGED_DECODE_WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-bkv-k6-selected-paged-decode"),
            layout: None,
            module: &shader,
            entry_point: Some("flat_attention_decode_boolean_selected_paged"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        match pollster::block_on(error_scope.pop()) {
            Some(error) => Err(BooleanSelectedPagedDecodeError::PipelineValidation(
                error.to_string(),
            )),
            None => Ok(Self { pipeline }),
        }
    }

    pub fn layout(
        q_heads: usize,
        head_dim: usize,
    ) -> Result<BooleanSelectedDecodeLayout, BooleanSelectedPagedDecodeError> {
        validate_geometry(q_heads, 1, head_dim)?;
        let q_elements = checked_mul(q_heads, head_dim)?;
        let lse_elements = q_heads;
        let combined_elements = q_elements
            .checked_add(lse_elements)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        Ok(BooleanSelectedDecodeLayout {
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
    ) -> Result<wgpu::Buffer, BooleanSelectedPagedDecodeError> {
        let layout = Self::layout(q_heads, head_dim)?;
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-bkv-k6-selected-paged-o-lse"),
            size: layout.combined_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: BooleanSelectedDecodePass<'_>,
    ) -> Result<BooleanSelectedDecodeLayout, BooleanSelectedPagedDecodeError> {
        pass.page_table.validate_against(pass.authoritative_table)?;
        if pass.page_table.entries.is_empty() || pass.page_table.selected_live_tokens == 0 {
            return Err(BooleanSelectedPagedDecodeError::EmptySelection);
        }
        if pass.page_table.entries.len() > BOOLEAN_SELECTED_PAGED_MAX_PAGES {
            return Err(BooleanSelectedPagedDecodeError::TooManySelectedPages {
                actual: pass.page_table.entries.len(),
                maximum: BOOLEAN_SELECTED_PAGED_MAX_PAGES,
            });
        }
        validate_geometry(pass.q_heads, pass.kv_heads, pass.head_dim)?;
        if !pass.theta.is_finite() || pass.theta <= 0.0 {
            return Err(BooleanSelectedPagedDecodeError::InvalidTheta(pass.theta));
        }
        if pass.config.causal
            && pass
                .q_causal_position
                .checked_add(1)
                .ok_or(FlatAttentionError::PositionOverflow)?
                < pass.page_table.full_live_tokens
        {
            return Err(BooleanSelectedPagedDecodeError::CausalVisibilityMismatch {
                query_position: pass.q_causal_position,
                full_kv_len: pass.page_table.full_live_tokens,
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
            return Err(BooleanSelectedPagedDecodeError::DispatchLimit {
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
        let mut params = Vec::with_capacity(16 + 4 * BOOLEAN_SELECTED_PAGED_MAX_PAGES);
        params.extend_from_slice(&[
            checked_u32(pass.page_table.full_live_tokens)?,
            checked_u32(pass.page_table.page_size)?,
            checked_u32(pass.page_table.physical_pages)?,
            checked_u32(pass.page_table.entries.len())?,
            checked_u32(pass.page_table.selected_live_tokens)?,
            checked_u32(pass.head_dim)?,
            checked_u32(pass.q_heads)?,
            checked_u32(pass.kv_heads)?,
            scale.to_bits(),
            pass.theta.to_bits(),
            checked_u32(pass.q_rope_position)?,
            checked_u32(pass.q_causal_position)?,
            u32::from(pass.config.causal),
            0,
            0,
            0,
        ]);
        for entry in &pass.page_table.entries {
            params.extend_from_slice(&[
                checked_u32(entry.physical_page)?,
                checked_u32(entry.logical_page)?,
                checked_u32(entry.live_tokens)?,
                0,
            ]);
        }
        params.resize(16 + 4 * BOOLEAN_SELECTED_PAGED_MAX_PAGES, 0);
        let params_bytes = encode_u32(&params);
        let params_len = params_bytes.len() as u64;
        let maximum_uniform_bytes = limits.max_uniform_buffer_binding_size;
        if params_len > maximum_uniform_bytes {
            return Err(BooleanSelectedPagedDecodeError::UniformBindingTooLarge {
                required_bytes: params_len,
                maximum_bytes: maximum_uniform_bytes,
            });
        }
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flat-bkv-k6-selected-paged-params"),
            contents: &params_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-bkv-k6-selected-paged-bind-group"),
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
            label: Some("flat-bkv-k6-selected-paged-decode"),
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
) -> Result<(), BooleanSelectedPagedDecodeError> {
    if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
        return Err(BooleanSelectedPagedDecodeError::InvalidHeadGrouping { q_heads, kv_heads });
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(FlatAttentionError::InvalidRotaryHeadDim { head_dim }.into());
    }
    if head_dim > WGSL_MAX_HEAD_DIM {
        return Err(BooleanSelectedPagedDecodeError::UnsupportedHeadDim {
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
) -> Result<(), BooleanSelectedPagedDecodeError> {
    if buffer.size() < required_bytes {
        return Err(BooleanSelectedPagedDecodeError::BufferTooSmall {
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
) -> Result<(), BooleanSelectedPagedDecodeError> {
    if required_bytes > maximum_bytes {
        return Err(BooleanSelectedPagedDecodeError::StorageBindingTooLarge {
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

fn checked_mul(a: usize, b: usize) -> Result<usize, BooleanSelectedPagedDecodeError> {
    a.checked_mul(b)
        .ok_or_else(|| FlatAttentionError::ShapeOverflow.into())
}

fn checked_u32(value: usize) -> Result<u32, BooleanSelectedPagedDecodeError> {
    u32::try_from(value)
        .map_err(|_| BooleanSelectedPagedDecodeError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32(len: usize) -> Result<u64, BooleanSelectedPagedDecodeError> {
    let bytes = len
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes)
        .map_err(|_| BooleanSelectedPagedDecodeError::IndexSpaceExceeded { elements: len })
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}
