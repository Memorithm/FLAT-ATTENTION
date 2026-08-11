//! M16 resident paged K/V cache for the portable paged decode path.
//!
//! Physical storage is `[physical_pages, page_size, kv_heads * head_dim]`.
//! Logical token placement is owned by [`PagedKvTable`]. Appends copy only newly
//! produced rows from caller-owned resident buffers into their final physical
//! page locations. Existing live K/V rows are never compacted or recopied.
//!
//! This type records device-to-device copies only. It never maps, polls,
//! submits, synchronizes, or performs a host round-trip.

use core::fmt;

use crate::paged_kv::{PagedKvConfig, PagedKvError, PagedKvTable};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuPagedKvCacheError {
    Table(PagedKvError),
    ZeroDimension,
    ShapeOverflow,
    BufferTooSmall {
        tensor: &'static str,
        actual_bytes: u64,
        required_bytes: u64,
    },
    MissingBufferUsage {
        tensor: &'static str,
        required: &'static str,
    },
    DeviceBufferLimit {
        required_bytes: u64,
        maximum_bytes: u64,
    },
}

impl fmt::Display for WgpuPagedKvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Table(error) => write!(f, "{error}"),
            Self::ZeroDimension => write!(f, "paged resident KV dimensions must be non-zero"),
            Self::ShapeOverflow => write!(f, "paged resident KV shape overflows the address space"),
            Self::BufferTooSmall {
                tensor,
                actual_bytes,
                required_bytes,
            } => write!(
                f,
                "paged append buffer {tensor} contains {actual_bytes} bytes, requires at least {required_bytes}"
            ),
            Self::MissingBufferUsage { tensor, required } => {
                write!(f, "paged append buffer {tensor} requires WGPU usage {required}")
            }
            Self::DeviceBufferLimit {
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "paged resident KV requires {required_bytes} bytes per tensor, device maximum is {maximum_bytes}"
            ),
        }
    }
}

impl std::error::Error for WgpuPagedKvCacheError {}

impl From<PagedKvError> for WgpuPagedKvCacheError {
    fn from(value: PagedKvError) -> Self {
        Self::Table(value)
    }
}

/// Single-sequence resident paged K/V storage.
///
/// K is expected to be RoPE-rotated before append, matching the qualified M16
/// paged decode contract. V remains raw. Native `kv_heads` cardinality is
/// preserved for GQA/MQA.
pub struct WgpuPagedKvCache {
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    table: PagedKvTable,
    kv_heads: usize,
    head_dim: usize,
    row_bytes: u64,
    tensor_bytes: u64,
}

impl fmt::Debug for WgpuPagedKvCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuPagedKvCache")
            .field("config", &self.table.config())
            .field("kv_heads", &self.kv_heads)
            .field("head_dim", &self.head_dim)
            .field("len", &self.table.len())
            .field("generation", &self.table.generation())
            .finish_non_exhaustive()
    }
}

impl WgpuPagedKvCache {
    pub fn new(
        device: &wgpu::Device,
        config: PagedKvConfig,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, WgpuPagedKvCacheError> {
        if kv_heads == 0 || head_dim == 0 {
            return Err(WgpuPagedKvCacheError::ZeroDimension);
        }
        let table = PagedKvTable::new(config)?;
        let row_elements = kv_heads
            .checked_mul(head_dim)
            .ok_or(WgpuPagedKvCacheError::ShapeOverflow)?;
        let row_bytes = bytes_for_f32(row_elements)?;
        let capacity_tokens = config.capacity_tokens()?;
        let tensor_bytes = checked_u64_mul(capacity_tokens, row_bytes)?;
        let maximum_bytes = device.limits().max_buffer_size;
        if tensor_bytes > maximum_bytes {
            return Err(WgpuPagedKvCacheError::DeviceBufferLimit {
                required_bytes: tensor_bytes,
                maximum_bytes,
            });
        }
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m16-paged-resident-k"),
            size: tensor_bytes,
            usage,
            mapped_at_creation: false,
        });
        let v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m16-paged-resident-v"),
            size: tensor_bytes,
            usage,
            mapped_at_creation: false,
        });
        Ok(Self {
            k,
            v,
            table,
            kv_heads,
            head_dim,
            row_bytes,
            tensor_bytes,
        })
    }

    #[must_use]
    pub fn config(&self) -> PagedKvConfig {
        self.table.config()
    }

    #[must_use]
    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.table.generation()
    }

    #[must_use]
    pub fn k_buffer(&self) -> &wgpu::Buffer {
        &self.k
    }

    #[must_use]
    pub fn v_buffer(&self) -> &wgpu::Buffer {
        &self.v
    }

    #[must_use]
    pub fn table(&self) -> &PagedKvTable {
        &self.table
    }

    /// Logical reset only. Physical bytes remain resident but the generation is
    /// invalidated before pages are deterministically reused.
    pub fn reset(&mut self) -> Result<(), WgpuPagedKvCacheError> {
        self.table.reset()?;
        Ok(())
    }

    /// Record an append from contiguous sequence-major projected K/V rows.
    ///
    /// Source layout is `[append_len, kv_heads * head_dim]`. K must already be
    /// RoPE-rotated. Only newly appended rows are copied; the live prefix is not
    /// moved or rewritten. Metadata is committed only after all copy commands
    /// have been recorded successfully.
    pub fn record_append(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        k_source: &wgpu::Buffer,
        v_source: &wgpu::Buffer,
        append_len: usize,
    ) -> Result<usize, WgpuPagedKvCacheError> {
        if append_len == 0 {
            return Err(WgpuPagedKvCacheError::ZeroDimension);
        }
        let required_source_bytes = checked_u64_mul(append_len, self.row_bytes)?;
        validate_source("K", k_source, required_source_bytes)?;
        validate_source("V", v_source, required_source_bytes)?;

        let old_len = self.table.len();
        let mut staged_table = self.table.clone();
        staged_table.append(append_len)?;
        let new_len = staged_table.len();

        let mut logical = old_len;
        let mut source_row = 0usize;
        while logical < new_len {
            let address = staged_table
                .address(logical)
                .ok_or(WgpuPagedKvCacheError::ShapeOverflow)?;
            let page_remaining = self
                .table
                .config()
                .page_size
                .checked_sub(address.offset_in_page)
                .ok_or(WgpuPagedKvCacheError::ShapeOverflow)?;
            let rows = page_remaining.min(new_len - logical);

            let physical_row = address
                .physical_page
                .checked_mul(self.table.config().page_size)
                .and_then(|row| row.checked_add(address.offset_in_page))
                .ok_or(WgpuPagedKvCacheError::ShapeOverflow)?;
            let source_offset = checked_u64_mul(source_row, self.row_bytes)?;
            let destination_offset = checked_u64_mul(physical_row, self.row_bytes)?;
            let copy_bytes = checked_u64_mul(rows, self.row_bytes)?;
            let destination_end = destination_offset
                .checked_add(copy_bytes)
                .ok_or(WgpuPagedKvCacheError::ShapeOverflow)?;
            if destination_end > self.tensor_bytes {
                return Err(WgpuPagedKvCacheError::ShapeOverflow);
            }

            encoder.copy_buffer_to_buffer(
                k_source,
                source_offset,
                &self.k,
                destination_offset,
                copy_bytes,
            );
            encoder.copy_buffer_to_buffer(
                v_source,
                source_offset,
                &self.v,
                destination_offset,
                copy_bytes,
            );

            logical = logical
                .checked_add(rows)
                .ok_or(WgpuPagedKvCacheError::ShapeOverflow)?;
            source_row = source_row
                .checked_add(rows)
                .ok_or(WgpuPagedKvCacheError::ShapeOverflow)?;
        }

        self.table = staged_table;
        Ok(new_len)
    }
}

fn validate_source(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), WgpuPagedKvCacheError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(WgpuPagedKvCacheError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    if !buffer.usage().contains(wgpu::BufferUsages::COPY_SRC) {
        return Err(WgpuPagedKvCacheError::MissingBufferUsage {
            tensor,
            required: "COPY_SRC",
        });
    }
    Ok(())
}

fn checked_u64_mul<T>(a: T, b: u64) -> Result<u64, WgpuPagedKvCacheError>
where
    T: TryInto<u64>,
{
    let a = a
        .try_into()
        .map_err(|_| WgpuPagedKvCacheError::ShapeOverflow)?;
    a.checked_mul(b).ok_or(WgpuPagedKvCacheError::ShapeOverflow)
}

fn bytes_for_f32(elements: usize) -> Result<u64, WgpuPagedKvCacheError> {
    checked_u64_mul(elements, core::mem::size_of::<f32>() as u64)
}
