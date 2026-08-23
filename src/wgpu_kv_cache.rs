//! M14 caller-owned resident K/V cache storage for portable WGPU decode.
//!
//! The cache owns only its K/V storage buffers and logical length metadata. New
//! projected K/V rows are appended from caller-owned resident buffers through
//! `copy_buffer_to_buffer` commands recorded into the caller's encoder. No map,
//! poll, queue submission, or host round-trip is performed by this type.
//!
//! Physical layout is sequence-major projection layout with fixed capacity:
//!
//! ```text
//! K/V: [batch, capacity, kv_heads * head_dim]
//! ```
//!
//! Only rows `[0, len)` are logically live. Resetting the cache changes metadata
//! only; subsequent appends overwrite reused rows before they become live again.

use core::fmt;

/// Explicit resident-KV cache failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuResidentKvCacheError {
    ZeroDimension,
    ShapeOverflow,
    CapacityExceeded {
        current_len: usize,
        append_len: usize,
        capacity: usize,
    },
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

impl fmt::Display for WgpuResidentKvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "resident KV cache dimensions must be non-zero"),
            Self::ShapeOverflow => write!(f, "resident KV cache shape overflows the address space"),
            Self::CapacityExceeded {
                current_len,
                append_len,
                capacity,
            } => write!(
                f,
                "resident KV cache append {append_len} at length {current_len} exceeds capacity {capacity}"
            ),
            Self::BufferTooSmall {
                tensor,
                actual_bytes,
                required_bytes,
            } => write!(
                f,
                "resident append buffer {tensor} contains {actual_bytes} bytes, requires at least {required_bytes}"
            ),
            Self::MissingBufferUsage { tensor, required } => write!(
                f,
                "resident append buffer {tensor} must declare the {required} usage"
            ),
            Self::DeviceBufferLimit {
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "resident KV cache requires {required_bytes} bytes per tensor, device maximum is {maximum_bytes}"
            ),
        }
    }
}

impl std::error::Error for WgpuResidentKvCacheError {}

/// Fixed-capacity device-resident K/V cache.
///
/// The cache stores K and V with native `kv_heads`; GQA/MQA never expands them
/// to query-head cardinality. Appending copies only the newly produced rows into
/// their final capacity-strided locations and never copies the existing prefix.
pub struct WgpuResidentKvCache {
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    batch: usize,
    kv_heads: usize,
    capacity: usize,
    head_dim: usize,
    len: usize,
    row_bytes: u64,
}

impl fmt::Debug for WgpuResidentKvCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuResidentKvCache")
            .field("batch", &self.batch)
            .field("kv_heads", &self.kv_heads)
            .field("capacity", &self.capacity)
            .field("head_dim", &self.head_dim)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl WgpuResidentKvCache {
    /// Allocate fixed-capacity resident K and V storage.
    pub fn new(
        device: &wgpu::Device,
        batch: usize,
        kv_heads: usize,
        capacity: usize,
        head_dim: usize,
    ) -> Result<Self, WgpuResidentKvCacheError> {
        if batch == 0 || kv_heads == 0 || capacity == 0 || head_dim == 0 {
            return Err(WgpuResidentKvCacheError::ZeroDimension);
        }
        let row_elements = kv_heads
            .checked_mul(head_dim)
            .ok_or(WgpuResidentKvCacheError::ShapeOverflow)?;
        let row_bytes = bytes_for_f32(row_elements)?;
        let tensor_bytes = checked_u64_mul(checked_usize_mul(batch, capacity)?, row_bytes)?;
        let maximum_bytes = device.limits().max_buffer_size;
        if tensor_bytes > maximum_bytes {
            return Err(WgpuResidentKvCacheError::DeviceBufferLimit {
                required_bytes: tensor_bytes,
                maximum_bytes,
            });
        }
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m14-resident-k"),
            size: tensor_bytes,
            usage,
            mapped_at_creation: false,
        });
        let v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-m14-resident-v"),
            size: tensor_bytes,
            usage,
            mapped_at_creation: false,
        });
        Ok(Self {
            k,
            v,
            batch,
            kv_heads,
            capacity,
            head_dim,
            len: 0,
            row_bytes,
        })
    }

    #[must_use]
    pub fn batch(&self) -> usize {
        self.batch
    }

    #[must_use]
    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.capacity - self.len
    }

    #[must_use]
    pub fn k_buffer(&self) -> &wgpu::Buffer {
        &self.k
    }

    #[must_use]
    pub fn v_buffer(&self) -> &wgpu::Buffer {
        &self.v
    }

    /// Logical reset. Stale bytes remain physically resident but are outside
    /// `[0, len)` and therefore must not be read by a conforming decode path.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Record an append from resident sequence-major projected K/V buffers.
    ///
    /// Source layout is `[batch, append_len, kv_heads * head_dim]`. Source
    /// buffers must have `COPY_SRC` usage. The operation copies only appended
    /// rows; the existing cache prefix is never moved or rewritten.
    ///
    /// The logical length advances after all copy commands have been recorded.
    /// The caller must submit the encoder before using the returned new length.
    pub fn record_append(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        k_source: &wgpu::Buffer,
        v_source: &wgpu::Buffer,
        append_len: usize,
    ) -> Result<usize, WgpuResidentKvCacheError> {
        if append_len == 0 {
            return Err(WgpuResidentKvCacheError::ZeroDimension);
        }
        let new_len = self
            .len
            .checked_add(append_len)
            .ok_or(WgpuResidentKvCacheError::ShapeOverflow)?;
        if new_len > self.capacity {
            return Err(WgpuResidentKvCacheError::CapacityExceeded {
                current_len: self.len,
                append_len,
                capacity: self.capacity,
            });
        }

        let source_batch_bytes = checked_u64_mul(append_len, self.row_bytes)?;
        let required_source_bytes = checked_u64_mul(self.batch, source_batch_bytes)?;
        validate_source("K", k_source, required_source_bytes)?;
        validate_source("V", v_source, required_source_bytes)?;

        for batch in 0..self.batch {
            let source_offset = checked_u64_mul(batch, source_batch_bytes)?;
            let destination_row = checked_usize_mul(batch, self.capacity)?
                .checked_add(self.len)
                .ok_or(WgpuResidentKvCacheError::ShapeOverflow)?;
            let destination_offset = checked_u64_mul(destination_row, self.row_bytes)?;
            encoder.copy_buffer_to_buffer(
                k_source,
                source_offset,
                &self.k,
                destination_offset,
                source_batch_bytes,
            );
            encoder.copy_buffer_to_buffer(
                v_source,
                source_offset,
                &self.v,
                destination_offset,
                source_batch_bytes,
            );
        }
        self.len = new_len;
        Ok(new_len)
    }
}

fn validate_source(
    tensor: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), WgpuResidentKvCacheError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(WgpuResidentKvCacheError::BufferTooSmall {
            tensor,
            actual_bytes,
            required_bytes,
        });
    }
    if !buffer.usage().contains(wgpu::BufferUsages::COPY_SRC) {
        return Err(WgpuResidentKvCacheError::MissingBufferUsage {
            tensor,
            required: "COPY_SRC",
        });
    }
    Ok(())
}

fn checked_usize_mul(a: usize, b: usize) -> Result<usize, WgpuResidentKvCacheError> {
    a.checked_mul(b)
        .ok_or(WgpuResidentKvCacheError::ShapeOverflow)
}

fn checked_u64_mul<T>(a: T, b: u64) -> Result<u64, WgpuResidentKvCacheError>
where
    T: TryInto<u64>,
{
    let a = a
        .try_into()
        .map_err(|_| WgpuResidentKvCacheError::ShapeOverflow)?;
    a.checked_mul(b)
        .ok_or(WgpuResidentKvCacheError::ShapeOverflow)
}

fn bytes_for_f32(elements: usize) -> Result<u64, WgpuResidentKvCacheError> {
    checked_u64_mul(elements, core::mem::size_of::<f32>() as u64)
}
