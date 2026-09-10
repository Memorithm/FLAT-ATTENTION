//! M16 resident paged K/V cache for the portable paged decode path.
//!
//! Physical storage is `[physical_pages, page_size, kv_heads * head_dim]`.
//! Logical token placement is owned by [`PagedKvTable`]. Appends copy only newly
//! produced rows from caller-owned resident buffers into their final physical
//! page locations. Existing live K/V rows are never compacted or recopied.
//!
//! This type records device-to-device copies only. It never maps, polls,
//! submits, synchronizes, or performs a host round-trip unless the caller uses
//! [`WgpuPagedKvCache::append_and_submit`], the managed convenience path.

use core::fmt;
use std::sync::Arc;

use crate::paged_kv::{PagedKvConfig, PagedKvError, PagedKvTable};

/// Opaque logical checkpoint for one [`WgpuPagedKvCache`] lineage.
///
/// A checkpoint records metadata only; it does not copy or snapshot K/V bytes.
/// Appends preserve checkpoint validity because they leave the captured prefix
/// untouched. Any truncate or reset changes the cache lineage and invalidates
/// checkpoints from the previous branch before released rows/pages may be
/// reused.
#[derive(Clone)]
pub struct WgpuPagedKvCheckpoint {
    origin: Arc<()>,
    len: usize,
    generation: u64,
    branch_epoch: u64,
}

impl fmt::Debug for WgpuPagedKvCheckpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuPagedKvCheckpoint")
            .field("len", &self.len)
            .field("generation", &self.generation)
            .field("branch_epoch", &self.branch_epoch)
            .finish_non_exhaustive()
    }
}

impl WgpuPagedKvCheckpoint {
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[must_use]
    pub fn branch_epoch(&self) -> u64 {
        self.branch_epoch
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
    BranchEpochOverflow,
    ForeignCheckpoint,
    UnsubmittedRecordedWrites,
    CheckpointGenerationMismatch {
        checkpoint_generation: u64,
        current_generation: u64,
    },
    CheckpointBranchMismatch {
        checkpoint_branch_epoch: u64,
        current_branch_epoch: u64,
    },
    CheckpointAhead {
        checkpoint_len: usize,
        current_len: usize,
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
            Self::BranchEpochOverflow => {
                write!(f, "paged resident KV branch epoch counter overflowed")
            }
            Self::ForeignCheckpoint => write!(
                f,
                "paged resident KV checkpoint belongs to a different cache instance"
            ),
            Self::UnsubmittedRecordedWrites => write!(
                f,
                "destructive paged resident KV transition is blocked because externally recorded GPU writes may still be submit-able"
            ),
            Self::CheckpointGenerationMismatch {
                checkpoint_generation,
                current_generation,
            } => write!(
                f,
                "paged resident KV checkpoint generation {checkpoint_generation} does not match current generation {current_generation}"
            ),
            Self::CheckpointBranchMismatch {
                checkpoint_branch_epoch,
                current_branch_epoch,
            } => write!(
                f,
                "paged resident KV checkpoint branch epoch {checkpoint_branch_epoch} does not match current branch epoch {current_branch_epoch}"
            ),
            Self::CheckpointAhead {
                checkpoint_len,
                current_len,
            } => write!(
                f,
                "paged resident KV checkpoint length {checkpoint_len} exceeds current live length {current_len}"
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
    checkpoint_origin: Arc<()>,
    branch_epoch: u64,
    unsubmitted_recorded_writes: bool,
}

impl fmt::Debug for WgpuPagedKvCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuPagedKvCache")
            .field("config", &self.table.config())
            .field("kv_heads", &self.kv_heads)
            .field("head_dim", &self.head_dim)
            .field("len", &self.table.len())
            .field("generation", &self.table.generation())
            .field("branch_epoch", &self.branch_epoch)
            .field(
                "unsubmitted_recorded_writes",
                &self.unsubmitted_recorded_writes,
            )
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
            checkpoint_origin: Arc::new(()),
            branch_epoch: 0,
            unsubmitted_recorded_writes: false,
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

    /// Current append-only branch epoch within this table generation.
    #[must_use]
    pub fn branch_epoch(&self) -> u64 {
        self.branch_epoch
    }

    /// Whether externally recorded append commands may still be submit-able.
    ///
    /// Once [`Self::record_append`] is used directly, this remains true for the
    /// lifetime of this cache because WGPU exposes no safe way for the cache to
    /// prove that every caller-owned encoder containing its writes has been
    /// consumed. Destructive lifecycle operations therefore stay fail-closed.
    /// Use [`Self::append_and_submit`] when later truncate/reset/restore is
    /// required.
    #[must_use]
    pub fn has_unsubmitted_recorded_writes(&self) -> bool {
        self.unsubmitted_recorded_writes
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

    /// Capture a metadata-only checkpoint of the current append-only lineage.
    ///
    /// The returned checkpoint does not preserve bytes independently. It stays
    /// valid across appends because existing live rows are not rewritten. A
    /// truncate/restore that actually shrinks the cache advances the branch
    /// epoch; reset advances the table generation. Either transition makes old
    /// checkpoints fail closed.
    #[must_use]
    pub fn checkpoint(&self) -> WgpuPagedKvCheckpoint {
        WgpuPagedKvCheckpoint {
            origin: Arc::clone(&self.checkpoint_origin),
            len: self.len(),
            generation: self.generation(),
            branch_epoch: self.branch_epoch,
        }
    }

    /// Validate that a checkpoint still names a live prefix of this exact cache.
    ///
    /// This is a metadata-only provenance check. It performs no device I/O and
    /// does not mutate the cache. Append-only growth preserves validity because
    /// the captured prefix remains live. A foreign cache, reset generation,
    /// destructive branch change, or checkpoint length ahead of the current
    /// cache fails closed.
    ///
    /// This method deliberately does not reject
    /// [`Self::has_unsubmitted_recorded_writes`]. That flag governs whether a
    /// destructive transition is safe; it does not by itself invalidate the
    /// already-captured prefix lineage. [`Self::restore`] still fails through
    /// [`Self::truncate`] when a shrinking restore is blocked by such writes.
    pub fn validate_checkpoint(
        &self,
        checkpoint: &WgpuPagedKvCheckpoint,
    ) -> Result<(), WgpuPagedKvCacheError> {
        if !Arc::ptr_eq(&self.checkpoint_origin, &checkpoint.origin) {
            return Err(WgpuPagedKvCacheError::ForeignCheckpoint);
        }

        let current_generation = self.generation();
        if checkpoint.generation != current_generation {
            return Err(WgpuPagedKvCacheError::CheckpointGenerationMismatch {
                checkpoint_generation: checkpoint.generation,
                current_generation,
            });
        }
        if checkpoint.branch_epoch != self.branch_epoch {
            return Err(WgpuPagedKvCacheError::CheckpointBranchMismatch {
                checkpoint_branch_epoch: checkpoint.branch_epoch,
                current_branch_epoch: self.branch_epoch,
            });
        }

        let current_len = self.len();
        if checkpoint.len > current_len {
            return Err(WgpuPagedKvCacheError::CheckpointAhead {
                checkpoint_len: checkpoint.len,
                current_len,
            });
        }
        Ok(())
    }

    /// Restore an append-only checkpoint by logically rewinding to its length.
    ///
    /// This is not a physical snapshot restore. It succeeds only when the
    /// checkpoint belongs to this cache, its generation and branch epoch are
    /// unchanged, and its captured prefix is still live. Any restore that would
    /// shrink the cache also requires that no caller-owned append encoder can
    /// remain submit-able. A successful shrink advances the branch epoch,
    /// intentionally making the consumed checkpoint and all peers from the old
    /// branch stale.
    pub fn restore(
        &mut self,
        checkpoint: &WgpuPagedKvCheckpoint,
    ) -> Result<(), WgpuPagedKvCacheError> {
        self.validate_checkpoint(checkpoint)?;

        let current_len = self.len();
        if checkpoint.len == current_len {
            return Ok(());
        }

        self.truncate(checkpoint.len)
    }

    /// Logical reset only. Physical bytes remain resident but the generation is
    /// invalidated before pages are deterministically reused. Reset starts a new
    /// checkpoint branch at epoch zero.
    ///
    /// Reset fails closed after any externally managed [`Self::record_append`],
    /// because an older caller-owned encoder could otherwise be submitted after
    /// page reuse and overwrite the new generation.
    pub fn reset(&mut self) -> Result<(), WgpuPagedKvCacheError> {
        if self.unsubmitted_recorded_writes {
            return Err(WgpuPagedKvCacheError::UnsubmittedRecordedWrites);
        }
        self.table.reset()?;
        self.branch_epoch = 0;
        Ok(())
    }

    /// Rewind the live paged cache without recording GPU work.
    ///
    /// The surviving prefix keeps its physical mappings and generation. Tail
    /// pages released by the table become available for deterministic reuse by
    /// subsequent appends. Physical K/V bytes are neither cleared nor copied;
    /// bytes outside the new logical length are non-live until overwritten.
    /// A successful shrink advances the branch epoch so checkpoints from the
    /// pre-truncate lineage cannot be restored after page reuse.
    ///
    /// Shrinks fail closed after any externally managed [`Self::record_append`].
    /// A no-op truncate to the current length is still permitted because it
    /// releases and reuses no storage.
    pub fn truncate(&mut self, new_len: usize) -> Result<(), WgpuPagedKvCacheError> {
        let current_len = self.len();
        if new_len >= current_len {
            self.table.truncate(new_len)?;
            return Ok(());
        }
        if self.unsubmitted_recorded_writes {
            return Err(WgpuPagedKvCacheError::UnsubmittedRecordedWrites);
        }
        let next_branch_epoch = self
            .branch_epoch
            .checked_add(1)
            .ok_or(WgpuPagedKvCacheError::BranchEpochOverflow)?;
        self.table.truncate(new_len)?;
        self.branch_epoch = next_branch_epoch;
        Ok(())
    }

    /// Record and submit one append using an encoder owned by this method.
    ///
    /// This is the safe path when later truncate/reset/restore may be required.
    /// The method owns the encoder from creation through [`wgpu::Queue::submit`],
    /// so no abandoned append command buffer can remain submit-able after the
    /// method returns. Queue submission establishes ordering; this method does
    /// not wait for GPU completion.
    ///
    /// If an externally managed append has already tainted this cache, this
    /// method fails closed rather than clearing that state based on a different
    /// encoder.
    pub fn append_and_submit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        k_source: &wgpu::Buffer,
        v_source: &wgpu::Buffer,
        append_len: usize,
    ) -> Result<(usize, wgpu::SubmissionIndex), WgpuPagedKvCacheError> {
        if self.unsubmitted_recorded_writes {
            return Err(WgpuPagedKvCacheError::UnsubmittedRecordedWrites);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m16-paged-kv-managed-append"),
        });
        let new_len = match self.record_append(&mut encoder, k_source, v_source, append_len) {
            Ok(new_len) => new_len,
            Err(error) => {
                // The encoder is owned by this method and is dropped without
                // submission on error, so no stale write can escape.
                self.unsubmitted_recorded_writes = false;
                return Err(error);
            }
        };

        let submission = queue.submit(Some(encoder.finish()));
        self.unsubmitted_recorded_writes = false;
        Ok((new_len, submission))
    }

    /// Record an append into a caller-owned command encoder.
    ///
    /// Source layout is `[append_len, kv_heads * head_dim]`. K must already be
    /// RoPE-rotated. Only newly appended rows are copied; the live prefix is not
    /// moved or rewritten. Metadata is committed only after all copy commands
    /// have been recorded successfully. Appends preserve the current branch
    /// epoch, so checkpoints remain valid while their captured prefix is intact.
    ///
    /// This low-level path is intended for append-only caller-side batching.
    /// After it is used, the cache cannot safely prove that the caller-owned
    /// encoder has been consumed, so destructive truncate/reset/restore remain
    /// blocked for the lifetime of this cache. Use [`Self::append_and_submit`]
    /// for caches that need rollback or page reuse.
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

        // From this point onward an error may leave copy commands recorded in
        // the caller-owned encoder. WGPU provides no safe provenance query that
        // can later prove every such encoder was consumed, so the cache remains
        // fail-closed for destructive lifecycle transitions.
        self.unsubmitted_recorded_writes = true;

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
