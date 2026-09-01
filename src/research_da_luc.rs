//! Research-only DA-LUC attention-facing KV view contract.
//!
//! FDAL0 defines metadata and validation only. It does not encode, decode,
//! dequantize, score, allocate, route, or change FLAT's dense resident/paged KV
//! paths. Backend adapters remain responsible for binding concrete storage to a
//! validated descriptor and for rejecting payloads that do not satisfy it.

use core::fmt;

/// Current schema version of the research-only DA-LUC KV view contract.
pub const DA_LUC_KV_VIEW_SCHEMA_VERSION: u16 = 1;

/// Floating-point storage used by codebooks, scales, dense values or residuals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucFloatDType {
    F16,
    Bf16,
    F32,
}

/// Ordering of token/head rows inside representation planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucRowOrder {
    /// `[batch, token, kv_head, ...]`.
    BatchTokenHead,
    /// `[batch, kv_head, token, ...]`.
    BatchHeadToken,
}

/// Bit numbering inside packed index/value streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucBitOrder {
    Lsb0,
    Msb0,
}

/// Whether one K codebook set is shared across KV heads or replicated per head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucCodebookScope {
    SharedAcrossKvHeads,
    PerKvHead,
}

/// Sparse outlier index representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucResidualIndexing {
    /// Fixed-width coordinates into one logical K or V vector.
    Coordinates {
        index_bits: u8,
        bit_order: DalucBitOrder,
    },
    /// One bit per logical scalar in the vector.
    Bitmap { bit_order: DalucBitOrder },
}

/// Optional sparse residual plane applied after the primary representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucResidualSemantics {
    None,
    Sparse {
        value_dtype: DalucFloatDType,
        indexing: DalucResidualIndexing,
        max_entries_per_vector: usize,
    },
}

/// Uniform-subspace codebook representation for K.
///
/// Codebook shape is derived from this descriptor. For shared scope it is
/// `[subspaces_per_head, codebook_entries, subspace_dim]`; for per-head scope it
/// is `[kv_heads, subspaces_per_head, codebook_entries, subspace_dim]`.
/// The packed index stream follows [`DalucPhysicalLayout::row_order`] with
/// subspace as the innermost logical dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucKeyRepresentation {
    pub subspace_dim: usize,
    pub codebook_entries: usize,
    pub codebook_dtype: DalucFloatDType,
    pub codebook_scope: DalucCodebookScope,
    pub index_bits: u8,
    pub index_bit_order: DalucBitOrder,
    pub residual: DalucResidualSemantics,
}

impl DalucKeyRepresentation {
    /// Validate one decoded codebook index against this descriptor.
    ///
    /// Payload adapters can use this before exposing an index to an attention
    /// consumer; out-of-range indices fail closed instead of being clamped.
    pub fn validate_codebook_index(self, index: u32) -> Result<(), DalucKvViewError> {
        if u64::from(index) >= usize_to_u64(self.codebook_entries)? {
            return Err(DalucKvViewError::CodebookIndexOutOfRange {
                index,
                codebook_entries: self.codebook_entries,
            });
        }
        Ok(())
    }
}

/// Storage for an affine per-group zero point. `None` means symmetric scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucZeroPointStorage {
    None,
    U8,
    U16,
}

/// V is intentionally independent from K.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucValueRepresentation {
    /// Uncompressed value oracle/candidate component.
    Dense { dtype: DalucFloatDType },
    /// Fixed-width groupwise low-bit values with one scale per group.
    GroupwiseAffine {
        storage_bits: u8,
        group_size: usize,
        scale_dtype: DalucFloatDType,
        zero_point: DalucZeroPointStorage,
        bit_order: DalucBitOrder,
        residual: DalucResidualSemantics,
    },
}

/// Logical attention geometry described by one per-layer DA-LUC view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucLogicalKvShape {
    pub batch: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub kv_len: usize,
    /// Q/K feature width.
    pub key_head_dim: usize,
    /// V/output feature width.
    pub value_head_dim: usize,
}

/// Physical capacity model. Page-table entries remain owned by the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucStorageTopology {
    Contiguous { capacity_tokens: usize },
    Paged {
        page_size: usize,
        physical_pages_per_batch: usize,
    },
}

/// Padding semantics for representation planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucPaddingRule {
    /// No padding bytes are part of the representation.
    None,
    /// Each plane is zero-padded to `plane_alignment_bytes`.
    ZeroFilledToAlignment,
}

/// Backend-neutral physical organization shared by K/V representation planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucPhysicalLayout {
    pub row_order: DalucRowOrder,
    pub topology: DalucStorageTopology,
    /// Required byte alignment for every bound representation plane.
    pub plane_alignment_bytes: usize,
    pub padding: DalucPaddingRule,
}

/// Versioned research descriptor for one attention-facing compressed KV view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucKvViewContract {
    pub schema_version: u16,
    pub shape: DalucLogicalKvShape,
    pub keys: DalucKeyRepresentation,
    pub values: DalucValueRepresentation,
    pub layout: DalucPhysicalLayout,
}

/// Fail-closed DA-LUC contract validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DalucKvViewError {
    UnsupportedSchemaVersion { actual: u16, supported: u16 },
    ZeroDimension { field: &'static str },
    ShapeOverflow,
    InvalidHeadGrouping { q_heads: usize, kv_heads: usize },
    KeySubspaceMismatch { key_head_dim: usize, subspace_dim: usize },
    InvalidCodebookEntries { codebook_entries: usize },
    InvalidBitWidth { field: &'static str, bits: u8 },
    PackedCapacityTooSmall { field: &'static str, bits: u8, required: usize },
    CodebookIndexOutOfRange { index: u32, codebook_entries: usize },
    ValueGroupMismatch { value_head_dim: usize, group_size: usize },
    InvalidZeroPointStorage { storage_bits: u8, zero_point: DalucZeroPointStorage },
    InvalidResidualBudget { dimension: usize, max_entries: usize },
    ResidualIndexOutOfRange { index: usize, dimension: usize },
    InsufficientCapacity { kv_len: usize, capacity_tokens: usize },
    InvalidAlignment { alignment_bytes: usize },
}

impl fmt::Display for DalucKvViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { actual, supported } => write!(
                f,
                "DA-LUC KV view schema {actual} is unsupported; expected {supported}"
            ),
            Self::ZeroDimension { field } => write!(f, "DA-LUC {field} must be non-zero"),
            Self::ShapeOverflow => write!(f, "DA-LUC shape or packed-capacity arithmetic overflowed"),
            Self::InvalidHeadGrouping { q_heads, kv_heads } => write!(
                f,
                "DA-LUC q_heads ({q_heads}) must be exactly divisible by kv_heads ({kv_heads})"
            ),
            Self::KeySubspaceMismatch { key_head_dim, subspace_dim } => write!(
                f,
                "DA-LUC key_head_dim {key_head_dim} is not divisible by subspace_dim {subspace_dim}"
            ),
            Self::InvalidCodebookEntries { codebook_entries } => write!(
                f,
                "DA-LUC codebook must contain at least two entries, got {codebook_entries}"
            ),
            Self::InvalidBitWidth { field, bits } => {
                write!(f, "DA-LUC {field} bit width {bits} is unsupported")
            }
            Self::PackedCapacityTooSmall { field, bits, required } => write!(
                f,
                "DA-LUC {field} width {bits} bits cannot address {required} values"
            ),
            Self::CodebookIndexOutOfRange { index, codebook_entries } => write!(
                f,
                "DA-LUC codebook index {index} is outside {codebook_entries} entries"
            ),
            Self::ValueGroupMismatch { value_head_dim, group_size } => write!(
                f,
                "DA-LUC value_head_dim {value_head_dim} is not divisible by group_size {group_size}"
            ),
            Self::InvalidZeroPointStorage { storage_bits, zero_point } => write!(
                f,
                "DA-LUC zero-point storage {zero_point:?} cannot represent a {storage_bits}-bit affine value domain"
            ),
            Self::InvalidResidualBudget { dimension, max_entries } => write!(
                f,
                "DA-LUC sparse residual budget {max_entries} is invalid for vector dimension {dimension}"
            ),
            Self::ResidualIndexOutOfRange { index, dimension } => write!(
                f,
                "DA-LUC residual index {index} is outside vector dimension {dimension}"
            ),
            Self::InsufficientCapacity { kv_len, capacity_tokens } => write!(
                f,
                "DA-LUC physical capacity {capacity_tokens} tokens is smaller than live KV length {kv_len}"
            ),
            Self::InvalidAlignment { alignment_bytes } => write!(
                f,
                "DA-LUC plane alignment {alignment_bytes} must be a non-zero power of two"
            ),
        }
    }
}

impl std::error::Error for DalucKvViewError {}

impl DalucKvViewContract {
    /// Validate the complete v1 descriptor without assuming a concrete backend.
    pub fn validate(self) -> Result<(), DalucKvViewError> {
        if self.schema_version != DA_LUC_KV_VIEW_SCHEMA_VERSION {
            return Err(DalucKvViewError::UnsupportedSchemaVersion {
                actual: self.schema_version,
                supported: DA_LUC_KV_VIEW_SCHEMA_VERSION,
            });
        }
        validate_shape(self.shape)?;
        validate_keys(self.shape, self.keys)?;
        validate_values(self.shape, self.values)?;
        validate_layout(self.shape, self.layout)?;
        Ok(())
    }

    /// Validate one sparse K residual coordinate against the declared vector.
    pub fn validate_key_residual_index(self, index: usize) -> Result<(), DalucKvViewError> {
        validate_residual_payload_index(self.keys.residual, self.shape.key_head_dim, index)
    }

    /// Validate one sparse V residual coordinate against the declared vector.
    pub fn validate_value_residual_index(self, index: usize) -> Result<(), DalucKvViewError> {
        let residual = match self.values {
            DalucValueRepresentation::Dense { .. } => DalucResidualSemantics::None,
            DalucValueRepresentation::GroupwiseAffine { residual, .. } => residual,
        };
        validate_residual_payload_index(residual, self.shape.value_head_dim, index)
    }

    /// Deterministic representation identity for evidence/adapters.
    ///
    /// It contains only contract fields; no source revision, device identity,
    /// performance result or inferred compression claim is synthesized.
    #[must_use]
    pub fn canonical_record(self) -> String {
        format!(
            "flat-da-luc-kv-view-v{};shape=b:{},qh:{},kvh:{},n:{},kd:{},vd:{};keys=sub:{},entries:{},dtype:{},scope:{},ib:{},ibo:{},res:{};values={};layout=row:{},topology:{},align:{},padding:{}",
            self.schema_version,
            self.shape.batch,
            self.shape.q_heads,
            self.shape.kv_heads,
            self.shape.kv_len,
            self.shape.key_head_dim,
            self.shape.value_head_dim,
            self.keys.subspace_dim,
            self.keys.codebook_entries,
            float_tag(self.keys.codebook_dtype),
            scope_tag(self.keys.codebook_scope),
            self.keys.index_bits,
            bit_tag(self.keys.index_bit_order),
            residual_tag(self.keys.residual),
            value_tag(self.values),
            row_tag(self.layout.row_order),
            topology_tag(self.layout.topology),
            self.layout.plane_alignment_bytes,
            padding_tag(self.layout.padding),
        )
    }
}

fn validate_shape(shape: DalucLogicalKvShape) -> Result<(), DalucKvViewError> {
    for (field, value) in [
        ("batch", shape.batch),
        ("q_heads", shape.q_heads),
        ("kv_heads", shape.kv_heads),
        ("kv_len", shape.kv_len),
        ("key_head_dim", shape.key_head_dim),
        ("value_head_dim", shape.value_head_dim),
    ] {
        if value == 0 {
            return Err(DalucKvViewError::ZeroDimension { field });
        }
    }
    if !shape.q_heads.is_multiple_of(shape.kv_heads) {
        return Err(DalucKvViewError::InvalidHeadGrouping {
            q_heads: shape.q_heads,
            kv_heads: shape.kv_heads,
        });
    }
    shape
        .batch
        .checked_mul(shape.kv_heads)
        .and_then(|v| v.checked_mul(shape.kv_len))
        .and_then(|v| v.checked_mul(shape.key_head_dim))
        .ok_or(DalucKvViewError::ShapeOverflow)?;
    Ok(())
}

fn validate_keys(
    shape: DalucLogicalKvShape,
    keys: DalucKeyRepresentation,
) -> Result<(), DalucKvViewError> {
    if keys.subspace_dim == 0 {
        return Err(DalucKvViewError::ZeroDimension { field: "key subspace_dim" });
    }
    if !shape.key_head_dim.is_multiple_of(keys.subspace_dim) {
        return Err(DalucKvViewError::KeySubspaceMismatch {
            key_head_dim: shape.key_head_dim,
            subspace_dim: keys.subspace_dim,
        });
    }
    if keys.codebook_entries < 2 {
        return Err(DalucKvViewError::InvalidCodebookEntries {
            codebook_entries: keys.codebook_entries,
        });
    }
    validate_packed_width("K index", keys.index_bits, keys.codebook_entries, 32)?;
    validate_residual(keys.residual, shape.key_head_dim)?;
    Ok(())
}

fn validate_values(
    shape: DalucLogicalKvShape,
    values: DalucValueRepresentation,
) -> Result<(), DalucKvViewError> {
    if let DalucValueRepresentation::GroupwiseAffine {
        storage_bits,
        group_size,
        zero_point,
        residual,
        ..
    } = values
    {
        if group_size == 0 {
            return Err(DalucKvViewError::ZeroDimension { field: "V group_size" });
        }
        if !shape.value_head_dim.is_multiple_of(group_size) {
            return Err(DalucKvViewError::ValueGroupMismatch {
                value_head_dim: shape.value_head_dim,
                group_size,
            });
        }
        validate_packed_width("V value", storage_bits, 2, 16)?;
        match zero_point {
            DalucZeroPointStorage::None => {}
            DalucZeroPointStorage::U8 if storage_bits <= 8 => {}
            DalucZeroPointStorage::U16 if storage_bits <= 16 => {}
            _ => {
                return Err(DalucKvViewError::InvalidZeroPointStorage {
                    storage_bits,
                    zero_point,
                });
            }
        }
        validate_residual(residual, shape.value_head_dim)?;
    }
    Ok(())
}

fn validate_residual(
    residual: DalucResidualSemantics,
    dimension: usize,
) -> Result<(), DalucKvViewError> {
    let DalucResidualSemantics::Sparse {
        indexing,
        max_entries_per_vector,
        ..
    } = residual
    else {
        return Ok(());
    };
    if max_entries_per_vector == 0 || max_entries_per_vector > dimension {
        return Err(DalucKvViewError::InvalidResidualBudget {
            dimension,
            max_entries: max_entries_per_vector,
        });
    }
    if let DalucResidualIndexing::Coordinates { index_bits, .. } = indexing {
        validate_packed_width("residual coordinate", index_bits, dimension, 32)?;
    }
    Ok(())
}

fn validate_residual_payload_index(
    residual: DalucResidualSemantics,
    dimension: usize,
    index: usize,
) -> Result<(), DalucKvViewError> {
    if matches!(residual, DalucResidualSemantics::None) || index >= dimension {
        return Err(DalucKvViewError::ResidualIndexOutOfRange { index, dimension });
    }
    Ok(())
}

fn validate_layout(
    shape: DalucLogicalKvShape,
    layout: DalucPhysicalLayout,
) -> Result<(), DalucKvViewError> {
    let alignment = layout.plane_alignment_bytes;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(DalucKvViewError::InvalidAlignment {
            alignment_bytes: alignment,
        });
    }
    let capacity_tokens = match layout.topology {
        DalucStorageTopology::Contiguous { capacity_tokens } => {
            if capacity_tokens == 0 {
                return Err(DalucKvViewError::ZeroDimension {
                    field: "contiguous capacity_tokens",
                });
            }
            capacity_tokens
        }
        DalucStorageTopology::Paged {
            page_size,
            physical_pages_per_batch,
        } => {
            if page_size == 0 {
                return Err(DalucKvViewError::ZeroDimension { field: "page_size" });
            }
            if physical_pages_per_batch == 0 {
                return Err(DalucKvViewError::ZeroDimension {
                    field: "physical_pages_per_batch",
                });
            }
            page_size
                .checked_mul(physical_pages_per_batch)
                .ok_or(DalucKvViewError::ShapeOverflow)?
        }
    };
    if capacity_tokens < shape.kv_len {
        return Err(DalucKvViewError::InsufficientCapacity {
            kv_len: shape.kv_len,
            capacity_tokens,
        });
    }
    Ok(())
}

fn validate_packed_width(
    field: &'static str,
    bits: u8,
    required: usize,
    maximum_bits: u8,
) -> Result<(), DalucKvViewError> {
    if bits == 0 || bits > maximum_bits {
        return Err(DalucKvViewError::InvalidBitWidth { field, bits });
    }
    let capacity = 1u64
        .checked_shl(u32::from(bits))
        .ok_or(DalucKvViewError::ShapeOverflow)?;
    if capacity < usize_to_u64(required)? {
        return Err(DalucKvViewError::PackedCapacityTooSmall {
            field,
            bits,
            required,
        });
    }
    Ok(())
}

fn usize_to_u64(value: usize) -> Result<u64, DalucKvViewError> {
    u64::try_from(value).map_err(|_| DalucKvViewError::ShapeOverflow)
}

const fn float_tag(value: DalucFloatDType) -> &'static str {
    match value {
        DalucFloatDType::F16 => "f16",
        DalucFloatDType::Bf16 => "bf16",
        DalucFloatDType::F32 => "f32",
    }
}

const fn bit_tag(value: DalucBitOrder) -> &'static str {
    match value {
        DalucBitOrder::Lsb0 => "lsb0",
        DalucBitOrder::Msb0 => "msb0",
    }
}

const fn scope_tag(value: DalucCodebookScope) -> &'static str {
    match value {
        DalucCodebookScope::SharedAcrossKvHeads => "shared",
        DalucCodebookScope::PerKvHead => "per-kv-head",
    }
}

const fn row_tag(value: DalucRowOrder) -> &'static str {
    match value {
        DalucRowOrder::BatchTokenHead => "batch-token-head",
        DalucRowOrder::BatchHeadToken => "batch-head-token",
    }
}

const fn padding_tag(value: DalucPaddingRule) -> &'static str {
    match value {
        DalucPaddingRule::None => "none",
        DalucPaddingRule::ZeroFilledToAlignment => "zero-to-alignment",
    }
}

fn residual_tag(value: DalucResidualSemantics) -> String {
    match value {
        DalucResidualSemantics::None => "none".to_owned(),
        DalucResidualSemantics::Sparse {
            value_dtype,
            indexing,
            max_entries_per_vector,
        } => match indexing {
            DalucResidualIndexing::Coordinates {
                index_bits,
                bit_order,
            } => format!(
                "sparse-coord(dtype:{},ib:{index_bits},ibo:{},max:{max_entries_per_vector})",
                float_tag(value_dtype),
                bit_tag(bit_order)
            ),
            DalucResidualIndexing::Bitmap { bit_order } => format!(
                "sparse-bitmap(dtype:{},bo:{},max:{max_entries_per_vector})",
                float_tag(value_dtype),
                bit_tag(bit_order)
            ),
        },
    }
}

fn value_tag(value: DalucValueRepresentation) -> String {
    match value {
        DalucValueRepresentation::Dense { dtype } => {
            format!("dense(dtype:{})", float_tag(dtype))
        }
        DalucValueRepresentation::GroupwiseAffine {
            storage_bits,
            group_size,
            scale_dtype,
            zero_point,
            bit_order,
            residual,
        } => format!(
            "groupwise(bits:{storage_bits},group:{group_size},scale:{},zp:{},bo:{},res:{})",
            float_tag(scale_dtype),
            zero_point_tag(zero_point),
            bit_tag(bit_order),
            residual_tag(residual)
        ),
    }
}

const fn zero_point_tag(value: DalucZeroPointStorage) -> &'static str {
    match value {
        DalucZeroPointStorage::None => "none",
        DalucZeroPointStorage::U8 => "u8",
        DalucZeroPointStorage::U16 => "u16",
    }
}

fn topology_tag(value: DalucStorageTopology) -> String {
    match value {
        DalucStorageTopology::Contiguous { capacity_tokens } => {
            format!("contiguous(cap:{capacity_tokens})")
        }
        DalucStorageTopology::Paged {
            page_size,
            physical_pages_per_batch,
        } => format!("paged(size:{page_size},pages-per-batch:{physical_pages_per_batch})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> DalucKvViewContract {
        DalucKvViewContract {
            schema_version: DA_LUC_KV_VIEW_SCHEMA_VERSION,
            shape: DalucLogicalKvShape {
                batch: 1,
                q_heads: 32,
                kv_heads: 8,
                kv_len: 128,
                key_head_dim: 64,
                value_head_dim: 64,
            },
            keys: DalucKeyRepresentation {
                subspace_dim: 8,
                codebook_entries: 256,
                codebook_dtype: DalucFloatDType::F16,
                codebook_scope: DalucCodebookScope::PerKvHead,
                index_bits: 8,
                index_bit_order: DalucBitOrder::Lsb0,
                residual: DalucResidualSemantics::Sparse {
                    value_dtype: DalucFloatDType::F16,
                    indexing: DalucResidualIndexing::Coordinates {
                        index_bits: 6,
                        bit_order: DalucBitOrder::Lsb0,
                    },
                    max_entries_per_vector: 4,
                },
            },
            values: DalucValueRepresentation::GroupwiseAffine {
                storage_bits: 4,
                group_size: 16,
                scale_dtype: DalucFloatDType::F16,
                zero_point: DalucZeroPointStorage::U8,
                bit_order: DalucBitOrder::Lsb0,
                residual: DalucResidualSemantics::None,
            },
            layout: DalucPhysicalLayout {
                row_order: DalucRowOrder::BatchTokenHead,
                topology: DalucStorageTopology::Paged {
                    page_size: 16,
                    physical_pages_per_batch: 8,
                },
                plane_alignment_bytes: 16,
                padding: DalucPaddingRule::ZeroFilledToAlignment,
            },
        }
    }

    #[test]
    fn representative_asymmetric_view_is_valid_and_deterministic() {
        let view = contract();
        view.validate().unwrap();
        assert_eq!(view.canonical_record(), view.canonical_record());
        assert!(view.canonical_record().starts_with("flat-da-luc-kv-view-v1;"));
    }

    #[test]
    fn future_schema_fails_closed() {
        let mut view = contract();
        view.schema_version += 1;
        assert!(matches!(
            view.validate(),
            Err(DalucKvViewError::UnsupportedSchemaVersion { .. })
        ));
    }

    #[test]
    fn ambiguous_head_mapping_is_rejected() {
        let mut view = contract();
        view.shape.q_heads = 30;
        assert!(matches!(
            view.validate(),
            Err(DalucKvViewError::InvalidHeadGrouping { .. })
        ));
    }

    #[test]
    fn malformed_key_partition_and_index_capacity_are_rejected() {
        let mut view = contract();
        view.keys.subspace_dim = 7;
        assert!(matches!(
            view.validate(),
            Err(DalucKvViewError::KeySubspaceMismatch { .. })
        ));

        let mut view = contract();
        view.keys.index_bits = 7;
        assert!(matches!(
            view.validate(),
            Err(DalucKvViewError::PackedCapacityTooSmall { field: "K index", .. })
        ));
    }

    #[test]
    fn payload_indices_fail_closed() {
        let view = contract();
        view.keys.validate_codebook_index(255).unwrap();
        assert!(matches!(
            view.keys.validate_codebook_index(256),
            Err(DalucKvViewError::CodebookIndexOutOfRange { .. })
        ));
        view.validate_key_residual_index(63).unwrap();
        assert!(matches!(
            view.validate_key_residual_index(64),
            Err(DalucKvViewError::ResidualIndexOutOfRange { .. })
        ));
        assert!(matches!(
            view.validate_value_residual_index(0),
            Err(DalucKvViewError::ResidualIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn value_group_zero_point_and_capacity_errors_are_explicit() {
        let mut view = contract();
        if let DalucValueRepresentation::GroupwiseAffine { group_size, .. } = &mut view.values {
            *group_size = 10;
        }
        assert!(matches!(
            view.validate(),
            Err(DalucKvViewError::ValueGroupMismatch { .. })
        ));

        let mut view = contract();
        if let DalucValueRepresentation::GroupwiseAffine {
            storage_bits,
            zero_point,
            ..
        } = &mut view.values
        {
            *storage_bits = 12;
            *zero_point = DalucZeroPointStorage::U8;
        }
        assert!(matches!(
            view.validate(),
            Err(DalucKvViewError::InvalidZeroPointStorage { .. })
        ));

        let mut view = contract();
        view.layout.topology = DalucStorageTopology::Paged {
            page_size: 16,
            physical_pages_per_batch: 7,
        };
        assert!(matches!(
            view.validate(),
            Err(DalucKvViewError::InsufficientCapacity { .. })
        ));
    }

    #[test]
    fn sparse_residual_metadata_is_validated() {
        let mut view = contract();
        view.keys.residual = DalucResidualSemantics::Sparse {
            value_dtype: DalucFloatDType::F16,
            indexing: DalucResidualIndexing::Coordinates {
                index_bits: 5,
                bit_order: DalucBitOrder::Lsb0,
            },
            max_entries_per_vector: 4,
        };
        assert!(matches!(
            view.validate(),
            Err(DalucKvViewError::PackedCapacityTooSmall {
                field: "residual coordinate",
                ..
            })
        ));

        let mut view = contract();
        view.keys.residual = DalucResidualSemantics::Sparse {
            value_dtype: DalucFloatDType::F16,
            indexing: DalucResidualIndexing::Bitmap {
                bit_order: DalucBitOrder::Lsb0,
            },
            max_entries_per_vector: 65,
        };
        assert!(matches!(
            view.validate(),
            Err(DalucKvViewError::InvalidResidualBudget { .. })
        ));
    }
}