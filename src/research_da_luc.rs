//! Research-only DA-LUC attention-facing KV view contract.
//!
//! FDAL0 defines metadata and fail-closed validation only. It does not encode,
//! decode, score, allocate, route, or change FLAT's dense resident/paged paths.

use core::fmt;

/// Current schema version of the research-only DA-LUC KV view contract.
pub const DA_LUC_KV_VIEW_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucFloatDType {
    F16,
    Bf16,
    F32,
}

/// Logical row order for packed K indices and V values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucRowOrder {
    BatchTokenHead,
    BatchHeadToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucBitOrder {
    Lsb0,
    Msb0,
}

/// Codebook shape is either
/// `[subspaces, entries, subspace_dim]` or
/// `[kv_heads, subspaces, entries, subspace_dim]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucCodebookScope {
    SharedAcrossKvHeads,
    PerKvHead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucResidualIndexing {
    /// Residual values pair one-for-one with unique coordinates in stored order.
    Coordinates {
        index_bits: u8,
        bit_order: DalucBitOrder,
    },
    /// One bitmap bit per logical scalar. Residual values correspond to set
    /// bits in increasing logical-coordinate order.
    Bitmap { bit_order: DalucBitOrder },
}

/// Additive sparse correction applied after the primary K/V reconstruction.
///
/// Each residual entry is added to exactly one logical scalar in the full K or
/// V vector. Duplicate coordinate entries are not part of the v1 semantics and
/// must be rejected by payload-producing/consuming adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucSparseResidual {
    pub value_dtype: DalucFloatDType,
    pub indexing: DalucResidualIndexing,
    pub max_entries_per_vector: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucResidualSemantics {
    None,
    Sparse(DalucSparseResidual),
}

/// Uniform-subspace codebook representation for K.
///
/// The index stream follows [`DalucPhysicalLayout::row_order`] and uses subspace
/// as the innermost logical dimension. Each primary K subspace is reconstructed
/// by selecting exactly one complete codebook vector, then applying any sparse
/// residual entries additively in full-vector coordinates.
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

/// V low-bit zero-point semantics.
///
/// `None` means the packed `storage_bits` field is interpreted as a signed
/// two's-complement integer and the primary reconstruction is `scale * q`.
/// `U8`/`U16` mean the packed field is unsigned and one zero-point payload is
/// stored per group in that container; reconstruction is `scale * (q - zp)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucZeroPointStorage {
    None,
    U8,
    U16,
}

/// V is deliberately independent from the K representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucValueRepresentation {
    Dense {
        dtype: DalucFloatDType,
    },
    /// Contiguous feature groups. Every group owns one floating scale and,
    /// when affine, one zero point. Sparse residuals are added after the primary
    /// group reconstruction.
    GroupwiseAffine {
        storage_bits: u8,
        group_size: usize,
        scale_dtype: DalucFloatDType,
        zero_point: DalucZeroPointStorage,
        bit_order: DalucBitOrder,
        residual: DalucResidualSemantics,
    },
}

/// Logical geometry for one per-layer attention-facing KV view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucLogicalKvShape {
    pub batch: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub kv_len: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucStorageTopology {
    Contiguous {
        capacity_tokens: usize,
    },
    /// Page-table entries are adapter-owned. This contract fixes page geometry
    /// and available physical capacity only.
    Paged {
        page_size: usize,
        physical_pages_per_batch: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucPaddingRule {
    None,
    ZeroFilledToAlignment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucPhysicalLayout {
    pub row_order: DalucRowOrder,
    pub topology: DalucStorageTopology,
    pub plane_alignment_bytes: usize,
    pub padding: DalucPaddingRule,
}

/// Versioned FDAL0 descriptor. No field implies runtime promotion or fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucKvViewContract {
    pub schema_version: u16,
    pub shape: DalucLogicalKvShape,
    pub keys: DalucKeyRepresentation,
    pub values: DalucValueRepresentation,
    pub layout: DalucPhysicalLayout,
}

/// Backend ability to consume a validated view directly.
///
/// Every representation/layout choice that changes byte interpretation is an
/// explicit capability. Missing capability always means rejection; it never
/// authorizes an implicit dense conversion or layout rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucBackendCapabilities {
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub supports_f32: bool,
    pub supports_lsb0: bool,
    pub supports_msb0: bool,
    pub supports_batch_token_head: bool,
    pub supports_batch_head_token: bool,
    pub supports_shared_codebook: bool,
    pub supports_per_kv_head_codebook: bool,
    pub supports_contiguous: bool,
    pub supports_paged: bool,
    pub supports_groupwise_affine_values: bool,
    pub supports_signed_symmetric_values: bool,
    pub supports_u8_zero_points: bool,
    pub supports_u16_zero_points: bool,
    pub supports_sparse_coordinates: bool,
    pub supports_sparse_bitmap: bool,
    pub supports_no_padding: bool,
    pub supports_zero_filled_padding: bool,
    /// Minimum physical plane alignment accepted for direct consumption.
    /// Must itself be a non-zero power of two.
    pub minimum_plane_alignment_bytes: usize,
    pub max_packed_index_bits: u8,
    pub max_groupwise_value_bits: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucKvSide {
    Key,
    Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DalucKvViewError {
    UnsupportedSchemaVersion {
        actual: u16,
        supported: u16,
    },
    InvalidMetadata(&'static str),
    CodebookIndexOutOfRange {
        index: u32,
        codebook_entries: usize,
    },
    ResidualIndexOutOfRange {
        side: DalucKvSide,
        index: usize,
        dimension: usize,
    },
    UnsupportedBackendCapability(&'static str),
}

impl fmt::Display for DalucKvViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { actual, supported } => write!(
                f,
                "DA-LUC KV view schema {actual} is unsupported; expected {supported}"
            ),
            Self::InvalidMetadata(reason) => write!(f, "invalid DA-LUC metadata: {reason}"),
            Self::CodebookIndexOutOfRange {
                index,
                codebook_entries,
            } => write!(
                f,
                "DA-LUC codebook index {index} is outside {codebook_entries} entries"
            ),
            Self::ResidualIndexOutOfRange {
                side,
                index,
                dimension,
            } => write!(
                f,
                "DA-LUC {side:?} residual index {index} is outside dimension {dimension}"
            ),
            Self::UnsupportedBackendCapability(capability) => {
                write!(f, "DA-LUC backend does not support {capability}")
            }
        }
    }
}

impl std::error::Error for DalucKvViewError {}

impl DalucKvViewContract {
    /// Validate all backend-independent v1 invariants.
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
        validate_layout(self.shape, self.layout)
    }

    /// Validate the descriptor against one backend's declared direct-consume
    /// capabilities. No fallback, repacking or materialization is implied on
    /// failure.
    pub fn validate_for_backend(
        self,
        backend: DalucBackendCapabilities,
    ) -> Result<(), DalucKvViewError> {
        self.validate()?;
        validate_backend_capabilities(backend)?;

        require_row_order(backend, self.layout.row_order)?;
        require_topology(backend, self.layout.topology)?;
        require_padding(backend, self.layout.padding)?;
        if self.layout.plane_alignment_bytes < backend.minimum_plane_alignment_bytes
            || !self
                .layout
                .plane_alignment_bytes
                .is_multiple_of(backend.minimum_plane_alignment_bytes)
        {
            return Err(DalucKvViewError::UnsupportedBackendCapability(
                "plane alignment",
            ));
        }

        require_dtype(backend, self.keys.codebook_dtype, "K codebook dtype")?;
        require_codebook_scope(backend, self.keys.codebook_scope)?;
        require_bit_order(backend, self.keys.index_bit_order, "K index bit order")?;
        if self.keys.index_bits > backend.max_packed_index_bits {
            return Err(DalucKvViewError::UnsupportedBackendCapability(
                "K packed index width",
            ));
        }
        require_residual(backend, self.keys.residual, "K residual")?;

        match self.values {
            DalucValueRepresentation::Dense { dtype } => {
                require_dtype(backend, dtype, "dense V dtype")?;
            }
            DalucValueRepresentation::GroupwiseAffine {
                storage_bits,
                scale_dtype,
                zero_point,
                bit_order,
                residual,
                ..
            } => {
                if !backend.supports_groupwise_affine_values {
                    return Err(DalucKvViewError::UnsupportedBackendCapability(
                        "groupwise affine V",
                    ));
                }
                if storage_bits > backend.max_groupwise_value_bits {
                    return Err(DalucKvViewError::UnsupportedBackendCapability(
                        "V packed value width",
                    ));
                }
                require_v_zero_point(backend, zero_point)?;
                require_bit_order(backend, bit_order, "V packed value bit order")?;
                require_dtype(backend, scale_dtype, "V scale dtype")?;
                require_residual(backend, residual, "V residual")?;
            }
        }
        Ok(())
    }

    /// Reject a decoded K codebook index outside the declared codebook.
    pub fn validate_codebook_index(self, index: u32) -> Result<(), DalucKvViewError> {
        self.validate()?;
        if u64::from(index) >= to_u64(self.keys.codebook_entries)? {
            return Err(DalucKvViewError::CodebookIndexOutOfRange {
                index,
                codebook_entries: self.keys.codebook_entries,
            });
        }
        Ok(())
    }

    pub fn validate_key_residual_index(self, index: usize) -> Result<(), DalucKvViewError> {
        self.validate()?;
        validate_payload_residual(
            self.keys.residual,
            DalucKvSide::Key,
            self.shape.key_head_dim,
            index,
        )
    }

    pub fn validate_value_residual_index(self, index: usize) -> Result<(), DalucKvViewError> {
        self.validate()?;
        let residual = match self.values {
            DalucValueRepresentation::Dense { .. } => DalucResidualSemantics::None,
            DalucValueRepresentation::GroupwiseAffine { residual, .. } => residual,
        };
        validate_payload_residual(
            residual,
            DalucKvSide::Value,
            self.shape.value_head_dim,
            index,
        )
    }
}

fn validate_shape(shape: DalucLogicalKvShape) -> Result<(), DalucKvViewError> {
    if [
        shape.batch,
        shape.q_heads,
        shape.kv_heads,
        shape.kv_len,
        shape.key_head_dim,
        shape.value_head_dim,
    ]
    .contains(&0)
    {
        return Err(DalucKvViewError::InvalidMetadata(
            "logical dimensions must be non-zero",
        ));
    }
    if !shape.q_heads.is_multiple_of(shape.kv_heads) {
        return Err(DalucKvViewError::InvalidMetadata(
            "q_heads must be exactly divisible by kv_heads",
        ));
    }
    checked_product(&[
        shape.batch,
        shape.kv_heads,
        shape.kv_len,
        shape.key_head_dim,
    ])?;
    checked_product(&[
        shape.batch,
        shape.kv_heads,
        shape.kv_len,
        shape.value_head_dim,
    ])?;
    Ok(())
}

fn validate_keys(
    shape: DalucLogicalKvShape,
    keys: DalucKeyRepresentation,
) -> Result<(), DalucKvViewError> {
    if keys.subspace_dim == 0 || !shape.key_head_dim.is_multiple_of(keys.subspace_dim) {
        return Err(DalucKvViewError::InvalidMetadata(
            "K subspace_dim must exactly partition key_head_dim",
        ));
    }
    if keys.codebook_entries < 2 {
        return Err(DalucKvViewError::InvalidMetadata(
            "K codebook must contain at least two entries",
        ));
    }
    validate_address_width(
        keys.index_bits,
        keys.codebook_entries,
        "K index width cannot address the codebook",
    )?;
    validate_residual(keys.residual, shape.key_head_dim)
}

fn validate_values(
    shape: DalucLogicalKvShape,
    values: DalucValueRepresentation,
) -> Result<(), DalucKvViewError> {
    let DalucValueRepresentation::GroupwiseAffine {
        storage_bits,
        group_size,
        zero_point,
        residual,
        ..
    } = values
    else {
        return Ok(());
    };
    if storage_bits == 0 || storage_bits > 16 {
        return Err(DalucKvViewError::InvalidMetadata(
            "V storage_bits must be in 1..=16",
        ));
    }
    if group_size == 0 || !shape.value_head_dim.is_multiple_of(group_size) {
        return Err(DalucKvViewError::InvalidMetadata(
            "V group_size must exactly partition value_head_dim",
        ));
    }
    if zero_point == DalucZeroPointStorage::U8 && storage_bits > 8 {
        return Err(DalucKvViewError::InvalidMetadata(
            "V zero-point container is too small for storage_bits",
        ));
    }
    validate_residual(residual, shape.value_head_dim)
}

fn validate_residual(
    residual: DalucResidualSemantics,
    dimension: usize,
) -> Result<(), DalucKvViewError> {
    let DalucResidualSemantics::Sparse(residual) = residual else {
        return Ok(());
    };
    if residual.max_entries_per_vector == 0 || residual.max_entries_per_vector > dimension {
        return Err(DalucKvViewError::InvalidMetadata(
            "sparse residual budget must be in 1..=vector dimension",
        ));
    }
    if let DalucResidualIndexing::Coordinates { index_bits, .. } = residual.indexing {
        validate_address_width(
            index_bits,
            dimension,
            "residual coordinate width cannot address the vector",
        )?;
    }
    Ok(())
}

fn validate_layout(
    shape: DalucLogicalKvShape,
    layout: DalucPhysicalLayout,
) -> Result<(), DalucKvViewError> {
    if layout.plane_alignment_bytes == 0 || !layout.plane_alignment_bytes.is_power_of_two() {
        return Err(DalucKvViewError::InvalidMetadata(
            "plane alignment must be a non-zero power of two",
        ));
    }
    let capacity = match layout.topology {
        DalucStorageTopology::Contiguous { capacity_tokens } => capacity_tokens,
        DalucStorageTopology::Paged {
            page_size,
            physical_pages_per_batch,
        } => page_size.checked_mul(physical_pages_per_batch).ok_or(
            DalucKvViewError::InvalidMetadata("paged capacity arithmetic overflow"),
        )?,
    };
    if capacity == 0 || capacity < shape.kv_len {
        return Err(DalucKvViewError::InvalidMetadata(
            "physical token capacity must cover live kv_len",
        ));
    }
    Ok(())
}

fn validate_address_width(
    bits: u8,
    required: usize,
    capacity_error: &'static str,
) -> Result<(), DalucKvViewError> {
    if bits == 0 || bits > 32 {
        return Err(DalucKvViewError::InvalidMetadata(
            "packed address width must be in 1..=32",
        ));
    }
    let capacity = 1u64 << bits;
    if capacity < to_u64(required)? {
        return Err(DalucKvViewError::InvalidMetadata(capacity_error));
    }
    Ok(())
}

fn validate_payload_residual(
    residual: DalucResidualSemantics,
    side: DalucKvSide,
    dimension: usize,
    index: usize,
) -> Result<(), DalucKvViewError> {
    if matches!(residual, DalucResidualSemantics::None) || index >= dimension {
        return Err(DalucKvViewError::ResidualIndexOutOfRange {
            side,
            index,
            dimension,
        });
    }
    Ok(())
}

fn validate_backend_capabilities(
    backend: DalucBackendCapabilities,
) -> Result<(), DalucKvViewError> {
    if backend.minimum_plane_alignment_bytes == 0
        || !backend.minimum_plane_alignment_bytes.is_power_of_two()
    {
        return Err(DalucKvViewError::UnsupportedBackendCapability(
            "valid plane alignment requirement",
        ));
    }
    Ok(())
}

fn require_dtype(
    backend: DalucBackendCapabilities,
    dtype: DalucFloatDType,
    label: &'static str,
) -> Result<(), DalucKvViewError> {
    let supported = match dtype {
        DalucFloatDType::F16 => backend.supports_f16,
        DalucFloatDType::Bf16 => backend.supports_bf16,
        DalucFloatDType::F32 => backend.supports_f32,
    };
    require_supported(supported, label)
}

fn require_bit_order(
    backend: DalucBackendCapabilities,
    bit_order: DalucBitOrder,
    label: &'static str,
) -> Result<(), DalucKvViewError> {
    let supported = match bit_order {
        DalucBitOrder::Lsb0 => backend.supports_lsb0,
        DalucBitOrder::Msb0 => backend.supports_msb0,
    };
    require_supported(supported, label)
}

fn require_row_order(
    backend: DalucBackendCapabilities,
    row_order: DalucRowOrder,
) -> Result<(), DalucKvViewError> {
    let supported = match row_order {
        DalucRowOrder::BatchTokenHead => backend.supports_batch_token_head,
        DalucRowOrder::BatchHeadToken => backend.supports_batch_head_token,
    };
    require_supported(supported, "row order")
}

fn require_codebook_scope(
    backend: DalucBackendCapabilities,
    scope: DalucCodebookScope,
) -> Result<(), DalucKvViewError> {
    let supported = match scope {
        DalucCodebookScope::SharedAcrossKvHeads => backend.supports_shared_codebook,
        DalucCodebookScope::PerKvHead => backend.supports_per_kv_head_codebook,
    };
    require_supported(supported, "K codebook scope")
}

fn require_topology(
    backend: DalucBackendCapabilities,
    topology: DalucStorageTopology,
) -> Result<(), DalucKvViewError> {
    let supported = match topology {
        DalucStorageTopology::Contiguous { .. } => backend.supports_contiguous,
        DalucStorageTopology::Paged { .. } => backend.supports_paged,
    };
    require_supported(supported, "KV storage topology")
}

fn require_padding(
    backend: DalucBackendCapabilities,
    padding: DalucPaddingRule,
) -> Result<(), DalucKvViewError> {
    let supported = match padding {
        DalucPaddingRule::None => backend.supports_no_padding,
        DalucPaddingRule::ZeroFilledToAlignment => backend.supports_zero_filled_padding,
    };
    require_supported(supported, "padding rule")
}

fn require_v_zero_point(
    backend: DalucBackendCapabilities,
    zero_point: DalucZeroPointStorage,
) -> Result<(), DalucKvViewError> {
    let (supported, label) = match zero_point {
        DalucZeroPointStorage::None => (
            backend.supports_signed_symmetric_values,
            "signed symmetric V values",
        ),
        DalucZeroPointStorage::U8 => (backend.supports_u8_zero_points, "u8 V zero points"),
        DalucZeroPointStorage::U16 => (backend.supports_u16_zero_points, "u16 V zero points"),
    };
    require_supported(supported, label)
}

fn require_residual(
    backend: DalucBackendCapabilities,
    residual: DalucResidualSemantics,
    label: &'static str,
) -> Result<(), DalucKvViewError> {
    let DalucResidualSemantics::Sparse(residual) = residual else {
        return Ok(());
    };
    require_dtype(backend, residual.value_dtype, label)?;
    match residual.indexing {
        DalucResidualIndexing::Coordinates {
            index_bits,
            bit_order,
        } => {
            if !backend.supports_sparse_coordinates || index_bits > backend.max_packed_index_bits {
                return Err(DalucKvViewError::UnsupportedBackendCapability(label));
            }
            require_bit_order(backend, bit_order, label)
        }
        DalucResidualIndexing::Bitmap { bit_order } => {
            if !backend.supports_sparse_bitmap {
                return Err(DalucKvViewError::UnsupportedBackendCapability(label));
            }
            require_bit_order(backend, bit_order, label)
        }
    }
}

fn require_supported(supported: bool, label: &'static str) -> Result<(), DalucKvViewError> {
    if !supported {
        return Err(DalucKvViewError::UnsupportedBackendCapability(label));
    }
    Ok(())
}

fn checked_product(values: &[usize]) -> Result<usize, DalucKvViewError> {
    values
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or(DalucKvViewError::InvalidMetadata(
            "logical shape arithmetic overflow",
        ))
}

fn to_u64(value: usize) -> Result<u64, DalucKvViewError> {
    u64::try_from(value).map_err(|_| DalucKvViewError::InvalidMetadata("usize exceeds u64"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> DalucKvViewContract {
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
                residual: DalucResidualSemantics::Sparse(DalucSparseResidual {
                    value_dtype: DalucFloatDType::F16,
                    indexing: DalucResidualIndexing::Coordinates {
                        index_bits: 6,
                        bit_order: DalucBitOrder::Lsb0,
                    },
                    max_entries_per_vector: 4,
                }),
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

    fn backend() -> DalucBackendCapabilities {
        DalucBackendCapabilities {
            supports_f16: true,
            supports_bf16: false,
            supports_f32: true,
            supports_lsb0: true,
            supports_msb0: false,
            supports_batch_token_head: true,
            supports_batch_head_token: false,
            supports_shared_codebook: false,
            supports_per_kv_head_codebook: true,
            supports_contiguous: true,
            supports_paged: true,
            supports_groupwise_affine_values: true,
            supports_signed_symmetric_values: true,
            supports_u8_zero_points: true,
            supports_u16_zero_points: false,
            supports_sparse_coordinates: true,
            supports_sparse_bitmap: false,
            supports_no_padding: true,
            supports_zero_filled_padding: true,
            minimum_plane_alignment_bytes: 16,
            max_packed_index_bits: 8,
            max_groupwise_value_bits: 4,
        }
    }

    #[test]
    fn representative_asymmetric_view_validates() {
        let contract = view();
        contract.validate().unwrap();
        contract.validate_for_backend(backend()).unwrap();
        contract.validate_codebook_index(255).unwrap();
        contract.validate_key_residual_index(63).unwrap();
    }

    #[test]
    fn versions_head_mapping_and_subspaces_fail_closed() {
        let mut invalid = view();
        invalid.schema_version += 1;
        assert!(matches!(
            invalid.validate(),
            Err(DalucKvViewError::UnsupportedSchemaVersion { .. })
        ));

        let mut invalid = view();
        invalid.shape.q_heads = 30;
        assert!(matches!(
            invalid.validate(),
            Err(DalucKvViewError::InvalidMetadata(_))
        ));

        let mut invalid = view();
        invalid.keys.subspace_dim = 7;
        assert!(matches!(
            invalid.validate(),
            Err(DalucKvViewError::InvalidMetadata(_))
        ));
    }

    #[test]
    fn malformed_packed_and_residual_metadata_fail_closed() {
        let mut invalid = view();
        invalid.keys.index_bits = 7;
        assert!(matches!(
            invalid.validate(),
            Err(DalucKvViewError::InvalidMetadata(_))
        ));

        let mut invalid = view();
        invalid.keys.residual = DalucResidualSemantics::Sparse(DalucSparseResidual {
            value_dtype: DalucFloatDType::F16,
            indexing: DalucResidualIndexing::Coordinates {
                index_bits: 5,
                bit_order: DalucBitOrder::Lsb0,
            },
            max_entries_per_vector: 4,
        });
        assert!(matches!(
            invalid.validate(),
            Err(DalucKvViewError::InvalidMetadata(_))
        ));
    }

    #[test]
    fn payload_indices_and_capacity_fail_closed() {
        let valid = view();
        assert!(matches!(
            valid.validate_codebook_index(256),
            Err(DalucKvViewError::CodebookIndexOutOfRange { .. })
        ));
        assert!(matches!(
            valid.validate_key_residual_index(64),
            Err(DalucKvViewError::ResidualIndexOutOfRange { .. })
        ));
        assert!(matches!(
            valid.validate_value_residual_index(0),
            Err(DalucKvViewError::ResidualIndexOutOfRange { .. })
        ));

        let mut invalid = view();
        invalid.layout.topology = DalucStorageTopology::Paged {
            page_size: 16,
            physical_pages_per_batch: 7,
        };
        assert!(matches!(
            invalid.validate(),
            Err(DalucKvViewError::InvalidMetadata(_))
        ));
    }

    #[test]
    fn payload_validators_cannot_bypass_contract_validation() {
        let mut invalid = view();
        invalid.schema_version += 1;
        assert!(matches!(
            invalid.validate_key_residual_index(0),
            Err(DalucKvViewError::UnsupportedSchemaVersion { .. })
        ));

        let mut invalid = view();
        invalid.keys.residual = DalucResidualSemantics::Sparse(DalucSparseResidual {
            value_dtype: DalucFloatDType::F16,
            indexing: DalucResidualIndexing::Coordinates {
                index_bits: 6,
                bit_order: DalucBitOrder::Lsb0,
            },
            max_entries_per_vector: 0,
        });
        assert!(matches!(
            invalid.validate_key_residual_index(0),
            Err(DalucKvViewError::InvalidMetadata(_))
        ));
    }

    #[test]
    fn unsupported_backend_capability_never_implies_fallback() {
        let mut caps = backend();
        caps.supports_paged = false;
        assert_eq!(
            view().validate_for_backend(caps),
            Err(DalucKvViewError::UnsupportedBackendCapability(
                "KV storage topology"
            ))
        );

        let mut caps = backend();
        caps.max_packed_index_bits = 6;
        assert_eq!(
            view().validate_for_backend(caps),
            Err(DalucKvViewError::UnsupportedBackendCapability(
                "K packed index width"
            ))
        );
    }

    #[test]
    fn backend_encoding_and_layout_variants_fail_closed() {
        let mut caps = backend();
        caps.supports_batch_token_head = false;
        assert_eq!(
            view().validate_for_backend(caps),
            Err(DalucKvViewError::UnsupportedBackendCapability("row order"))
        );

        let mut caps = backend();
        caps.supports_per_kv_head_codebook = false;
        assert_eq!(
            view().validate_for_backend(caps),
            Err(DalucKvViewError::UnsupportedBackendCapability(
                "K codebook scope"
            ))
        );

        let mut caps = backend();
        caps.supports_lsb0 = false;
        assert_eq!(
            view().validate_for_backend(caps),
            Err(DalucKvViewError::UnsupportedBackendCapability(
                "K index bit order"
            ))
        );

        let mut caps = backend();
        caps.supports_u8_zero_points = false;
        assert_eq!(
            view().validate_for_backend(caps),
            Err(DalucKvViewError::UnsupportedBackendCapability(
                "u8 V zero points"
            ))
        );

        let mut caps = backend();
        caps.supports_zero_filled_padding = false;
        assert_eq!(
            view().validate_for_backend(caps),
            Err(DalucKvViewError::UnsupportedBackendCapability(
                "padding rule"
            ))
        );

        let mut caps = backend();
        caps.minimum_plane_alignment_bytes = 32;
        assert_eq!(
            view().validate_for_backend(caps),
            Err(DalucKvViewError::UnsupportedBackendCapability(
                "plane alignment"
            ))
        );
    }
}
