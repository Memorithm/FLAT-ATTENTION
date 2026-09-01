//! Deterministic host reference representation for the research-only DA-LUC KV contract.
//!
//! This module is an oracle and evidence surface only. It does not define a runtime
//! backend layout, train codebooks, dispatch GPU work, or promote compressed KV.

use super::research_da_luc::{
    DalucBitOrder, DalucCodebookScope, DalucFloatDType, DalucKvViewContract, DalucKvViewError,
    DalucPaddingRule, DalucResidualIndexing, DalucResidualSemantics, DalucRowOrder,
    DalucStorageTopology, DalucValueRepresentation, DalucZeroPointStorage,
};
use crate::F16;
use core::fmt;

/// Version of the host-only deterministic oracle payload semantics.
pub const DA_LUC_ORACLE_PAYLOAD_VERSION: u16 = 1;

/// Research-only direct q_len=1 compressed-attention oracle (FDAL2).
#[path = "research_da_luc_decode_oracle.rs"]
pub mod decode;

/// Research-only portable direct-compressed WGPU candidate (FDAL3).
#[cfg(feature = "wgpu")]
#[path = "research_da_luc_wgpu.rs"]
pub mod wgpu;

/// Research-only deterministic dynamic precision tier routing (FDAL5).
#[path = "research_da_luc_tiering.rs"]
pub mod tiering;

/// One byte-backed oracle plane.
///
/// `logical_bits` excludes byte-tail and alignment padding. `bytes` includes both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DalucOraclePlane {
    bytes: Vec<u8>,
    logical_bits: usize,
    alignment_padding_bytes: usize,
}

impl DalucOraclePlane {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn logical_bits(&self) -> usize {
        self.logical_bits
    }

    pub fn logical_bytes(&self) -> usize {
        bytes_for_bits(self.logical_bits).expect("validated oracle plane length")
    }

    pub const fn alignment_padding_bytes(&self) -> usize {
        self.alignment_padding_bytes
    }

    pub fn byte_tail_padding_bits(&self) -> usize {
        self.logical_bytes() * 8 - self.logical_bits
    }

    pub fn total_bytes(&self) -> usize {
        self.bytes.len()
    }

    fn empty() -> Self {
        Self {
            bytes: Vec::new(),
            logical_bits: 0,
            alignment_padding_bytes: 0,
        }
    }
}

/// Exact byte accounting for one oracle payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DalucOracleStorageReport {
    pub logical_kv_scalar_count: usize,
    pub key_codebook_payload_bytes: usize,
    pub key_index_payload_bytes: usize,
    pub key_residual_value_payload_bytes: usize,
    pub key_residual_index_payload_bytes: usize,
    pub value_payload_bytes: usize,
    pub value_scale_payload_bytes: usize,
    pub value_zero_point_payload_bytes: usize,
    pub value_residual_value_payload_bytes: usize,
    pub value_residual_index_payload_bytes: usize,
    pub page_metadata_payload_bytes: usize,
    pub packing_tail_padding_bits: usize,
    pub alignment_padding_bytes: usize,
    pub external_metadata_bytes: usize,
    pub total_representation_bytes: usize,
    pub dense_baseline_dtype: DalucFloatDType,
    pub dense_baseline_bytes: usize,
    pub effective_bits_per_value: f64,
    pub compression_ratio_against_dense: f64,
}

/// Error statistics against the caller-supplied dense values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DalucOracleErrorStats {
    pub scalar_count: usize,
    pub max_abs: f32,
    pub mean_abs: f64,
    pub root_mean_square: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DalucOracleReconstructionReport {
    pub keys: DalucOracleErrorStats,
    pub values: DalucOracleErrorStats,
}

/// Versioned deterministic representation payload.
///
/// This is intentionally not a stable runtime ABI. The planes make storage
/// accounting and scalar reconstruction reproducible while leaving physical
/// runtime ownership to the selected backend/adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DalucOraclePayload {
    pub payload_version: u16,
    pub contract: DalucKvViewContract,
    key_codebook: DalucOraclePlane,
    key_indices: DalucOraclePlane,
    key_residual_values: DalucOraclePlane,
    key_residual_indexing: DalucOraclePlane,
    values: DalucOraclePlane,
    value_scales: DalucOraclePlane,
    value_zero_points: DalucOraclePlane,
    value_residual_values: DalucOraclePlane,
    value_residual_indexing: DalucOraclePlane,
    page_table: DalucOraclePlane,
}

impl DalucOraclePayload {
    /// Deterministically encode a validated contract from dense canonical K/V.
    ///
    /// `key_codebook` is supplied by the caller: FDAL1 does not train or calibrate
    /// codebooks. Its length follows the FDAL0 codebook scope and shape.
    pub fn encode(
        contract: DalucKvViewContract,
        key_codebook: &[f32],
        dense_keys: &[f32],
        dense_values: &[f32],
    ) -> Result<Self, DalucOracleError> {
        Self::encode_impl(contract, None, key_codebook, dense_keys, dense_values)
    }

    /// Encode with an explicit logical-page to physical-page mapping.
    ///
    /// Only paged contracts accept a page table. Contiguous contracts reject one
    /// rather than silently ignoring adapter-owned placement evidence.
    pub fn encode_with_page_table(
        contract: DalucKvViewContract,
        page_table: &[u32],
        key_codebook: &[f32],
        dense_keys: &[f32],
        dense_values: &[f32],
    ) -> Result<Self, DalucOracleError> {
        Self::encode_impl(
            contract,
            Some(page_table),
            key_codebook,
            dense_keys,
            dense_values,
        )
    }

    fn encode_impl(
        contract: DalucKvViewContract,
        page_table: Option<&[u32]>,
        key_codebook: &[f32],
        dense_keys: &[f32],
        dense_values: &[f32],
    ) -> Result<Self, DalucOracleError> {
        contract.validate()?;
        validate_finite_slice("K input", dense_keys)?;
        validate_finite_slice("V input", dense_values)?;
        require_len("K input", dense_keys.len(), logical_side_len(contract, true)?)?;
        require_len(
            "V input",
            dense_values.len(),
            logical_side_len(contract, false)?,
        )?;

        let geometry = OracleGeometry::new(contract, page_table)?;
        let codebook = StoredCodebook::new(contract, key_codebook)?;
        let key_indices = encode_keys(contract, &geometry, &codebook, dense_keys)?;
        let encoded_values = encode_values(contract, &geometry, dense_values)?;
        let key_primary = decode_primary_keys(contract, &geometry, &codebook, &key_indices)?;
        let value_primary = decode_primary_values(contract, &geometry, &encoded_values)?;
        let key_residual = encode_residuals(
            contract,
            &geometry,
            DalucKvSide::Key,
            dense_keys,
            &key_primary,
            key_residual(contract),
        )?;
        let value_residual = encode_residuals(
            contract,
            &geometry,
            DalucKvSide::Value,
            dense_values,
            &value_primary,
            value_residual(contract),
        )?;

        let payload = Self {
            payload_version: DA_LUC_ORACLE_PAYLOAD_VERSION,
            contract,
            key_codebook: codebook.plane,
            key_indices,
            key_residual_values: key_residual.values,
            key_residual_indexing: key_residual.indexing,
            values: encoded_values.values,
            value_scales: encoded_values.scales,
            value_zero_points: encoded_values.zero_points,
            value_residual_values: value_residual.values,
            value_residual_indexing: value_residual.indexing,
            page_table: geometry.page_table_plane,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn validate(&self) -> Result<(), DalucOracleError> {
        if self.payload_version != DA_LUC_ORACLE_PAYLOAD_VERSION {
            return Err(DalucOracleError::UnsupportedPayloadVersion {
                actual: self.payload_version,
                supported: DA_LUC_ORACLE_PAYLOAD_VERSION,
            });
        }
        self.contract.validate()?;
        let geometry = OracleGeometry::from_payload(self.contract, &self.page_table)?;
        validate_stored_codebook(self.contract, &self.key_codebook)?;
        validate_key_indices(self.contract, &geometry, &self.key_indices)?;
        validate_stored_values(
            self.contract,
            &geometry,
            &self.values,
            &self.value_scales,
            &self.value_zero_points,
        )?;
        validate_stored_residuals(
            self.contract,
            &geometry,
            DalucKvSide::Key,
            &self.key_residual_values,
            &self.key_residual_indexing,
            key_residual(self.contract),
        )?;
        validate_stored_residuals(
            self.contract,
            &geometry,
            DalucKvSide::Value,
            &self.value_residual_values,
            &self.value_residual_indexing,
            value_residual(self.contract),
        )?;
        Ok(())
    }

    pub fn decode_keys(&self) -> Result<Vec<f32>, DalucOracleError> {
        self.validate()?;
        let geometry = OracleGeometry::from_payload(self.contract, &self.page_table)?;
        let codebook = StoredCodebook::from_plane(self.contract, self.key_codebook.clone())?;
        let mut output = decode_primary_keys(self.contract, &geometry, &codebook, &self.key_indices)?;
        apply_residuals(
            self.contract,
            &geometry,
            DalucKvSide::Key,
            &mut output,
            &self.key_residual_values,
            &self.key_residual_indexing,
            key_residual(self.contract),
        )?;
        Ok(output)
    }

    pub fn decode_values(&self) -> Result<Vec<f32>, DalucOracleError> {
        self.validate()?;
        let geometry = OracleGeometry::from_payload(self.contract, &self.page_table)?;
        let encoded = EncodedValues {
            values: self.values.clone(),
            scales: self.value_scales.clone(),
            zero_points: self.value_zero_points.clone(),
        };
        let mut output = decode_primary_values(self.contract, &geometry, &encoded)?;
        apply_residuals(
            self.contract,
            &geometry,
            DalucKvSide::Value,
            &mut output,
            &self.value_residual_values,
            &self.value_residual_indexing,
            value_residual(self.contract),
        )?;
        Ok(output)
    }

    pub fn reconstruction_report(
        &self,
        dense_keys: &[f32],
        dense_values: &[f32],
    ) -> Result<DalucOracleReconstructionReport, DalucOracleError> {
        require_len(
            "dense K comparison",
            dense_keys.len(),
            logical_side_len(self.contract, true)?,
        )?;
        require_len(
            "dense V comparison",
            dense_values.len(),
            logical_side_len(self.contract, false)?,
        )?;
        validate_finite_slice("K input", dense_keys)?;
        validate_finite_slice("V input", dense_values)?;
        let decoded_keys = self.decode_keys()?;
        let decoded_values = self.decode_values()?;
        Ok(DalucOracleReconstructionReport {
            keys: error_stats(dense_keys, &decoded_keys),
            values: error_stats(dense_values, &decoded_values),
        })
    }

    /// Exact allocated-byte accounting for this oracle payload.
    ///
    /// `external_metadata_bytes` is explicit because FDAL0 intentionally did not
    /// standardize a runtime descriptor serialization. Callers must count their
    /// surrounding protocol rather than letting the oracle invent one.
    pub fn storage_report(
        &self,
        dense_baseline_dtype: DalucFloatDType,
        external_metadata_bytes: usize,
    ) -> Result<DalucOracleStorageReport, DalucOracleError> {
        self.validate()?;
        let geometry = OracleGeometry::from_payload(self.contract, &self.page_table)?;
        let planes = self.all_planes();
        let alignment_padding_bytes = planes
            .iter()
            .try_fold(0usize, |acc, plane| {
                acc.checked_add(plane.alignment_padding_bytes())
            })
            .ok_or(DalucOracleError::ArithmeticOverflow(
                "alignment padding sum",
            ))?;
        let packing_tail_padding_bits = planes
            .iter()
            .try_fold(0usize, |acc, plane| {
                acc.checked_add(plane.byte_tail_padding_bits())
            })
            .ok_or(DalucOracleError::ArithmeticOverflow("packing tail sum"))?;
        let payload_bytes = planes
            .iter()
            .try_fold(0usize, |acc, plane| acc.checked_add(plane.total_bytes()))
            .ok_or(DalucOracleError::ArithmeticOverflow("payload byte sum"))?;
        let total_representation_bytes = payload_bytes
            .checked_add(external_metadata_bytes)
            .ok_or(DalucOracleError::ArithmeticOverflow("representation bytes"))?;
        let logical_kv_scalar_count = logical_kv_scalar_count(self.contract)?;
        let dense_baseline_bytes = dense_baseline_bytes(
            self.contract,
            &geometry,
            dense_baseline_dtype,
            external_metadata_bytes,
        )?;
        let effective_bits_per_value =
            total_representation_bytes as f64 * 8.0 / logical_kv_scalar_count as f64;
        let compression_ratio_against_dense =
            dense_baseline_bytes as f64 / total_representation_bytes as f64;

        Ok(DalucOracleStorageReport {
            logical_kv_scalar_count,
            key_codebook_payload_bytes: self.key_codebook.logical_bytes(),
            key_index_payload_bytes: self.key_indices.logical_bytes(),
            key_residual_value_payload_bytes: self.key_residual_values.logical_bytes(),
            key_residual_index_payload_bytes: self.key_residual_indexing.logical_bytes(),
            value_payload_bytes: self.values.logical_bytes(),
            value_scale_payload_bytes: self.value_scales.logical_bytes(),
            value_zero_point_payload_bytes: self.value_zero_points.logical_bytes(),
            value_residual_value_payload_bytes: self.value_residual_values.logical_bytes(),
            value_residual_index_payload_bytes: self.value_residual_indexing.logical_bytes(),
            page_metadata_payload_bytes: self.page_table.logical_bytes(),
            packing_tail_padding_bits,
            alignment_padding_bytes,
            external_metadata_bytes,
            total_representation_bytes,
            dense_baseline_dtype,
            dense_baseline_bytes,
            effective_bits_per_value,
            compression_ratio_against_dense,
        })
    }

    fn all_planes(&self) -> [&DalucOraclePlane; 10] {
        [
            &self.key_codebook,
            &self.key_indices,
            &self.key_residual_values,
            &self.key_residual_indexing,
            &self.values,
            &self.value_scales,
            &self.value_zero_points,
            &self.value_residual_values,
            &self.value_residual_indexing,
            &self.page_table,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DalucOracleError {
    Contract(DalucKvViewError),
    UnsupportedPayloadVersion {
        actual: u16,
        supported: u16,
    },
    LengthMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        what: &'static str,
        index: usize,
    },
    ArithmeticOverflow(&'static str),
    MalformedPayload(&'static str),
    InvalidPageTable(&'static str),
    ScaleUnderflow {
        row: usize,
        group: usize,
    },
}

impl fmt::Display for DalucOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(f, "{error}"),
            Self::UnsupportedPayloadVersion { actual, supported } => write!(
                f,
                "DA-LUC oracle payload version {actual} is unsupported; expected {supported}"
            ),
            Self::LengthMismatch {
                what,
                expected,
                actual,
            } => write!(
                f,
                "{what} length {actual} does not match expected {expected}"
            ),
            Self::NonFinite { what, index } => {
                write!(f, "{what} contains non-finite value at {index}")
            }
            Self::ArithmeticOverflow(label) => {
                write!(f, "DA-LUC oracle arithmetic overflow: {label}")
            }
            Self::MalformedPayload(label) => {
                write!(f, "malformed DA-LUC oracle payload: {label}")
            }
            Self::InvalidPageTable(label) => {
                write!(f, "invalid DA-LUC oracle page table: {label}")
            }
            Self::ScaleUnderflow { row, group } => write!(
                f,
                "DA-LUC V scale underflow/non-finite value at physical row {row}, group {group}"
            ),
        }
    }
}

impl std::error::Error for DalucOracleError {}

impl From<DalucKvViewError> for DalucOracleError {
    fn from(value: DalucKvViewError) -> Self {
        Self::Contract(value)
    }
}

#[derive(Debug, Clone)]
struct OracleGeometry {
    capacity_tokens: usize,
    physical_rows: usize,
    logical_pages_per_batch: usize,
    page_table: Vec<u32>,
    page_table_plane: DalucOraclePlane,
}

impl OracleGeometry {
    fn new(
        contract: DalucKvViewContract,
        supplied_page_table: Option<&[u32]>,
    ) -> Result<Self, DalucOracleError> {
        let (capacity_tokens, logical_pages_per_batch, page_table) = match contract.layout.topology
        {
            DalucStorageTopology::Contiguous { capacity_tokens } => {
                if supplied_page_table.is_some_and(|table| !table.is_empty()) {
                    return Err(DalucOracleError::InvalidPageTable(
                        "contiguous topology does not carry a page table",
                    ));
                }
                (capacity_tokens, 0, Vec::new())
            }
            DalucStorageTopology::Paged {
                page_size,
                physical_pages_per_batch,
            } => {
                let logical_pages = div_ceil(contract.shape.kv_len, page_size)?;
                let entries =
                    checked_product(&[contract.shape.batch, logical_pages], "page entries")?;
                let table = match supplied_page_table {
                    Some(table) => {
                        require_len("page table", table.len(), entries)?;
                        table.to_vec()
                    }
                    None => {
                        if logical_pages > physical_pages_per_batch {
                            return Err(DalucOracleError::InvalidPageTable(
                                "identity map exceeds physical page capacity",
                            ));
                        }
                        let mut table = Vec::with_capacity(entries);
                        for _batch in 0..contract.shape.batch {
                            for page in 0..logical_pages {
                                table.push(u32::try_from(page).map_err(|_| {
                                    DalucOracleError::InvalidPageTable("page index exceeds u32")
                                })?);
                            }
                        }
                        table
                    }
                };
                validate_page_table(
                    &table,
                    contract.shape.batch,
                    logical_pages,
                    physical_pages_per_batch,
                )?;
                let capacity_tokens = page_size
                    .checked_mul(physical_pages_per_batch)
                    .ok_or(DalucOracleError::ArithmeticOverflow("paged capacity"))?;
                (capacity_tokens, logical_pages, table)
            }
        };
        let physical_rows = checked_product(
            &[
                contract.shape.batch,
                contract.shape.kv_heads,
                capacity_tokens,
            ],
            "physical rows",
        )?;
        let page_table_plane = page_table_plane(contract, &page_table)?;
        Ok(Self {
            capacity_tokens,
            physical_rows,
            logical_pages_per_batch,
            page_table,
            page_table_plane,
        })
    }

    fn from_payload(
        contract: DalucKvViewContract,
        page_table_plane: &DalucOraclePlane,
    ) -> Result<Self, DalucOracleError> {
        match contract.layout.topology {
            DalucStorageTopology::Contiguous { .. } => {
                if page_table_plane.logical_bits != 0 || !page_table_plane.bytes.is_empty() {
                    return Err(DalucOracleError::InvalidPageTable(
                        "contiguous payload contains page metadata",
                    ));
                }
                Self::new(contract, None)
            }
            DalucStorageTopology::Paged { page_size, .. } => {
                if page_table_plane.logical_bits % 32 != 0 {
                    return Err(DalucOracleError::InvalidPageTable(
                        "page table is not a whole u32 stream",
                    ));
                }
                let logical_pages = div_ceil(contract.shape.kv_len, page_size)?;
                let expected_entries =
                    checked_product(&[contract.shape.batch, logical_pages], "page table entries")?;
                let mut table = Vec::with_capacity(expected_entries);
                let logical_bytes = page_table_plane.logical_bytes();
                let expected_bytes = expected_entries
                    .checked_mul(4)
                    .ok_or(DalucOracleError::ArithmeticOverflow("page table bytes"))?;
                require_len("page table bytes", logical_bytes, expected_bytes)?;
                for chunk in page_table_plane.bytes[..logical_bytes].chunks_exact(4) {
                    table.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Self::new(contract, Some(&table))
            }
        }
    }

    fn physical_row(
        &self,
        contract: DalucKvViewContract,
        batch: usize,
        head: usize,
        token: usize,
    ) -> Result<usize, DalucOracleError> {
        let physical_token = match contract.layout.topology {
            DalucStorageTopology::Contiguous { .. } => token,
            DalucStorageTopology::Paged { page_size, .. } => {
                let logical_page = token / page_size;
                let offset = token % page_size;
                let table_index = batch
                    .checked_mul(self.logical_pages_per_batch)
                    .and_then(|base| base.checked_add(logical_page))
                    .ok_or(DalucOracleError::ArithmeticOverflow("page table lookup"))?;
                let page = usize::try_from(self.page_table[table_index]).map_err(|_| {
                    DalucOracleError::InvalidPageTable("u32 page does not fit usize")
                })?;
                page.checked_mul(page_size)
                    .and_then(|base| base.checked_add(offset))
                    .ok_or(DalucOracleError::ArithmeticOverflow("physical token"))?
            }
        };
        let row = match contract.layout.row_order {
            DalucRowOrder::BatchTokenHead => batch
                .checked_mul(self.capacity_tokens)
                .and_then(|v| v.checked_add(physical_token))
                .and_then(|v| v.checked_mul(contract.shape.kv_heads))
                .and_then(|v| v.checked_add(head)),
            DalucRowOrder::BatchHeadToken => batch
                .checked_mul(contract.shape.kv_heads)
                .and_then(|v| v.checked_add(head))
                .and_then(|v| v.checked_mul(self.capacity_tokens))
                .and_then(|v| v.checked_add(physical_token)),
        }
        .ok_or(DalucOracleError::ArithmeticOverflow("physical row index"))?;
        if row >= self.physical_rows {
            return Err(DalucOracleError::MalformedPayload(
                "physical row outside capacity",
            ));
        }
        Ok(row)
    }
}

#[derive(Debug, Clone)]
struct StoredCodebook {
    plane: DalucOraclePlane,
    decoded: Vec<f32>,
}

impl StoredCodebook {
    fn new(contract: DalucKvViewContract, input: &[f32]) -> Result<Self, DalucOracleError> {
        let expected = codebook_elements(contract)?;
        require_len("K codebook", input.len(), expected)?;
        let mut raw = Vec::with_capacity(
            expected
                .checked_mul(dtype_bytes(contract.keys.codebook_dtype))
                .ok_or(DalucOracleError::ArithmeticOverflow("codebook bytes"))?,
        );
        let mut decoded = Vec::with_capacity(expected);
        for (index, &value) in input.iter().enumerate() {
            append_float(&mut raw, value, contract.keys.codebook_dtype);
            let stored = round_float(value, contract.keys.codebook_dtype);
            if !stored.is_finite() {
                return Err(DalucOracleError::NonFinite {
                    what: "stored K codebook",
                    index,
                });
            }
            decoded.push(stored);
        }
        let plane = finalize_byte_plane(contract, raw)?;
        Ok(Self { plane, decoded })
    }

    fn from_plane(
        contract: DalucKvViewContract,
        plane: DalucOraclePlane,
    ) -> Result<Self, DalucOracleError> {
        validate_stored_codebook(contract, &plane)?;
        let expected = codebook_elements(contract)?;
        let mut decoded = Vec::with_capacity(expected);
        for index in 0..expected {
            let value = read_float(&plane, index, contract.keys.codebook_dtype)?;
            if !value.is_finite() {
                return Err(DalucOracleError::MalformedPayload(
                    "non-finite codebook value",
                ));
            }
            decoded.push(value);
        }
        Ok(Self { plane, decoded })
    }

    fn vector_offset(
        &self,
        contract: DalucKvViewContract,
        head: usize,
        subspace: usize,
        entry: usize,
    ) -> Result<usize, DalucOracleError> {
        let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
        let scope_head = match contract.keys.codebook_scope {
            DalucCodebookScope::SharedAcrossKvHeads => 0,
            DalucCodebookScope::PerKvHead => head,
        };
        scope_head
            .checked_mul(subspaces)
            .and_then(|v| v.checked_add(subspace))
            .and_then(|v| v.checked_mul(contract.keys.codebook_entries))
            .and_then(|v| v.checked_add(entry))
            .and_then(|v| v.checked_mul(contract.keys.subspace_dim))
            .ok_or(DalucOracleError::ArithmeticOverflow(
                "codebook vector offset",
            ))
    }
}

#[derive(Debug, Clone)]
struct EncodedValues {
    values: DalucOraclePlane,
    scales: DalucOraclePlane,
    zero_points: DalucOraclePlane,
}

fn encode_keys(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    codebook: &StoredCodebook,
    dense_keys: &[f32],
) -> Result<DalucOraclePlane, DalucOracleError> {
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    let count = checked_product(&[geometry.physical_rows, subspaces], "K index count")?;
    let mut indices = vec![0u32; count];
    for batch in 0..contract.shape.batch {
        for head in 0..contract.shape.kv_heads {
            for token in 0..contract.shape.kv_len {
                let row = geometry.physical_row(contract, batch, head, token)?;
                let input_base = canonical_vector_offset(contract, batch, head, token, true)?;
                for subspace in 0..subspaces {
                    let start = input_base + subspace * contract.keys.subspace_dim;
                    let input = &dense_keys[start..start + contract.keys.subspace_dim];
                    let mut best_entry = 0usize;
                    let mut best_distance = f64::INFINITY;
                    for entry in 0..contract.keys.codebook_entries {
                        let codebook_start =
                            codebook.vector_offset(contract, head, subspace, entry)?;
                        let candidate = &codebook.decoded
                            [codebook_start..codebook_start + contract.keys.subspace_dim];
                        let distance = input.iter().zip(candidate).fold(0.0f64, |acc, (&a, &b)| {
                            let delta = f64::from(a) - f64::from(b);
                            acc + delta * delta
                        });
                        if distance < best_distance {
                            best_distance = distance;
                            best_entry = entry;
                        }
                    }
                    let index = u32::try_from(best_entry).map_err(|_| {
                        DalucOracleError::ArithmeticOverflow("K codebook index u32")
                    })?;
                    indices[row * subspaces + subspace] = index;
                }
            }
        }
    }
    pack_integer_plane(
        contract,
        &indices,
        contract.keys.index_bits,
        contract.keys.index_bit_order,
    )
}

fn decode_primary_keys(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    codebook: &StoredCodebook,
    indices: &DalucOraclePlane,
) -> Result<Vec<f32>, DalucOracleError> {
    let output_len = logical_side_len(contract, true)?;
    let mut output = vec![0.0f32; output_len];
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    for batch in 0..contract.shape.batch {
        for head in 0..contract.shape.kv_heads {
            for token in 0..contract.shape.kv_len {
                let row = geometry.physical_row(contract, batch, head, token)?;
                let output_base = canonical_vector_offset(contract, batch, head, token, true)?;
                for subspace in 0..subspaces {
                    let packed_index = row * subspaces + subspace;
                    let entry = usize::try_from(unpack_integer(
                        indices,
                        packed_index,
                        contract.keys.index_bits,
                        contract.keys.index_bit_order,
                    )?)
                    .map_err(|_| {
                        DalucOracleError::MalformedPayload("K index does not fit usize")
                    })?;
                    if entry >= contract.keys.codebook_entries {
                        return Err(DalucOracleError::MalformedPayload(
                            "K codebook index out of range",
                        ));
                    }
                    let codebook_start = codebook.vector_offset(contract, head, subspace, entry)?;
                    let target_start = output_base + subspace * contract.keys.subspace_dim;
                    output[target_start..target_start + contract.keys.subspace_dim]
                        .copy_from_slice(
                            &codebook.decoded
                                [codebook_start..codebook_start + contract.keys.subspace_dim],
                        );
                }
            }
        }
    }
    Ok(output)
}

fn encode_values(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    dense_values: &[f32],
) -> Result<EncodedValues, DalucOracleError> {
    match contract.values {
        DalucValueRepresentation::Dense { dtype } => {
            let scalar_count = checked_product(
                &[geometry.physical_rows, contract.shape.value_head_dim],
                "dense V scalar count",
            )?;
            let bytes = scalar_count
                .checked_mul(dtype_bytes(dtype))
                .ok_or(DalucOracleError::ArithmeticOverflow("dense V bytes"))?;
            let mut raw = vec![0u8; bytes];
            for batch in 0..contract.shape.batch {
                for head in 0..contract.shape.kv_heads {
                    for token in 0..contract.shape.kv_len {
                        let row = geometry.physical_row(contract, batch, head, token)?;
                        let input_base =
                            canonical_vector_offset(contract, batch, head, token, false)?;
                        for feature in 0..contract.shape.value_head_dim {
                            write_float(
                                &mut raw,
                                row * contract.shape.value_head_dim + feature,
                                dense_values[input_base + feature],
                                dtype,
                            )?;
                        }
                    }
                }
            }
            Ok(EncodedValues {
                values: finalize_byte_plane(contract, raw)?,
                scales: DalucOraclePlane::empty(),
                zero_points: DalucOraclePlane::empty(),
            })
        }
        DalucValueRepresentation::GroupwiseAffine {
            storage_bits,
            group_size,
            scale_dtype,
            zero_point,
            bit_order,
            ..
        } => {
            let scalar_count = checked_product(
                &[geometry.physical_rows, contract.shape.value_head_dim],
                "quantized V scalar count",
            )?;
            let groups = contract.shape.value_head_dim / group_size;
            let group_count = checked_product(&[geometry.physical_rows, groups], "V group count")?;
            let mut packed_values = vec![0u32; scalar_count];
            let scale_bytes = group_count
                .checked_mul(dtype_bytes(scale_dtype))
                .ok_or(DalucOracleError::ArithmeticOverflow("V scale bytes"))?;
            let mut scales_raw = vec![0u8; scale_bytes];
            let zero_point_bytes = match zero_point {
                DalucZeroPointStorage::None => 0,
                DalucZeroPointStorage::U8 => 1,
                DalucZeroPointStorage::U16 => 2,
            };
            let zero_point_total_bytes = group_count
                .checked_mul(zero_point_bytes)
                .ok_or(DalucOracleError::ArithmeticOverflow("V zero-point bytes"))?;
            let mut zero_points_raw = vec![0u8; zero_point_total_bytes];
            for batch in 0..contract.shape.batch {
                for head in 0..contract.shape.kv_heads {
                    for token in 0..contract.shape.kv_len {
                        let row = geometry.physical_row(contract, batch, head, token)?;
                        let input_base =
                            canonical_vector_offset(contract, batch, head, token, false)?;
                        for group in 0..groups {
                            let start = input_base + group * group_size;
                            let values = &dense_values[start..start + group_size];
                            let params =
                                quantization_params(values, storage_bits, scale_dtype, zero_point)?;
                            let group_index = row * groups + group;
                            write_float(&mut scales_raw, group_index, params.scale, scale_dtype)?;
                            write_zero_point(
                                &mut zero_points_raw,
                                group_index,
                                params.zero_point,
                                zero_point,
                            )?;
                            for (inner, &value) in values.iter().enumerate() {
                                let q = quantize_value(value, params, storage_bits, zero_point);
                                packed_values[row * contract.shape.value_head_dim
                                    + group * group_size
                                    + inner] = q;
                            }
                        }
                    }
                }
            }
            Ok(EncodedValues {
                values: pack_integer_plane(contract, &packed_values, storage_bits, bit_order)?,
                scales: finalize_byte_plane(contract, scales_raw)?,
                zero_points: finalize_byte_plane(contract, zero_points_raw)?,
            })
        }
    }
}

fn decode_primary_values(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    encoded: &EncodedValues,
) -> Result<Vec<f32>, DalucOracleError> {
    let output_len = logical_side_len(contract, false)?;
    let mut output = vec![0.0f32; output_len];
    match contract.values {
        DalucValueRepresentation::Dense { dtype } => {
            for batch in 0..contract.shape.batch {
                for head in 0..contract.shape.kv_heads {
                    for token in 0..contract.shape.kv_len {
                        let row = geometry.physical_row(contract, batch, head, token)?;
                        let output_base =
                            canonical_vector_offset(contract, batch, head, token, false)?;
                        for feature in 0..contract.shape.value_head_dim {
                            output[output_base + feature] = read_float(
                                &encoded.values,
                                row * contract.shape.value_head_dim + feature,
                                dtype,
                            )?;
                        }
                    }
                }
            }
        }
        DalucValueRepresentation::GroupwiseAffine {
            storage_bits,
            group_size,
            scale_dtype,
            zero_point,
            bit_order,
            ..
        } => {
            let groups = contract.shape.value_head_dim / group_size;
            for batch in 0..contract.shape.batch {
                for head in 0..contract.shape.kv_heads {
                    for token in 0..contract.shape.kv_len {
                        let row = geometry.physical_row(contract, batch, head, token)?;
                        let output_base =
                            canonical_vector_offset(contract, batch, head, token, false)?;
                        for group in 0..groups {
                            let group_index = row * groups + group;
                            let scale = read_float(&encoded.scales, group_index, scale_dtype)?;
                            let zp = read_zero_point(&encoded.zero_points, group_index, zero_point)?;
                            for inner in 0..group_size {
                                let feature = group * group_size + inner;
                                let q = unpack_integer(
                                    &encoded.values,
                                    row * contract.shape.value_head_dim + feature,
                                    storage_bits,
                                    bit_order,
                                )?;
                                output[output_base + feature] =
                                    dequantize_value(q, scale, zp, storage_bits, zero_point);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

fn encode_residuals(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    side: DalucKvSide,
    dense: &[f32],
    primary: &[f32],
    semantics: DalucResidualSemantics,
) -> Result<EncodedResiduals, DalucOracleError> {
    let DalucResidualSemantics::Sparse(sparse) = semantics else {
        return Ok(EncodedResiduals {
            values: DalucOraclePlane::empty(),
            indexing: DalucOraclePlane::empty(),
        });
    };
    let dimension = side_dimension(contract, side);
    let logical_vectors = checked_product(
        &[
            contract.shape.batch,
            contract.shape.kv_heads,
            contract.shape.kv_len,
        ],
        "residual logical vectors",
    )?;
    let physical_scalars = geometry
        .physical_rows
        .checked_mul(dimension)
        .ok_or(DalucOracleError::ArithmeticOverflow(
            "residual physical scalars",
        ))?;
    let selected_per_vector = sparse.max_entries_per_vector.min(dimension);
    let mut selected = vec![Vec::<(usize, f32)>::new(); geometry.physical_rows];
    for vector in 0..logical_vectors {
        let (batch, head, token) = logical_vector_indices(contract, vector)?;
        let row = geometry.physical_row(contract, batch, head, token)?;
        let dense_base = canonical_vector_offset(contract, batch, head, token, side == DalucKvSide::Key)?;
        let primary_base = dense_base;
        let mut candidates = (0..dimension)
            .map(|coordinate| {
                let delta = dense[dense_base + coordinate] - primary[primary_base + coordinate];
                (coordinate, delta)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|(left_index, left), (right_index, right)| {
            right
                .abs()
                .total_cmp(&left.abs())
                .then(left_index.cmp(right_index))
        });
        candidates.truncate(selected_per_vector);
        candidates.sort_by_key(|(coordinate, _)| *coordinate);
        selected[row] = candidates;
    }

    let mut residual_values = Vec::new();
    for row in &selected {
        for (_, value) in row {
            append_float(&mut residual_values, *value, sparse.value_dtype);
        }
    }
    let value_plane = finalize_byte_plane(contract, residual_values)?;

    let indexing_plane = match sparse.indexing {
        DalucResidualIndexing::Coordinates {
            index_bits,
            bit_order,
        } => {
            let sentinel = (1u64 << index_bits) - 1;
            let count = geometry
                .physical_rows
                .checked_mul(sparse.max_entries_per_vector)
                .ok_or(DalucOracleError::ArithmeticOverflow(
                    "residual coordinate count",
                ))?;
            let mut coordinates = vec![
                u32::try_from(sentinel).map_err(|_| DalucOracleError::ArithmeticOverflow(
                    "residual sentinel"
                ))?;
                count
            ];
            for (row_index, row) in selected.iter().enumerate() {
                for (slot, (coordinate, _)) in row.iter().enumerate() {
                    coordinates[row_index * sparse.max_entries_per_vector + slot] =
                        u32::try_from(*coordinate).map_err(|_| {
                            DalucOracleError::ArithmeticOverflow("residual coordinate")
                        })?;
                }
            }
            pack_integer_plane(contract, &coordinates, index_bits, bit_order)?
        }
        DalucResidualIndexing::Bitmap { bit_order } => {
            let mut bitmap = vec![0u32; physical_scalars];
            for (row_index, row) in selected.iter().enumerate() {
                for (coordinate, _) in row {
                    bitmap[row_index * dimension + coordinate] = 1;
                }
            }
            pack_integer_plane(contract, &bitmap, 1, bit_order)?
        }
    };

    Ok(EncodedResiduals {
        values: value_plane,
        indexing: indexing_plane,
    })
}

fn apply_residuals(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    side: DalucKvSide,
    output: &mut [f32],
    values: &DalucOraclePlane,
    indexing: &DalucOraclePlane,
    semantics: DalucResidualSemantics,
) -> Result<(), DalucOracleError> {
    let DalucResidualSemantics::Sparse(sparse) = semantics else {
        return Ok(());
    };
    let dimension = side_dimension(contract, side);
    let value_count = residual_value_count(contract, geometry, indexing, sparse)?;
    let values_decoded = read_float_stream(values, value_count, sparse.value_dtype)?;
    let mut value_index = 0usize;
    for batch in 0..contract.shape.batch {
        for head in 0..contract.shape.kv_heads {
            for token in 0..contract.shape.kv_len {
                let row = geometry.physical_row(contract, batch, head, token)?;
                let output_base = canonical_vector_offset(contract, batch, head, token, side == DalucKvSide::Key)?;
                match sparse.indexing {
                    DalucResidualIndexing::Coordinates {
                        index_bits,
                        bit_order,
                    } => {
                        let sentinel = (1u64 << index_bits) - 1;
                        for slot in 0..sparse.max_entries_per_vector {
                            let raw = unpack_integer(
                                indexing,
                                row * sparse.max_entries_per_vector + slot,
                                index_bits,
                                bit_order,
                            )?;
                            if u64::from(raw) == sentinel {
                                continue;
                            }
                            let coordinate = usize::try_from(raw).map_err(|_| {
                                DalucOracleError::MalformedPayload(
                                    "residual coordinate does not fit usize",
                                )
                            })?;
                            if coordinate >= dimension {
                                return Err(DalucOracleError::MalformedPayload(
                                    "residual coordinate out of range",
                                ));
                            }
                            output[output_base + coordinate] += values_decoded[value_index];
                            value_index += 1;
                        }
                    }
                    DalucResidualIndexing::Bitmap { bit_order } => {
                        for coordinate in 0..dimension {
                            if unpack_integer(
                                indexing,
                                row * dimension + coordinate,
                                1,
                                bit_order,
                            )? != 0
                            {
                                output[output_base + coordinate] += values_decoded[value_index];
                                value_index += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    if value_index != values_decoded.len() {
        return Err(DalucOracleError::MalformedPayload(
            "residual value/index count mismatch",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct EncodedResiduals {
    values: DalucOraclePlane,
    indexing: DalucOraclePlane,
}

#[derive(Debug, Clone, Copy)]
struct QuantizationParams {
    scale: f32,
    zero_point: i32,
}

fn quantization_params(
    values: &[f32],
    storage_bits: u8,
    scale_dtype: DalucFloatDType,
    zero_point: DalucZeroPointStorage,
) -> Result<QuantizationParams, DalucOracleError> {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for &value in values {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    let (q_min, q_max) = integer_range(storage_bits, zero_point);
    let mut scale = if zero_point == DalucZeroPointStorage::None {
        let max_abs = minimum.abs().max(maximum.abs());
        if max_abs == 0.0 {
            1.0
        } else {
            max_abs / (q_max as f32)
        }
    } else if maximum == minimum {
        1.0
    } else {
        (maximum - minimum) / (q_max - q_min) as f32
    };
    scale = round_float(scale, scale_dtype);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(DalucOracleError::ScaleUnderflow { row: 0, group: 0 });
    }
    let zero_point_value = if zero_point == DalucZeroPointStorage::None {
        0
    } else {
        let raw = (q_min as f64 - f64::from(minimum) / f64::from(scale)).round();
        raw.clamp(q_min as f64, q_max as f64) as i32
    };
    Ok(QuantizationParams {
        scale,
        zero_point: zero_point_value,
    })
}

fn quantize_value(
    value: f32,
    params: QuantizationParams,
    storage_bits: u8,
    zero_point: DalucZeroPointStorage,
) -> u32 {
    let (q_min, q_max) = integer_range(storage_bits, zero_point);
    let quantized = (f64::from(value) / f64::from(params.scale) + f64::from(params.zero_point))
        .round()
        .clamp(q_min as f64, q_max as f64) as i32;
    if zero_point == DalucZeroPointStorage::None {
        signed_to_bits(quantized, storage_bits)
    } else {
        u32::try_from(quantized).expect("validated unsigned quantized value")
    }
}

pub(super) fn dequantize_value(
    packed: u32,
    scale: f32,
    zero_point: i32,
    storage_bits: u8,
    zero_point_storage: DalucZeroPointStorage,
) -> f32 {
    let q = if zero_point_storage == DalucZeroPointStorage::None {
        bits_to_signed(packed, storage_bits)
    } else {
        i32::try_from(packed).expect("packed value fits signed i32")
    };
    scale * (q - zero_point) as f32
}

fn integer_range(storage_bits: u8, zero_point: DalucZeroPointStorage) -> (i32, i32) {
    if zero_point == DalucZeroPointStorage::None {
        let half = 1i32 << (storage_bits - 1);
        (-half, half - 1)
    } else {
        (0, (1i32 << storage_bits) - 1)
    }
}

fn signed_to_bits(value: i32, bits: u8) -> u32 {
    let mask = (1u32 << bits) - 1;
    (value as u32) & mask
}

fn bits_to_signed(value: u32, bits: u8) -> i32 {
    let sign = 1u32 << (bits - 1);
    let mask = (1u32 << bits) - 1;
    let value = value & mask;
    if value & sign == 0 {
        value as i32
    } else {
        (value | !mask) as i32
    }
}

fn key_residual(contract: DalucKvViewContract) -> DalucResidualSemantics {
    contract.keys.residual
}

fn value_residual(contract: DalucKvViewContract) -> DalucResidualSemantics {
    match contract.values {
        DalucValueRepresentation::Dense { .. } => DalucResidualSemantics::None,
        DalucValueRepresentation::GroupwiseAffine { residual, .. } => residual,
    }
}

fn side_dimension(contract: DalucKvViewContract, side: DalucKvSide) -> usize {
    match side {
        DalucKvSide::Key => contract.shape.key_head_dim,
        DalucKvSide::Value => contract.shape.value_head_dim,
    }
}

fn logical_side_len(
    contract: DalucKvViewContract,
    key: bool,
) -> Result<usize, DalucOracleError> {
    checked_product(
        &[
            contract.shape.batch,
            contract.shape.kv_heads,
            contract.shape.kv_len,
            if key {
                contract.shape.key_head_dim
            } else {
                contract.shape.value_head_dim
            },
        ],
        "logical side length",
    )
}

fn logical_kv_scalar_count(contract: DalucKvViewContract) -> Result<usize, DalucOracleError> {
    let vectors = checked_product(
        &[
            contract.shape.batch,
            contract.shape.kv_heads,
            contract.shape.kv_len,
        ],
        "logical vectors",
    )?;
    vectors
        .checked_mul(
            contract
                .shape
                .key_head_dim
                .checked_add(contract.shape.value_head_dim)
                .ok_or(DalucOracleError::ArithmeticOverflow("K+V dimensions"))?,
        )
        .ok_or(DalucOracleError::ArithmeticOverflow("logical KV scalars"))
}

fn dense_baseline_bytes(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    dtype: DalucFloatDType,
    external_metadata_bytes: usize,
) -> Result<usize, DalucOracleError> {
    let dimensions = contract
        .shape
        .key_head_dim
        .checked_add(contract.shape.value_head_dim)
        .ok_or(DalucOracleError::ArithmeticOverflow("dense K+V dimensions"))?;
    let scalar_bytes = geometry
        .physical_rows
        .checked_mul(dimensions)
        .and_then(|v| v.checked_mul(dtype_bytes(dtype)))
        .ok_or(DalucOracleError::ArithmeticOverflow(
            "dense baseline bytes",
        ))?;
    scalar_bytes
        .checked_add(geometry.page_table_plane.total_bytes())
        .and_then(|v| v.checked_add(external_metadata_bytes))
        .ok_or(DalucOracleError::ArithmeticOverflow(
            "dense baseline metadata",
        ))
}

fn error_stats(expected: &[f32], actual: &[f32]) -> DalucOracleErrorStats {
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_square = 0.0f64;
    for (&expected, &actual) in expected.iter().zip(actual) {
        let error = (actual - expected).abs();
        max_abs = max_abs.max(error);
        sum_abs += f64::from(error);
        sum_square += f64::from(error) * f64::from(error);
    }
    let count = expected.len();
    DalucOracleErrorStats {
        scalar_count: count,
        max_abs,
        mean_abs: sum_abs / count as f64,
        root_mean_square: (sum_square / count as f64).sqrt(),
    }
}

fn validate_finite_slice(what: &'static str, values: &[f32]) -> Result<(), DalucOracleError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(DalucOracleError::NonFinite { what, index });
    }
    Ok(())
}

fn validate_stored_codebook(
    contract: DalucKvViewContract,
    plane: &DalucOraclePlane,
) -> Result<(), DalucOracleError> {
    let count = codebook_elements(contract)?;
    validate_float_plane(
        contract,
        plane,
        count,
        contract.keys.codebook_dtype,
        "K codebook",
    )
}

fn validate_key_indices(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    plane: &DalucOraclePlane,
) -> Result<(), DalucOracleError> {
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    let count = checked_product(&[geometry.physical_rows, subspaces], "K index count")?;
    validate_integer_plane(
        contract,
        plane,
        count,
        contract.keys.index_bits,
        "K indices",
    )?;
    for index in 0..count {
        let entry = unpack_integer(
            plane,
            index,
            contract.keys.index_bits,
            contract.keys.index_bit_order,
        )?;
        if usize::try_from(entry).map_or(true, |entry| entry >= contract.keys.codebook_entries) {
            return Err(DalucOracleError::MalformedPayload(
                "K codebook index out of range",
            ));
        }
    }
    Ok(())
}

fn validate_stored_values(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    values: &DalucOraclePlane,
    scales: &DalucOraclePlane,
    zero_points: &DalucOraclePlane,
) -> Result<(), DalucOracleError> {
    match contract.values {
        DalucValueRepresentation::Dense { dtype } => {
            let count = checked_product(
                &[geometry.physical_rows, contract.shape.value_head_dim],
                "dense V count",
            )?;
            validate_float_plane(contract, values, count, dtype, "dense V")?;
            validate_empty_plane(scales, "dense V scales")?;
            validate_empty_plane(zero_points, "dense V zero points")
        }
        DalucValueRepresentation::GroupwiseAffine {
            storage_bits,
            group_size,
            scale_dtype,
            zero_point,
            ..
        } => {
            let value_count = checked_product(
                &[geometry.physical_rows, contract.shape.value_head_dim],
                "quantized V count",
            )?;
            validate_integer_plane(contract, values, value_count, storage_bits, "quantized V")?;
            let groups = contract.shape.value_head_dim / group_size;
            let group_count = checked_product(&[geometry.physical_rows, groups], "V groups")?;
            validate_float_plane(contract, scales, group_count, scale_dtype, "V scales")?;
            match zero_point {
                DalucZeroPointStorage::None => validate_empty_plane(zero_points, "V zero points"),
                DalucZeroPointStorage::U8 => validate_integer_bytes(
                    contract,
                    zero_points,
                    group_count,
                    1,
                    "V U8 zero points",
                ),
                DalucZeroPointStorage::U16 => validate_integer_bytes(
                    contract,
                    zero_points,
                    group_count,
                    2,
                    "V U16 zero points",
                ),
            }
        }
    }
}

fn validate_stored_residuals(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    side: DalucKvSide,
    values: &DalucOraclePlane,
    indexing: &DalucOraclePlane,
    semantics: DalucResidualSemantics,
) -> Result<(), DalucOracleError> {
    let DalucResidualSemantics::Sparse(sparse) = semantics else {
        validate_empty_plane(values, "residual values")?;
        return validate_empty_plane(indexing, "residual indexing");
    };
    let count = residual_value_count(contract, geometry, indexing, sparse)?;
    validate_float_plane(contract, values, count, sparse.value_dtype, "residual values")
}

fn residual_value_count(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    indexing: &DalucOraclePlane,
    sparse: super::research_da_luc::DalucSparseResidual,
) -> Result<usize, DalucOracleError> {
    let dimension = match sparse.indexing {
        DalucResidualIndexing::Coordinates { .. } | DalucResidualIndexing::Bitmap { .. } => {
            let key_matches = sparse == match key_residual(contract) {
                DalucResidualSemantics::Sparse(value) => value,
                DalucResidualSemantics::None => sparse,
            };
            if key_matches {
                contract.shape.key_head_dim
            } else {
                contract.shape.value_head_dim
            }
        }
    };
    match sparse.indexing {
        DalucResidualIndexing::Coordinates {
            index_bits,
            bit_order,
        } => {
            let count = geometry
                .physical_rows
                .checked_mul(sparse.max_entries_per_vector)
                .ok_or(DalucOracleError::ArithmeticOverflow(
                    "residual coordinate count",
                ))?;
            validate_integer_plane(contract, indexing, count, index_bits, "residual coordinates")?;
            let sentinel = (1u64 << index_bits) - 1;
            let mut values = 0usize;
            for row in 0..geometry.physical_rows {
                let mut previous = None;
                for slot in 0..sparse.max_entries_per_vector {
                    let raw = unpack_integer(
                        indexing,
                        row * sparse.max_entries_per_vector + slot,
                        index_bits,
                        bit_order,
                    )?;
                    if u64::from(raw) == sentinel {
                        continue;
                    }
                    let coordinate = usize::try_from(raw).map_err(|_| {
                        DalucOracleError::MalformedPayload("residual coordinate does not fit usize")
                    })?;
                    if coordinate >= dimension {
                        return Err(DalucOracleError::MalformedPayload(
                            "residual coordinate out of range",
                        ));
                    }
                    if previous.is_some_and(|previous| coordinate <= previous) {
                        return Err(DalucOracleError::MalformedPayload(
                            "residual coordinates are not unique/increasing",
                        ));
                    }
                    previous = Some(coordinate);
                    values = values.checked_add(1).ok_or(DalucOracleError::ArithmeticOverflow(
                        "residual value count",
                    ))?;
                }
            }
            Ok(values)
        }
        DalucResidualIndexing::Bitmap { bit_order } => {
            let count = geometry
                .physical_rows
                .checked_mul(dimension)
                .ok_or(DalucOracleError::ArithmeticOverflow(
                    "residual bitmap count",
                ))?;
            validate_integer_plane(contract, indexing, count, 1, "residual bitmap")?;
            let mut values = 0usize;
            for index in 0..count {
                if unpack_integer(indexing, index, 1, bit_order)? != 0 {
                    values = values.checked_add(1).ok_or(DalucOracleError::ArithmeticOverflow(
                        "residual value count",
                    ))?;
                }
            }
            Ok(values)
        }
    }
}

fn validate_float_plane(
    contract: DalucKvViewContract,
    plane: &DalucOraclePlane,
    count: usize,
    dtype: DalucFloatDType,
    label: &'static str,
) -> Result<(), DalucOracleError> {
    let logical_bits = count
        .checked_mul(dtype_bytes(dtype))
        .and_then(|bytes| bytes.checked_mul(8))
        .ok_or(DalucOracleError::ArithmeticOverflow("float plane bits"))?;
    validate_plane_shape(contract, plane, logical_bits, label)?;
    for index in 0..count {
        if !read_float(plane, index, dtype)?.is_finite() {
            return Err(DalucOracleError::MalformedPayload(label));
        }
    }
    Ok(())
}

fn validate_integer_plane(
    contract: DalucKvViewContract,
    plane: &DalucOraclePlane,
    count: usize,
    bits: u8,
    label: &'static str,
) -> Result<(), DalucOracleError> {
    let logical_bits = count
        .checked_mul(usize::from(bits))
        .ok_or(DalucOracleError::ArithmeticOverflow("integer plane bits"))?;
    validate_plane_shape(contract, plane, logical_bits, label)
}

fn validate_integer_bytes(
    contract: DalucKvViewContract,
    plane: &DalucOraclePlane,
    count: usize,
    bytes: usize,
    label: &'static str,
) -> Result<(), DalucOracleError> {
    let logical_bits = count
        .checked_mul(bytes)
        .and_then(|value| value.checked_mul(8))
        .ok_or(DalucOracleError::ArithmeticOverflow("integer byte plane bits"))?;
    validate_plane_shape(contract, plane, logical_bits, label)
}

fn validate_plane_shape(
    contract: DalucKvViewContract,
    plane: &DalucOraclePlane,
    logical_bits: usize,
    label: &'static str,
) -> Result<(), DalucOracleError> {
    if plane.logical_bits != logical_bits {
        return Err(DalucOracleError::MalformedPayload(label));
    }
    let logical_bytes = bytes_for_bits(logical_bits)?;
    let expected_total = aligned_bytes(
        logical_bytes,
        contract.layout.plane_alignment_bytes,
        contract.layout.padding,
    )?;
    if plane.bytes.len() != expected_total {
        return Err(DalucOracleError::MalformedPayload(label));
    }
    let expected_alignment_padding = expected_total - logical_bytes;
    if plane.alignment_padding_bytes != expected_alignment_padding {
        return Err(DalucOracleError::MalformedPayload(label));
    }
    let tail_mask = tail_padding_mask(logical_bits, contract.keys.index_bit_order);
    if logical_bits % 8 != 0 && logical_bytes != 0 {
        let last = plane.bytes[logical_bytes - 1];
        if last & tail_mask != 0 {
            return Err(DalucOracleError::MalformedPayload(
                "non-zero packing tail padding",
            ));
        }
    }
    if plane.bytes[logical_bytes..].iter().any(|byte| *byte != 0) {
        return Err(DalucOracleError::MalformedPayload(
            "non-zero alignment padding",
        ));
    }
    Ok(())
}

fn validate_empty_plane(plane: &DalucOraclePlane, label: &'static str) -> Result<(), DalucOracleError> {
    if plane.logical_bits != 0 || !plane.bytes.is_empty() || plane.alignment_padding_bytes != 0 {
        return Err(DalucOracleError::MalformedPayload(label));
    }
    Ok(())
}

fn codebook_elements(contract: DalucKvViewContract) -> Result<usize, DalucOracleError> {
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    let scopes = match contract.keys.codebook_scope {
        DalucCodebookScope::SharedAcrossKvHeads => 1,
        DalucCodebookScope::PerKvHead => contract.shape.kv_heads,
    };
    checked_product(
        &[
            scopes,
            subspaces,
            contract.keys.codebook_entries,
            contract.keys.subspace_dim,
        ],
        "codebook elements",
    )
}

fn canonical_vector_offset(
    contract: DalucKvViewContract,
    batch: usize,
    head: usize,
    token: usize,
    key: bool,
) -> Result<usize, DalucOracleError> {
    let dimension = if key {
        contract.shape.key_head_dim
    } else {
        contract.shape.value_head_dim
    };
    batch
        .checked_mul(contract.shape.kv_heads)
        .and_then(|v| v.checked_add(head))
        .and_then(|v| v.checked_mul(contract.shape.kv_len))
        .and_then(|v| v.checked_add(token))
        .and_then(|v| v.checked_mul(dimension))
        .ok_or(DalucOracleError::ArithmeticOverflow("canonical offset"))
}

fn logical_vector_indices(
    contract: DalucKvViewContract,
    vector: usize,
) -> Result<(usize, usize, usize), DalucOracleError> {
    let per_batch = contract
        .shape
        .kv_heads
        .checked_mul(contract.shape.kv_len)
        .ok_or(DalucOracleError::ArithmeticOverflow("vectors per batch"))?;
    let batch = vector / per_batch;
    let remainder = vector % per_batch;
    let head = remainder / contract.shape.kv_len;
    let token = remainder % contract.shape.kv_len;
    Ok((batch, head, token))
}

fn page_table_plane(
    contract: DalucKvViewContract,
    table: &[u32],
) -> Result<DalucOraclePlane, DalucOracleError> {
    if table.is_empty() {
        return Ok(DalucOraclePlane::empty());
    }
    let mut raw = Vec::with_capacity(table.len() * 4);
    for value in table {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    finalize_byte_plane(contract, raw)
}

fn validate_page_table(
    table: &[u32],
    batch: usize,
    logical_pages: usize,
    physical_pages_per_batch: usize,
) -> Result<(), DalucOracleError> {
    for batch_index in 0..batch {
        let start = batch_index * logical_pages;
        let end = start + logical_pages;
        let mut seen = vec![false; physical_pages_per_batch];
        for &page in &table[start..end] {
            let page = usize::try_from(page)
                .map_err(|_| DalucOracleError::InvalidPageTable("page does not fit usize"))?;
            if page >= physical_pages_per_batch {
                return Err(DalucOracleError::InvalidPageTable(
                    "page exceeds physical capacity",
                ));
            }
            if seen[page] {
                return Err(DalucOracleError::InvalidPageTable(
                    "logical pages alias the same physical page",
                ));
            }
            seen[page] = true;
        }
    }
    Ok(())
}

fn pack_integer_plane(
    contract: DalucKvViewContract,
    values: &[u32],
    bits: u8,
    order: DalucBitOrder,
) -> Result<DalucOraclePlane, DalucOracleError> {
    let logical_bits = values
        .len()
        .checked_mul(usize::from(bits))
        .ok_or(DalucOracleError::ArithmeticOverflow("packed bits"))?;
    let logical_bytes = bytes_for_bits(logical_bits)?;
    let mut raw = vec![0u8; logical_bytes];
    for (index, &value) in values.iter().enumerate() {
        if u64::from(value) >= (1u64 << bits) {
            return Err(DalucOracleError::MalformedPayload(
                "packed integer exceeds declared bit width",
            ));
        }
        write_packed(&mut raw, index, bits, order, value)?;
    }
    finalize_bits_plane(contract, raw, logical_bits)
}

pub(super) fn unpack_integer(
    plane: &DalucOraclePlane,
    index: usize,
    bits: u8,
    order: DalucBitOrder,
) -> Result<u32, DalucOracleError> {
    let start_bit = index
        .checked_mul(usize::from(bits))
        .ok_or(DalucOracleError::ArithmeticOverflow("unpack start bit"))?;
    let end_bit = start_bit
        .checked_add(usize::from(bits))
        .ok_or(DalucOracleError::ArithmeticOverflow("unpack end bit"))?;
    if end_bit > plane.logical_bits {
        return Err(DalucOracleError::MalformedPayload(
            "packed integer read exceeds logical bits",
        ));
    }
    let mut value = 0u32;
    for bit in 0..usize::from(bits) {
        let absolute = start_bit + bit;
        let byte = plane.bytes[absolute / 8];
        let in_byte = absolute % 8;
        let source_bit = match order {
            DalucBitOrder::Lsb0 => in_byte,
            DalucBitOrder::Msb0 => 7 - in_byte,
        };
        let bit_value = (byte >> source_bit) & 1;
        let target_bit = match order {
            DalucBitOrder::Lsb0 => bit,
            DalucBitOrder::Msb0 => usize::from(bits) - 1 - bit,
        };
        value |= u32::from(bit_value) << target_bit;
    }
    Ok(value)
}

fn write_packed(
    bytes: &mut [u8],
    index: usize,
    bits: u8,
    order: DalucBitOrder,
    value: u32,
) -> Result<(), DalucOracleError> {
    let start_bit = index
        .checked_mul(usize::from(bits))
        .ok_or(DalucOracleError::ArithmeticOverflow("pack start bit"))?;
    for bit in 0..usize::from(bits) {
        let source_bit = match order {
            DalucBitOrder::Lsb0 => bit,
            DalucBitOrder::Msb0 => usize::from(bits) - 1 - bit,
        };
        let bit_value = (value >> source_bit) & 1;
        let absolute = start_bit + bit;
        let byte_index = absolute / 8;
        if byte_index >= bytes.len() {
            return Err(DalucOracleError::MalformedPayload(
                "packed integer write exceeds allocation",
            ));
        }
        let in_byte = absolute % 8;
        let target_bit = match order {
            DalucBitOrder::Lsb0 => in_byte,
            DalucBitOrder::Msb0 => 7 - in_byte,
        };
        bytes[byte_index] |= u8::try_from(bit_value).expect("one bit") << target_bit;
    }
    Ok(())
}

fn append_float(bytes: &mut Vec<u8>, value: f32, dtype: DalucFloatDType) {
    match dtype {
        DalucFloatDType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
        DalucFloatDType::F16 => bytes.extend_from_slice(&F16::from_f32(value).to_bits().to_le_bytes()),
        DalucFloatDType::Bf16 => {
            let bits = value.to_bits();
            let rounding_bias = 0x7fff + ((bits >> 16) & 1);
            let rounded = bits.wrapping_add(rounding_bias);
            bytes.extend_from_slice(&(rounded >> 16).to_le_bytes()[..2]);
        }
    }
}

fn round_float(value: f32, dtype: DalucFloatDType) -> f32 {
    match dtype {
        DalucFloatDType::F32 => value,
        DalucFloatDType::F16 => F16::from_f32(value).to_f32(),
        DalucFloatDType::Bf16 => {
            let bits = value.to_bits();
            let rounding_bias = 0x7fff + ((bits >> 16) & 1);
            f32::from_bits((bits.wrapping_add(rounding_bias) >> 16) << 16)
        }
    }
}

fn read_float(
    plane: &DalucOraclePlane,
    index: usize,
    dtype: DalucFloatDType,
) -> Result<f32, DalucOracleError> {
    let width = dtype_bytes(dtype);
    let start = index
        .checked_mul(width)
        .ok_or(DalucOracleError::ArithmeticOverflow("float read offset"))?;
    let end = start
        .checked_add(width)
        .ok_or(DalucOracleError::ArithmeticOverflow("float read end"))?;
    if end > plane.logical_bytes() {
        return Err(DalucOracleError::MalformedPayload(
            "float read exceeds logical bytes",
        ));
    }
    Ok(match dtype {
        DalucFloatDType::F32 => f32::from_le_bytes([
            plane.bytes[start],
            plane.bytes[start + 1],
            plane.bytes[start + 2],
            plane.bytes[start + 3],
        ]),
        DalucFloatDType::F16 => F16::from_bits(u16::from_le_bytes([
            plane.bytes[start],
            plane.bytes[start + 1],
        ]))
        .to_f32(),
        DalucFloatDType::Bf16 => {
            let upper = u16::from_le_bytes([plane.bytes[start], plane.bytes[start + 1]]);
            f32::from_bits(u32::from(upper) << 16)
        }
    })
}

fn read_float_stream(
    plane: &DalucOraclePlane,
    count: usize,
    dtype: DalucFloatDType,
) -> Result<Vec<f32>, DalucOracleError> {
    (0..count).map(|index| read_float(plane, index, dtype)).collect()
}

fn write_float(
    raw: &mut [u8],
    index: usize,
    value: f32,
    dtype: DalucFloatDType,
) -> Result<(), DalucOracleError> {
    let width = dtype_bytes(dtype);
    let start = index
        .checked_mul(width)
        .ok_or(DalucOracleError::ArithmeticOverflow("float write offset"))?;
    let end = start
        .checked_add(width)
        .ok_or(DalucOracleError::ArithmeticOverflow("float write end"))?;
    if end > raw.len() {
        return Err(DalucOracleError::MalformedPayload(
            "float write exceeds allocation",
        ));
    }
    match dtype {
        DalucFloatDType::F32 => raw[start..end].copy_from_slice(&value.to_le_bytes()),
        DalucFloatDType::F16 => raw[start..end]
            .copy_from_slice(&F16::from_f32(value).to_bits().to_le_bytes()),
        DalucFloatDType::Bf16 => {
            let rounded = round_float(value, DalucFloatDType::Bf16).to_bits();
            raw[start..end].copy_from_slice(&((rounded >> 16) as u16).to_le_bytes());
        }
    }
    Ok(())
}

fn write_zero_point(
    raw: &mut [u8],
    index: usize,
    zero_point: i32,
    storage: DalucZeroPointStorage,
) -> Result<(), DalucOracleError> {
    match storage {
        DalucZeroPointStorage::None => Ok(()),
        DalucZeroPointStorage::U8 => {
            let value = u8::try_from(zero_point)
                .map_err(|_| DalucOracleError::MalformedPayload("U8 zero point overflow"))?;
            raw[index] = value;
            Ok(())
        }
        DalucZeroPointStorage::U16 => {
            let value = u16::try_from(zero_point)
                .map_err(|_| DalucOracleError::MalformedPayload("U16 zero point overflow"))?;
            let start = index
                .checked_mul(2)
                .ok_or(DalucOracleError::ArithmeticOverflow("zero point offset"))?;
            raw[start..start + 2].copy_from_slice(&value.to_le_bytes());
            Ok(())
        }
    }
}

pub(super) fn read_zero_point(
    plane: &DalucOraclePlane,
    index: usize,
    storage: DalucZeroPointStorage,
) -> Result<i32, DalucOracleError> {
    Ok(match storage {
        DalucZeroPointStorage::None => 0,
        DalucZeroPointStorage::U8 => {
            if index >= plane.logical_bytes() {
                return Err(DalucOracleError::MalformedPayload(
                    "U8 zero point read exceeds plane",
                ));
            }
            i32::from(plane.bytes[index])
        }
        DalucZeroPointStorage::U16 => {
            let start = index
                .checked_mul(2)
                .ok_or(DalucOracleError::ArithmeticOverflow("zero point read offset"))?;
            if start + 2 > plane.logical_bytes() {
                return Err(DalucOracleError::MalformedPayload(
                    "U16 zero point read exceeds plane",
                ));
            }
            i32::from(u16::from_le_bytes([plane.bytes[start], plane.bytes[start + 1]]))
        }
    })
}

fn finalize_byte_plane(
    contract: DalucKvViewContract,
    raw: Vec<u8>,
) -> Result<DalucOraclePlane, DalucOracleError> {
    let logical_bits = raw
        .len()
        .checked_mul(8)
        .ok_or(DalucOracleError::ArithmeticOverflow("byte plane bits"))?;
    finalize_bits_plane(contract, raw, logical_bits)
}

fn finalize_bits_plane(
    contract: DalucKvViewContract,
    mut raw: Vec<u8>,
    logical_bits: usize,
) -> Result<DalucOraclePlane, DalucOracleError> {
    let logical_bytes = bytes_for_bits(logical_bits)?;
    if raw.len() != logical_bytes {
        return Err(DalucOracleError::MalformedPayload(
            "logical bytes do not match packed allocation",
        ));
    }
    let total = aligned_bytes(
        logical_bytes,
        contract.layout.plane_alignment_bytes,
        contract.layout.padding,
    )?;
    let alignment_padding_bytes = total - logical_bytes;
    raw.resize(total, 0);
    Ok(DalucOraclePlane {
        bytes: raw,
        logical_bits,
        alignment_padding_bytes,
    })
}

fn aligned_bytes(
    logical: usize,
    alignment: usize,
    padding: DalucPaddingRule,
) -> Result<usize, DalucOracleError> {
    if logical == 0 {
        return Ok(0);
    }
    match padding {
        DalucPaddingRule::None => Ok(logical),
        DalucPaddingRule::ZeroFilledToAlignment => {
            let add = alignment - 1;
            logical
                .checked_add(add)
                .map(|value| value / alignment * alignment)
                .ok_or(DalucOracleError::ArithmeticOverflow("aligned bytes"))
        }
    }
}

fn bytes_for_bits(bits: usize) -> Result<usize, DalucOracleError> {
    bits.checked_add(7)
        .map(|value| value / 8)
        .ok_or(DalucOracleError::ArithmeticOverflow("bytes for bits"))
}

fn tail_padding_mask(logical_bits: usize, order: DalucBitOrder) -> u8 {
    let used = logical_bits % 8;
    if used == 0 {
        return 0;
    }
    match order {
        DalucBitOrder::Lsb0 => !((1u8 << used) - 1),
        DalucBitOrder::Msb0 => (1u8 << (8 - used)) - 1,
    }
}

fn dtype_bytes(dtype: DalucFloatDType) -> usize {
    match dtype {
        DalucFloatDType::F16 | DalucFloatDType::Bf16 => 2,
        DalucFloatDType::F32 => 4,
    }
}

fn require_len(
    what: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), DalucOracleError> {
    if actual != expected {
        return Err(DalucOracleError::LengthMismatch {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

fn checked_product(values: &[usize], label: &'static str) -> Result<usize, DalucOracleError> {
    values.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or(DalucOracleError::ArithmeticOverflow(label))
    })
}

fn div_ceil(value: usize, divisor: usize) -> Result<usize, DalucOracleError> {
    value
        .checked_add(divisor - 1)
        .map(|adjusted| adjusted / divisor)
        .ok_or(DalucOracleError::ArithmeticOverflow("div ceil"))
}
