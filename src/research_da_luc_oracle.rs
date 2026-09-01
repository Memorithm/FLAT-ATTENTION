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
#[derive(Debug, Clone, PartialEq)]
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
    /// Explicit metadata bytes supplied by the caller's surrounding protocol.
    /// The oracle never guesses a runtime descriptor size.
    pub external_metadata_bytes: usize,
    pub total_representation_bytes: usize,
    pub dense_baseline_dtype: DalucFloatDType,
    pub dense_baseline_bytes: usize,
    pub effective_bits_per_value: f64,
    pub compression_ratio_against_dense: f64,
}

/// Deterministic reconstruction error summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DalucOracleErrorStats {
    pub samples: usize,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub rmse: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DalucOracleReconstructionReport {
    pub keys: DalucOracleErrorStats,
    pub values: DalucOracleErrorStats,
}

/// Fully owned host-oracle payload.
///
/// Dense input/output uses canonical `[batch, kv_heads, kv_len, feature]` order.
/// Physical planes follow the FDAL0 row order/topology contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DalucOraclePayload {
    payload_version: u16,
    contract: DalucKvViewContract,
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
    /// Encode with a deterministic identity logical-page -> physical-page map.
    pub fn encode(
        contract: DalucKvViewContract,
        codebook: &[f32],
        dense_keys: &[f32],
        dense_values: &[f32],
    ) -> Result<Self, DalucOracleError> {
        Self::encode_with_page_table(contract, codebook, dense_keys, dense_values, None)
    }

    /// Encode with an explicit per-batch page table when the contract is paged.
    ///
    /// Entries are physical page indices local to each batch. For contiguous
    /// topology `page_table` must be `None` or empty. The oracle serializes each
    /// entry as little-endian `u32` solely for deterministic host evidence; this
    /// is not a backend page-table ABI.
    pub fn encode_with_page_table(
        contract: DalucKvViewContract,
        codebook: &[f32],
        dense_keys: &[f32],
        dense_values: &[f32],
        page_table: Option<&[u32]>,
    ) -> Result<Self, DalucOracleError> {
        contract.validate()?;
        validate_finite_slice("K input", dense_keys)?;
        validate_finite_slice("V input", dense_values)?;
        validate_finite_slice("K codebook", codebook)?;

        let key_len = logical_side_len(contract, true)?;
        let value_len = logical_side_len(contract, false)?;
        require_len("K input", dense_keys.len(), key_len)?;
        require_len("V input", dense_values.len(), value_len)?;

        let geometry = OracleGeometry::new(contract, page_table)?;
        let stored_codebook = StoredCodebook::new(contract, codebook)?;
        let key_indices = encode_keys(contract, &geometry, &stored_codebook, dense_keys)?;
        let (key_residual_values, key_residual_indexing) = encode_residuals(
            contract,
            &geometry,
            dense_keys,
            &decode_primary_keys(contract, &geometry, &stored_codebook, &key_indices)?,
            true,
        )?;
        let encoded_values = encode_values(contract, &geometry, dense_values)?;
        let primary_values = decode_primary_values(contract, &geometry, &encoded_values)?;
        let (value_residual_values, value_residual_indexing) =
            encode_residuals(contract, &geometry, dense_values, &primary_values, false)?;

        let payload = Self {
            payload_version: DA_LUC_ORACLE_PAYLOAD_VERSION,
            contract,
            key_codebook: stored_codebook.plane,
            key_indices,
            key_residual_values,
            key_residual_indexing,
            values: encoded_values.values,
            value_scales: encoded_values.scales,
            value_zero_points: encoded_values.zero_points,
            value_residual_values,
            value_residual_indexing,
            page_table: geometry.page_table_plane,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub const fn payload_version(&self) -> u16 {
        self.payload_version
    }

    pub const fn contract(&self) -> DalucKvViewContract {
        self.contract
    }

    pub fn key_codebook_plane(&self) -> &DalucOraclePlane {
        &self.key_codebook
    }

    pub fn key_index_plane(&self) -> &DalucOraclePlane {
        &self.key_indices
    }

    pub fn page_table_plane(&self) -> &DalucOraclePlane {
        &self.page_table
    }

    /// Validate all plane lengths, padding, page mappings and live payload indices.
    pub fn validate(&self) -> Result<(), DalucOracleError> {
        self.contract.validate()?;
        if self.payload_version != DA_LUC_ORACLE_PAYLOAD_VERSION {
            return Err(DalucOracleError::UnsupportedPayloadVersion {
                actual: self.payload_version,
                supported: DA_LUC_ORACLE_PAYLOAD_VERSION,
            });
        }
        let geometry = OracleGeometry::from_payload(self.contract, &self.page_table)?;
        validate_plane_padding(self.contract, &self.key_codebook)?;
        validate_plane_padding(self.contract, &self.key_indices)?;
        validate_plane_padding(self.contract, &self.key_residual_values)?;
        validate_plane_padding(self.contract, &self.key_residual_indexing)?;
        validate_plane_padding(self.contract, &self.values)?;
        validate_plane_padding(self.contract, &self.value_scales)?;
        validate_plane_padding(self.contract, &self.value_zero_points)?;
        validate_plane_padding(self.contract, &self.value_residual_values)?;
        validate_plane_padding(self.contract, &self.value_residual_indexing)?;
        validate_plane_padding(self.contract, &self.page_table)?;

        validate_expected_plane_lengths(self, &geometry)?;
        validate_stored_codebook(self.contract, &self.key_codebook)?;
        validate_live_key_indices(self.contract, &geometry, &self.key_indices)?;
        validate_live_residuals(
            self.contract,
            &geometry,
            &self.key_residual_values,
            &self.key_residual_indexing,
            true,
        )?;
        validate_value_planes(self.contract, &geometry, self)?;
        validate_live_residuals(
            self.contract,
            &geometry,
            &self.value_residual_values,
            &self.value_residual_indexing,
            false,
        )
    }

    pub fn decode_keys(&self) -> Result<Vec<f32>, DalucOracleError> {
        self.validate()?;
        let geometry = OracleGeometry::from_payload(self.contract, &self.page_table)?;
        let codebook = StoredCodebook::from_plane(self.contract, self.key_codebook.clone())?;
        let mut decoded =
            decode_primary_keys(self.contract, &geometry, &codebook, &self.key_indices)?;
        apply_residuals(
            self.contract,
            &geometry,
            &self.key_residual_values,
            &self.key_residual_indexing,
            true,
            &mut decoded,
        )?;
        Ok(decoded)
    }

    pub fn decode_values(&self) -> Result<Vec<f32>, DalucOracleError> {
        self.validate()?;
        let geometry = OracleGeometry::from_payload(self.contract, &self.page_table)?;
        let encoded = EncodedValues {
            values: self.values.clone(),
            scales: self.value_scales.clone(),
            zero_points: self.value_zero_points.clone(),
        };
        let mut decoded = decode_primary_values(self.contract, &geometry, &encoded)?;
        apply_residuals(
            self.contract,
            &geometry,
            &self.value_residual_values,
            &self.value_residual_indexing,
            false,
            &mut decoded,
        )?;
        Ok(decoded)
    }

    pub fn reconstruction_report(
        &self,
        dense_keys: &[f32],
        dense_values: &[f32],
    ) -> Result<DalucOracleReconstructionReport, DalucOracleError> {
        let expected_keys = logical_side_len(self.contract, true)?;
        let expected_values = logical_side_len(self.contract, false)?;
        require_len("K input", dense_keys.len(), expected_keys)?;
        require_len("V input", dense_values.len(), expected_values)?;
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
                            if !scale.is_finite() || scale <= 0.0 {
                                return Err(DalucOracleError::ScaleUnderflow { row, group });
                            }
                            let zp =
                                read_zero_point(&encoded.zero_points, group_index, zero_point)?;
                            for inner in 0..group_size {
                                let feature = group * group_size + inner;
                                let packed_index = row * contract.shape.value_head_dim + feature;
                                let raw = unpack_integer(
                                    &encoded.values,
                                    packed_index,
                                    storage_bits,
                                    bit_order,
                                )?;
                                output[output_base + feature] =
                                    dequantize_value(raw, scale, zp, storage_bits, zero_point);
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
    dense: &[f32],
    primary: &[f32],
    key_side: bool,
) -> Result<(DalucOraclePlane, DalucOraclePlane), DalucOracleError> {
    let (dimension, semantics) = side_residual(contract, key_side);
    let DalucResidualSemantics::Sparse(residual) = semantics else {
        return Ok((DalucOraclePlane::empty(), DalucOraclePlane::empty()));
    };
    let k = residual.max_entries_per_vector;
    let value_count = checked_product(&[geometry.physical_rows, k], "residual values")?;
    let residual_value_bytes = value_count
        .checked_mul(dtype_bytes(residual.value_dtype))
        .ok_or(DalucOracleError::ArithmeticOverflow("residual value bytes"))?;
    let mut value_raw = vec![0u8; residual_value_bytes];
    let mut coordinate_values = match residual.indexing {
        DalucResidualIndexing::Coordinates { .. } => vec![0u32; value_count],
        DalucResidualIndexing::Bitmap { .. } => Vec::new(),
    };
    if !coordinate_values.is_empty() {
        for row in 0..geometry.physical_rows {
            for slot in 0..k {
                coordinate_values[row * k + slot] = u32::try_from(slot).map_err(|_| {
                    DalucOracleError::ArithmeticOverflow("residual default coordinate")
                })?;
            }
        }
    }
    let bitmap_bits = match residual.indexing {
        DalucResidualIndexing::Bitmap { .. } => {
            checked_product(&[geometry.physical_rows, dimension], "residual bitmap bits")?
        }
        DalucResidualIndexing::Coordinates { .. } => 0,
    };
    let mut bitmap_raw = vec![0u8; bytes_for_bits(bitmap_bits)?];

    for batch in 0..contract.shape.batch {
        for head in 0..contract.shape.kv_heads {
            for token in 0..contract.shape.kv_len {
                let row = geometry.physical_row(contract, batch, head, token)?;
                let base = canonical_vector_offset(contract, batch, head, token, key_side)?;
                let errors: Vec<f32> = (0..dimension)
                    .map(|feature| dense[base + feature] - primary[base + feature])
                    .collect();
                let selected = select_residual_coordinates(&errors, k);
                let mut storage_coordinates = selected;
                storage_coordinates.sort_unstable();
                for (slot, &coordinate) in storage_coordinates.iter().enumerate() {
                    write_float(
                        &mut value_raw,
                        row * k + slot,
                        errors[coordinate],
                        residual.value_dtype,
                    )?;
                    match residual.indexing {
                        DalucResidualIndexing::Coordinates { .. } => {
                            coordinate_values[row * k + slot] =
                                u32::try_from(coordinate).map_err(|_| {
                                    DalucOracleError::ArithmeticOverflow("residual coordinate u32")
                                })?;
                        }
                        DalucResidualIndexing::Bitmap { bit_order } => {
                            set_stream_bit(
                                &mut bitmap_raw,
                                row * dimension + coordinate,
                                bit_order,
                                true,
                            )?;
                        }
                    }
                }
            }
        }
    }

    let values = finalize_byte_plane(contract, value_raw)?;
    let indexing = match residual.indexing {
        DalucResidualIndexing::Coordinates {
            index_bits,
            bit_order,
        } => pack_integer_plane(contract, &coordinate_values, index_bits, bit_order)?,
        DalucResidualIndexing::Bitmap { .. } => {
            finalize_bit_plane(contract, bitmap_raw, bitmap_bits)?
        }
    };
    Ok((values, indexing))
}

fn apply_residuals(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    values: &DalucOraclePlane,
    indexing: &DalucOraclePlane,
    key_side: bool,
    dense: &mut [f32],
) -> Result<(), DalucOracleError> {
    let (dimension, semantics) = side_residual(contract, key_side);
    let DalucResidualSemantics::Sparse(residual) = semantics else {
        return Ok(());
    };
    let k = residual.max_entries_per_vector;
    for batch in 0..contract.shape.batch {
        for head in 0..contract.shape.kv_heads {
            for token in 0..contract.shape.kv_len {
                let row = geometry.physical_row(contract, batch, head, token)?;
                let base = canonical_vector_offset(contract, batch, head, token, key_side)?;
                match residual.indexing {
                    DalucResidualIndexing::Coordinates {
                        index_bits,
                        bit_order,
                    } => {
                        for slot in 0..k {
                            let coordinate = usize::try_from(unpack_integer(
                                indexing,
                                row * k + slot,
                                index_bits,
                                bit_order,
                            )?)
                            .map_err(|_| {
                                DalucOracleError::MalformedPayload("residual coordinate usize")
                            })?;
                            if coordinate >= dimension {
                                return Err(DalucOracleError::MalformedPayload(
                                    "residual coordinate out of range",
                                ));
                            }
                            let correction =
                                read_float(values, row * k + slot, residual.value_dtype)?;
                            dense[base + coordinate] += correction;
                        }
                    }
                    DalucResidualIndexing::Bitmap { bit_order } => {
                        let mut slot = 0usize;
                        for coordinate in 0..dimension {
                            if get_stream_bit(indexing, row * dimension + coordinate, bit_order)? {
                                if slot >= k {
                                    return Err(DalucOracleError::MalformedPayload(
                                        "residual bitmap exceeds budget",
                                    ));
                                }
                                let correction =
                                    read_float(values, row * k + slot, residual.value_dtype)?;
                                dense[base + coordinate] += correction;
                                slot += 1;
                            }
                        }
                        if slot != k {
                            return Err(DalucOracleError::MalformedPayload(
                                "residual bitmap does not fill fixed oracle budget",
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct QuantParams {
    scale: f32,
    zero_point: u32,
}

fn quantization_params(
    values: &[f32],
    bits: u8,
    scale_dtype: DalucFloatDType,
    zero_point: DalucZeroPointStorage,
) -> Result<QuantParams, DalucOracleError> {
    let qmax_unsigned = bit_mask(bits)?;
    let params = match zero_point {
        DalucZeroPointStorage::None => {
            let qmax = (1u32 << (bits - 1)) - 1;
            let max_abs = values
                .iter()
                .fold(0.0f32, |acc, value| acc.max(value.abs()));
            let raw_scale = if max_abs == 0.0 || qmax == 0 {
                1.0
            } else {
                max_abs / qmax as f32
            };
            QuantParams {
                scale: round_float(raw_scale, scale_dtype),
                zero_point: 0,
            }
        }
        DalucZeroPointStorage::U8 | DalucZeroPointStorage::U16 => {
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if min == max {
                if min == 0.0 {
                    QuantParams {
                        scale: round_float(1.0, scale_dtype),
                        zero_point: 0,
                    }
                } else {
                    let raw_scale = min.abs() / qmax_unsigned as f32;
                    QuantParams {
                        scale: round_float(raw_scale, scale_dtype),
                        zero_point: if min < 0.0 { qmax_unsigned } else { 0 },
                    }
                }
            } else {
                let raw_scale = (max - min) / qmax_unsigned as f32;
                let stored_scale = round_float(raw_scale, scale_dtype);
                let zp = (-min / stored_scale)
                    .round()
                    .clamp(0.0, qmax_unsigned as f32) as u32;
                QuantParams {
                    scale: stored_scale,
                    zero_point: zp,
                }
            }
        }
    };
    if !params.scale.is_finite() || params.scale <= 0.0 {
        return Err(DalucOracleError::MalformedPayload(
            "quantization scale underflow/non-finite",
        ));
    }
    Ok(params)
}

fn quantize_value(
    value: f32,
    params: QuantParams,
    bits: u8,
    zero_point: DalucZeroPointStorage,
) -> u32 {
    match zero_point {
        DalucZeroPointStorage::None => {
            let qmin = -(1i32 << (bits - 1));
            let qmax = (1i32 << (bits - 1)) - 1;
            let q = (value / params.scale)
                .round()
                .clamp(qmin as f32, qmax as f32) as i32;
            (q as u32) & bit_mask(bits).expect("validated bits")
        }
        DalucZeroPointStorage::U8 | DalucZeroPointStorage::U16 => {
            let qmax = bit_mask(bits).expect("validated bits");
            (value / params.scale + params.zero_point as f32)
                .round()
                .clamp(0.0, qmax as f32) as u32
        }
    }
}

fn dequantize_value(
    raw: u32,
    scale: f32,
    zero_point: u32,
    bits: u8,
    mode: DalucZeroPointStorage,
) -> f32 {
    match mode {
        DalucZeroPointStorage::None => sign_extend(raw, bits) as f32 * scale,
        DalucZeroPointStorage::U8 | DalucZeroPointStorage::U16 => {
            (raw as f32 - zero_point as f32) * scale
        }
    }
}

fn select_residual_coordinates(errors: &[f32], k: usize) -> Vec<usize> {
    let mut coordinates: Vec<usize> = (0..errors.len()).collect();
    coordinates.sort_by(|&a, &b| {
        errors[b]
            .abs()
            .total_cmp(&errors[a].abs())
            .then_with(|| a.cmp(&b))
    });
    coordinates.truncate(k);
    coordinates
}

fn validate_expected_plane_lengths(
    payload: &DalucOraclePayload,
    geometry: &OracleGeometry,
) -> Result<(), DalucOracleError> {
    let contract = payload.contract;
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    let expected_key_index_bits = checked_product(
        &[
            geometry.physical_rows,
            subspaces,
            usize::from(contract.keys.index_bits),
        ],
        "K index bits",
    )?;
    require_bits(
        "K index plane",
        payload.key_indices.logical_bits,
        expected_key_index_bits,
    )?;
    let codebook_bits = codebook_elements(contract)?
        .checked_mul(dtype_bytes(contract.keys.codebook_dtype) * 8)
        .ok_or(DalucOracleError::ArithmeticOverflow("codebook bits"))?;
    require_bits(
        "K codebook plane",
        payload.key_codebook.logical_bits,
        codebook_bits,
    )?;

    validate_residual_plane_lengths(
        contract,
        geometry,
        true,
        &payload.key_residual_values,
        &payload.key_residual_indexing,
    )?;
    validate_residual_plane_lengths(
        contract,
        geometry,
        false,
        &payload.value_residual_values,
        &payload.value_residual_indexing,
    )?;

    match contract.values {
        DalucValueRepresentation::Dense { dtype } => {
            let bits = checked_product(
                &[
                    geometry.physical_rows,
                    contract.shape.value_head_dim,
                    dtype_bytes(dtype) * 8,
                ],
                "dense V bits",
            )?;
            require_bits("V plane", payload.values.logical_bits, bits)?;
            require_bits("V scale plane", payload.value_scales.logical_bits, 0)?;
            require_bits(
                "V zero-point plane",
                payload.value_zero_points.logical_bits,
                0,
            )?;
        }
        DalucValueRepresentation::GroupwiseAffine {
            storage_bits,
            group_size,
            scale_dtype,
            zero_point,
            ..
        } => {
            let value_bits = checked_product(
                &[
                    geometry.physical_rows,
                    contract.shape.value_head_dim,
                    usize::from(storage_bits),
                ],
                "V packed bits",
            )?;
            let groups = contract.shape.value_head_dim / group_size;
            let scale_bits = checked_product(
                &[geometry.physical_rows, groups, dtype_bytes(scale_dtype) * 8],
                "V scale bits",
            )?;
            let zp_bits_per_group = match zero_point {
                DalucZeroPointStorage::None => 0,
                DalucZeroPointStorage::U8 => 8,
                DalucZeroPointStorage::U16 => 16,
            };
            let zp_bits = checked_product(
                &[geometry.physical_rows, groups, zp_bits_per_group],
                "V zero-point bits",
            )?;
            require_bits("V plane", payload.values.logical_bits, value_bits)?;
            require_bits(
                "V scale plane",
                payload.value_scales.logical_bits,
                scale_bits,
            )?;
            require_bits(
                "V zero-point plane",
                payload.value_zero_points.logical_bits,
                zp_bits,
            )?;
        }
    }
    let expected_page_bits = geometry
        .page_table
        .len()
        .checked_mul(32)
        .ok_or(DalucOracleError::ArithmeticOverflow("page metadata bits"))?;
    require_bits(
        "page table",
        payload.page_table.logical_bits,
        expected_page_bits,
    )
}

fn validate_residual_plane_lengths(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    key_side: bool,
    values: &DalucOraclePlane,
    indexing: &DalucOraclePlane,
) -> Result<(), DalucOracleError> {
    let (dimension, semantics) = side_residual(contract, key_side);
    let DalucResidualSemantics::Sparse(residual) = semantics else {
        require_bits("residual values", values.logical_bits, 0)?;
        return require_bits("residual indexing", indexing.logical_bits, 0);
    };
    let value_bits = checked_product(
        &[
            geometry.physical_rows,
            residual.max_entries_per_vector,
            dtype_bytes(residual.value_dtype) * 8,
        ],
        "residual value bits",
    )?;
    let index_bits = match residual.indexing {
        DalucResidualIndexing::Coordinates { index_bits, .. } => checked_product(
            &[
                geometry.physical_rows,
                residual.max_entries_per_vector,
                usize::from(index_bits),
            ],
            "residual coordinate bits",
        )?,
        DalucResidualIndexing::Bitmap { .. } => {
            checked_product(&[geometry.physical_rows, dimension], "residual bitmap bits")?
        }
    };
    require_bits("residual values", values.logical_bits, value_bits)?;
    require_bits("residual indexing", indexing.logical_bits, index_bits)
}

fn validate_live_key_indices(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    plane: &DalucOraclePlane,
) -> Result<(), DalucOracleError> {
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    for batch in 0..contract.shape.batch {
        for head in 0..contract.shape.kv_heads {
            for token in 0..contract.shape.kv_len {
                let row = geometry.physical_row(contract, batch, head, token)?;
                for subspace in 0..subspaces {
                    let index = unpack_integer(
                        plane,
                        row * subspaces + subspace,
                        contract.keys.index_bits,
                        contract.keys.index_bit_order,
                    )?;
                    contract.validate_codebook_index(index)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_value_planes(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    payload: &DalucOraclePayload,
) -> Result<(), DalucOracleError> {
    match contract.values {
        DalucValueRepresentation::Dense { dtype } => {
            for batch in 0..contract.shape.batch {
                for head in 0..contract.shape.kv_heads {
                    for token in 0..contract.shape.kv_len {
                        let row = geometry.physical_row(contract, batch, head, token)?;
                        for feature in 0..contract.shape.value_head_dim {
                            let value = read_float(
                                &payload.values,
                                row * contract.shape.value_head_dim + feature,
                                dtype,
                            )?;
                            if !value.is_finite() {
                                return Err(DalucOracleError::MalformedPayload(
                                    "non-finite dense V",
                                ));
                            }
                        }
                    }
                }
            }
        }
        DalucValueRepresentation::GroupwiseAffine {
            group_size,
            scale_dtype,
            zero_point,
            ..
        } => {
            let groups = contract.shape.value_head_dim / group_size;
            for batch in 0..contract.shape.batch {
                for head in 0..contract.shape.kv_heads {
                    for token in 0..contract.shape.kv_len {
                        let row = geometry.physical_row(contract, batch, head, token)?;
                        for group in 0..groups {
                            let group_index = row * groups + group;
                            let scale =
                                read_float(&payload.value_scales, group_index, scale_dtype)?;
                            if !scale.is_finite() || scale <= 0.0 {
                                return Err(DalucOracleError::ScaleUnderflow { row, group });
                            }
                            let _ = read_zero_point(
                                &payload.value_zero_points,
                                group_index,
                                zero_point,
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_live_residuals(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    values: &DalucOraclePlane,
    indexing: &DalucOraclePlane,
    key_side: bool,
) -> Result<(), DalucOracleError> {
    let (dimension, semantics) = side_residual(contract, key_side);
    let DalucResidualSemantics::Sparse(residual) = semantics else {
        return Ok(());
    };
    let k = residual.max_entries_per_vector;
    for batch in 0..contract.shape.batch {
        for head in 0..contract.shape.kv_heads {
            for token in 0..contract.shape.kv_len {
                let row = geometry.physical_row(contract, batch, head, token)?;
                let mut seen = vec![false; dimension];
                let coordinates = match residual.indexing {
                    DalucResidualIndexing::Coordinates {
                        index_bits,
                        bit_order,
                    } => {
                        let mut coordinates = Vec::with_capacity(k);
                        for slot in 0..k {
                            let coordinate = usize::try_from(unpack_integer(
                                indexing,
                                row * k + slot,
                                index_bits,
                                bit_order,
                            )?)
                            .map_err(|_| {
                                DalucOracleError::MalformedPayload("residual coordinate usize")
                            })?;
                            if coordinate >= dimension || seen[coordinate] {
                                return Err(DalucOracleError::MalformedPayload(
                                    "residual coordinates must be unique and in range",
                                ));
                            }
                            seen[coordinate] = true;
                            coordinates.push(coordinate);
                        }
                        coordinates
                    }
                    DalucResidualIndexing::Bitmap { bit_order } => {
                        let mut coordinates = Vec::with_capacity(k);
                        for coordinate in 0..dimension {
                            if get_stream_bit(indexing, row * dimension + coordinate, bit_order)? {
                                coordinates.push(coordinate);
                            }
                        }
                        if coordinates.len() != k {
                            return Err(DalucOracleError::MalformedPayload(
                                "residual bitmap must contain the fixed oracle budget",
                            ));
                        }
                        coordinates
                    }
                };
                for slot in 0..coordinates.len() {
                    let value = read_float(values, row * k + slot, residual.value_dtype)?;
                    if !value.is_finite() {
                        return Err(DalucOracleError::MalformedPayload(
                            "non-finite residual value",
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_stored_codebook(
    contract: DalucKvViewContract,
    plane: &DalucOraclePlane,
) -> Result<(), DalucOracleError> {
    let elements = codebook_elements(contract)?;
    let expected_bits = elements
        .checked_mul(dtype_bytes(contract.keys.codebook_dtype) * 8)
        .ok_or(DalucOracleError::ArithmeticOverflow(
            "codebook validation bits",
        ))?;
    require_bits("K codebook", plane.logical_bits, expected_bits)?;
    for index in 0..elements {
        if !read_float(plane, index, contract.keys.codebook_dtype)?.is_finite() {
            return Err(DalucOracleError::MalformedPayload(
                "non-finite stored K codebook value",
            ));
        }
    }
    Ok(())
}

fn validate_plane_padding(
    contract: DalucKvViewContract,
    plane: &DalucOraclePlane,
) -> Result<(), DalucOracleError> {
    let logical_bytes = plane.logical_bytes();
    if logical_bytes > plane.bytes.len() {
        return Err(DalucOracleError::MalformedPayload(
            "plane shorter than logical bits",
        ));
    }
    let expected_alignment = match contract.layout.padding {
        DalucPaddingRule::None => 0,
        DalucPaddingRule::ZeroFilledToAlignment => {
            if logical_bytes == 0 {
                0
            } else {
                align_up(logical_bytes, contract.layout.plane_alignment_bytes)? - logical_bytes
            }
        }
    };
    if plane.alignment_padding_bytes != expected_alignment
        || plane.bytes.len() != logical_bytes + expected_alignment
    {
        return Err(DalucOracleError::MalformedPayload(
            "alignment padding length mismatch",
        ));
    }
    if plane.bytes[logical_bytes..].iter().any(|byte| *byte != 0) {
        return Err(DalucOracleError::MalformedPayload(
            "alignment padding is not zero-filled",
        ));
    }
    Ok(())
}

fn page_table_plane(
    contract: DalucKvViewContract,
    table: &[u32],
) -> Result<DalucOraclePlane, DalucOracleError> {
    if table.is_empty() {
        return Ok(DalucOraclePlane::empty());
    }
    let capacity = table
        .len()
        .checked_mul(4)
        .ok_or(DalucOracleError::ArithmeticOverflow(
            "page table serialization",
        ))?;
    let mut raw = Vec::with_capacity(capacity);
    for &entry in table {
        raw.extend_from_slice(&entry.to_le_bytes());
    }
    finalize_byte_plane(contract, raw)
}

fn pack_integer_plane(
    contract: DalucKvViewContract,
    values: &[u32],
    width: u8,
    order: DalucBitOrder,
) -> Result<DalucOraclePlane, DalucOracleError> {
    let logical_bits = values
        .len()
        .checked_mul(usize::from(width))
        .ok_or(DalucOracleError::ArithmeticOverflow("packed integer bits"))?;
    let mut raw = vec![0u8; bytes_for_bits(logical_bits)?];
    let mask = bit_mask(width)?;
    for (index, &value) in values.iter().enumerate() {
        if value > mask {
            return Err(DalucOracleError::MalformedPayload(
                "integer exceeds packed width",
            ));
        }
        for bit in 0..width {
            let source_bit = match order {
                DalucBitOrder::Lsb0 => bit,
                DalucBitOrder::Msb0 => width - 1 - bit,
            };
            let set = (value >> source_bit) & 1 != 0;
            set_stream_bit(
                &mut raw,
                index * usize::from(width) + usize::from(bit),
                order,
                set,
            )?;
        }
    }
    finalize_bit_plane(contract, raw, logical_bits)
}

fn unpack_integer(
    plane: &DalucOraclePlane,
    index: usize,
    width: u8,
    order: DalucBitOrder,
) -> Result<u32, DalucOracleError> {
    let start =
        index
            .checked_mul(usize::from(width))
            .ok_or(DalucOracleError::ArithmeticOverflow(
                "packed integer offset",
            ))?;
    let end = start
        .checked_add(usize::from(width))
        .ok_or(DalucOracleError::ArithmeticOverflow("packed integer end"))?;
    if end > plane.logical_bits {
        return Err(DalucOracleError::MalformedPayload(
            "packed integer read out of range",
        ));
    }
    let mut value = 0u32;
    for bit in 0..width {
        if get_stream_bit(plane, start + usize::from(bit), order)? {
            let target = match order {
                DalucBitOrder::Lsb0 => bit,
                DalucBitOrder::Msb0 => width - 1 - bit,
            };
            value |= 1u32 << target;
        }
    }
    Ok(value)
}

fn set_stream_bit(
    bytes: &mut [u8],
    stream_bit: usize,
    order: DalucBitOrder,
    value: bool,
) -> Result<(), DalucOracleError> {
    let byte_index = stream_bit / 8;
    let within = stream_bit % 8;
    let Some(byte) = bytes.get_mut(byte_index) else {
        return Err(DalucOracleError::MalformedPayload("bit write out of range"));
    };
    let bit = match order {
        DalucBitOrder::Lsb0 => within,
        DalucBitOrder::Msb0 => 7 - within,
    };
    if value {
        *byte |= 1u8 << bit;
    } else {
        *byte &= !(1u8 << bit);
    }
    Ok(())
}

fn get_stream_bit(
    plane: &DalucOraclePlane,
    stream_bit: usize,
    order: DalucBitOrder,
) -> Result<bool, DalucOracleError> {
    if stream_bit >= plane.logical_bits {
        return Err(DalucOracleError::MalformedPayload("bit read out of range"));
    }
    let byte = plane.bytes[stream_bit / 8];
    let within = stream_bit % 8;
    let bit = match order {
        DalucBitOrder::Lsb0 => within,
        DalucBitOrder::Msb0 => 7 - within,
    };
    Ok((byte >> bit) & 1 != 0)
}

fn finalize_byte_plane(
    contract: DalucKvViewContract,
    raw: Vec<u8>,
) -> Result<DalucOraclePlane, DalucOracleError> {
    let logical_bits = raw
        .len()
        .checked_mul(8)
        .ok_or(DalucOracleError::ArithmeticOverflow("byte plane bits"))?;
    finalize_bit_plane(contract, raw, logical_bits)
}

fn finalize_bit_plane(
    contract: DalucKvViewContract,
    mut raw: Vec<u8>,
    logical_bits: usize,
) -> Result<DalucOraclePlane, DalucOracleError> {
    let logical_bytes = bytes_for_bits(logical_bits)?;
    require_len("raw plane bytes", raw.len(), logical_bytes)?;
    let padding = match contract.layout.padding {
        DalucPaddingRule::None => 0,
        DalucPaddingRule::ZeroFilledToAlignment => {
            if logical_bytes == 0 {
                0
            } else {
                align_up(logical_bytes, contract.layout.plane_alignment_bytes)? - logical_bytes
            }
        }
    };
    raw.resize(logical_bytes + padding, 0);
    Ok(DalucOraclePlane {
        bytes: raw,
        logical_bits,
        alignment_padding_bytes: padding,
    })
}

fn append_float(raw: &mut Vec<u8>, value: f32, dtype: DalucFloatDType) {
    match dtype {
        DalucFloatDType::F16 => {
            raw.extend_from_slice(&F16::from_f32(value).to_bits().to_le_bytes())
        }
        DalucFloatDType::Bf16 => raw.extend_from_slice(&f32_to_bf16(value).to_le_bytes()),
        DalucFloatDType::F32 => raw.extend_from_slice(&value.to_bits().to_le_bytes()),
    }
}

fn write_float(
    raw: &mut [u8],
    index: usize,
    value: f32,
    dtype: DalucFloatDType,
) -> Result<(), DalucOracleError> {
    let size = dtype_bytes(dtype);
    let offset = index
        .checked_mul(size)
        .ok_or(DalucOracleError::ArithmeticOverflow("float write offset"))?;
    let end = offset
        .checked_add(size)
        .ok_or(DalucOracleError::ArithmeticOverflow("float write end"))?;
    let Some(target) = raw.get_mut(offset..end) else {
        return Err(DalucOracleError::MalformedPayload(
            "float write out of range",
        ));
    };
    match dtype {
        DalucFloatDType::F16 => {
            target.copy_from_slice(&F16::from_f32(value).to_bits().to_le_bytes())
        }
        DalucFloatDType::Bf16 => target.copy_from_slice(&f32_to_bf16(value).to_le_bytes()),
        DalucFloatDType::F32 => target.copy_from_slice(&value.to_bits().to_le_bytes()),
    }
    Ok(())
}

fn read_float(
    plane: &DalucOraclePlane,
    index: usize,
    dtype: DalucFloatDType,
) -> Result<f32, DalucOracleError> {
    let size = dtype_bytes(dtype);
    let offset = index
        .checked_mul(size)
        .ok_or(DalucOracleError::ArithmeticOverflow("float read offset"))?;
    let end = offset
        .checked_add(size)
        .ok_or(DalucOracleError::ArithmeticOverflow("float read end"))?;
    if end > plane.logical_bytes() {
        return Err(DalucOracleError::MalformedPayload(
            "float read out of range",
        ));
    }
    let source = &plane.bytes[offset..end];
    Ok(match dtype {
        DalucFloatDType::F16 => F16::from_bits(u16::from_le_bytes([source[0], source[1]])).to_f32(),
        DalucFloatDType::Bf16 => bf16_to_f32(u16::from_le_bytes([source[0], source[1]])),
        DalucFloatDType::F32 => f32::from_bits(u32::from_le_bytes([
            source[0], source[1], source[2], source[3],
        ])),
    })
}

fn round_float(value: f32, dtype: DalucFloatDType) -> f32 {
    match dtype {
        DalucFloatDType::F16 => F16::from_f32(value).to_f32(),
        DalucFloatDType::Bf16 => bf16_to_f32(f32_to_bf16(value)),
        DalucFloatDType::F32 => value,
    }
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn write_zero_point(
    raw: &mut [u8],
    index: usize,
    zero_point: u32,
    mode: DalucZeroPointStorage,
) -> Result<(), DalucOracleError> {
    match mode {
        DalucZeroPointStorage::None => Ok(()),
        DalucZeroPointStorage::U8 => {
            let byte = u8::try_from(zero_point)
                .map_err(|_| DalucOracleError::MalformedPayload("u8 zero point overflow"))?;
            let Some(target) = raw.get_mut(index) else {
                return Err(DalucOracleError::MalformedPayload(
                    "u8 zero point write out of range",
                ));
            };
            *target = byte;
            Ok(())
        }
        DalucZeroPointStorage::U16 => {
            let value = u16::try_from(zero_point)
                .map_err(|_| DalucOracleError::MalformedPayload("u16 zero point overflow"))?;
            let offset = index
                .checked_mul(2)
                .ok_or(DalucOracleError::ArithmeticOverflow(
                    "u16 zero point offset",
                ))?;
            let Some(target) = raw.get_mut(offset..offset + 2) else {
                return Err(DalucOracleError::MalformedPayload(
                    "u16 zero point write out of range",
                ));
            };
            target.copy_from_slice(&value.to_le_bytes());
            Ok(())
        }
    }
}

fn read_zero_point(
    plane: &DalucOraclePlane,
    index: usize,
    mode: DalucZeroPointStorage,
) -> Result<u32, DalucOracleError> {
    match mode {
        DalucZeroPointStorage::None => Ok(0),
        DalucZeroPointStorage::U8 => plane.bytes.get(index).copied().map(u32::from).ok_or(
            DalucOracleError::MalformedPayload("u8 zero point read out of range"),
        ),
        DalucZeroPointStorage::U16 => {
            let offset = index
                .checked_mul(2)
                .ok_or(DalucOracleError::ArithmeticOverflow(
                    "u16 zero point read offset",
                ))?;
            if offset + 2 > plane.logical_bytes() {
                return Err(DalucOracleError::MalformedPayload(
                    "u16 zero point read out of range",
                ));
            }
            Ok(u32::from(u16::from_le_bytes([
                plane.bytes[offset],
                plane.bytes[offset + 1],
            ])))
        }
    }
}

fn side_residual(contract: DalucKvViewContract, key_side: bool) -> (usize, DalucResidualSemantics) {
    if key_side {
        (contract.shape.key_head_dim, contract.keys.residual)
    } else {
        match contract.values {
            DalucValueRepresentation::Dense { .. } => {
                (contract.shape.value_head_dim, DalucResidualSemantics::None)
            }
            DalucValueRepresentation::GroupwiseAffine { residual, .. } => {
                (contract.shape.value_head_dim, residual)
            }
        }
    }
}

fn canonical_vector_offset(
    contract: DalucKvViewContract,
    batch: usize,
    head: usize,
    token: usize,
    key_side: bool,
) -> Result<usize, DalucOracleError> {
    let dimension = if key_side {
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
        .ok_or(DalucOracleError::ArithmeticOverflow(
            "canonical vector offset",
        ))
}

fn logical_side_len(
    contract: DalucKvViewContract,
    key_side: bool,
) -> Result<usize, DalucOracleError> {
    checked_product(
        &[
            contract.shape.batch,
            contract.shape.kv_heads,
            contract.shape.kv_len,
            if key_side {
                contract.shape.key_head_dim
            } else {
                contract.shape.value_head_dim
            },
        ],
        "logical side length",
    )
}

fn logical_kv_scalar_count(contract: DalucKvViewContract) -> Result<usize, DalucOracleError> {
    let per_vector = contract
        .shape
        .key_head_dim
        .checked_add(contract.shape.value_head_dim)
        .ok_or(DalucOracleError::ArithmeticOverflow("K+V dimension"))?;
    checked_product(
        &[
            contract.shape.batch,
            contract.shape.kv_heads,
            contract.shape.kv_len,
            per_vector,
        ],
        "logical KV scalar count",
    )
}

fn codebook_elements(contract: DalucKvViewContract) -> Result<usize, DalucOracleError> {
    let scopes = match contract.keys.codebook_scope {
        DalucCodebookScope::SharedAcrossKvHeads => 1,
        DalucCodebookScope::PerKvHead => contract.shape.kv_heads,
    };
    checked_product(
        &[
            scopes,
            contract.shape.key_head_dim / contract.keys.subspace_dim,
            contract.keys.codebook_entries,
            contract.keys.subspace_dim,
        ],
        "codebook elements",
    )
}

fn dense_baseline_bytes(
    contract: DalucKvViewContract,
    geometry: &OracleGeometry,
    dtype: DalucFloatDType,
    external_metadata_bytes: usize,
) -> Result<usize, DalucOracleError> {
    let scalar_bytes = dtype_bytes(dtype);
    let key_raw = checked_product(
        &[
            geometry.physical_rows,
            contract.shape.key_head_dim,
            scalar_bytes,
        ],
        "dense baseline K bytes",
    )?;
    let value_raw = checked_product(
        &[
            geometry.physical_rows,
            contract.shape.value_head_dim,
            scalar_bytes,
        ],
        "dense baseline V bytes",
    )?;
    let key_plane = padded_plane_bytes(contract, key_raw)?;
    let value_plane = padded_plane_bytes(contract, value_raw)?;
    key_plane
        .checked_add(value_plane)
        .and_then(|v| v.checked_add(geometry.page_table_plane.total_bytes()))
        .and_then(|v| v.checked_add(external_metadata_bytes))
        .ok_or(DalucOracleError::ArithmeticOverflow(
            "dense baseline total bytes",
        ))
}

fn padded_plane_bytes(
    contract: DalucKvViewContract,
    logical_bytes: usize,
) -> Result<usize, DalucOracleError> {
    match contract.layout.padding {
        DalucPaddingRule::None => Ok(logical_bytes),
        DalucPaddingRule::ZeroFilledToAlignment => {
            if logical_bytes == 0 {
                Ok(0)
            } else {
                align_up(logical_bytes, contract.layout.plane_alignment_bytes)
            }
        }
    }
}

fn error_stats(expected: &[f32], actual: &[f32]) -> DalucOracleErrorStats {
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for (&expected, &actual) in expected.iter().zip(actual) {
        let delta = f64::from(expected) - f64::from(actual);
        let abs = delta.abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        sum_sq += delta * delta;
    }
    let samples = expected.len();
    DalucOracleErrorStats {
        samples,
        max_abs,
        mean_abs: sum_abs / samples as f64,
        rmse: (sum_sq / samples as f64).sqrt(),
    }
}

fn validate_page_table(
    table: &[u32],
    batch: usize,
    logical_pages: usize,
    physical_pages: usize,
) -> Result<(), DalucOracleError> {
    let expected = checked_product(&[batch, logical_pages], "page table validation entries")?;
    require_len("page table", table.len(), expected)?;
    for batch_index in 0..batch {
        let start = batch_index * logical_pages;
        let slice = &table[start..start + logical_pages];
        let mut seen = vec![false; physical_pages];
        for &page in slice {
            let page = usize::try_from(page)
                .map_err(|_| DalucOracleError::InvalidPageTable("page does not fit usize"))?;
            if page >= physical_pages {
                return Err(DalucOracleError::InvalidPageTable(
                    "physical page out of range",
                ));
            }
            if seen[page] {
                return Err(DalucOracleError::InvalidPageTable(
                    "logical pages alias one physical page",
                ));
            }
            seen[page] = true;
        }
    }
    Ok(())
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

fn require_len(what: &'static str, actual: usize, expected: usize) -> Result<(), DalucOracleError> {
    if actual != expected {
        return Err(DalucOracleError::LengthMismatch {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

fn require_bits(
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
    values
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or(DalucOracleError::ArithmeticOverflow(label))
}

fn div_ceil(value: usize, divisor: usize) -> Result<usize, DalucOracleError> {
    value
        .checked_add(divisor - 1)
        .map(|v| v / divisor)
        .ok_or(DalucOracleError::ArithmeticOverflow("ceil division"))
}

fn bytes_for_bits(bits: usize) -> Result<usize, DalucOracleError> {
    if bits == 0 {
        return Ok(0);
    }
    bits.checked_add(7)
        .map(|v| v / 8)
        .ok_or(DalucOracleError::ArithmeticOverflow("bits to bytes"))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, DalucOracleError> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or(DalucOracleError::ArithmeticOverflow("alignment"))
}

const fn dtype_bytes(dtype: DalucFloatDType) -> usize {
    match dtype {
        DalucFloatDType::F16 | DalucFloatDType::Bf16 => 2,
        DalucFloatDType::F32 => 4,
    }
}

fn bit_mask(bits: u8) -> Result<u32, DalucOracleError> {
    match bits {
        1..=31 => Ok((1u32 << bits) - 1),
        32 => Ok(u32::MAX),
        _ => Err(DalucOracleError::MalformedPayload(
            "packed width outside 1..=32",
        )),
    }
}

fn sign_extend(value: u32, bits: u8) -> i32 {
    if bits == 32 {
        return value as i32;
    }
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}
