//! FDAL2 direct q_len=1 attention oracle over the FDAL1 compressed payload.
//!
//! The direct path reads codebook indices, sparse residuals, low-bit V values,
//! scales and zero points from the validated oracle planes. It never calls the
//! full dense K/V reconstruction helpers. Scalar conversion/dequantization still
//! occurs, so this module does not claim "zero dequantization" or performance.

use super::*;
use crate::{FlatAttentionConfig, FlatAttentionError};
use core::fmt;

/// Version of the research-only compressed q_len=1 attention oracle semantics.
pub const DA_LUC_COMPRESSED_DECODE_ORACLE_VERSION: u16 = 1;

/// q_len=1 attention configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DalucQlen1DecodeConfig {
    /// Causal policy and score scaling shared with FLAT's dense scalar oracle.
    pub attention: FlatAttentionConfig,
    /// Absolute position of the single query token.
    pub query_position: usize,
}

impl DalucQlen1DecodeConfig {
    /// Canonical autoregressive decode configuration for the last live KV token.
    pub fn for_last_token(
        contract: DalucKvViewContract,
        attention: FlatAttentionConfig,
    ) -> Result<Self, DalucDecodeOracleError> {
        contract.validate().map_err(DalucOracleError::from)?;
        Ok(Self {
            attention,
            query_position: contract.shape.kv_len - 1,
        })
    }
}

/// Structural execution evidence from the scalar compressed-consumption oracle.
///
/// These are deterministic logical operation counts. They are not physical
/// memory transactions, bandwidth, latency, throughput, or a performance claim.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DalucQlen1DecodeTrace {
    pub lut_entry_dot_products: usize,
    pub key_index_lookups: usize,
    pub key_residual_corrections: usize,
    pub attended_kv_rows: usize,
    pub value_primary_scalar_reads: usize,
    pub value_quantized_scalar_conversions: usize,
    pub value_residual_corrections: usize,
}

/// q_len=1 attention result in canonical `[batch, q_heads, value_head_dim]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct DalucQlen1DecodeOutput {
    pub output: Vec<f32>,
    /// `[batch, q_heads]` log-sum-exp values.
    pub lse: Vec<f32>,
    pub trace: DalucQlen1DecodeTrace,
}

/// Numerical difference between the direct compressed oracle and the explicitly
/// reconstructed dense mathematical reference over the same FDAL1 payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DalucQlen1EquivalenceReport {
    pub output: DalucOracleErrorStats,
    pub lse: DalucOracleErrorStats,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum DalucDecodeOracleError {
    Oracle(DalucOracleError),
    Attention(FlatAttentionError),
}

impl fmt::Display for DalucDecodeOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Oracle(error) => write!(f, "{error}"),
            Self::Attention(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for DalucDecodeOracleError {}

impl From<DalucOracleError> for DalucDecodeOracleError {
    fn from(value: DalucOracleError) -> Self {
        Self::Oracle(value)
    }
}

impl From<FlatAttentionError> for DalucDecodeOracleError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Attention(value)
    }
}

impl DalucOraclePayload {
    /// Consume the compressed payload directly for q_len=1 attention.
    ///
    /// K scoring uses a query-to-codebook LUT plus sparse K residual correction.
    /// V accumulation reads dense or groupwise-affine scalars directly from the
    /// payload and applies sparse V residuals without constructing a dense K/V
    /// cache. A small sparse residual list and one query-local LUT are temporary
    /// scalar-oracle scratch state; neither is a dense KV materialization.
    pub fn q_len1_attention_direct(
        &self,
        query: &[f32],
        config: DalucQlen1DecodeConfig,
    ) -> Result<DalucQlen1DecodeOutput, DalucDecodeOracleError> {
        self.validate()?;
        validate_query(self.contract, query)?;
        let scale = config
            .attention
            .resolved_scale(self.contract.shape.key_head_dim)?;
        let geometry = OracleGeometry::from_payload(self.contract, &self.page_table)?;
        let codebook = StoredCodebook::from_plane(self.contract, self.key_codebook.clone())?;
        let group_size = self.contract.shape.q_heads / self.contract.shape.kv_heads;
        let output_len = checked_product(
            &[
                self.contract.shape.batch,
                self.contract.shape.q_heads,
                self.contract.shape.value_head_dim,
            ],
            "FDAL2 output length",
        )?;
        let lse_len = checked_product(
            &[self.contract.shape.batch, self.contract.shape.q_heads],
            "FDAL2 LSE length",
        )?;
        let mut output = vec![0.0f32; output_len];
        let mut lse = vec![0.0f32; lse_len];
        let mut trace = DalucQlen1DecodeTrace::default();

        for batch in 0..self.contract.shape.batch {
            for q_head in 0..self.contract.shape.q_heads {
                let kv_head = q_head / group_size;
                let q_base = query_offset(self.contract, batch, q_head)?;
                let out_base = output_offset(self.contract, batch, q_head)?;
                let lut = build_query_codebook_lut(
                    self.contract,
                    &codebook,
                    query,
                    q_base,
                    kv_head,
                    &mut trace,
                )?;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;

                for key_pos in 0..self.contract.shape.kv_len {
                    if config.attention.causal && key_pos > config.query_position {
                        break;
                    }
                    bump(&mut trace.attended_kv_rows, 1, "attended KV rows")?;
                    let row = geometry.physical_row(self.contract, batch, kv_head, key_pos)?;
                    let mut dot = direct_key_primary_score(self, row, &lut, &mut trace)?;
                    let key_residuals = sparse_residual_entries(self, row, true)?;
                    for &(coordinate, correction) in &key_residuals {
                        dot += query[q_base + coordinate] * correction;
                        bump(
                            &mut trace.key_residual_corrections,
                            1,
                            "K residual corrections",
                        )?;
                    }

                    let score = dot * scale;
                    let new_max = running_max.max(score);
                    let alpha = if running_max.is_infinite() {
                        0.0
                    } else {
                        (running_max - new_max).exp()
                    };
                    let probability_numerator = (score - new_max).exp();
                    let value_residuals = sparse_residual_entries(self, row, false)?;

                    for feature in 0..self.contract.shape.value_head_dim {
                        let mut value = direct_primary_value(self, row, feature, &mut trace)?;
                        if let Some((_, correction)) = value_residuals
                            .iter()
                            .find(|(coordinate, _)| *coordinate == feature)
                        {
                            value += *correction;
                            bump(
                                &mut trace.value_residual_corrections,
                                1,
                                "V residual corrections",
                            )?;
                        }
                        let target = out_base + feature;
                        output[target] = output[target] * alpha + probability_numerator * value;
                    }
                    running_sum = running_sum * alpha + probability_numerator;
                    running_max = new_max;
                }

                let inv_sum = running_sum.recip();
                for feature in 0..self.contract.shape.value_head_dim {
                    output[out_base + feature] *= inv_sum;
                }
                lse[batch * self.contract.shape.q_heads + q_head] = running_max + running_sum.ln();
            }
        }

        Ok(DalucQlen1DecodeOutput { output, lse, trace })
    }

    /// Dense mathematical comparator for FDAL2.
    ///
    /// This method intentionally reconstructs the FDAL1 K/V payload first and is
    /// therefore the semantic reference, not the direct-compressed candidate.
    pub fn q_len1_attention_dense_reference(
        &self,
        query: &[f32],
        config: DalucQlen1DecodeConfig,
    ) -> Result<DalucQlen1DecodeOutput, DalucDecodeOracleError> {
        self.validate()?;
        validate_query(self.contract, query)?;
        let scale = config
            .attention
            .resolved_scale(self.contract.shape.key_head_dim)?;
        let keys = self.decode_keys()?;
        let values = self.decode_values()?;
        let group_size = self.contract.shape.q_heads / self.contract.shape.kv_heads;
        let output_len = checked_product(
            &[
                self.contract.shape.batch,
                self.contract.shape.q_heads,
                self.contract.shape.value_head_dim,
            ],
            "FDAL2 dense output length",
        )?;
        let lse_len = checked_product(
            &[self.contract.shape.batch, self.contract.shape.q_heads],
            "FDAL2 dense LSE length",
        )?;
        let mut output = vec![0.0f32; output_len];
        let mut lse = vec![0.0f32; lse_len];

        for batch in 0..self.contract.shape.batch {
            for q_head in 0..self.contract.shape.q_heads {
                let kv_head = q_head / group_size;
                let q_base = query_offset(self.contract, batch, q_head)?;
                let out_base = output_offset(self.contract, batch, q_head)?;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;

                for key_pos in 0..self.contract.shape.kv_len {
                    if config.attention.causal && key_pos > config.query_position {
                        break;
                    }
                    let key_base =
                        canonical_vector_offset(self.contract, batch, kv_head, key_pos, true)?;
                    let value_base =
                        canonical_vector_offset(self.contract, batch, kv_head, key_pos, false)?;
                    let mut dot = 0.0f32;
                    for feature in 0..self.contract.shape.key_head_dim {
                        dot += query[q_base + feature] * keys[key_base + feature];
                    }
                    let score = dot * scale;
                    let new_max = running_max.max(score);
                    let alpha = if running_max.is_infinite() {
                        0.0
                    } else {
                        (running_max - new_max).exp()
                    };
                    let probability_numerator = (score - new_max).exp();
                    for feature in 0..self.contract.shape.value_head_dim {
                        let target = out_base + feature;
                        output[target] = output[target] * alpha
                            + probability_numerator * values[value_base + feature];
                    }
                    running_sum = running_sum * alpha + probability_numerator;
                    running_max = new_max;
                }

                let inv_sum = running_sum.recip();
                for feature in 0..self.contract.shape.value_head_dim {
                    output[out_base + feature] *= inv_sum;
                }
                lse[batch * self.contract.shape.q_heads + q_head] = running_max + running_sum.ln();
            }
        }

        Ok(DalucQlen1DecodeOutput {
            output,
            lse,
            trace: DalucQlen1DecodeTrace::default(),
        })
    }

    /// Compare direct compressed consumption with the reconstructed dense
    /// mathematical reference. Differences reflect floating accumulation order,
    /// not representation error: both paths consume the same encoded payload.
    pub fn q_len1_attention_equivalence_report(
        &self,
        query: &[f32],
        config: DalucQlen1DecodeConfig,
    ) -> Result<DalucQlen1EquivalenceReport, DalucDecodeOracleError> {
        let direct = self.q_len1_attention_direct(query, config)?;
        let dense = self.q_len1_attention_dense_reference(query, config)?;
        Ok(DalucQlen1EquivalenceReport {
            output: error_stats(&dense.output, &direct.output),
            lse: error_stats(&dense.lse, &direct.lse),
        })
    }
}

fn validate_query(contract: DalucKvViewContract, query: &[f32]) -> Result<(), DalucOracleError> {
    let expected = checked_product(
        &[
            contract.shape.batch,
            contract.shape.q_heads,
            contract.shape.key_head_dim,
        ],
        "FDAL2 query length",
    )?;
    require_len("FDAL2 query", query.len(), expected)?;
    validate_finite_slice("FDAL2 query", query)
}

fn query_offset(
    contract: DalucKvViewContract,
    batch: usize,
    q_head: usize,
) -> Result<usize, DalucOracleError> {
    batch
        .checked_mul(contract.shape.q_heads)
        .and_then(|value| value.checked_add(q_head))
        .and_then(|value| value.checked_mul(contract.shape.key_head_dim))
        .ok_or(DalucOracleError::ArithmeticOverflow("FDAL2 query offset"))
}

fn output_offset(
    contract: DalucKvViewContract,
    batch: usize,
    q_head: usize,
) -> Result<usize, DalucOracleError> {
    batch
        .checked_mul(contract.shape.q_heads)
        .and_then(|value| value.checked_add(q_head))
        .and_then(|value| value.checked_mul(contract.shape.value_head_dim))
        .ok_or(DalucOracleError::ArithmeticOverflow("FDAL2 output offset"))
}

fn build_query_codebook_lut(
    contract: DalucKvViewContract,
    codebook: &StoredCodebook,
    query: &[f32],
    q_base: usize,
    kv_head: usize,
    trace: &mut DalucQlen1DecodeTrace,
) -> Result<Vec<f32>, DalucOracleError> {
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    let entries = contract.keys.codebook_entries;
    let lut_len = checked_product(&[subspaces, entries], "FDAL2 LUT length")?;
    let mut lut = vec![0.0f32; lut_len];
    for subspace in 0..subspaces {
        let query_start = q_base + subspace * contract.keys.subspace_dim;
        for entry in 0..entries {
            let codebook_start = codebook.vector_offset(contract, kv_head, subspace, entry)?;
            let mut dot = 0.0f32;
            for inner in 0..contract.keys.subspace_dim {
                dot += query[query_start + inner] * codebook.decoded[codebook_start + inner];
            }
            lut[subspace * entries + entry] = dot;
            bump(
                &mut trace.lut_entry_dot_products,
                1,
                "LUT entry dot products",
            )?;
        }
    }
    Ok(lut)
}

fn direct_key_primary_score(
    payload: &DalucOraclePayload,
    row: usize,
    lut: &[f32],
    trace: &mut DalucQlen1DecodeTrace,
) -> Result<f32, DalucOracleError> {
    let contract = payload.contract;
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    let entries = contract.keys.codebook_entries;
    let mut dot = 0.0f32;
    for subspace in 0..subspaces {
        let packed_index = row
            .checked_mul(subspaces)
            .and_then(|value| value.checked_add(subspace))
            .ok_or(DalucOracleError::ArithmeticOverflow("FDAL2 K packed index"))?;
        let entry = usize::try_from(unpack_integer(
            &payload.key_indices,
            packed_index,
            contract.keys.index_bits,
            contract.keys.index_bit_order,
        )?)
        .map_err(|_| DalucOracleError::MalformedPayload("FDAL2 K index usize"))?;
        if entry >= entries {
            return Err(DalucOracleError::MalformedPayload(
                "FDAL2 K codebook index out of range",
            ));
        }
        dot += lut[subspace * entries + entry];
        bump(&mut trace.key_index_lookups, 1, "K index lookups")?;
    }
    Ok(dot)
}

fn direct_primary_value(
    payload: &DalucOraclePayload,
    row: usize,
    feature: usize,
    trace: &mut DalucQlen1DecodeTrace,
) -> Result<f32, DalucOracleError> {
    bump(
        &mut trace.value_primary_scalar_reads,
        1,
        "V primary scalar reads",
    )?;
    match payload.contract.values {
        DalucValueRepresentation::Dense { dtype } => {
            let index = row
                .checked_mul(payload.contract.shape.value_head_dim)
                .and_then(|value| value.checked_add(feature))
                .ok_or(DalucOracleError::ArithmeticOverflow("FDAL2 dense V index"))?;
            read_float(&payload.values, index, dtype)
        }
        DalucValueRepresentation::GroupwiseAffine {
            storage_bits,
            group_size,
            scale_dtype,
            zero_point,
            bit_order,
            ..
        } => {
            let groups = payload.contract.shape.value_head_dim / group_size;
            let group = feature / group_size;
            let group_index = row
                .checked_mul(groups)
                .and_then(|value| value.checked_add(group))
                .ok_or(DalucOracleError::ArithmeticOverflow("FDAL2 V group index"))?;
            let scale = read_float(&payload.value_scales, group_index, scale_dtype)?;
            if !scale.is_finite() || scale <= 0.0 {
                return Err(DalucOracleError::ScaleUnderflow { row, group });
            }
            let zp = read_zero_point(&payload.value_zero_points, group_index, zero_point)?;
            let packed_index = row
                .checked_mul(payload.contract.shape.value_head_dim)
                .and_then(|value| value.checked_add(feature))
                .ok_or(DalucOracleError::ArithmeticOverflow("FDAL2 packed V index"))?;
            let raw = unpack_integer(&payload.values, packed_index, storage_bits, bit_order)?;
            bump(
                &mut trace.value_quantized_scalar_conversions,
                1,
                "V quantized scalar conversions",
            )?;
            Ok(dequantize_value(raw, scale, zp, storage_bits, zero_point))
        }
    }
}

fn sparse_residual_entries(
    payload: &DalucOraclePayload,
    row: usize,
    key_side: bool,
) -> Result<Vec<(usize, f32)>, DalucOracleError> {
    let (dimension, semantics) = side_residual(payload.contract, key_side);
    let DalucResidualSemantics::Sparse(residual) = semantics else {
        return Ok(Vec::new());
    };
    let (values, indexing) = if key_side {
        (&payload.key_residual_values, &payload.key_residual_indexing)
    } else {
        (
            &payload.value_residual_values,
            &payload.value_residual_indexing,
        )
    };
    let k = residual.max_entries_per_vector;
    let mut entries = Vec::with_capacity(k);
    match residual.indexing {
        DalucResidualIndexing::Coordinates {
            index_bits,
            bit_order,
        } => {
            for slot in 0..k {
                let packed_index = row
                    .checked_mul(k)
                    .and_then(|value| value.checked_add(slot))
                    .ok_or(DalucOracleError::ArithmeticOverflow(
                        "FDAL2 residual coordinate index",
                    ))?;
                let coordinate = usize::try_from(unpack_integer(
                    indexing,
                    packed_index,
                    index_bits,
                    bit_order,
                )?)
                .map_err(|_| {
                    DalucOracleError::MalformedPayload("FDAL2 residual coordinate usize")
                })?;
                if coordinate >= dimension {
                    return Err(DalucOracleError::MalformedPayload(
                        "FDAL2 residual coordinate out of range",
                    ));
                }
                let correction = read_float(values, packed_index, residual.value_dtype)?;
                entries.push((coordinate, correction));
            }
        }
        DalucResidualIndexing::Bitmap { bit_order } => {
            let bit_base =
                row.checked_mul(dimension)
                    .ok_or(DalucOracleError::ArithmeticOverflow(
                        "FDAL2 residual bitmap base",
                    ))?;
            let value_base = row
                .checked_mul(k)
                .ok_or(DalucOracleError::ArithmeticOverflow(
                    "FDAL2 residual value base",
                ))?;
            let mut slot = 0usize;
            for coordinate in 0..dimension {
                if get_stream_bit(indexing, bit_base + coordinate, bit_order)? {
                    if slot >= k {
                        return Err(DalucOracleError::MalformedPayload(
                            "FDAL2 residual bitmap exceeds budget",
                        ));
                    }
                    let correction = read_float(values, value_base + slot, residual.value_dtype)?;
                    entries.push((coordinate, correction));
                    slot += 1;
                }
            }
            if slot != k {
                return Err(DalucOracleError::MalformedPayload(
                    "FDAL2 residual bitmap does not fill fixed oracle budget",
                ));
            }
        }
    }
    Ok(entries)
}

fn bump(counter: &mut usize, amount: usize, label: &'static str) -> Result<(), DalucOracleError> {
    *counter = counter
        .checked_add(amount)
        .ok_or(DalucOracleError::ArithmeticOverflow(label))?;
    Ok(())
}
