//! FLAT-ATTENTION: Rust-native, IO-aware fused attention for SciRust.
//!
//! The project keeps a deterministic scalar oracle and portable fused WGSL
//! kernels under one explicit contract. Optimized generations are qualified
//! against the oracle before becoming the default.

#![forbid(unsafe_code)]

use core::fmt;

/// Maximum head dimension supported by the portable WGSL kernels.
pub const WGSL_MAX_HEAD_DIM: usize = 128;
/// Number of invocations in one WGSL workgroup.
pub const WGSL_WORKGROUP_SIZE: usize = 64;
/// Number of K/V rows staged in workgroup memory at once.
pub const WGSL_KV_TILE: usize = 8;
/// Number of query rows sharing each K/V tile in the M4 default kernel.
pub const WGSL_QUERY_ROWS: usize = 4;

/// M4 portable fused forward kernel: four query rows per workgroup.
pub const FLAT_FWD_WGSL: &str = include_str!("../shaders/flat_fwd.wgsl");
/// Qualified M2/M3 one-query-row kernel retained as a baseline source.
pub const FLAT_FWD_SINGLE_WGSL: &str = include_str!("../shaders/flat_fwd_single.wgsl");

#[cfg(feature = "wgpu")]
mod wgpu_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_backend::{
    WgpuFlatAttention, WgpuFlatAttentionError, WgpuResidentAttentionOutput, WgpuResidentBuffer,
};

/// Contiguous tensor shape used by the current MHA contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionShape {
    pub batch: usize,
    pub heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
}

impl AttentionShape {
    /// Number of scalar elements in Q, K, V, and O.
    pub fn tensor_len(self) -> Result<usize, FlatAttentionError> {
        self.batch
            .checked_mul(self.heads)
            .and_then(|n| n.checked_mul(self.seq_len))
            .and_then(|n| n.checked_mul(self.head_dim))
            .ok_or(FlatAttentionError::ShapeOverflow)
    }

    /// Number of log-sum-exp statistics produced by forward.
    pub fn lse_len(self) -> Result<usize, FlatAttentionError> {
        self.batch
            .checked_mul(self.heads)
            .and_then(|n| n.checked_mul(self.seq_len))
            .ok_or(FlatAttentionError::ShapeOverflow)
    }

    fn validate(self) -> Result<(), FlatAttentionError> {
        if self.batch == 0 || self.heads == 0 || self.seq_len == 0 || self.head_dim == 0 {
            return Err(FlatAttentionError::ZeroDimension);
        }
        self.tensor_len()?;
        self.lse_len()?;
        Ok(())
    }
}

/// Forward configuration shared by reference and GPU kernels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FlatAttentionConfig {
    /// Apply autoregressive masking (`key_position > query_position`).
    pub causal: bool,
    /// Optional score multiplier. Defaults to `1 / sqrt(head_dim)`.
    pub softmax_scale: Option<f32>,
}

impl FlatAttentionConfig {
    pub fn resolved_scale(self, head_dim: usize) -> Result<f32, FlatAttentionError> {
        if head_dim == 0 {
            return Err(FlatAttentionError::ZeroDimension);
        }
        let scale = self
            .softmax_scale
            .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
        if !scale.is_finite() || scale <= 0.0 {
            return Err(FlatAttentionError::InvalidScale(scale));
        }
        Ok(scale)
    }
}

/// Static memory-traffic model for a fused kernel generation.
///
/// `kv_storage_scalar_loads` counts logical scalar loads of K plus V from
/// storage into workgroup memory according to the kernel's explicit staging
/// loops. It is an architectural count, not a claim about physical DRAM
/// transactions, cache hits, bandwidth, or runtime speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoModel {
    pub query_workgroups: usize,
    pub kv_storage_scalar_loads: usize,
}

/// Analytical IO model for the qualified single-row baseline.
pub fn single_row_io_model(
    shape: AttentionShape,
    _causal: bool,
) -> Result<IoModel, FlatAttentionError> {
    shape.validate()?;
    let batch_heads = shape
        .batch
        .checked_mul(shape.heads)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let query_workgroups = batch_heads
        .checked_mul(shape.seq_len)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let loads_per_workgroup = 2usize
        .checked_mul(shape.seq_len)
        .and_then(|n| n.checked_mul(shape.head_dim))
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    Ok(IoModel {
        query_workgroups,
        kv_storage_scalar_loads: query_workgroups
            .checked_mul(loads_per_workgroup)
            .ok_or(FlatAttentionError::ShapeOverflow)?,
    })
}

/// Analytical IO model for the M4 four-query-row tiled kernel.
pub fn tiled_q4_io_model(
    shape: AttentionShape,
    causal: bool,
) -> Result<IoModel, FlatAttentionError> {
    shape.validate()?;
    let batch_heads = shape
        .batch
        .checked_mul(shape.heads)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let query_tiles_per_head = shape.seq_len.div_ceil(WGSL_QUERY_ROWS);
    let query_workgroups = batch_heads
        .checked_mul(query_tiles_per_head)
        .ok_or(FlatAttentionError::ShapeOverflow)?;

    let mut kv_rows_per_head = 0usize;
    for tile in 0..query_tiles_per_head {
        let query_start = tile
            .checked_mul(WGSL_QUERY_ROWS)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let staged_rows = if causal {
            query_start
                .checked_add(WGSL_QUERY_ROWS)
                .ok_or(FlatAttentionError::ShapeOverflow)?
                .min(shape.seq_len)
        } else {
            shape.seq_len
        };
        kv_rows_per_head = kv_rows_per_head
            .checked_add(staged_rows)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
    }

    let kv_storage_scalar_loads = batch_heads
        .checked_mul(kv_rows_per_head)
        .and_then(|n| n.checked_mul(shape.head_dim))
        .and_then(|n| n.checked_mul(2))
        .ok_or(FlatAttentionError::ShapeOverflow)?;

    Ok(IoModel {
        query_workgroups,
        kv_storage_scalar_loads,
    })
}

/// Result of a forward attention pass.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatAttentionOutput {
    /// Context tensor with the same shape as Q.
    pub output: Vec<f32>,
    /// Per-query `log(sum(exp(scores)))`, shape `[batch, heads, seq_len]`.
    pub lse: Vec<f32>,
}

/// Errors are explicit: FLAT-ATTENTION never fabricates a fallback result.
#[derive(Debug, Clone, PartialEq)]
pub enum FlatAttentionError {
    ZeroDimension,
    ShapeOverflow,
    LengthMismatch {
        tensor: &'static str,
        actual: usize,
        expected: usize,
    },
    InvalidScale(f32),
    NonFiniteInput {
        tensor: &'static str,
        index: usize,
    },
}

impl fmt::Display for FlatAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "attention dimensions must all be non-zero"),
            Self::ShapeOverflow => write!(f, "attention shape overflows the address space"),
            Self::LengthMismatch {
                tensor,
                actual,
                expected,
            } => write!(
                f,
                "tensor {tensor} contains {actual} elements, expected {expected}"
            ),
            Self::InvalidScale(scale) => {
                write!(f, "softmax scale must be finite and positive, got {scale}")
            }
            Self::NonFiniteInput { tensor, index } => {
                write!(
                    f,
                    "tensor {tensor} contains a non-finite value at index {index}"
                )
            }
        }
    }
}

impl std::error::Error for FlatAttentionError {}

fn validate_input(
    name: &'static str,
    data: &[f32],
    expected: usize,
) -> Result<(), FlatAttentionError> {
    if data.len() != expected {
        return Err(FlatAttentionError::LengthMismatch {
            tensor: name,
            actual: data.len(),
            expected,
        });
    }
    if let Some(index) = data.iter().position(|x| !x.is_finite()) {
        return Err(FlatAttentionError::NonFiniteInput {
            tensor: name,
            index,
        });
    }
    Ok(())
}

/// Deterministic online-softmax reference forward pass.
///
/// This implementation is intentionally scalar and simple. It is the numerical
/// oracle for optimized kernels. It has O(N * D) auxiliary state per active
/// query and never allocates the O(N²) attention matrix.
pub fn forward_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
) -> Result<FlatAttentionOutput, FlatAttentionError> {
    shape.validate()?;
    let tensor_len = shape.tensor_len()?;
    validate_input("Q", q, tensor_len)?;
    validate_input("K", k, tensor_len)?;
    validate_input("V", v, tensor_len)?;
    let scale = config.resolved_scale(shape.head_dim)?;

    let mut output = vec![0.0f32; tensor_len];
    let mut lse = vec![0.0f32; shape.lse_len()?];
    let head_stride = shape.seq_len * shape.head_dim;

    for batch in 0..shape.batch {
        for head in 0..shape.heads {
            let bh = batch * shape.heads + head;
            let head_base = bh * head_stride;
            let lse_base = bh * shape.seq_len;

            for query_pos in 0..shape.seq_len {
                let q_base = head_base + query_pos * shape.head_dim;
                let out_base = q_base;
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;

                for key_pos in 0..shape.seq_len {
                    if config.causal && key_pos > query_pos {
                        break;
                    }

                    let kv_base = head_base + key_pos * shape.head_dim;
                    let mut dot = 0.0f32;
                    for dim in 0..shape.head_dim {
                        dot += q[q_base + dim] * k[kv_base + dim];
                    }
                    let score = dot * scale;
                    let new_max = running_max.max(score);
                    let alpha = if running_max.is_infinite() {
                        0.0
                    } else {
                        (running_max - new_max).exp()
                    };
                    let probability_numerator = (score - new_max).exp();

                    for dim in 0..shape.head_dim {
                        output[out_base + dim] = output[out_base + dim] * alpha
                            + probability_numerator * v[kv_base + dim];
                    }
                    running_sum = running_sum * alpha + probability_numerator;
                    running_max = new_max;
                }

                let inv_sum = running_sum.recip();
                for dim in 0..shape.head_dim {
                    output[out_base + dim] *= inv_sum;
                }
                lse[lse_base + query_pos] = running_max + running_sum.ln();
            }
        }
    }

    Ok(FlatAttentionOutput { output, lse })
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn default_scale_matches_inverse_sqrt_head_dim() {
        let scale = FlatAttentionConfig::default().resolved_scale(64).unwrap();
        assert_eq!(scale, 0.125);
    }

    #[test]
    fn rejects_bad_lengths() {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 2,
            head_dim: 2,
        };
        let err = forward_reference(
            &[0.0; 3],
            &[0.0; 4],
            &[0.0; 4],
            shape,
            FlatAttentionConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            FlatAttentionError::LengthMismatch { tensor: "Q", .. }
        ));
    }

    #[test]
    fn q4_io_model_reuses_kv_across_query_rows() {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 128,
            head_dim: 64,
        };
        let baseline = single_row_io_model(shape, false).unwrap();
        let tiled = tiled_q4_io_model(shape, false).unwrap();
        assert_eq!(baseline.query_workgroups, 128);
        assert_eq!(tiled.query_workgroups, 32);
        assert_eq!(baseline.kv_storage_scalar_loads, 4 * tiled.kv_storage_scalar_loads);
    }

    #[test]
    fn q4_causal_io_model_skips_fully_future_kv_rows() {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 8,
            head_dim: 1,
        };
        let baseline = single_row_io_model(shape, true).unwrap();
        let tiled = tiled_q4_io_model(shape, true).unwrap();
        assert_eq!(baseline.kv_storage_scalar_loads, 128);
        assert_eq!(tiled.kv_storage_scalar_loads, 24);
    }
}
