//! FLAT-ATTENTION: Rust-native, IO-aware fused attention for SciRust.
//!
//! The first milestone deliberately contains no CUDA, C, C++, vendor SDK, or
//! project-authored FFI. It establishes:
//! - a deterministic Rust reference implementation using online softmax;
//! - a fused WGSL forward kernel that never materializes the N x N score matrix;
//! - a small public API whose tensor layout matches SciRust's contiguous
//!   `[batch, heads, sequence, head_dim]` convention.

#![forbid(unsafe_code)]

use core::fmt;

/// Maximum head dimension supported by the first portable WGSL kernel.
pub const WGSL_MAX_HEAD_DIM: usize = 128;
/// Number of invocations in one WGSL workgroup.
pub const WGSL_WORKGROUP_SIZE: usize = 64;
/// Number of K/V rows staged in workgroup memory at once.
pub const WGSL_KV_TILE: usize = 16;

/// Portable fused forward kernel source.
pub const FLAT_FWD_WGSL: &str = include_str!("../shaders/flat_fwd.wgsl");

/// Contiguous tensor shape used by FLAT-ATTENTION.
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatAttentionConfig {
    /// Apply autoregressive masking (`key_position > query_position`).
    pub causal: bool,
    /// Optional score multiplier. Defaults to `1 / sqrt(head_dim)`.
    pub softmax_scale: Option<f32>,
}

impl Default for FlatAttentionConfig {
    fn default() -> Self {
        Self {
            causal: false,
            softmax_scale: None,
        }
    }
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
}
