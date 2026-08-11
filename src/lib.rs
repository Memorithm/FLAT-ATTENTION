//! FLAT-ATTENTION: Rust-native, IO-aware fused attention for SciRust.
//!
//! The project keeps a deterministic scalar oracle and portable fused WGSL
//! kernels under one explicit contract. Optimized generations are qualified
//! against the oracle before becoming the default.

#![forbid(unsafe_code)]

use core::fmt;

mod f16;
pub use f16::{FlatAttentionF16Output, F16};

mod grouped;
pub use grouped::{forward_reference_grouped, GroupedAttentionShape};

mod asymmetric_grouped;
pub use asymmetric_grouped::{
    forward_reference_grouped_asymmetric, AsymmetricGroupedAttentionShape,
};

mod rotary_grouped;
pub use rotary_grouped::{forward_reference_grouped_rope, RotaryEmbeddingConfig};

mod projection_grouped;
pub use projection_grouped::forward_reference_projection_grouped_rope;

mod projection_asymmetric;
pub use projection_asymmetric::{
    forward_reference_projection_grouped_rope_asymmetric, AsymmetricRotaryEmbeddingConfig,
};

mod attention_bias;
pub use attention_bias::{
    forward_reference_projection_grouped_rope_asymmetric_biased, AttentionBias,
};

mod backward;
pub use backward::{backward_reference, FlatAttentionBackwardOutput};

mod backward_grouped;
pub use backward_grouped::backward_reference_grouped;

mod numerical;
pub use numerical::{
    AccumulationPolicy, NumericalBackendKind, NumericalError, NumericalExecutor,
    NumericalGuarantees, NumericalMode, ReductionPolicy, SoftmaxUpdatePolicy,
};

pub mod chunked_projection_prefill;
pub mod paged_kv;

/// Maximum head dimension supported by the portable WGSL kernels.
pub const WGSL_MAX_HEAD_DIM: usize = 128;
/// Number of invocations in one WGSL workgroup.
pub const WGSL_WORKGROUP_SIZE: usize = 64;
/// Number of K/V rows staged in workgroup memory at once.
pub const WGSL_KV_TILE: usize = 8;
/// Number of query rows sharing each K/V tile in the M4 default kernel.
pub const WGSL_QUERY_ROWS: usize = 4;

/// Qualified M4 portable fused forward kernel: four query rows per workgroup.
pub const FLAT_FWD_WGSL: &str = include_str!("../shaders/flat_fwd.wgsl");
/// M10 native GQA/MQA kernel without physical K/V head expansion.
pub const FLAT_FWD_GROUPED_WGSL: &str = include_str!("../shaders/flat_fwd_grouped.wgsl");
/// FLAT-R1 native GQA/MQA kernel with head-local RoPE fused into Q/K staging.
pub const FLAT_FWD_GROUPED_ROPE_WGSL: &str = include_str!("../shaders/flat_fwd_grouped_rope.wgsl");
/// FLAT-R2 direct sequence-major projection-layout RoPE + GQA/MQA kernel.
pub const FLAT_FWD_PROJECTION_ROPE_WGSL: &str =
    include_str!("../shaders/flat_fwd_projection_rope.wgsl");
/// M11 rectangular sequence-major projection-layout RoPE + GQA/MQA kernel.
pub const FLAT_FWD_PROJECTION_ROPE_ASYMMETRIC_WGSL: &str =
    include_str!("../shaders/flat_fwd_projection_rope_asymmetric.wgsl");
/// M12 padded variable-length projection-layout RoPE + GQA/MQA kernel.
pub const FLAT_FWD_PROJECTION_ROPE_VARIABLE_WGSL: &str =
    include_str!("../shaders/flat_fwd_projection_rope_variable.wgsl");
/// M15 q_len=1 decode kernel over fixed-capacity resident K/V storage.
pub const FLAT_DECODE_RESIDENT_WGSL: &str = include_str!("../shaders/flat_decode_resident.wgsl");
/// M16 q_len=1 decode kernel over paged resident K/V storage.
pub const FLAT_DECODE_PAGED_WGSL: &str = include_str!("../shaders/flat_decode_paged.wgsl");
/// M18 portable correctness-first backward recomputation kernel.
pub const FLAT_BACKWARD_RECOMPUTE_WGSL: &str =
    include_str!("../shaders/flat_backward_recompute.wgsl");
/// M5 subgroup-assisted Q4 kernel, selected only after runtime capability checks.
pub const FLAT_FWD_SUBGROUP_WGSL: &str = include_str!("../shaders/flat_fwd_subgroup.wgsl");
/// M8 packed-binary16 forward kernel with FP32 accumulation and FP32 LSE.
pub const FLAT_FWD_F16_WGSL: &str = include_str!("../shaders/flat_fwd_f16.wgsl");
/// Qualified M2/M3 one-query-row kernel retained as a baseline source.
pub const FLAT_FWD_SINGLE_WGSL: &str = include_str!("../shaders/flat_fwd_single.wgsl");

#[cfg(feature = "wgpu")]
mod wgpu_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_backend::{
    WgpuFlatAttention, WgpuFlatAttentionError, WgpuKernelVariant, WgpuResidentAttentionOutput,
    WgpuResidentBuffer, WgpuSubgroupPolicy,
};

#[cfg(feature = "wgpu")]
mod wgpu_f16_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_f16_backend::{
    WgpuF16Attention, WgpuF16AttentionError, WgpuIoPrecision, WgpuPreferredAttention,
    WgpuResidentF16AttentionOutput, WgpuResidentF16Buffer,
};

#[cfg(feature = "wgpu")]
mod wgpu_grouped_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_grouped_backend::{
    WgpuGroupedAttention, WgpuGroupedResidentAttentionOutput, WgpuGroupedResidentBuffer,
};

#[cfg(feature = "wgpu")]
mod wgpu_rotary_grouped_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_rotary_grouped_backend::{
    WgpuRotaryGroupedAttention, WgpuRotaryGroupedResidentBuffer, WgpuRotaryGroupedResidentOutput,
};

#[cfg(feature = "wgpu")]
mod wgpu_external;
#[cfg(feature = "wgpu")]
pub use wgpu_external::{
    ExternalProjectionLayout, ExternalProjectionPass, ExternalProjectionRotaryGroupedPipeline,
    ExternalWgpuError,
};

#[cfg(feature = "wgpu")]
mod wgpu_external_asymmetric;
#[cfg(feature = "wgpu")]
pub use wgpu_external_asymmetric::{
    ExternalAsymmetricProjectionPass, ExternalAsymmetricProjectionRotaryGroupedPipeline,
    WGSL_ALIBI_MAX_HEADS,
};

#[cfg(feature = "wgpu")]
mod wgpu_external_variable;
#[cfg(feature = "wgpu")]
pub use wgpu_external_variable::{
    ExternalVariableProjectionPass, ExternalVariableProjectionRotaryGroupedPipeline,
    VariableLengthRotaryEmbeddingConfig, VariableLengthSequenceMetadata, WGSL_VARIABLE_MAX_BATCH,
};

#[cfg(feature = "wgpu")]
mod wgpu_kv_cache;
#[cfg(feature = "wgpu")]
pub use wgpu_kv_cache::{WgpuResidentKvCache, WgpuResidentKvCacheError};

#[cfg(feature = "wgpu")]
mod wgpu_decode;
#[cfg(feature = "wgpu")]
pub use wgpu_decode::{
    ResidentDecodeError, ResidentDecodeLayout, ResidentDecodePass, WgpuResidentDecodePipeline,
};

#[cfg(feature = "wgpu")]
mod wgpu_paged_decode;
#[cfg(feature = "wgpu")]
pub use wgpu_paged_decode::{
    PagedDecodeError, PagedDecodeLayout, PagedDecodePass, WgpuPagedDecodePipeline,
};

#[cfg(feature = "wgpu")]
mod wgpu_paged_kv_cache;
#[cfg(feature = "wgpu")]
pub use wgpu_paged_kv_cache::{WgpuPagedKvCache, WgpuPagedKvCacheError};

#[cfg(feature = "wgpu")]
mod wgpu_chunked_prefill;
#[cfg(feature = "wgpu")]
pub use wgpu_chunked_prefill::{
    ChunkedProjectionPrefillError, ChunkedProjectionPrefillPass,
    WgpuChunkedProjectionPrefillPipeline,
};

#[cfg(feature = "wgpu")]
mod wgpu_backward;
#[cfg(feature = "wgpu")]
pub use wgpu_backward::{
    pack_backward_recompute_inputs, BackwardRecomputeError, BackwardRecomputeLayout,
    BackwardRecomputePass, WgpuBackwardRecomputePipeline,
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

    pub(crate) fn validate(self) -> Result<(), FlatAttentionError> {
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
    InvalidHeadGrouping {
        q_heads: usize,
        kv_heads: usize,
    },
    InvalidRotaryHeadDim {
        head_dim: usize,
    },
    InvalidRotaryTheta(f32),
    PositionOverflow,
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
            Self::InvalidHeadGrouping { q_heads, kv_heads } => write!(
                f,
                "q_heads ({q_heads}) must be exactly divisible by kv_heads ({kv_heads})"
            ),
            Self::InvalidRotaryHeadDim { head_dim } => write!(
                f,
                "rotary attention head_dim must be non-zero and even, got {head_dim}"
            ),
            Self::InvalidRotaryTheta(theta) => {
                write!(f, "rotary theta must be finite and positive, got {theta}")
            }
            Self::PositionOverflow => write!(f, "rotary position offset overflows the index space"),
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
        let q = vec![0.0; 4];
        let k = vec![0.0; 3];
        let v = vec![0.0; 4];
        let error = forward_reference(&q, &k, &v, shape, FlatAttentionConfig::default())
            .expect_err("invalid K length must fail");
        assert!(matches!(
            error,
            FlatAttentionError::LengthMismatch { tensor: "K", .. }
        ));
    }
}
