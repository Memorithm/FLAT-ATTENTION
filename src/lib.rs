//! FLAT-ATTENTION: Rust-native, IO-aware fused attention for SciRust.
//!
//! The project keeps a deterministic scalar oracle and portable fused WGSL
//! kernels under one explicit contract. Optimized generations are qualified
//! against the oracle before becoming the default.

#![forbid(unsafe_code)]

use core::fmt;

/// Versioned backend-neutral reusable API.
pub mod api;

/// Machine-readable reproducible benchmark provenance.
pub mod benchmark_manifest;

/// Machine-readable evidence retention for the research-only nonlocal semantic.
mod research_nonlocal_evidence;
pub use research_nonlocal_evidence::{
    NonlocalEvidenceError, NonlocalEvidenceManifest, ResearchEvidenceDisposition,
    ResearchEvidenceScope, NONLOCAL_EVIDENCE_SCHEMA_VERSION,
};

mod fingerprint;

/// Experimental FLAT Kernel IR: validated structural descriptions of
/// qualified kernel architectures. Compiler infrastructure; changes no
/// runtime routing and makes no performance claim.
pub mod kernel_ir;

/// Deterministic WGSL emission from the FLAT Kernel IR. Compiler
/// infrastructure; generated sources require Naga/device qualification before
/// any routing use and carry no performance claim.
pub mod kernel_wgsl;

/// Static capability prefilter completing the M24 model: candidate
/// requirements are checked against explicit device limits before pipeline
/// creation, with typed rejections and no silent substitution.
pub mod kernel_prefilter;

/// Deterministic bounded candidate generation over the registered active
/// kernel realizations. Planning surface only; carries no performance claim.
pub mod kernel_candidates;

/// Correctness-gated bounded autotuner core with pluggable timing/correctness
/// surfaces. Selection evidence is reproducible; this core itself makes no
/// performance claim and performs no routing.
pub mod kernel_autotune;

#[cfg(feature = "wgpu")]
mod kernel_autotune_wgpu;
#[cfg(feature = "wgpu")]
pub use kernel_autotune_wgpu::{probe_capabilities, OracleParityGate, ResidentForwardHarness};

/// Safe persistent tuning cache for autotuning evidence (advisory, not
/// authoritative). Corruption or staleness invalidates entries safely.
pub mod kernel_cache;

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
    within_tol, AccumulationPolicy, NumericalBackendKind, NumericalError, NumericalExecutor,
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
/// M53 opt-in vec4 loads for rectangular projection-layout RoPE + GQA/MQA.
pub const FLAT_FWD_PROJECTION_ROPE_ASYMMETRIC_VEC4_WGSL: &str =
    include_str!("../shaders/flat_fwd_projection_rope_asymmetric_vec4.wgsl");
/// M58 opt-in Q1 vec4 MHA kernel: one workgroup per query row, register Q.
pub const FLAT_FWD_Q1_VEC4_WGSL: &str = include_str!("../shaders/flat_fwd_q1_vec4.wgsl");
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
/// M19 native GQA/MQA correctness-first backward recomputation kernel.
pub const FLAT_BACKWARD_GROUPED_RECOMPUTE_WGSL: &str =
    include_str!("../shaders/flat_backward_grouped_recompute.wgsl");
/// M5 subgroup-assisted Q4 kernel, selected only after runtime capability checks.
pub const FLAT_FWD_SUBGROUP_WGSL: &str = include_str!("../shaders/flat_fwd_subgroup.wgsl");
/// M8 packed-binary16 forward kernel with FP32 accumulation and FP32 LSE.
pub const FLAT_FWD_F16_WGSL: &str = include_str!("../shaders/flat_fwd_f16.wgsl");
/// Qualified M2/M3 one-query-row kernel retained as a baseline source.
pub const FLAT_FWD_SINGLE_WGSL: &str = include_str!("../shaders/flat_fwd_single.wgsl");

mod device_model;

/// Host-side device identity/capability model and passive dispatch telemetry.
/// Available without the `wgpu` feature so capability prefiltering, candidate
/// planning and evidence records run without a GPU runtime.
pub use device_model::{
    AutotunerCacheStatus, RuntimeDeviceCapabilities, RuntimeDeviceFingerprint,
    RuntimeDispatchTelemetry, RuntimeKernelId, RuntimeTileGeometry,
};

#[cfg(feature = "wgpu")]
mod runtime_telemetry;
#[cfg(feature = "wgpu")]
mod wgpu_internal;

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
pub use wgpu_grouped_backend::{WgpuGroupedAttention, WgpuGroupedAttentionError};

#[cfg(feature = "wgpu")]
mod wgpu_projection_grouped_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_projection_grouped_backend::{
    WgpuProjectionGroupedAttention, WgpuProjectionGroupedAttentionError,
};

#[cfg(feature = "wgpu")]
mod wgpu_projection_asymmetric_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_projection_asymmetric_backend::{
    WgpuProjectionAsymmetricAttention, WgpuProjectionAsymmetricAttentionError,
};

#[cfg(feature = "wgpu")]
mod wgpu_projection_variable_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_projection_variable_backend::{
    WgpuProjectionVariableAttention, WgpuProjectionVariableAttentionError,
};

#[cfg(feature = "wgpu")]
mod wgpu_backward_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_backward_backend::{WgpuBackwardAttention, WgpuBackwardAttentionError};

#[cfg(feature = "wgpu")]
mod wgpu_backward_grouped_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_backward_grouped_backend::{
    WgpuBackwardGroupedAttention, WgpuBackwardGroupedAttentionError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionShape {
    pub batch: usize,
    pub heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatAttentionConfig {
    pub scale: f32,
    pub causal: bool,
}

impl Default for FlatAttentionConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            causal: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlatAttentionOutput {
    pub output: Vec<f32>,
    pub logsumexp: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatAttentionError {
    ZeroDimension,
    UnsupportedHeadDim,
    ShapeMismatch,
    NonFiniteScale,
    InvalidGroupShape,
    InvalidBiasShape,
}

impl fmt::Display for FlatAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::ZeroDimension => "attention dimensions must be non-zero",
            Self::UnsupportedHeadDim => "head_dim exceeds portable kernel limit",
            Self::ShapeMismatch => "input length does not match attention shape",
            Self::NonFiniteScale => "attention scale must be finite",
            Self::InvalidGroupShape => "query and key/value head grouping is invalid",
            Self::InvalidBiasShape => "attention bias shape is invalid",
        };
        f.write_str(message)
    }
}

impl std::error::Error for FlatAttentionError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBias<'a> {
    None,
    Additive {
        values: &'a [f32],
        query_len: usize,
        key_len: usize,
    },
}

impl Default for AttentionBias<'_> {
    fn default() -> Self {
        Self::None
    }
}

impl<'a> AttentionBias<'a> {
    fn validate(self, shape: AttentionShape) -> Result<(), FlatAttentionError> {
        match self {
            Self::None => Ok(()),
            Self::Additive {
                values,
                query_len,
                key_len,
            } => {
                if query_len != shape.seq_len || key_len != shape.seq_len {
                    return Err(FlatAttentionError::InvalidBiasShape);
                }
                let expected = query_len
                    .checked_mul(key_len)
                    .ok_or(FlatAttentionError::InvalidBiasShape)?;
                if values.len() != expected {
                    return Err(FlatAttentionError::InvalidBiasShape);
                }
                Ok(())
            }
        }
    }
}

fn validate(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
) -> Result<(), FlatAttentionError> {
    if shape.batch == 0 || shape.heads == 0 || shape.seq_len == 0 || shape.head_dim == 0 {
        return Err(FlatAttentionError::ZeroDimension);
    }
    if shape.head_dim > WGSL_MAX_HEAD_DIM {
        return Err(FlatAttentionError::UnsupportedHeadDim);
    }
    if !config.scale.is_finite() {
        return Err(FlatAttentionError::NonFiniteScale);
    }
    let expected = shape
        .batch
        .checked_mul(shape.heads)
        .and_then(|n| n.checked_mul(shape.seq_len))
        .and_then(|n| n.checked_mul(shape.head_dim))
        .ok_or(FlatAttentionError::ShapeMismatch)?;
    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(FlatAttentionError::ShapeMismatch);
    }
    Ok(())
}

fn softmax_scores(scores: &mut [f32]) -> f32 {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0_f32;
    for score in scores.iter_mut() {
        *score = (*score - max).exp();
        denom += *score;
    }
    for score in scores.iter_mut() {
        *score /= denom;
    }
    max + denom.ln()
}

pub fn forward_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
) -> Result<FlatAttentionOutput, FlatAttentionError> {
    validate(q, k, v, shape, config)?;

    let rows = shape.batch * shape.heads * shape.seq_len;
    let mut output = vec![0.0_f32; rows * shape.head_dim];
    let mut logsumexp = vec![0.0_f32; rows];

    for batch in 0..shape.batch {
        for head in 0..shape.heads {
            for query in 0..shape.seq_len {
                let query_row = ((batch * shape.heads + head) * shape.seq_len + query)
                    * shape.head_dim;
                let allowed_keys = if config.causal {
                    query + 1
                } else {
                    shape.seq_len
                };
                let mut scores = vec![0.0_f32; allowed_keys];
                for (key, score) in scores.iter_mut().enumerate() {
                    let key_row =
                        ((batch * shape.heads + head) * shape.seq_len + key) * shape.head_dim;
                    let mut dot = 0.0_f32;
                    for dim in 0..shape.head_dim {
                        dot += q[query_row + dim] * k[key_row + dim];
                    }
                    *score = dot * config.scale;
                }
                logsumexp[(batch * shape.heads + head) * shape.seq_len + query] =
                    softmax_scores(&mut scores);
                for (key, weight) in scores.iter().copied().enumerate() {
                    let value_row =
                        ((batch * shape.heads + head) * shape.seq_len + key) * shape.head_dim;
                    for dim in 0..shape.head_dim {
                        output[query_row + dim] += weight * v[value_row + dim];
                    }
                }
            }
        }
    }

    Ok(FlatAttentionOutput {
        output,
        logsumexp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    #[test]
    fn non_causal_two_token_reference() {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 2,
            head_dim: 2,
        };
        let config = FlatAttentionConfig {
            scale: 1.0,
            causal: false,
        };
        let q = [1.0, 0.0, 0.0, 1.0];
        let k = q;
        let v = [1.0, 2.0, 3.0, 4.0];
        let result = forward_reference(&q, &k, &v, shape, config).unwrap();
        assert_eq!(result.output.len(), 4);
        assert!(approx_eq(result.output[0], 1.5378828));
        assert!(approx_eq(result.output[1], 2.5378828));
        assert!(approx_eq(result.output[2], 2.4621172));
        assert!(approx_eq(result.output[3], 3.4621172));
        assert_eq!(result.logsumexp.len(), 2);
    }

    #[test]
    fn causal_masks_future_tokens() {
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 2,
            head_dim: 1,
        };
        let q = [1.0, 1.0];
        let k = [1.0, 2.0];
        let v = [5.0, 9.0];
        let result = forward_reference(
            &q,
            &k,
            &v,
            shape,
            FlatAttentionConfig {
                scale: 1.0,
                causal: true,
            },
        )
        .unwrap();
        assert!(approx_eq(result.output[0], 5.0));
        assert!(result.output[1] > 7.0);
    }

    #[test]
    fn invalid_shape_is_rejected() {
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
        assert_eq!(err, FlatAttentionError::ShapeMismatch);
    }
}
