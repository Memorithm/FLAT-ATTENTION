//! M9 numerical policy layer.
//!
//! The policy is explicit about execution and reduction semantics. Reference
//! execution is never presented as a GPU fallback, and deterministic GPU mode
//! deliberately disables capability-dependent subgroup reductions.

use core::fmt;

use super::{
    forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
};

#[cfg(feature = "wgpu")]
use super::{WgpuFlatAttention, WgpuFlatAttentionError, WgpuKernelVariant, WgpuSubgroupPolicy};

/// Public numerical execution modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericalMode {
    /// Scalar Rust oracle with serial FP32 accumulation.
    ExactReference,
    /// Qualified portable GPU path with capability-based optimization.
    #[default]
    FastPortable,
    /// Portable GPU path with fixed workgroup reduction topology.
    DeterministicPortable,
}

/// FP32 accumulation contract associated with a numerical mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulationPolicy {
    /// Scalar serial FP32 dot products and output accumulation.
    SerialFp32,
    /// Two dimensions per lane followed by a fixed 64-lane shared-memory tree.
    FixedTreeFp32,
    /// Backend may use the qualified subgroup first stage or the fixed tree.
    CapabilityOptimizedFp32,
}

/// Reduction-order contract associated with a numerical mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionPolicy {
    /// Key and head dimensions are consumed in scalar index order.
    SerialReference,
    /// Runtime capability may select subgroup or fixed-tree reduction.
    CapabilityOptimized,
    /// Fixed 64-lane workgroup tree; subgroup reduction is disabled.
    FixedWorkgroupTree,
}

/// Stable max/exponential update contract shared by all modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxUpdatePolicy {
    /// Running maximum is updated before exponentiation and prior state is
    /// rescaled by `exp(old_max - new_max)`.
    StableOnlineFp32,
}

/// Static guarantees for a numerical mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericalGuarantees {
    pub accumulation: AccumulationPolicy,
    pub reduction: ReductionPolicy,
    pub softmax_update: SoftmaxUpdatePolicy,
    /// Repeated identical calls are required to reproduce identical FP32 bit
    /// patterns under the same qualified backend/device/context contract.
    pub repeatable_same_backend_device: bool,
    /// Whether native subgroup reduction may be selected.
    pub allows_subgroup: bool,
}

impl NumericalMode {
    pub const fn guarantees(self) -> NumericalGuarantees {
        match self {
            Self::ExactReference => NumericalGuarantees {
                accumulation: AccumulationPolicy::SerialFp32,
                reduction: ReductionPolicy::SerialReference,
                softmax_update: SoftmaxUpdatePolicy::StableOnlineFp32,
                repeatable_same_backend_device: true,
                allows_subgroup: false,
            },
            Self::FastPortable => NumericalGuarantees {
                accumulation: AccumulationPolicy::CapabilityOptimizedFp32,
                reduction: ReductionPolicy::CapabilityOptimized,
                softmax_update: SoftmaxUpdatePolicy::StableOnlineFp32,
                repeatable_same_backend_device: false,
                allows_subgroup: true,
            },
            Self::DeterministicPortable => NumericalGuarantees {
                accumulation: AccumulationPolicy::FixedTreeFp32,
                reduction: ReductionPolicy::FixedWorkgroupTree,
                softmax_update: SoftmaxUpdatePolicy::StableOnlineFp32,
                repeatable_same_backend_device: true,
                allows_subgroup: false,
            },
        }
    }
}

/// Concrete execution backend selected by [`NumericalExecutor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalBackendKind {
    ReferenceCpu,
    Wgpu,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NumericalError {
    Core(FlatAttentionError),
    /// The crate was compiled without its optional WGPU feature.
    GpuFeatureDisabled,
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuFlatAttentionError),
}

impl fmt::Display for NumericalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::GpuFeatureDisabled => write!(
                f,
                "portable GPU numerical modes require the crate's `wgpu` feature"
            ),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(error) => write!(f, "WGPU numerical execution failed: {error}"),
        }
    }
}

impl std::error::Error for NumericalError {}

impl From<FlatAttentionError> for NumericalError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

#[cfg(feature = "wgpu")]
impl From<WgpuFlatAttentionError> for NumericalError {
    fn from(value: WgpuFlatAttentionError) -> Self {
        Self::Wgpu(value)
    }
}

enum NumericalBackend {
    Reference,
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuFlatAttention),
}

/// Explicit numerical-policy executor.
///
/// Construction chooses exactly one backend. A failed GPU construction returns
/// an error; it never switches to the Rust reference implementation.
pub struct NumericalExecutor {
    mode: NumericalMode,
    backend: NumericalBackend,
}

impl NumericalExecutor {
    pub fn new(mode: NumericalMode) -> Result<Self, NumericalError> {
        match mode {
            NumericalMode::ExactReference => Ok(Self {
                mode,
                backend: NumericalBackend::Reference,
            }),
            NumericalMode::FastPortable => Self::new_fast_portable(),
            NumericalMode::DeterministicPortable => Self::new_deterministic_portable(),
        }
    }

    pub const fn mode(&self) -> NumericalMode {
        self.mode
    }

    pub const fn guarantees(&self) -> NumericalGuarantees {
        self.mode.guarantees()
    }

    pub fn backend_kind(&self) -> NumericalBackendKind {
        match &self.backend {
            NumericalBackend::Reference => NumericalBackendKind::ReferenceCpu,
            #[cfg(feature = "wgpu")]
            NumericalBackend::Wgpu(_) => NumericalBackendKind::Wgpu,
        }
    }

    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<FlatAttentionOutput, NumericalError> {
        match &self.backend {
            NumericalBackend::Reference => {
                forward_reference(q, k, v, shape, config).map_err(Into::into)
            }
            #[cfg(feature = "wgpu")]
            NumericalBackend::Wgpu(context) => {
                context.forward(q, k, v, shape, config).map_err(Into::into)
            }
        }
    }

    #[cfg(feature = "wgpu")]
    pub fn adapter_name(&self) -> Option<&str> {
        match &self.backend {
            NumericalBackend::Reference => None,
            NumericalBackend::Wgpu(context) => Some(context.adapter_name()),
        }
    }

    #[cfg(feature = "wgpu")]
    pub fn kernel_variant_for_head_dim(&self, head_dim: usize) -> Option<WgpuKernelVariant> {
        match &self.backend {
            NumericalBackend::Reference => None,
            NumericalBackend::Wgpu(context) => Some(context.kernel_variant_for_head_dim(head_dim)),
        }
    }

    fn new_fast_portable() -> Result<Self, NumericalError> {
        #[cfg(feature = "wgpu")]
        {
            Ok(Self {
                mode: NumericalMode::FastPortable,
                backend: NumericalBackend::Wgpu(WgpuFlatAttention::new()?),
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            Err(NumericalError::GpuFeatureDisabled)
        }
    }

    fn new_deterministic_portable() -> Result<Self, NumericalError> {
        #[cfg(feature = "wgpu")]
        {
            // Fixed reduction topology: no subgroup, M6 vec4 storage remains
            // allowed because it does not alter the 64-lane reduction tree.
            let context = WgpuFlatAttention::with_subgroup_vectorization_and_double_buffering(
                WgpuSubgroupPolicy::Disable,
                true,
                false,
            )?;
            Ok(Self {
                mode: NumericalMode::DeterministicPortable,
                backend: NumericalBackend::Wgpu(context),
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            Err(NumericalError::GpuFeatureDisabled)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_guarantees_are_explicit_and_non_overlapping() {
        let exact = NumericalMode::ExactReference.guarantees();
        assert_eq!(exact.accumulation, AccumulationPolicy::SerialFp32);
        assert_eq!(exact.reduction, ReductionPolicy::SerialReference);
        assert!(exact.repeatable_same_backend_device);
        assert!(!exact.allows_subgroup);

        let fast = NumericalMode::FastPortable.guarantees();
        assert_eq!(
            fast.accumulation,
            AccumulationPolicy::CapabilityOptimizedFp32
        );
        assert_eq!(fast.reduction, ReductionPolicy::CapabilityOptimized);
        assert!(!fast.repeatable_same_backend_device);
        assert!(fast.allows_subgroup);

        let deterministic = NumericalMode::DeterministicPortable.guarantees();
        assert_eq!(
            deterministic.accumulation,
            AccumulationPolicy::FixedTreeFp32
        );
        assert_eq!(deterministic.reduction, ReductionPolicy::FixedWorkgroupTree);
        assert!(deterministic.repeatable_same_backend_device);
        assert!(!deterministic.allows_subgroup);
    }

    #[test]
    fn exact_executor_is_explicit_reference_execution() {
        let executor = NumericalExecutor::new(NumericalMode::ExactReference).unwrap();
        assert_eq!(executor.mode(), NumericalMode::ExactReference);
        assert_eq!(executor.backend_kind(), NumericalBackendKind::ReferenceCpu);

        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 2,
            head_dim: 2,
        };
        let q = [1.0, 0.0, 0.0, 1.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let v = [2.0, 3.0, 5.0, 7.0];
        let expected =
            forward_reference(&q, &k, &v, shape, FlatAttentionConfig::default()).unwrap();
        let actual = executor
            .forward(&q, &k, &v, shape, FlatAttentionConfig::default())
            .unwrap();
        assert_eq!(actual, expected);
    }
}
