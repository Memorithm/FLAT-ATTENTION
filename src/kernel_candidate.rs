//! Deterministic FLAT kernel-candidate generation (roadmap M25, first slice).
//!
//! Given an [`AttentionProblem`], a [`DeviceLimitsView`], and a
//! [`CandidatePolicy`], this module produces the ordered set of legal Q4
//! candidate realizations with statically checkable requirements, plus the
//! deterministic rejection list of pruned families.
//!
//! Candidate generation belongs to FLAT; selection among candidates belongs
//! to the Elastic layer. This module never ranks by preference beyond the
//! static family order — it decides *legality*, not *desirability*.
//!
//! The capability view is deliberately host-neutral: `RuntimeDeviceCapabilities`
//! converts into [`DeviceLimitsView`] behind the `wgpu` feature, keeping IR +
//! generation usable in host-only builds.

use crate::kernel_ir::{
    ExecutionPlan, FlatKernelIr, KernelIrError, KernelVariantIdentity, PrecisionPolicy,
    ReductionStrategy, TileConfig, WorkgroupGeometry,
};
use std::fmt;

/// Host-neutral view of the device limits candidates must respect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimitsView {
    /// Maximum invocations along each workgroup axis `[x, y, z]`.
    pub max_workgroup_size: [u32; 3],
    /// Workgroup-addressable storage in bytes.
    pub max_workgroup_storage_bytes: u64,
    /// Maximum bind groups per dispatch.
    pub max_bind_groups: u32,
    /// Largest storage-buffer binding in bytes.
    pub max_storage_buffer_binding_bytes: u64,
    /// Maximum workgroups along one dispatch-grid axis.
    pub max_workgroups_per_dimension: u32,
    /// Whether subgroup operations are available.
    pub subgroup_supported: bool,
}

/// Static requirements one candidate imposes on a boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateRequirements {
    /// Invocations per workgroup along X (Y and Z are 1).
    pub workgroup_size_x: u32,
    /// Staged workgroup-memory bytes.
    pub workgroup_storage_bytes: u64,
    /// Bind groups used by the Q4 dispatch contract.
    pub bind_groups: u32,
    /// Whether the candidate executes subgroup operations.
    pub requires_subgroup: bool,
    /// Required head-dimension divisibility (1 for scalar, 4 for vec4).
    pub head_dim_multiple: u32,
}

/// Where the executable realization of a candidate comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RealizationSource {
    /// WGSL is deterministically generated from this candidate's IR by the
    /// M21 emitter subset.
    GeneratedFromIr,
    /// The realization uses the existing handwritten shader that already
    /// passed qualification; the IR documents its architecture until codegen
    /// covers it.
    HandwrittenQualified,
}

/// One legal candidate realization of a logical attention problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelCandidateSpec {
    ir: FlatKernelIr,
    source: RealizationSource,
    requirements: CandidateRequirements,
    static_kv_storage_scalar_loads: u64,
    query_workgroups: u64,
}

impl KernelCandidateSpec {
    /// Normalized kernel IR.
    #[must_use]
    pub const fn ir(&self) -> &FlatKernelIr {
        &self.ir
    }

    /// Realization source of the executable form.
    #[must_use]
    pub const fn source(&self) -> RealizationSource {
        self.source
    }

    /// Boundary requirements of this candidate.
    #[must_use]
    pub const fn requirements(&self) -> &CandidateRequirements {
        &self.requirements
    }

    /// Architectural K/V staging-load count from the explicit static IO model
    /// (see [`crate::tiled_q4_io_model`]). Not a DRAM-traffic claim.
    #[must_use]
    pub const fn static_kv_storage_scalar_loads(&self) -> u64 {
        self.static_kv_storage_scalar_loads
    }

    /// Query-axis workgroups this candidate dispatches.
    #[must_use]
    pub const fn query_workgroups(&self) -> u64 {
        self.query_workgroups
    }
}

/// Why one candidate family was pruned during generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrunedReason {
    /// Disabled by caller policy.
    PolicyDisabled,
    /// Requires subgroups the boundary does not report.
    SubgroupUnavailable,
    /// Head dimension violates the family's divisibility requirement.
    HeadDimNotMultiple {
        /// Required divisor.
        required_multiple: u32,
        /// Offending head dimension.
        actual: u32,
    },
    /// Staged workgroup memory exceeds the boundary limit.
    WorkgroupStorageExceeded {
        /// Required bytes.
        required_bytes: u64,
        /// Available bytes.
        available_bytes: u64,
    },
    /// Workgroup size exceeds the boundary limit.
    WorkgroupSizeExceeded {
        /// Required invocations.
        required: u32,
        /// Available invocations.
        available: u32,
    },
    /// Bind groups exceed the boundary limit.
    BindGroupsExceeded {
        /// Required entries.
        required: u32,
        /// Available entries.
        available: u32,
    },
    /// A dispatch-grid axis exceeds the boundary limit.
    DispatchAxisExceeded {
        /// Axis name.
        axis: &'static str,
        /// Required workgroups.
        required: u64,
        /// Available workgroups.
        available: u32,
    },
}

impl fmt::Display for PrunedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolicyDisabled => write!(f, "disabled by candidate policy"),
            Self::SubgroupUnavailable => write!(f, "subgroup support unavailable"),
            Self::HeadDimNotMultiple {
                required_multiple,
                actual,
            } => write!(
                f,
                "head_dim {actual} is not a multiple of {required_multiple}"
            ),
            Self::WorkgroupStorageExceeded {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "requires {required_bytes} bytes of workgroup storage, device allows {available_bytes}"
            ),
            Self::WorkgroupSizeExceeded { required, available } => write!(
                f,
                "requires workgroup size {required}, device allows {available}"
            ),
            Self::BindGroupsExceeded { required, available } => write!(
                f,
                "requires {required} bind groups, device allows {available}"
            ),
            Self::DispatchAxisExceeded {
                axis,
                required,
                available,
            } => write!(
                f,
                "dispatch axis {axis} requires {required} workgroups, device allows {available}"
            ),
        }
    }
}

/// Result of one deterministic generation pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationReport {
    candidates: Vec<KernelCandidateSpec>,
    pruned: Vec<(&'static str, PrunedReason)>,
}

impl GenerationReport {
    /// Legal candidates in deterministic family order.
    #[must_use]
    pub fn candidates(&self) -> &[KernelCandidateSpec] {
        &self.candidates
    }

    /// Pruned families with explicit reasons, in evaluation order.
    #[must_use]
    pub fn pruned(&self) -> &[(&'static str, PrunedReason)] {
        &self.pruned
    }

    /// Whether at least one candidate survived.
    #[must_use]
    pub fn has_candidates(&self) -> bool {
        !self.candidates.is_empty()
    }
}

/// knobs of the generation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidatePolicy {
    /// Consider the subgroup-assisted family.
    pub allow_subgroup: bool,
    /// Consider the experimental double-buffered family (requires vec4).
    pub allow_double_buffered: bool,
    /// Consider the vec4 portable family.
    pub allow_vec4: bool,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            allow_subgroup: true,
            allow_double_buffered: false,
            allow_vec4: true,
        }
    }
}

/// Canonical tile/workgroup geometry of the qualified Q4 family.
const FAMILY_QUERY_ROWS: u32 = 4;
const FAMILY_KV_TILE: u32 = 8;
const FAMILY_WORKGROUP: u32 = 64;
/// Bind groups used by every Q4 forward dispatch (Q, K, V, out_and_lse, params).
const FAMILY_BIND_GROUPS: u32 = 5;

/// Generate the ordered legal Q4 candidates for `(problem, limits, policy)`.
///
/// Family order is fixed: `q4-subgroup`, `q4-double-buffer`, `q4-vec4`,
/// `q4-portable-generated`. Every pruning decision is recorded with an
/// explicit reason. Identical inputs always produce identical reports.
///
/// # Errors
///
/// Returns [`GenerateError`] when the problem itself cannot run on the given
/// limits regardless of family (binding-size or dispatch-space violations),
/// or when IR construction fails.
pub fn generate_q4_candidates(
    problem: &crate::kernel_ir::AttentionProblem,
    limits: &DeviceLimitsView,
    policy: CandidatePolicy,
) -> Result<GenerationReport, GenerateError> {
    problem.validate()?;
    #[allow(clippy::cast_possible_truncation)]
    let shape = crate::AttentionShape {
        batch: usize::try_from(problem.batch_heads)
            .map_err(|_| GenerateError::BindingSizeOverflow)?,
        heads: 1,
        seq_len: usize::try_from(problem.seq_len)
            .map_err(|_| GenerateError::BindingSizeOverflow)?,
        head_dim: usize::try_from(problem.head_dim)
            .map_err(|_| GenerateError::BindingSizeOverflow)?,
    };
    let io = crate::tiled_q4_io_model(shape, problem.causal)
        .map_err(|_| GenerateError::StaticModelOverflow)?;

    // Problem-wide binding feasibility (independent of family).
    let qkv_bytes = problem
        .tensor_elements()
        .checked_mul(3)
        .and_then(|v| v.checked_mul(4))
        .ok_or(GenerateError::BindingSizeOverflow)?;
    let out_bytes = problem
        .tensor_elements()
        .checked_add(problem.lse_elements())
        .and_then(|v| v.checked_mul(4))
        .ok_or(GenerateError::BindingSizeOverflow)?;
    if qkv_bytes > limits.max_storage_buffer_binding_bytes
        || out_bytes > limits.max_storage_buffer_binding_bytes
    {
        return Err(GenerateError::ProblemExceedsBufferBindingLimit {
            required_bytes: qkv_bytes.max(out_bytes),
            available_bytes: limits.max_storage_buffer_binding_bytes,
        });
    }
    let query_tiles = u64::from(problem.seq_len).div_ceil(u64::from(FAMILY_QUERY_ROWS));
    if query_tiles > u64::from(limits.max_workgroups_per_dimension)
        || u64::from(problem.batch_heads) > u64::from(limits.max_workgroups_per_dimension)
    {
        return Err(GenerateError::ProblemExceedsDispatchLimit {
            required_workgroups: query_tiles.max(u64::from(problem.batch_heads)),
            available_workgroups: limits.max_workgroups_per_dimension,
        });
    }

    let mut candidates = Vec::new();
    let mut pruned: Vec<(&'static str, PrunedReason)> = Vec::new();

    // --- q4-subgroup ---
    const SUBGROUP_FAMILY: &str = "flat.fwd.q4:subgroup";
    if !policy.allow_subgroup {
        pruned.push((SUBGROUP_FAMILY, PrunedReason::PolicyDisabled));
    } else if !limits.subgroup_supported {
        pruned.push((SUBGROUP_FAMILY, PrunedReason::SubgroupUnavailable));
    } else {
        match build_spec(
            KernelVariantIdentity::subgroup_q4(),
            problem,
            ReductionStrategy::SubgroupArithmetic,
            PrecisionPolicy::F32StorageF32Accumulate,
            RealizationSource::HandwrittenQualified,
            limits,
            &io,
        ) {
            Ok(spec) => candidates.push(spec),
            Err(reason) => pruned.push((SUBGROUP_FAMILY, reason)),
        }
    }

    // --- q4-double-buffer (experimental, opt-in, vec4-dependent) ---
    // Represented as a policy-gated portable-tree variant on the same
    // geometry; it remains non-default until benchmark evidence justifies it.
    const DOUBLE_BUFFER_FAMILY: &str = "flat.fwd.q4:double-buffer";
    if !policy.allow_double_buffered {
        pruned.push((DOUBLE_BUFFER_FAMILY, PrunedReason::PolicyDisabled));
    } else if problem.head_dim % 4 != 0 {
        pruned.push((
            DOUBLE_BUFFER_FAMILY,
            PrunedReason::HeadDimNotMultiple {
                required_multiple: 4,
                actual: problem.head_dim,
            },
        ));
    } else {
        match build_spec(
            KernelVariantIdentity {
                family: "flat.fwd.q4",
                variant: "double-buffer",
                schema_version: crate::kernel_ir::KERNEL_IR_SCHEMA_VERSION,
            },
            problem,
            ReductionStrategy::TreeInWorkgroup,
            PrecisionPolicy::F32StorageF32Accumulate,
            RealizationSource::HandwrittenQualified,
            limits,
            &io,
        ) {
            Ok(spec) => candidates.push(spec),
            Err(reason) => pruned.push((DOUBLE_BUFFER_FAMILY, reason)),
        }
    }

    // --- q4-vec4 (handwritten M6) ---
    const VEC4_FAMILY: &str = "flat.fwd.q4:vec4";
    if !policy.allow_vec4 {
        pruned.push((VEC4_FAMILY, PrunedReason::PolicyDisabled));
    } else if problem.head_dim % 4 != 0 {
        pruned.push((
            VEC4_FAMILY,
            PrunedReason::HeadDimNotMultiple {
                required_multiple: 4,
                actual: problem.head_dim,
            },
        ));
    } else {
        match build_spec(
            KernelVariantIdentity {
                family: "flat.fwd.q4",
                variant: "vec4",
                schema_version: crate::kernel_ir::KERNEL_IR_SCHEMA_VERSION,
            },
            problem,
            ReductionStrategy::TreeInWorkgroup,
            PrecisionPolicy::F32StorageF32Accumulate,
            RealizationSource::HandwrittenQualified,
            limits,
            &io,
        ) {
            Ok(spec) => candidates.push(spec),
            Err(reason) => pruned.push((VEC4_FAMILY, reason)),
        }
    }

    // --- q4-portable-generated (M20/M21 emitter subset) ---
    const PORTABLE_FAMILY: &str = "flat.fwd.q4:portable";
    match build_spec(
        KernelVariantIdentity::portable_q4(),
        problem,
        ReductionStrategy::TreeInWorkgroup,
        PrecisionPolicy::F32StorageF32Accumulate,
        RealizationSource::GeneratedFromIr,
        limits,
        &io,
    ) {
        Ok(spec) => candidates.push(spec),
        Err(reason) => pruned.push((PORTABLE_FAMILY, reason)),
    }

    Ok(GenerationReport { candidates, pruned })
}

/// Errors from candidate generation that are about the problem, not one
/// family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GenerateError {
    /// Problem validation failed.
    InvalidProblem(KernelIrError),
    /// Buffer bindings exceed the boundary limit for every family.
    ProblemExceedsBufferBindingLimit {
        /// Required bytes for the largest binding.
        required_bytes: u64,
        /// Available bytes.
        available_bytes: u64,
    },
    /// Dispatch-grid space exceeds the boundary limit for every family.
    ProblemExceedsDispatchLimit {
        /// Required workgroups along the largest axis.
        required_workgroups: u64,
        /// Available workgroups.
        available_workgroups: u32,
    },
    /// Binding-size accounting overflowed.
    BindingSizeOverflow,
    /// The static IO model overflowed while deriving architectural counts.
    StaticModelOverflow,
}

impl fmt::Display for GenerateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProblem(error) => write!(f, "invalid attention problem: {error}"),
            Self::ProblemExceedsBufferBindingLimit {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "problem requires a {required_bytes}-byte storage binding, device allows {available_bytes}"
            ),
            Self::ProblemExceedsDispatchLimit {
                required_workgroups,
                available_workgroups,
            } => write!(
                f,
                "problem requires {required_workgroups} workgroups per axis, device allows {available_workgroups}"
            ),
            Self::BindingSizeOverflow => write!(f, "binding accounting overflowed"),
            Self::StaticModelOverflow => {
                write!(f, "static IO model overflowed")
            }
        }
    }
}

impl std::error::Error for GenerateError {}

impl From<KernelIrError> for GenerateError {
    fn from(value: KernelIrError) -> Self {
        Self::InvalidProblem(value)
    }
}

fn build_spec(
    identity: KernelVariantIdentity,
    problem: &crate::kernel_ir::AttentionProblem,
    reduction: ReductionStrategy,
    precision: PrecisionPolicy,
    source: RealizationSource,
    limits: &DeviceLimitsView,
    io: &crate::IoModel,
) -> Result<KernelCandidateSpec, PrunedReason> {
    let tiles = TileConfig {
        query_rows: FAMILY_QUERY_ROWS,
        kv_tile: FAMILY_KV_TILE,
    };
    let workgroup = WorkgroupGeometry {
        invocations: FAMILY_WORKGROUP,
    };
    let plan = ExecutionPlan::build(
        tiles,
        workgroup,
        reduction,
        precision,
        reduction == ReductionStrategy::SubgroupArithmetic,
    )
    .map_err(|error| match error {
        // The only build failure reachable here is an invalid subgroup
        // declaration; every other geometry constant is family-fixed and
        // validated above.
        crate::kernel_ir::KernelIrError::SubgroupOperationWithoutRequirement => {
            PrunedReason::SubgroupUnavailable
        }
        _ => PrunedReason::WorkgroupSizeExceeded {
            required: FAMILY_WORKGROUP,
            available: limits.max_workgroup_size[0],
        },
    })?;
    if plan.workgroup().invocations > limits.max_workgroup_size[0] {
        return Err(PrunedReason::WorkgroupSizeExceeded {
            required: plan.workgroup().invocations,
            available: limits.max_workgroup_size[0],
        });
    }
    let storage_bytes =
        plan.workgroup_storage_bytes()
            .map_err(|_| PrunedReason::WorkgroupStorageExceeded {
                required_bytes: u64::MAX,
                available_bytes: limits.max_workgroup_storage_bytes,
            })?;
    if storage_bytes > limits.max_workgroup_storage_bytes {
        return Err(PrunedReason::WorkgroupStorageExceeded {
            required_bytes: storage_bytes,
            available_bytes: limits.max_workgroup_storage_bytes,
        });
    }
    if FAMILY_BIND_GROUPS > limits.max_bind_groups {
        return Err(PrunedReason::BindGroupsExceeded {
            required: FAMILY_BIND_GROUPS,
            available: limits.max_bind_groups,
        });
    }
    let ir =
        FlatKernelIr::build(identity, *problem, plan).map_err(|_| PrunedReason::PolicyDisabled)?;
    Ok(KernelCandidateSpec {
        ir,
        source,
        requirements: CandidateRequirements {
            workgroup_size_x: FAMILY_WORKGROUP,
            workgroup_storage_bytes: storage_bytes,
            bind_groups: FAMILY_BIND_GROUPS,
            requires_subgroup: reduction == ReductionStrategy::SubgroupArithmetic,
            head_dim_multiple: 1,
        },
        static_kv_storage_scalar_loads: io.kv_storage_scalar_loads as u64,
        query_workgroups: io.query_workgroups as u64,
    })
}
