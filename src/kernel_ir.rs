//! Experimental FLAT Kernel IR for the qualified dense Q4 forward family.
//!
//! The IR is the project-owned structural description that sits between
//! attention semantics and generated kernels. It deliberately stays small: it
//! expresses exactly the algorithm structure of the currently qualified dense
//! four-query-row forward architecture (scalar, vec4, double-buffered and
//! subgroup-assisted realizations) rather than a universal GPU language.
//!
//! Two separations are enforced by construction:
//!
//! - [`AttentionProblem`] describes *what* is computed (semantic geometry).
//! - [`KernelConfig`] describes *how* it is computed (tuning choices).
//!
//! Modules can only be produced by validated builders, so illegal kernel
//! descriptions (vector widths without an executable path, double buffering
//! off the vec4 realization, subgroup reduction without a subgroup capability
//! requirement, unchecked dimension arithmetic) are rejected before shader
//! generation or pipeline creation can observe them. The ordered program is
//! assembled internally; callers iterate it read-only, so illegal orderings
//! are not representable.
//!
//! Everything here is deterministic and host-only: equivalent modules produce
//! byte-identical [`KernelModule::canonical_record`] text and stable
//! [`KernelModule::structural_fingerprint`] values. Fingerprints follow the
//! repository's FNV-1a-64 discipline; they are cache-key/equality
//! accelerators, **not** cryptographic authentication, and correctness
//! decisions must rest on structural validation.
//!
//! This surface is experimental compiler infrastructure. It changes no
//! runtime routing and makes no performance claim.

use core::fmt;

use crate::{AttentionShape, FlatAttentionConfig};

/// Schema version of the Kernel IR. Any incompatible change to structure,
/// validation, canonical serialization or derived identities must bump this
/// version so cached keys from older schemas invalidate cleanly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelIrVersion {
    /// Incompatible schema change; invalidates every derived identity.
    pub major: u32,
    /// Backward-compatible refinement within one major schema.
    pub minor: u32,
}

impl KernelIrVersion {
    /// IR schema produced by this crate.
    pub const CURRENT: Self = Self { major: 1, minor: 0 };
}

impl fmt::Display for KernelIrVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Semantic description of one dense attention computation: the WHAT.
///
/// Geometry uses the folded `batch × heads` contract of the portable WGSL
/// family (`workgroup_id.y` domain) with the same `u32` index space as the
/// shaders, so validated problems cannot exceed what a dispatch can express.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionProblem {
    /// Folded batch × query-head count per problem set.
    pub batch_heads: u32,
    /// Query/key/value token count of the dense family.
    pub seq_len: u32,
    /// Feature width of every head row (1..=[`KERNEL_MAX_HEAD_DIM`]).
    pub head_dim: u32,
    /// Apply autoregressive masking (`key_position > query_position`).
    pub causal: bool,
}

/// Maximum head dimension representable by the dense Q4 family; matches the
/// portable WGSL guard.
pub const KERNEL_MAX_HEAD_DIM: u32 = 128;

impl AttentionProblem {
    /// Build a problem from the public MHA shape/config contract.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the public shape is invalid or its
    /// dimensions exceed the folded `u32` dispatch index space.
    pub fn from_shape(
        shape: &AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<Self, KernelIrError> {
        let batch_heads = checked_u32_product(shape.batch, shape.heads).ok_or(
            KernelIrError::IndexSpaceOverflow {
                field: "batch_heads",
            },
        )?;
        if shape.seq_len == 0 || shape.head_dim == 0 {
            // Preserve the public zero-dimension semantics before u32 casts.
            return Err(KernelIrError::ZeroDimension {
                field: if shape.seq_len == 0 {
                    "seq_len"
                } else {
                    "head_dim"
                },
            });
        }
        if shape.batch == 0 || shape.heads == 0 {
            return Err(KernelIrError::ZeroDimension {
                field: if shape.batch == 0 { "batch" } else { "heads" },
            });
        }
        let seq_len = u32::try_from(shape.seq_len)
            .map_err(|_| KernelIrError::IndexSpaceOverflow { field: "seq_len" })?;
        let head_dim = u32::try_from(shape.head_dim)
            .map_err(|_| KernelIrError::IndexSpaceOverflow { field: "head_dim" })?;
        let problem = Self {
            batch_heads,
            seq_len,
            head_dim,
            causal: config.causal,
        };
        problem.validate()?;
        Ok(problem)
    }

    /// Validate the semantic geometry.
    pub fn validate(&self) -> Result<(), KernelIrError> {
        if self.batch_heads == 0 {
            return Err(KernelIrError::ZeroDimension {
                field: "batch_heads",
            });
        }
        if self.seq_len == 0 {
            return Err(KernelIrError::ZeroDimension { field: "seq_len" });
        }
        if self.head_dim == 0 {
            return Err(KernelIrError::ZeroDimension { field: "head_dim" });
        }
        if self.head_dim > KERNEL_MAX_HEAD_DIM {
            return Err(KernelIrError::HeadDimBeyondFamilyMaximum {
                actual: self.head_dim,
                maximum: KERNEL_MAX_HEAD_DIM,
            });
        }
        self.tensor_elements()?;
        self.output_elements()?;
        Ok(())
    }

    /// f32 elements in one Q/K/V tensor, checked.
    ///
    /// The product is accumulated in `u64`: individual dimensions fit the
    /// shader `u32` index space while their product legitimately exceeds it.
    pub fn tensor_elements(&self) -> Result<u64, KernelIrError> {
        u64::from(self.batch_heads)
            .checked_mul(u64::from(self.seq_len))
            .and_then(|n| n.checked_mul(u64::from(self.head_dim)))
            .ok_or(KernelIrError::ElementCountOverflow { tensor: "qkv" })
    }

    /// Packed output elements (`O` rows plus one LSE scalar per token),
    /// checked.
    pub fn output_elements(&self) -> Result<u64, KernelIrError> {
        let o = self.tensor_elements()?;
        let lse = u64::from(self.batch_heads)
            .checked_mul(u64::from(self.seq_len))
            .ok_or(KernelIrError::ElementCountOverflow { tensor: "lse" })?;
        o.checked_add(lse)
            .ok_or(KernelIrError::ElementCountOverflow { tensor: "output" })
    }

    /// Deterministic canonical record of the semantic problem.
    pub fn canonical_record(&self) -> String {
        format!(
            "bh={};n={};d={};causal={}",
            self.batch_heads,
            self.seq_len,
            self.head_dim,
            u8::from(self.causal)
        )
    }
}

fn checked_u32_product(a: usize, b: usize) -> Option<u32> {
    a.checked_mul(b).and_then(|n| u32::try_from(n).ok())
}

/// Storage access width of one realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorWidth {
    /// Scalar `f32` accesses (qualified portable fallback).
    Scalar,
    /// `vec4<f32>` packed accesses for aligned head dimensions.
    Vec4,
}

impl VectorWidth {
    /// Components per access.
    #[must_use]
    pub const fn components(self) -> u32 {
        match self {
            Self::Scalar => 1,
            Self::Vec4 => 4,
        }
    }

    const fn canonical(self) -> &'static str {
        match self {
            Self::Scalar => "x1",
            Self::Vec4 => "x4",
        }
    }
}

/// K/V workgroup staging strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvStaging {
    /// One staged K/V tile per iteration.
    SingleBuffered,
    /// Ping/pong tiles enabling load/compute overlap (M7 lineage, opt-in).
    DoubleBuffered,
}

impl KvStaging {
    /// Workgroup K/V arrays required by the strategy.
    #[must_use]
    pub const fn buffers(self) -> u32 {
        match self {
            Self::SingleBuffered => 1,
            Self::DoubleBuffered => 2,
        }
    }

    const fn canonical(self) -> &'static str {
        match self {
            Self::SingleBuffered => "single",
            Self::DoubleBuffered => "double",
        }
    }
}

/// Score-reduction mechanism for the per-query dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoreReduction {
    /// Deterministic workgroup-memory tree reduction.
    WorkgroupTree,
    /// Subgroup-arithmetic assisted reduction with a deterministic tree over
    /// subgroup totals. Requires the subgroup capability at runtime.
    SubgroupAssisted,
}

impl ScoreReduction {
    pub(crate) const fn canonical(self) -> &'static str {
        match self {
            Self::WorkgroupTree => "tree",
            Self::SubgroupAssisted => "subgroup",
        }
    }
}

/// Query rows computed per workgroup; closed over existing machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryRows {
    /// Four consecutive query rows share each staged K/V tile (Q4).
    Four,
}

impl QueryRows {
    /// Row count as a plain integer.
    #[must_use]
    pub const fn get(self) -> u32 {
        match self {
            Self::Four => 4,
        }
    }

    const fn canonical(self) -> &'static str {
        match self {
            Self::Four => "4",
        }
    }
}

/// K/V rows staged per tile iteration; closed over existing machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvTileRows {
    /// Eight K/V rows per staged tile.
    Eight,
}

impl KvTileRows {
    /// Row count as a plain integer.
    #[must_use]
    pub const fn get(self) -> u32 {
        match self {
            Self::Eight => 8,
        }
    }
}

/// Workgroup invocations along X; closed over existing machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkgroupSize {
    /// 64 invocations cover two 32-lane dot halves.
    SixtyFour,
}

impl WorkgroupSize {
    /// Invocation count as a plain integer.
    #[must_use]
    pub const fn get(self) -> u32 {
        match self {
            Self::SixtyFour => 64,
        }
    }
}

/// Implementation/tuning choices for one realization: the HOW.
///
/// Every field is a closed enumeration over values with actual executable
/// machinery on `main`; free-form integers would allow describing candidates
/// that can never be generated or run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelConfig {
    /// Query rows computed per workgroup.
    pub query_rows: QueryRows,
    /// K/V rows streamed per tile.
    pub kv_tile_rows: KvTileRows,
    /// Invocations along the workgroup X axis.
    pub workgroup_size: WorkgroupSize,
    /// Storage access width for Q/K/V.
    pub vector_width: VectorWidth,
    /// K/V workgroup staging strategy.
    pub kv_staging: KvStaging,
    /// Score reduction mechanism.
    pub score_reduction: ScoreReduction,
}

impl KernelConfig {
    /// Qualified portable scalar baseline (M2/M4 lineage).
    pub const PORTABLE_SCALAR: Self = Self {
        query_rows: QueryRows::Four,
        kv_tile_rows: KvTileRows::Eight,
        workgroup_size: WorkgroupSize::SixtyFour,
        vector_width: VectorWidth::Scalar,
        kv_staging: KvStaging::SingleBuffered,
        score_reduction: ScoreReduction::WorkgroupTree,
    };

    /// Qualified vec4 realization (M6 lineage).
    pub const PORTABLE_VEC4: Self = Self {
        vector_width: VectorWidth::Vec4,
        ..Self::PORTABLE_SCALAR
    };

    /// Experimental double-buffered vec4 realization (M7 lineage).
    pub const DOUBLE_BUFFERED_VEC4: Self = Self {
        kv_staging: KvStaging::DoubleBuffered,
        ..Self::PORTABLE_VEC4
    };

    /// Subgroup-assisted realization (M5 lineage).
    pub const SUBGROUP_ASSISTED: Self = Self {
        score_reduction: ScoreReduction::SubgroupAssisted,
        ..Self::PORTABLE_SCALAR
    };

    /// Workgroup f32 scalars declared by this configuration (problem
    /// independent), or overflow.
    pub(crate) fn workgroup_storage_scalars(&self) -> Option<u64> {
        let max_head_dim = u64::from(KERNEL_MAX_HEAD_DIM);
        let query_rows = u64::from(self.query_rows.get());
        let kv_tile = u64::from(self.kv_tile_rows.get());
        let kv_buffers = u64::from(self.kv_staging.buffers());
        let invocations = u64::from(self.workgroup_size.get());
        let q = query_rows.checked_mul(max_head_dim)?;
        let kv = kv_buffers
            .checked_mul(2)?
            .checked_mul(kv_tile)?
            .checked_mul(max_head_dim)?;
        let reduce = query_rows.checked_mul(invocations)?;
        let state = 4_u64.checked_mul(query_rows)?;
        q.checked_add(kv)?.checked_add(reduce)?.checked_add(state)
    }

    /// Capability requirements implied by the configuration alone, in
    /// deterministic order. Problem-dependent facts (dispatch extents,
    /// output binding size) are deliberately excluded; see
    /// [`KernelModule::capability_requirements`].
    #[must_use]
    pub fn static_requirements(&self) -> Vec<CapabilityRequirement> {
        let mut requirements = Vec::new();
        if self.score_reduction == ScoreReduction::SubgroupAssisted {
            requirements.push(CapabilityRequirement::SubgroupOperations);
        }
        requirements.push(CapabilityRequirement::MinWorkgroupInvocations(
            self.workgroup_size.get(),
        ));
        if let Some(scalars) = self.workgroup_storage_scalars() {
            requirements.push(CapabilityRequirement::MinWorkgroupStorageBytes(
                u32::try_from(scalars.saturating_mul(4)).unwrap_or(u32::MAX),
            ));
        }
        requirements.push(CapabilityRequirement::MinBindingEntries(5));
        requirements
    }

    /// Cross-checks configuration legality against the semantic problem.
    fn validate_against(&self, problem: &AttentionProblem) -> Result<(), KernelIrError> {
        if self.vector_width == VectorWidth::Vec4
            && problem.head_dim != 64
            && problem.head_dim != 128
        {
            // The qualified vec4 machinery serves exactly D64/D128; any other
            // width has no executable path.
            return Err(KernelIrError::HeadDimUnsupportedByVectorWidth {
                head_dim: problem.head_dim,
                components: self.vector_width.components(),
            });
        }
        if self.kv_staging == KvStaging::DoubleBuffered && self.vector_width != VectorWidth::Vec4 {
            return Err(KernelIrError::DoubleBufferingRequiresVectorPath);
        }
        Ok(())
    }

    fn canonical_record(&self) -> String {
        format!(
            "qr={};kt={};wg={};vw={};stg={};red={}",
            self.query_rows.canonical(),
            self.kv_tile_rows.get(),
            self.workgroup_size.get(),
            self.vector_width.canonical(),
            self.kv_staging.canonical(),
            self.score_reduction.canonical(),
        )
    }
}

/// Kernel architecture realized by a module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelFamily {
    /// Dense fused forward with four query rows per workgroup (MHA only;
    /// grouped GQA/MQA realizations belong to their own future families).
    DenseQ4Forward,
}

impl KernelFamily {
    const fn canonical(self) -> &'static str {
        match self {
            Self::DenseQ4Forward => "dense_q4_forward",
        }
    }
}

/// Ordered typed phases of the dense forward program.
///
/// Assembled internally in fixed order by [`KernelModule::build`]; callers
/// iterate read-only via [`KernelModule::phases`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelPhase {
    /// Guard out-of-range dispatch coordinates and unsupported head dims.
    DispatchGuard,
    /// Initialize online-softmax state for every query row.
    InitOnlineSoftmaxState {
        /// Query rows holding independent state.
        rows: QueryRows,
    },
    /// Stage the query tile into workgroup storage.
    StageQuery {
        /// Access width of the stage.
        width: VectorWidth,
    },
    /// Stream one K/V tile into workgroup storage.
    StreamKvTile {
        /// Rows per staged tile.
        tile_rows: KvTileRows,
        /// Access width of the stage.
        width: VectorWidth,
        /// Staging strategy controlling buffer count and barriers.
        staging: KvStaging,
    },
    /// Compute per-lane partial products of the scaled dot product.
    ScorePartials {
        /// Access width feeding the partials.
        width: VectorWidth,
    },
    /// Reduce per-lane partials into one score per query row.
    ReduceScore {
        /// Reduction mechanism.
        strategy: ScoreReduction,
    },
    /// Lane-leader online-softmax state update and broadcast.
    SoftmaxStateUpdate,
    /// Rescale accumulators and accumulate the probability-weighted V row.
    AccumulateValue {
        /// Access width of the V accumulation.
        width: VectorWidth,
    },
    /// Normalize accumulated output and store the O rows.
    NormalizeStoreOutput {
        /// Access width of the output store.
        width: VectorWidth,
    },
    /// Store the log-sum-exp statistic for every query token.
    StoreLse,
}

impl KernelPhase {
    const fn tag(self) -> &'static str {
        match self {
            Self::DispatchGuard => "guard",
            Self::InitOnlineSoftmaxState { .. } => "softmax_init",
            Self::StageQuery { .. } => "stage_q",
            Self::StreamKvTile { .. } => "stream_kv",
            Self::ScorePartials { .. } => "score_partials",
            Self::ReduceScore { .. } => "reduce_score",
            Self::SoftmaxStateUpdate => "softmax_update",
            Self::AccumulateValue { .. } => "accumulate_v",
            Self::NormalizeStoreOutput { .. } => "normalize_store",
            Self::StoreLse => "store_lse",
        }
    }
}

/// Abstract device capability declared by a module before execution.
///
/// Requirements are pure IR facts; the runtime capability model decides
/// whether a concrete adapter satisfies them. WGPU runtime objects stay out
/// of the IR entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityRequirement {
    /// Native subgroup operations must be available.
    SubgroupOperations,
    /// Minimum invocations addressable inside one workgroup.
    MinWorkgroupInvocations(u32),
    /// Minimum workgroup-addressable storage in bytes.
    MinWorkgroupStorageBytes(u32),
    /// Minimum bind-group entries usable by one dispatch.
    MinBindingEntries(u32),
}

/// Static resource footprint of one configured module for one problem.
///
/// Every quantity is computed with checked arithmetic during validation;
/// overflow is a construction error instead of a wrapped value reaching
/// allocation or dispatch arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelResources {
    /// Workgroup-addressable storage declared by the realization, in bytes.
    pub workgroup_storage_bytes: u64,
    /// Invocations per workgroup along all axes.
    pub invocations_per_workgroup: u32,
    /// Bind-group entries used by the entry point.
    pub binding_entries: u32,
    /// Resolved dispatch extents `[x, y, z]` for the bound problem.
    pub dispatch_extents: [u32; 3],
    /// Packed output elements (`O | LSE`) written by the entry point.
    pub output_elements: u64,
}

/// Validated Kernel IR module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelModule {
    ir_version: KernelIrVersion,
    family: KernelFamily,
    problem: AttentionProblem,
    config: KernelConfig,
    phases: Vec<KernelPhase>,
}

impl KernelModule {
    /// Build and fully validate a dense Q4 forward module.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] for invalid problems, configurations without
    /// an executable path, or resource accounting overflow.
    pub fn build(
        family: KernelFamily,
        problem: AttentionProblem,
        config: KernelConfig,
    ) -> Result<Self, KernelIrError> {
        // KernelFamily is a closed enumeration; only registered families with
        // a program builder may construct modules.
        let phases = match family {
            KernelFamily::DenseQ4Forward => Self::program(config),
        };
        problem.validate()?;
        config.validate_against(&problem)?;
        // Resources are part of validation: a module whose static footprint
        // cannot be computed safely must not exist.
        Self::checked_resources(&problem, &config)?;
        Ok(Self {
            ir_version: KernelIrVersion::CURRENT,
            family,
            problem,
            config,
            phases,
        })
    }

    /// Fixed ordered program of the dense Q4 forward architecture.
    fn program(config: KernelConfig) -> Vec<KernelPhase> {
        vec![
            KernelPhase::DispatchGuard,
            KernelPhase::InitOnlineSoftmaxState {
                rows: config.query_rows,
            },
            KernelPhase::StageQuery {
                width: config.vector_width,
            },
            KernelPhase::StreamKvTile {
                tile_rows: config.kv_tile_rows,
                width: config.vector_width,
                staging: config.kv_staging,
            },
            KernelPhase::ScorePartials {
                width: config.vector_width,
            },
            KernelPhase::ReduceScore {
                strategy: config.score_reduction,
            },
            KernelPhase::SoftmaxStateUpdate,
            KernelPhase::AccumulateValue {
                width: config.vector_width,
            },
            KernelPhase::NormalizeStoreOutput {
                width: config.vector_width,
            },
            KernelPhase::StoreLse,
        ]
    }

    /// Immutable ordered phase program.
    #[must_use]
    pub fn phases(&self) -> &[KernelPhase] {
        &self.phases
    }

    /// Semantic problem realized by this module.
    #[must_use]
    pub fn problem(&self) -> &AttentionProblem {
        &self.problem
    }

    /// Tuning configuration of this module.
    #[must_use]
    pub fn config(&self) -> &KernelConfig {
        &self.config
    }

    /// Kernel family of this module.
    #[must_use]
    pub fn family(&self) -> KernelFamily {
        self.family
    }

    /// IR schema version of this module.
    #[must_use]
    pub fn ir_version(&self) -> KernelIrVersion {
        self.ir_version
    }

    /// Static resource footprint computed at build time.
    #[must_use]
    pub fn resources(&self) -> KernelResources {
        Self::checked_resources(&self.problem, &self.config)
            .expect("validated module resources recompute infallibly")
    }

    /// Capability requirements imposed on the executing adapter, in
    /// deterministic order. Combines configuration-static requirements with
    /// problem-derived dispatch/output facts expressed as requirement entries.
    #[must_use]
    pub fn capability_requirements(&self) -> Vec<CapabilityRequirement> {
        // The static prefix is exactly KernelConfig::static_requirements();
        // the module adds nothing else today because output/dispatch facts
        // are carried by `resources()` instead of duplicate entries.
        self.config.static_requirements()
    }

    fn checked_resources(
        problem: &AttentionProblem,
        config: &KernelConfig,
    ) -> Result<KernelResources, KernelIrError> {
        let max_head_dim = u64::from(KERNEL_MAX_HEAD_DIM);
        let query_rows = u64::from(config.query_rows.get());
        let kv_tile = u64::from(config.kv_tile_rows.get());
        let kv_buffers = u64::from(config.kv_staging.buffers());
        let invocations = u64::from(config.workgroup_size.get());

        // f32 scalars staged in workgroup memory: query tile + K/V tiles
        // (+ second ping/pong pair when double buffered) + reduction scratch
        // + four per-row softmax state vectors. Mirrors the handwritten
        // declarations so generated variants stay under the same limits.
        let q_scalars = query_rows.checked_mul(max_head_dim);
        let kv_scalars = kv_buffers
            .checked_mul(2)
            .and_then(|v| v.checked_mul(kv_tile))
            .and_then(|v| v.checked_mul(max_head_dim));
        let reduce_scalars = query_rows.checked_mul(invocations);
        let softmax_state_scalars = 4_u64.checked_mul(query_rows);
        let total_scalars = q_scalars
            .zip(kv_scalars)
            .and_then(|(q, kv)| q.checked_add(kv))
            .zip(reduce_scalars)
            .and_then(|(acc, r)| acc.checked_add(r))
            .zip(softmax_state_scalars)
            .and_then(|(acc, s)| acc.checked_add(s))
            .ok_or(KernelIrError::WorkgroupStorageOverflow)?;
        let workgroup_storage_bytes = total_scalars
            .checked_mul(4)
            .ok_or(KernelIrError::WorkgroupStorageOverflow)?;

        let query_tiles_x = problem.seq_len.div_ceil(config.query_rows.get());
        let y = problem.batch_heads;

        Ok(KernelResources {
            workgroup_storage_bytes,
            invocations_per_workgroup: config.workgroup_size.get(),
            binding_entries: 5,
            dispatch_extents: [query_tiles_x, y, 1],
            output_elements: problem.output_elements()?,
        })
    }

    /// Deterministic canonical representation including the IR version.
    ///
    /// Equivalent modules produce byte-identical records regardless of how
    /// they were constructed.
    #[must_use]
    pub fn canonical_record(&self) -> String {
        format!(
            "ir=v{};family={};problem={};config={};phases={}",
            self.ir_version,
            self.family.canonical(),
            self.problem.canonical_record(),
            self.config.canonical_record(),
            self.phases
                .iter()
                .map(|phase| phase.tag())
                .collect::<Vec<_>>()
                .join(">"),
        )
    }

    /// Stable FNV-1a-64 structural fingerprint of [`Self::canonical_record`].
    ///
    /// Identifies structure for cache keys and equality acceleration; this is
    /// explicitly **not** cryptographic authentication and must never bypass
    /// structural validation.
    #[must_use]
    pub fn structural_fingerprint(&self) -> u64 {
        crate::fingerprint::fnv1a64(self.canonical_record().as_bytes())
    }
}

/// Errors raised while validating Kernel IR problems/configurations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelIrError {
    /// A semantic dimension was zero.
    ZeroDimension {
        /// Offending dimension name.
        field: &'static str,
    },
    /// A dimension exceeded the folded `u32` dispatch index space.
    IndexSpaceOverflow {
        /// Offending dimension name.
        field: &'static str,
    },
    /// Tensor element accounting exceeded addressable bounds.
    ElementCountOverflow {
        /// Offending tensor name.
        tensor: &'static str,
    },
    /// Head dimension has no executable path for the requested vector width.
    HeadDimUnsupportedByVectorWidth {
        /// Requested head dimension.
        head_dim: u32,
        /// Requested vector components.
        components: u32,
    },
    /// Head dimension exceeds the family's guarded maximum.
    HeadDimBeyondFamilyMaximum {
        /// Requested head dimension.
        actual: u32,
        /// Family maximum head dimension.
        maximum: u32,
    },
    /// Double buffering exists only on the vec4 realization.
    DoubleBufferingRequiresVectorPath,
    /// Workgroup-storage accounting overflowed checked bounds.
    WorkgroupStorageOverflow,
}

impl fmt::Display for KernelIrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { field } => {
                write!(f, "attention dimension {field} must be non-zero")
            }
            Self::IndexSpaceOverflow { field } => write!(
                f,
                "attention dimension {field} exceeds the u32 dispatch index space"
            ),
            Self::ElementCountOverflow { tensor } => {
                write!(f, "{tensor} element accounting exceeds addressable bounds")
            }
            Self::HeadDimUnsupportedByVectorWidth {
                head_dim,
                components,
            } => write!(
                f,
                "head_dim {head_dim} has no vec{components} kernel path; use 64 or 128"
            ),
            Self::HeadDimBeyondFamilyMaximum { actual, maximum } => {
                write!(f, "head_dim {actual} exceeds the family maximum {maximum}")
            }
            Self::DoubleBufferingRequiresVectorPath => write!(
                f,
                "double-buffered K/V staging requires the vec4 realization"
            ),
            Self::WorkgroupStorageOverflow => {
                write!(f, "workgroup storage accounting overflowed checked bounds")
            }
        }
    }
}

impl std::error::Error for KernelIrError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(head_dim: u32) -> AttentionShape {
        AttentionShape {
            batch: 2,
            heads: 4,
            seq_len: 129,
            head_dim: head_dim as usize,
        }
    }

    fn problem(head_dim: u32) -> AttentionProblem {
        AttentionProblem::from_shape(
            &shape(head_dim),
            FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn build_is_deterministic_and_equality_stable() {
        let a = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(64),
            KernelConfig::PORTABLE_SCALAR,
        )
        .unwrap();
        let b = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(64),
            KernelConfig::PORTABLE_SCALAR,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.canonical_record(), b.canonical_record());
        assert_eq!(a.structural_fingerprint(), b.structural_fingerprint());
        assert_eq!(a.ir_version(), KernelIrVersion::CURRENT);
    }

    #[test]
    fn fingerprint_is_version_sensitive() {
        let module = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(64),
            KernelConfig::PORTABLE_SCALAR,
        )
        .unwrap();
        let mut mutated = module.clone();
        mutated.ir_version = KernelIrVersion { major: 9, minor: 9 };
        assert_ne!(
            module.structural_fingerprint(),
            mutated.structural_fingerprint()
        );
    }

    #[test]
    fn fingerprint_tracks_every_meaningful_specialization() {
        let base = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(64),
            KernelConfig::PORTABLE_SCALAR,
        )
        .unwrap();
        for variant in [
            KernelConfig::PORTABLE_VEC4,
            KernelConfig::DOUBLE_BUFFERED_VEC4,
            KernelConfig::SUBGROUP_ASSISTED,
        ] {
            let other =
                KernelModule::build(KernelFamily::DenseQ4Forward, problem(64), variant).unwrap();
            assert_ne!(
                base.structural_fingerprint(),
                other.structural_fingerprint(),
                "variant must change identity: {variant:?}"
            );
        }
        let non_causal_shape = shape(64);
        let non_causal = AttentionProblem::from_shape(
            &non_causal_shape,
            FlatAttentionConfig {
                causal: false,
                softmax_scale: None,
            },
        )
        .unwrap();
        let other = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            non_causal,
            KernelConfig::PORTABLE_SCALAR,
        )
        .unwrap();
        assert_ne!(
            base.structural_fingerprint(),
            other.structural_fingerprint()
        );
    }

    #[test]
    fn from_shape_rejects_zero_and_overflowing_dimensions() {
        let mut zero = shape(64);
        zero.seq_len = 0;
        assert_eq!(
            AttentionProblem::from_shape(&zero, FlatAttentionConfig::default()).unwrap_err(),
            KernelIrError::ZeroDimension { field: "seq_len" }
        );
        let mut huge = shape(64);
        huge.seq_len = usize::MAX;
        assert_eq!(
            AttentionProblem::from_shape(&huge, FlatAttentionConfig::default()).unwrap_err(),
            KernelIrError::IndexSpaceOverflow { field: "seq_len" }
        );
        let mut folded = shape(64);
        folded.batch = usize::MAX;
        folded.heads = 8;
        assert_eq!(
            AttentionProblem::from_shape(&folded, FlatAttentionConfig::default()).unwrap_err(),
            KernelIrError::IndexSpaceOverflow {
                field: "batch_heads"
            }
        );
    }

    #[test]
    fn head_dims_beyond_family_maximum_are_rejected() {
        let mut wide = shape(64);
        wide.head_dim = 256;
        let wide_problem = AttentionProblem::from_shape(&wide, FlatAttentionConfig::default());
        assert_eq!(
            wide_problem.unwrap_err(),
            KernelIrError::HeadDimBeyondFamilyMaximum {
                actual: 256,
                maximum: KERNEL_MAX_HEAD_DIM,
            }
        );
    }

    #[test]
    fn vec4_width_requires_supported_head_dims() {
        for head_dim in [64, 128] {
            KernelModule::build(
                KernelFamily::DenseQ4Forward,
                problem(head_dim),
                KernelConfig::PORTABLE_VEC4,
            )
            .unwrap();
        }
        assert_eq!(
            KernelModule::build(
                KernelFamily::DenseQ4Forward,
                problem(80),
                KernelConfig::PORTABLE_VEC4
            )
            .unwrap_err(),
            KernelIrError::HeadDimUnsupportedByVectorWidth {
                head_dim: 80,
                components: 4,
            }
        );
    }

    #[test]
    fn double_buffering_requires_the_vector_path() {
        assert_eq!(
            KernelModule::build(
                KernelFamily::DenseQ4Forward,
                problem(64),
                KernelConfig {
                    kv_staging: KvStaging::DoubleBuffered,
                    ..KernelConfig::PORTABLE_SCALAR
                }
            )
            .unwrap_err(),
            KernelIrError::DoubleBufferingRequiresVectorPath
        );
    }

    #[test]
    fn subgroup_reduction_declares_subgroup_capability_requirement() {
        let tree = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(64),
            KernelConfig::PORTABLE_SCALAR,
        )
        .unwrap();
        assert!(!tree
            .capability_requirements()
            .contains(&CapabilityRequirement::SubgroupOperations));

        let subgroup = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(64),
            KernelConfig::SUBGROUP_ASSISTED,
        )
        .unwrap();
        assert!(subgroup
            .capability_requirements()
            .contains(&CapabilityRequirement::SubgroupOperations));
    }

    #[test]
    fn scalar_storage_footprint_matches_handwritten_arrays() {
        // flat_fwd.wgsl declares 512 + 1024 + 1024 + 256 + 4*4 f32 scalars.
        let module = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(64),
            KernelConfig::PORTABLE_SCALAR,
        )
        .unwrap();
        let expected_bytes = (512 + 1024 + 1024 + 256 + 16) * 4;
        assert_eq!(
            module.resources().workgroup_storage_bytes,
            expected_bytes as u64
        );
        assert_eq!(module.resources().invocations_per_workgroup, 64);
        assert_eq!(module.resources().binding_entries, 5);
    }

    #[test]
    fn double_buffered_storage_doubles_kv_tiles() {
        let single = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(128),
            KernelConfig::PORTABLE_VEC4,
        )
        .unwrap();
        let double = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(128),
            KernelConfig::DOUBLE_BUFFERED_VEC4,
        )
        .unwrap();
        let kv_tile_bytes = 2 * 8 * 128 * 4;
        assert_eq!(
            double.resources().workgroup_storage_bytes,
            single.resources().workgroup_storage_bytes + kv_tile_bytes as u64
        );
    }

    #[test]
    fn dispatch_extents_follow_query_tiling_and_batch_heads() {
        let module = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(64),
            KernelConfig::PORTABLE_SCALAR,
        )
        .unwrap();
        // ceil(129/4) = 33 tiles; y folds 2 batches x 4 heads.
        assert_eq!(module.resources().dispatch_extents, [33, 8, 1]);
        assert_eq!(module.resources().output_elements, 2 * 4 * (129 * 64 + 129));
    }

    #[test]
    fn packed_output_overflow_is_explicit_not_wrapped() {
        // O and LSE fit individually while their packed sum overflows:
        // with d=64, o = 64X and o+lse = 65X where X = batch_heads*seq_len.
        // X = 2^58 - 1 = (2^29+1)(2^29-1) keeps o at u64::MAX-63 while
        // 65X exceeds u64::MAX.
        let p = AttentionProblem {
            batch_heads: (1_u32 << 29) + 1,
            seq_len: (1_u32 << 29) - 1,
            head_dim: 64,
            causal: false,
        };
        assert!(p.tensor_elements().is_ok());
        assert!(p.output_elements().is_err());
        assert!(KernelModule::build(
            KernelFamily::DenseQ4Forward,
            p,
            KernelConfig::PORTABLE_SCALAR
        )
        .is_err());
    }

    #[test]
    fn program_phase_order_is_fixed_and_complete() {
        let module = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(64),
            KernelConfig::PORTABLE_SCALAR,
        )
        .unwrap();
        let tags: Vec<&'static str> = module.phases().iter().map(|p| p.tag()).collect();
        assert_eq!(
            tags,
            vec![
                "guard",
                "softmax_init",
                "stage_q",
                "stream_kv",
                "score_partials",
                "reduce_score",
                "softmax_update",
                "accumulate_v",
                "normalize_store",
                "store_lse",
            ]
        );
    }

    #[test]
    fn canonical_record_is_reproducible_text() {
        let module = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem(64),
            KernelConfig::SUBGROUP_ASSISTED,
        )
        .unwrap();
        let record = module.canonical_record();
        assert!(record.starts_with("ir=v1.0;family=dense_q4_forward;"));
        assert!(record.contains("red=subgroup"));
        assert_eq!(record, module.clone().canonical_record());
    }
}
