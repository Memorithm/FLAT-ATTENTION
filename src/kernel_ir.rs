//! FLAT Kernel IR foundation (roadmap M20, first slice).
//!
//! This module owns a deliberately small, attention-oriented intermediate
//! representation of the qualified Q4 fused-forward architecture. It separates
//! the two things the roadmap insists stay separate:
//!
//! - *semantic attention configuration* ([`AttentionProblem`]): what must be
//!   computed (logical geometry + causal mode);
//! - *device execution plan* ([`ExecutionPlan`]): how the kernel is realized
//!   (tiles, workgroup geometry, reduction strategy, precision policy).
//!
//! The IR makes illegal kernel descriptions constructible only through
//! validation: [`FlatKernelIr::build`] rejects zero tile dimensions,
//! impossible workgroup geometry, subgroup operations without a subgroup
//! capability requirement, invalid precision combinations, malformed
//! online-softmax state, and inconsistent program structure before any
//! shader generation happens.
//!
//! Everything in this module is host-only: no `wgpu` dependency, fully
//! deterministic, and fingerprinted with the repository's framed FNV-1a
//! discipline so identical configurations always produce identical
//! identities.

use std::fmt;

/// Schema version of this IR. Bumped on any semantic change to the
/// representation or its fingerprint layout.
pub const KERNEL_IR_SCHEMA_VERSION: u16 = 1;

/// Canonical namespace tag absorbed first by every kernel-IR fingerprint.
const KERNEL_IR_FINGERPRINT_DOMAIN: &str = "flat-attention/kernel-ir/v1";

/// Maximum head dimension representable by the Q4 kernel family. Matches
/// [`crate::WGSL_MAX_HEAD_DIM`]; the constant is restated here so the IR
/// stays independent of shader-source constants.
pub const KERNEL_MAX_HEAD_DIM: u32 = 128;

/// Semantic description of one logical attention computation.
///
/// This is the WHAT: batch/head folding follows the flat forward contract
/// (`x = batch_heads` workgroup rows), and every field is a mandatory,
/// validated fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionProblem {
    /// Folded batch × query-head count (`workgroup_id.y` domain).
    pub batch_heads: u32,
    /// Query/key/value token count.
    pub seq_len: u32,
    /// Feature width of every head row (1..=[`KERNEL_MAX_HEAD_DIM`]).
    pub head_dim: u32,
    /// Apply autoregressive masking (`key_position > query_position`).
    pub causal: bool,
}

impl AttentionProblem {
    /// Validate the semantic geometry.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when any dimension is zero or `head_dim`
    /// exceeds [`KERNEL_MAX_HEAD_DIM`].
    pub fn validate(&self) -> Result<(), KernelIrError> {
        if self.batch_heads == 0 || self.seq_len == 0 || self.head_dim == 0 {
            return Err(KernelIrError::ZeroDimension);
        }
        if self.head_dim > KERNEL_MAX_HEAD_DIM {
            return Err(KernelIrError::HeadDimBeyondFamilyMaximum {
                actual: self.head_dim,
                maximum: KERNEL_MAX_HEAD_DIM,
            });
        }
        Ok(())
    }

    /// Build a problem from the public MHA shape/config contract.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the shape overflows the folded `u32`
    /// index space or fails geometric validation.
    pub fn from_shape(
        shape: &crate::AttentionShape,
        config: crate::FlatAttentionConfig,
    ) -> Result<Self, KernelIrError> {
        shape.validate().map_err(|error| match error {
            crate::FlatAttentionError::ZeroDimension => KernelIrError::ZeroDimension,
            _ => KernelIrError::IndexSpaceOverflow,
        })?;
        let batch_heads = u32::try_from(
            shape
                .batch
                .checked_mul(shape.heads)
                .ok_or(KernelIrError::IndexSpaceOverflow)?,
        )
        .map_err(|_| KernelIrError::IndexSpaceOverflow)?;
        let seq_len =
            u32::try_from(shape.seq_len).map_err(|_| KernelIrError::IndexSpaceOverflow)?;
        let head_dim =
            u32::try_from(shape.head_dim).map_err(|_| KernelIrError::IndexSpaceOverflow)?;
        let problem = Self {
            batch_heads,
            seq_len,
            head_dim,
            causal: config.causal,
        };
        problem.validate()?;
        Ok(problem)
    }

    /// Deterministic canonical record used by fingerprints and evidence.
    #[must_use]
    pub fn canonical_record(&self) -> String {
        format!(
            "bh={};n={};d={};causal={}",
            self.batch_heads,
            self.seq_len,
            self.head_dim,
            u8::from(self.causal)
        )
    }

    /// Number of f32 elements in one Q/K/V tensor for this problem.
    #[must_use]
    pub const fn tensor_elements(&self) -> u64 {
        self.batch_heads as u64 * self.seq_len as u64 * self.head_dim as u64
    }

    /// Number of f32 elements in the LSE statistics for this problem.
    #[must_use]
    pub const fn lse_elements(&self) -> u64 {
        self.batch_heads as u64 * self.seq_len as u64
    }
}

impl fmt::Display for AttentionProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.canonical_record())
    }
}

/// Stable identity of a kernel variant within the FLAT namespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelVariantIdentity {
    /// Kernel family, e.g. `"flat.fwd.q4"`.
    pub family: &'static str,
    /// Variant within the family, e.g. `"portable"` or `"subgroup"`.
    pub variant: &'static str,
    /// IR schema version baked into the identity.
    pub schema_version: u16,
}

impl KernelVariantIdentity {
    /// Identity of the portable Q4 tiled family member.
    #[must_use]
    pub const fn portable_q4() -> Self {
        Self {
            family: "flat.fwd.q4",
            variant: "portable",
            schema_version: KERNEL_IR_SCHEMA_VERSION,
        }
    }

    /// Identity of the subgroup-assisted Q4 family member.
    #[must_use]
    pub const fn subgroup_q4() -> Self {
        Self {
            family: "flat.fwd.q4",
            variant: "subgroup",
            schema_version: KERNEL_IR_SCHEMA_VERSION,
        }
    }

    /// Deterministic canonical record.
    #[must_use]
    pub fn canonical_record(&self) -> String {
        format!("{}:{}@v{}", self.family, self.variant, self.schema_version)
    }
}

impl fmt::Display for KernelVariantIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.canonical_record())
    }
}

/// Tile geometry of the Q4 pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileConfig {
    /// Query rows sharing one K/V tile per workgroup.
    pub query_rows: u32,
    /// K/V rows staged in workgroup memory per tile iteration.
    pub kv_tile: u32,
}

impl TileConfig {
    fn validate(&self) -> Result<(), KernelIrError> {
        if self.query_rows == 0 || self.kv_tile == 0 {
            return Err(KernelIrError::ZeroTileDimension);
        }
        Ok(())
    }
}

/// Workgroup geometry of the Q4 pipeline (one-dimensional).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkgroupGeometry {
    /// Invocations along the workgroup X axis; Y and Z are 1.
    pub invocations: u32,
}

impl WorkgroupGeometry {
    fn validate(&self) -> Result<(), KernelIrError> {
        if self.invocations == 0 {
            return Err(KernelIrError::InvalidWorkgroupGeometry);
        }
        // The Q4 reduction tree and shared buffers assume a power-of-two,
        // family-bounded lane count.
        if self.invocations > 1024 || !self.invocations.is_power_of_two() {
            return Err(KernelIrError::InvalidWorkgroupGeometry);
        }
        Ok(())
    }
}

/// Dot-product reduction strategy of the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionStrategy {
    /// Portable deterministic tree over workgroup memory. Always legal.
    TreeInWorkgroup,
    /// Subgroup-assisted reduction followed by a tree over subgroup totals.
    /// Requires the subgroup capability requirement.
    SubgroupArithmetic,
}

/// Precision policy of the plan.
///
/// Only the qualified FP32-storage/FP32-accumulate policy has executable
/// coverage today. The packed-f16 policy is representable because its
/// capability obligation is expressible; declaring it does not imply any
/// executable candidate exists (none is emitted yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionPolicy {
    /// FP32 storage, FP32 accumulation, FP32 LSE.
    F32StorageF32Accumulate,
    /// Packed binary16 inputs with FP32 accumulation. Requires native
    /// shader-f16 support from the boundary.
    PackedF16StorageF32Accumulate,
}

impl PrecisionPolicy {
    /// The capability obligation this policy imposes.
    #[must_use]
    pub const fn requires_native_f16(self) -> bool {
        matches!(self, Self::PackedF16StorageF32Accumulate)
    }
}

/// One structural operation of the Q4 pipeline program.
///
/// The set intentionally covers exactly what the qualified architecture
/// executes: tile staging, score computation, reductions, online-softmax
/// state transitions, output accumulation, normalization/storage, and the
/// barriers between them. It is not a universal GPU IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrOp {
    /// Coalesce-load up to `query_rows` Q rows into workgroup memory.
    StageQueryTile,
    /// Initialize running max/sum/alpha/p state for every query row.
    InitSoftmaxState,
    /// Full workgroup barrier.
    Barrier,
    /// Begin of the per-K/V-tile loop body.
    BeginKvTileLoop,
    /// Stage one K/V tile into workgroup memory.
    StageKvTile,
    /// Compute partial dot products `q_row · k_tile_row` into the reduce buffer.
    ComputeScorePartials,
    /// Portable tree reduction of score partials over workgroup memory.
    TreeReducePartials,
    /// Subgroup-arithmetic reduction of score partials.
    SubgroupReducePartials,
    /// Update online-softmax state (`max`, `alpha`, `p`, running sum).
    SoftmaxRescale,
    /// Accumulate `acc = acc * alpha + p * v_row` for every query row.
    AccumulateOutputRow,
    /// End of the per-K/V-tile loop body.
    EndKvTileLoop,
    /// Normalize accumulated output by the final sum and store O and LSE.
    FinalNormalizeAndStore,
}

/// Ordered structural program of one Q4 realization.
///
/// Constructed exclusively through [`ExecutionPlan::program_for`], which
/// appends the canonical op sequence with correct barrier placement, making
/// malformed programs (scores before staging, stores without a final barrier,
/// subgroup operations in a portable plan) unrepresentable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExecutionProgram {
    ops: Vec<IrOp>,
}

impl ExecutionProgram {
    /// The structural operations in execution order.
    #[must_use]
    pub fn ops(&self) -> &[IrOp] {
        &self.ops
    }

    fn build(reduction: ReductionStrategy) -> Self {
        use IrOp::*;
        let reduction_op = match reduction {
            ReductionStrategy::TreeInWorkgroup => TreeReducePartials,
            ReductionStrategy::SubgroupArithmetic => SubgroupReducePartials,
        };
        // Canonical Q4 structure: query staging + softmax-state prologue,
        // then a K/V-tile loop of {stage, barrier, score partials, barrier,
        // reduce, softmax rescale, barrier, output accumulate, barrier}, and
        // a final normalize/store epilogue.
        let ops = vec![
            StageQueryTile,
            InitSoftmaxState,
            Barrier,
            BeginKvTileLoop,
            StageKvTile,
            Barrier,
            ComputeScorePartials,
            Barrier,
            reduction_op,
            SoftmaxRescale,
            Barrier,
            AccumulateOutputRow,
            Barrier,
            EndKvTileLoop,
            FinalNormalizeAndStore,
        ];
        Self { ops }
    }

    fn validate(&self, reduction: ReductionStrategy) -> Result<(), KernelIrError> {
        let ops = &self.ops;
        let mut softmax_initialized = false;
        let mut staged_query = false;
        let mut staged_kv_since_barrier = false;
        let mut computed_partials = false;
        let mut reduced = false;
        let mut loop_depth = 0usize;
        for (index, op) in ops.iter().enumerate() {
            match op {
                IrOp::StageQueryTile => staged_query = true,
                IrOp::InitSoftmaxState => softmax_initialized = true,
                IrOp::Barrier => {
                    // Barriers synchronize access to already-staged data;
                    // they do not invalidate what is staged.
                }
                IrOp::BeginKvTileLoop => {
                    loop_depth += 1;
                    if loop_depth > 1 {
                        return Err(KernelIrError::MalformedProgramStructure);
                    }
                }
                IrOp::EndKvTileLoop => {
                    loop_depth = loop_depth.saturating_sub(1);
                }
                IrOp::StageKvTile => staged_kv_since_barrier = true,
                IrOp::ComputeScorePartials => {
                    if !staged_query || !softmax_initialized || !staged_kv_since_barrier {
                        return Err(KernelIrError::MalformedProgramStructure);
                    }
                    computed_partials = true;
                }
                IrOp::TreeReducePartials | IrOp::SubgroupReducePartials => {
                    if !computed_partials {
                        return Err(KernelIrError::MalformedProgramStructure);
                    }
                    if *op == IrOp::SubgroupReducePartials
                        && reduction != ReductionStrategy::SubgroupArithmetic
                    {
                        return Err(KernelIrError::SubgroupOperationWithoutRequirement);
                    }
                    reduced = true;
                }
                IrOp::SoftmaxRescale => {
                    if !reduced {
                        return Err(KernelIrError::MalformedProgramStructure);
                    }
                    reduced = false;
                    computed_partials = false;
                }
                IrOp::AccumulateOutputRow => {
                    if !staged_kv_since_barrier {
                        return Err(KernelIrError::MalformedProgramStructure);
                    }
                }
                IrOp::FinalNormalizeAndStore => {
                    if !softmax_initialized || index != ops.len() - 1 {
                        return Err(KernelIrError::MalformedProgramStructure);
                    }
                }
            }
        }
        if loop_depth != 0 {
            return Err(KernelIrError::MalformedProgramStructure);
        }
        Ok(())
    }
}

/// Device-tuning half of the IR: how the Q4 architecture is realized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    tiles: TileConfig,
    workgroup: WorkgroupGeometry,
    reduction: ReductionStrategy,
    precision: PrecisionPolicy,
    program: ExecutionProgram,
}

impl ExecutionPlan {
    /// Assemble and internally validate an execution plan.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when tile/workgroup geometry is invalid,
    /// when a subgroup reduction lacks its capability requirement, when a
    /// precision policy violates its obligations, or when the derived
    /// program is malformed.
    pub fn build(
        tiles: TileConfig,
        workgroup: WorkgroupGeometry,
        reduction: ReductionStrategy,
        precision: PrecisionPolicy,
        requires_subgroup: bool,
    ) -> Result<Self, KernelIrError> {
        tiles.validate()?;
        workgroup.validate()?;
        if reduction == ReductionStrategy::SubgroupArithmetic && !requires_subgroup {
            return Err(KernelIrError::SubgroupOperationWithoutRequirement);
        }
        if precision.requires_native_f16() && reduction == ReductionStrategy::SubgroupArithmetic {
            // No combined f16+subgroup architecture is qualified; refusing
            // the combination keeps the IR honest about what exists.
            return Err(KernelIrError::UnsupportedPrecisionCombination);
        }
        let program = ExecutionProgram::build(reduction);
        program.validate(reduction)?;
        Ok(Self {
            tiles,
            workgroup,
            reduction,
            precision,
            program,
        })
    }

    /// Tile configuration.
    #[must_use]
    pub const fn tiles(&self) -> TileConfig {
        self.tiles
    }

    /// Workgroup geometry.
    #[must_use]
    pub const fn workgroup(&self) -> WorkgroupGeometry {
        self.workgroup
    }

    /// Reduction strategy.
    #[must_use]
    pub const fn reduction(&self) -> ReductionStrategy {
        self.reduction
    }

    /// Precision policy.
    #[must_use]
    pub const fn precision(&self) -> PrecisionPolicy {
        self.precision
    }

    /// Structural program.
    #[must_use]
    pub const fn program(&self) -> &ExecutionProgram {
        &self.program
    }

    /// Whether the plan declares a subgroup capability requirement.
    #[must_use]
    pub const fn requires_subgroup(&self) -> bool {
        matches!(self.reduction, ReductionStrategy::SubgroupArithmetic)
    }

    /// Static workgroup-memory footprint in bytes implied by the geometry.
    ///
    /// Checked arithmetic throughout; overflow reports
    /// [`KernelIrError::StorageAccountingOverflow`].
    pub fn workgroup_storage_bytes(&self) -> Result<u64, KernelIrError> {
        let stride = u64::from(KERNEL_MAX_HEAD_DIM);
        let query_rows = u64::from(self.tiles.query_rows);
        let kv_tile = u64::from(self.tiles.kv_tile);
        let lanes = u64::from(self.workgroup.invocations);
        let q_shared = query_rows
            .checked_mul(stride)
            .ok_or(KernelIrError::StorageAccountingOverflow)?;
        let kv_shared = kv_tile
            .checked_mul(stride)
            .and_then(|rows| rows.checked_mul(2))
            .ok_or(KernelIrError::StorageAccountingOverflow)?;
        let reduce_shared = query_rows
            .checked_mul(lanes)
            .ok_or(KernelIrError::StorageAccountingOverflow)?;
        let softmax_state = query_rows.saturating_mul(4);
        let elements = q_shared
            .checked_add(kv_shared)
            .and_then(|v| v.checked_add(reduce_shared))
            .and_then(|v| v.checked_add(softmax_state))
            .ok_or(KernelIrError::StorageAccountingOverflow)?;
        elements
            .checked_mul(4)
            .ok_or(KernelIrError::StorageAccountingOverflow)
    }
}

/// Complete normalized description of one FLAT kernel realization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatKernelIr {
    identity: KernelVariantIdentity,
    problem: AttentionProblem,
    plan: ExecutionPlan,
}

impl FlatKernelIr {
    /// Assemble and validate one kernel IR.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when any component fails validation.
    pub fn build(
        identity: KernelVariantIdentity,
        problem: AttentionProblem,
        plan: ExecutionPlan,
    ) -> Result<Self, KernelIrError> {
        if identity.schema_version != KERNEL_IR_SCHEMA_VERSION {
            return Err(KernelIrError::SchemaVersionMismatch {
                actual: identity.schema_version,
                expected: KERNEL_IR_SCHEMA_VERSION,
            });
        }
        problem.validate()?;
        Ok(Self {
            identity,
            problem,
            plan,
        })
    }

    /// Variant identity.
    #[must_use]
    pub const fn identity(&self) -> &KernelVariantIdentity {
        &self.identity
    }

    /// Semantic attention problem.
    #[must_use]
    pub const fn problem(&self) -> &AttentionProblem {
        &self.problem
    }

    /// Device execution plan.
    #[must_use]
    pub const fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Deterministic structural fingerprint.
    ///
    /// Identical configurations always produce identical fingerprints; no
    /// hash-iteration order, memory address, or wall-clock value participates.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut hash = Fnv1a::new();
        hash.update(KERNEL_IR_FINGERPRINT_DOMAIN.as_bytes());
        hash.update(b"\0");
        hash.update(format!("{}\n", self.identity.canonical_record()).as_bytes());
        hash.update(self.problem.canonical_record().as_bytes());
        hash.update(b"\n");
        let plan = &self.plan;
        let record = format!(
            "qr={};kvt={};wg={};red={};prec={}",
            plan.tiles.query_rows,
            plan.tiles.kv_tile,
            plan.workgroup.invocations,
            match plan.reduction {
                ReductionStrategy::TreeInWorkgroup => "tree",
                ReductionStrategy::SubgroupArithmetic => "subgroup",
            },
            match plan.precision {
                PrecisionPolicy::F32StorageF32Accumulate => "f32f32",
                PrecisionPolicy::PackedF16StorageF32Accumulate => "f16acc-f32",
            },
        );
        hash.update(record.as_bytes());
        for op in plan.program.ops() {
            hash.update(b"|");
            hash.update(std::format!("{op:?}").as_bytes());
        }
        hash.finish()
    }
}

/// Errors produced while building or validating kernel IRs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelIrError {
    /// A mandatory dimension was zero.
    ZeroDimension,
    /// A tile dimension was zero.
    ZeroTileDimension,
    /// Head dimension exceeds the family maximum.
    HeadDimBeyondFamilyMaximum {
        /// Offending head dimension.
        actual: u32,
        /// Family maximum.
        maximum: u32,
    },
    /// Geometry could not be represented in the folded u32 index space.
    IndexSpaceOverflow,
    /// Workgroup geometry was invalid (zero, non-power-of-two, or beyond the
    /// family bound).
    InvalidWorkgroupGeometry,
    /// A subgroup-dependent operation appeared without a subgroup capability
    /// requirement.
    SubgroupOperationWithoutRequirement,
    /// A precision combination with no qualified architecture was requested.
    UnsupportedPrecisionCombination,
    /// Online-softmax state or program structure was malformed.
    MalformedProgramStructure,
    /// Static storage accounting overflowed.
    StorageAccountingOverflow,
    /// The identity carried a schema version other than
    /// [`KERNEL_IR_SCHEMA_VERSION`].
    SchemaVersionMismatch {
        /// Offending version.
        actual: u16,
        /// Supported version.
        expected: u16,
    },
}

impl fmt::Display for KernelIrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "attention dimensions must be non-zero"),
            Self::ZeroTileDimension => write!(f, "tile dimensions must be non-zero"),
            Self::HeadDimBeyondFamilyMaximum { actual, maximum } => write!(
                f,
                "head_dim {actual} exceeds the Q4 family maximum {maximum}"
            ),
            Self::IndexSpaceOverflow => {
                write!(f, "problem geometry overflows the folded u32 index space")
            }
            Self::InvalidWorkgroupGeometry => write!(
                f,
                "workgroup geometry must be a positive power of two within the family bound"
            ),
            Self::SubgroupOperationWithoutRequirement => write!(
                f,
                "subgroup reduction present without a subgroup capability requirement"
            ),
            Self::UnsupportedPrecisionCombination => write!(
                f,
                "precision/reduction combination has no qualified architecture"
            ),
            Self::MalformedProgramStructure => {
                write!(f, "kernel program structure is malformed")
            }
            Self::StorageAccountingOverflow => {
                write!(f, "static storage accounting overflowed")
            }
            Self::SchemaVersionMismatch { actual, expected } => write!(
                f,
                "kernel identity schema v{actual} does not match IR schema v{expected}"
            ),
        }
    }
}

impl std::error::Error for KernelIrError {}

/// Framed FNV-1a-64 used for deterministic IR fingerprints. Mirrors the
/// repository convention in `runtime_telemetry`.
struct Fnv1a(u64);

impl Fnv1a {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_problem() -> AttentionProblem {
        AttentionProblem {
            batch_heads: 8,
            seq_len: 128,
            head_dim: 64,
            causal: true,
        }
    }

    fn canonical_tiles() -> TileConfig {
        TileConfig {
            query_rows: 4,
            kv_tile: 8,
        }
    }

    fn canonical_plan(reduction: ReductionStrategy) -> ExecutionPlan {
        ExecutionPlan::build(
            canonical_tiles(),
            WorkgroupGeometry { invocations: 64 },
            reduction,
            PrecisionPolicy::F32StorageF32Accumulate,
            reduction == ReductionStrategy::SubgroupArithmetic,
        )
        .expect("canonical plan builds")
    }

    #[test]
    fn canonical_ir_builds_and_fingerprints_deterministically() {
        let ir = FlatKernelIr::build(
            KernelVariantIdentity::portable_q4(),
            canonical_problem(),
            canonical_plan(ReductionStrategy::TreeInWorkgroup),
        )
        .expect("canonical IR builds");
        assert_eq!(ir.fingerprint(), ir.fingerprint());

        let rebuilt = FlatKernelIr::build(
            KernelVariantIdentity::portable_q4(),
            canonical_problem(),
            canonical_plan(ReductionStrategy::TreeInWorkgroup),
        )
        .expect("rebuilt");
        assert_eq!(ir.fingerprint(), rebuilt.fingerprint());
    }

    #[test]
    fn fingerprint_distinguishes_semantics_and_tuning() {
        let base = FlatKernelIr::build(
            KernelVariantIdentity::portable_q4(),
            canonical_problem(),
            canonical_plan(ReductionStrategy::TreeInWorkgroup),
        )
        .expect("base");

        let mut other_problem = canonical_problem();
        other_problem.seq_len += 1;
        let changed_problem = FlatKernelIr::build(
            KernelVariantIdentity::portable_q4(),
            other_problem,
            canonical_plan(ReductionStrategy::TreeInWorkgroup),
        )
        .expect("changed problem");
        assert_ne!(base.fingerprint(), changed_problem.fingerprint());

        let subgroup = FlatKernelIr::build(
            KernelVariantIdentity::subgroup_q4(),
            canonical_problem(),
            canonical_plan(ReductionStrategy::SubgroupArithmetic),
        )
        .expect("subgroup");
        assert_ne!(base.fingerprint(), subgroup.fingerprint());
    }

    #[test]
    fn invalid_geometry_is_rejected_before_generation() {
        assert_eq!(
            AttentionProblem {
                head_dim: 0,
                ..canonical_problem()
            }
            .validate(),
            Err(KernelIrError::ZeroDimension)
        );
        assert_eq!(
            AttentionProblem {
                head_dim: 144,
                ..canonical_problem()
            }
            .validate(),
            Err(KernelIrError::HeadDimBeyondFamilyMaximum {
                actual: 144,
                maximum: 128,
            })
        );
        assert_eq!(
            TileConfig {
                query_rows: 0,
                kv_tile: 8
            }
            .validate(),
            Err(KernelIrError::ZeroTileDimension)
        );
        assert_eq!(
            ExecutionPlan::build(
                canonical_tiles(),
                WorkgroupGeometry { invocations: 63 },
                ReductionStrategy::TreeInWorkgroup,
                PrecisionPolicy::F32StorageF32Accumulate,
                false,
            ),
            Err(KernelIrError::InvalidWorkgroupGeometry)
        );
    }

    #[test]
    fn subgroup_reduction_requires_the_capability_requirement() {
        assert_eq!(
            ExecutionPlan::build(
                canonical_tiles(),
                WorkgroupGeometry { invocations: 64 },
                ReductionStrategy::SubgroupArithmetic,
                PrecisionPolicy::F32StorageF32Accumulate,
                false,
            ),
            Err(KernelIrError::SubgroupOperationWithoutRequirement)
        );
        assert!(ExecutionPlan::build(
            canonical_tiles(),
            WorkgroupGeometry { invocations: 64 },
            ReductionStrategy::SubgroupArithmetic,
            PrecisionPolicy::F32StorageF32Accumulate,
            true,
        )
        .is_ok());
    }

    #[test]
    fn program_structure_is_loop_consistent_and_validated() {
        let plan = canonical_plan(ReductionStrategy::TreeInWorkgroup);
        let ops = plan.program().ops();
        assert_eq!(ops.first(), Some(&IrOp::StageQueryTile));
        assert_eq!(ops.last(), Some(&IrOp::FinalNormalizeAndStore));
        assert!(ops.contains(&IrOp::BeginKvTileLoop));
        assert!(ops.contains(&IrOp::EndKvTileLoop));
        // Validation passes implicitly through build(); a hand-mangled
        // program must fail.
        let mangled = ExecutionProgram {
            ops: vec![IrOp::ComputeScorePartials, IrOp::FinalNormalizeAndStore],
        };
        assert_eq!(
            mangled.validate(ReductionStrategy::TreeInWorkgroup),
            Err(KernelIrError::MalformedProgramStructure)
        );
    }

    #[test]
    fn storage_accounting_matches_the_static_layout() {
        let plan = canonical_plan(ReductionStrategy::TreeInWorkgroup);
        // q_shared 4*128 + k/v shared 2*(8*128) + reduce 4*64 + state 4*4
        // = 512 + 2048 + 256 + 16 = 2832 f32 elements = 11328 bytes.
        assert_eq!(plan.workgroup_storage_bytes(), Ok(11_328));
    }

    #[test]
    fn from_shape_preserves_semantics_and_rejects_overflow() {
        let shape = crate::AttentionShape {
            batch: 2,
            heads: 4,
            seq_len: 33,
            head_dim: 80,
        };
        let problem = AttentionProblem::from_shape(
            &shape,
            crate::FlatAttentionConfig {
                causal: false,
                softmax_scale: None,
            },
        )
        .expect("converts");
        assert_eq!(problem.batch_heads, 8);
        assert_eq!(problem.seq_len, 33);
        assert_eq!(problem.head_dim, 80);
        assert!(!problem.causal);
        assert_eq!(problem.tensor_elements(), 8 * 33 * 80);

        let huge = crate::AttentionShape {
            batch: usize::MAX,
            heads: 1,
            seq_len: 1,
            head_dim: 1,
        };
        assert_eq!(
            AttentionProblem::from_shape(&huge, crate::FlatAttentionConfig::default()),
            Err(KernelIrError::IndexSpaceOverflow)
        );
    }
}
