//! Fail-closed ADA graduation import and scalar FLAT parity gate.
//!
//! This crate is an integration boundary, not a new semantic oracle. ADA owns
//! graduation decoding, qualification provenance, exact f64 replay, and fixture
//! identity. FLAT only accepts the narrow StandardSoftmax subset that its
//! current scalar reference can execute, explicitly narrows exact ADA f64 Q/K/V
//! inputs to f32, and checks that the FLAT scalar result remains within a
//! caller-declared integration tolerance of ADA's f64 semantic result.
//!
//! The FLAT parity tolerance is deliberately separate from every ADA
//! qualification tolerance. Passing this gate is finite-corpus integration
//! evidence only; it does not alter ADA's verdict, qualify a GPU kernel, mutate
//! production routing, or make a performance claim.

#![forbid(unsafe_code)]

use std::fmt::{Display, Formatter};

use ada_graduation::FlatGraduationBundle;
use ada_replay::{decode_graduation_fixtures, verify_ada_reference_replay};
use ada_semantic::{
    AffinityRule, InputTransform, MaskRule, OutputRule, SelectionRule, ValueMixRule, WeightRule,
};
use ada_workload::{AttentionTopology, HeadGrouping};
use flat_attention::{AttentionShape, FlatAttentionConfig, forward_reference};

/// Caller-owned numerical allowance for the explicit ADA-f64 to FLAT-f32
/// integration comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatParityConfig {
    /// Maximum absolute difference between the FLAT f32 scalar oracle and the
    /// ADA f64 semantic evaluator on one retained replay fixture.
    pub max_abs_difference: f64,
}

impl FlatParityConfig {
    /// Construct a finite, non-negative integration tolerance.
    ///
    /// # Errors
    ///
    /// Rejects NaN, infinity, and negative values.
    pub fn new(max_abs_difference: f64) -> Result<Self, GraduationImportError> {
        if !max_abs_difference.is_finite() || max_abs_difference < 0.0 {
            return Err(GraduationImportError::InvalidParityTolerance);
        }
        Ok(Self { max_abs_difference })
    }
}

/// Successful ADA-to-FLAT finite-corpus replay summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatParityReport {
    fixture_count: usize,
    ada_worst_max_abs_error: f64,
    flat_worst_max_abs_difference: f64,
    flat_max_abs_difference_allowed: f64,
}

impl FlatParityReport {
    /// Number of exact retained ADA fixtures replayed by both reference paths.
    #[must_use]
    pub const fn fixture_count(self) -> usize {
        self.fixture_count
    }

    /// Worst ADA semantic-vs-independent-truth error from ADA's exact f64
    /// replay. This is reported, not reinterpreted by FLAT.
    #[must_use]
    pub const fn ada_worst_max_abs_error(self) -> f64 {
        self.ada_worst_max_abs_error
    }

    /// Worst FLAT-f32-vs-ADA-f64 semantic difference across retained fixtures.
    #[must_use]
    pub const fn flat_worst_max_abs_difference(self) -> f64 {
        self.flat_worst_max_abs_difference
    }

    /// Caller-declared FLAT integration allowance used for this report.
    #[must_use]
    pub const fn flat_max_abs_difference_allowed(self) -> f64 {
        self.flat_max_abs_difference_allowed
    }
}

/// Canonically decoded ADA graduation artifact that has passed both ADA exact
/// replay and the narrow FLAT scalar parity gate.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedGraduation {
    bundle: FlatGraduationBundle,
    report: FlatParityReport,
}

impl ImportedGraduation {
    /// Fully revalidated ADA graduation bundle.
    #[must_use]
    pub const fn bundle(&self) -> &FlatGraduationBundle {
        &self.bundle
    }

    /// Finite-corpus scalar parity report.
    #[must_use]
    pub const fn report(&self) -> FlatParityReport {
        self.report
    }
}

/// Typed fail-closed import failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraduationImportError {
    /// Caller supplied an invalid FLAT-only integration tolerance.
    InvalidParityTolerance,
    /// ADA rejected the graduation artifact or one of its nested contracts.
    InvalidGraduation(String),
    /// ADA's exact replay layer rejected retained fixture identity or oracle
    /// replay.
    AdaReplay(String),
    /// The semantic is valid ADA research output but not the StandardSoftmax
    /// subset currently executable by FLAT's scalar reference bridge.
    UnsupportedSemantic(String),
    /// The workload is valid ADA research output but outside the current FLAT
    /// scalar bridge domain.
    UnsupportedWorkload(String),
    /// One finite f64 value overflows while narrowing to FLAT's f32 reference
    /// input domain.
    F32NarrowingOverflow {
        /// Tensor or scalar being narrowed.
        field: &'static str,
        /// Element index for tensors, zero for scalar fields.
        index: usize,
        /// Exact source f64 IEEE-754 bits.
        source_bits: u64,
    },
    /// FLAT's scalar reference rejected an input after the bridge had validated
    /// the ADA contract.
    FlatReference(String),
    /// FLAT f32 and ADA f64 semantic outputs disagree beyond the explicit
    /// integration tolerance.
    FlatParityMismatch {
        /// Retained CEGIS fixture identifier.
        fixture_id: String,
        /// Exact f64 bits of the measured maximum absolute difference.
        max_abs_difference_bits: u64,
        /// Exact f64 bits of the caller-declared integration tolerance.
        tolerance_bits: u64,
    },
}

impl Display for GraduationImportError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParityTolerance => formatter
                .write_str("FLAT parity tolerance must be finite and non-negative"),
            Self::InvalidGraduation(reason) => write!(formatter, "invalid ADA graduation: {reason}"),
            Self::AdaReplay(reason) => write!(formatter, "ADA replay failure: {reason}"),
            Self::UnsupportedSemantic(reason) => {
                write!(formatter, "unsupported ADA semantic for current FLAT bridge: {reason}")
            }
            Self::UnsupportedWorkload(reason) => {
                write!(formatter, "unsupported ADA workload for current FLAT bridge: {reason}")
            }
            Self::F32NarrowingOverflow {
                field,
                index,
                source_bits,
            } => write!(
                formatter,
                "{field}[{index}] overflows f32 during ADA-to-FLAT narrowing: {source_bits:016x}"
            ),
            Self::FlatReference(reason) => write!(formatter, "FLAT scalar reference failure: {reason}"),
            Self::FlatParityMismatch {
                fixture_id,
                max_abs_difference_bits,
                tolerance_bits,
            } => write!(
                formatter,
                "fixture {fixture_id} exceeds FLAT integration parity tolerance: difference_bits={max_abs_difference_bits:016x};tolerance_bits={tolerance_bits:016x}"
            ),
        }
    }
}

impl std::error::Error for GraduationImportError {}

/// Decode a canonical ADA graduation artifact, require exact ADA replay, then
/// compare the current FLAT scalar StandardSoftmax reference against ADA's f64
/// semantic evaluator on every retained replayable fixture.
///
/// # Errors
///
/// Fails closed on malformed/tampered ADA artifacts, opaque legacy fixtures,
/// unsupported semantics/workloads, f32 narrowing overflow, FLAT reference
/// rejection, or parity outside the caller-declared integration tolerance.
pub fn import_and_verify(
    canonical_bundle: &str,
    parity: FlatParityConfig,
) -> Result<ImportedGraduation, GraduationImportError> {
    if !parity.max_abs_difference.is_finite() || parity.max_abs_difference < 0.0 {
        return Err(GraduationImportError::InvalidParityTolerance);
    }

    let bundle = FlatGraduationBundle::from_canonical_text(canonical_bundle)
        .map_err(|error| GraduationImportError::InvalidGraduation(error.to_string()))?;
    validate_bridge_contract(&bundle)?;

    let ada_report = verify_ada_reference_replay(&bundle)
        .map_err(|error| GraduationImportError::AdaReplay(error.to_string()))?;
    let fixtures = decode_graduation_fixtures(&bundle)
        .map_err(|error| GraduationImportError::AdaReplay(error.to_string()))?;

    let scale = match bundle.semantic().affinity() {
        AffinityRule::ScaledDotProduct { scale } => narrow_scalar("softmax_scale", scale)?,
    };
    let causal = matches!(bundle.semantic().mask(), MaskRule::Causal);
    let mut flat_worst_max_abs_difference = 0.0_f64;

    for fixture in &fixtures {
        let replay_input = fixture.input();
        let reference_input = replay_input
            .to_reference_input()
            .map_err(|error| GraduationImportError::AdaReplay(error.to_string()))?;
        let ada_output = bundle
            .semantic()
            .evaluate(&reference_input)
            .map_err(|error| GraduationImportError::AdaReplay(error.to_string()))?;

        let q = narrow_tensor("Q", replay_input.queries())?;
        let k = narrow_tensor("K", replay_input.keys())?;
        let v = narrow_tensor("V", replay_input.values())?;
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: replay_input.query_count(),
            head_dim: replay_input.q_dimension(),
        };
        let flat_output = forward_reference(
            &q,
            &k,
            &v,
            shape,
            FlatAttentionConfig {
                causal,
                softmax_scale: Some(scale),
            },
        )
        .map_err(|error| GraduationImportError::FlatReference(error.to_string()))?;

        if flat_output.output.len() != ada_output.output().len() {
            return Err(GraduationImportError::UnsupportedWorkload(
                "FLAT and ADA output element counts differ".into(),
            ));
        }
        let max_abs_difference = flat_output
            .output
            .iter()
            .zip(ada_output.output())
            .map(|(&flat, &ada)| (f64::from(flat) - ada).abs())
            .fold(0.0_f64, f64::max);
        if max_abs_difference > parity.max_abs_difference {
            return Err(GraduationImportError::FlatParityMismatch {
                fixture_id: fixture.id().to_owned(),
                max_abs_difference_bits: max_abs_difference.to_bits(),
                tolerance_bits: parity.max_abs_difference.to_bits(),
            });
        }
        flat_worst_max_abs_difference =
            flat_worst_max_abs_difference.max(max_abs_difference);
    }

    let report = FlatParityReport {
        fixture_count: fixtures.len(),
        ada_worst_max_abs_error: ada_report.worst_max_abs_error(),
        flat_worst_max_abs_difference,
        flat_max_abs_difference_allowed: parity.max_abs_difference,
    };
    Ok(ImportedGraduation { bundle, report })
}

fn validate_bridge_contract(bundle: &FlatGraduationBundle) -> Result<(), GraduationImportError> {
    let semantic = bundle.semantic();
    if semantic.input_transform() != InputTransform::Identity {
        return Err(GraduationImportError::UnsupportedSemantic(
            "input transform must be identity".into(),
        ));
    }
    if semantic.selection() != SelectionRule::All {
        return Err(GraduationImportError::UnsupportedSemantic(
            "selection must retain all visible keys".into(),
        ));
    }
    if semantic.weight() != WeightRule::Softmax {
        return Err(GraduationImportError::UnsupportedSemantic(
            "weighting must be ordinary softmax".into(),
        ));
    }
    if semantic.value_mix() != ValueMixRule::WeightedSum || semantic.output() != OutputRule::Identity
    {
        return Err(GraduationImportError::UnsupportedSemantic(
            "value mixing/output must be weighted-sum plus identity".into(),
        ));
    }
    if !matches!(semantic.mask(), MaskRule::Unmasked | MaskRule::Causal) {
        return Err(GraduationImportError::UnsupportedSemantic(
            "only unmasked or causal visibility is executable".into(),
        ));
    }

    semantic
        .validate_for_workload(bundle.workload())
        .map_err(|error| GraduationImportError::UnsupportedWorkload(error.to_string()))?;

    let geometry = bundle.workload().geometry();
    if geometry.sequence_lengths().batch_count() != 1
        || geometry.query_heads() != 1
        || geometry.kv_heads() != 1
        || geometry.head_grouping() != HeadGrouping::MultiHead
    {
        return Err(GraduationImportError::UnsupportedWorkload(
            "current bridge is single-batch, single-head MHA".into(),
        ));
    }
    if geometry.topology() != AttentionTopology::SelfAttention {
        return Err(GraduationImportError::UnsupportedWorkload(
            "current FLAT scalar bridge requires self-attention".into(),
        ));
    }
    let query_length = geometry
        .sequence_lengths()
        .query_length_for(0)
        .ok_or_else(|| GraduationImportError::UnsupportedWorkload("missing query length".into()))?;
    let kv_length = geometry
        .sequence_lengths()
        .kv_length_for(0)
        .ok_or_else(|| GraduationImportError::UnsupportedWorkload("missing KV length".into()))?;
    if query_length != kv_length {
        return Err(GraduationImportError::UnsupportedWorkload(
            "current FLAT scalar bridge requires square Q/KV sequence lengths".into(),
        ));
    }
    let qk_dimension = geometry.qk_dimension().ok_or_else(|| {
        GraduationImportError::UnsupportedWorkload("explicit Q/K dimension is required".into())
    })?;
    if qk_dimension != geometry.value_dimension() {
        return Err(GraduationImportError::UnsupportedWorkload(
            "current FLAT scalar bridge requires Q/K and V dimensions to match".into(),
        ));
    }
    Ok(())
}

fn narrow_scalar(field: &'static str, value: f64) -> Result<f32, GraduationImportError> {
    let narrowed = value as f32;
    if !narrowed.is_finite() || narrowed <= 0.0 {
        return Err(GraduationImportError::F32NarrowingOverflow {
            field,
            index: 0,
            source_bits: value.to_bits(),
        });
    }
    Ok(narrowed)
}

fn narrow_tensor(field: &'static str, values: &[f64]) -> Result<Vec<f32>, GraduationImportError> {
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let narrowed = value as f32;
            if !narrowed.is_finite() {
                Err(GraduationImportError::F32NarrowingOverflow {
                    field,
                    index,
                    source_bits: value.to_bits(),
                })
            } else {
                Ok(narrowed)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
