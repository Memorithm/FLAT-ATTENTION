//! Deterministic research-only DA-LUC precision-tier routing (FDAL5).
//!
//! This module decides only which explicitly declared representation tier is
//! assigned to each logical token segment. It does not transcode a payload,
//! mutate cache storage, move pages, or select a runtime backend. A consumer
//! must apply every representation transition explicitly.

use super::super::research_da_luc::{
    DalucKeyRepresentation, DalucKvViewContract, DalucKvViewError, DalucValueRepresentation,
};
use core::fmt;

/// Version of the deterministic FDAL5 tier-routing semantics.
pub const DA_LUC_TIER_ROUTING_VERSION: u16 = 1;

/// Stable caller-defined tier identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DalucTierId(pub u16);

/// One explicit K/V representation tier.
///
/// The order of the tier slice supplied to routing functions is the selection
/// priority. No name such as "hot", "cold", "high" or "low" is inferred from
/// bit width, dtype or this identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucPrecisionTier {
    pub id: DalucTierId,
    pub keys: DalucKeyRepresentation,
    pub values: DalucValueRepresentation,
}

/// Exact number of logical segments assigned to one tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucTierQuota {
    pub tier_id: DalucTierId,
    pub segments: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucTierRoutingPolicy {
    /// Newest logical segment first, with higher segment index considered newer.
    Recency,
    /// Descending caller-supplied non-negative attention mass. Equal masses use
    /// lower segment index as the deterministic tie-break.
    AttentionMass,
}

/// Canonical assignment of one logical token segment to one declared tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucTierAssignment {
    pub segment_index: usize,
    pub start_token: usize,
    pub end_token_exclusive: usize,
    pub tier_id: DalucTierId,
}

/// Versioned deterministic routing result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DalucTierRoutingPlan {
    pub routing_version: u16,
    pub kv_view_schema_version: u16,
    pub policy: DalucTierRoutingPolicy,
    pub kv_len: usize,
    pub segment_size: usize,
    pub assignments: Vec<DalucTierAssignment>,
}

/// Explicit representation transition between two already-materialized plans.
///
/// Returning this record does not perform the transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DalucTierTransition {
    pub segment_index: usize,
    pub start_token: usize,
    pub end_token_exclusive: usize,
    pub from_tier: DalucTierId,
    pub to_tier: DalucTierId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DalucTierRoutingError {
    Contract(DalucKvViewError),
    UnsupportedRoutingVersion {
        actual: u16,
        supported: u16,
    },
    InvalidSegmentSize,
    EmptyTierCatalog,
    DuplicateTierId(DalucTierId),
    DuplicateQuotaTier(DalucTierId),
    UnknownQuotaTier(DalucTierId),
    MissingQuotaTier(DalucTierId),
    QuotaSumMismatch {
        expected_segments: usize,
        actual_segments: usize,
    },
    AttentionMassLength {
        expected: usize,
        actual: usize,
    },
    InvalidAttentionMass {
        segment_index: usize,
    },
    MalformedPlan(&'static str),
    IncompatiblePlans(&'static str),
    ArithmeticOverflow(&'static str),
}

impl fmt::Display for DalucTierRoutingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(formatter, "{error}"),
            Self::UnsupportedRoutingVersion { actual, supported } => write!(
                formatter,
                "DA-LUC tier routing version {actual} is unsupported; expected {supported}"
            ),
            Self::InvalidSegmentSize => {
                write!(formatter, "DA-LUC tier segment size must be non-zero")
            }
            Self::EmptyTierCatalog => write!(formatter, "DA-LUC tier catalog must not be empty"),
            Self::DuplicateTierId(id) => write!(formatter, "duplicate DA-LUC tier id {}", id.0),
            Self::DuplicateQuotaTier(id) => {
                write!(formatter, "duplicate DA-LUC tier quota for id {}", id.0)
            }
            Self::UnknownQuotaTier(id) => {
                write!(
                    formatter,
                    "DA-LUC tier quota references unknown id {}",
                    id.0
                )
            }
            Self::MissingQuotaTier(id) => {
                write!(formatter, "DA-LUC tier id {} has no explicit quota", id.0)
            }
            Self::QuotaSumMismatch {
                expected_segments,
                actual_segments,
            } => write!(
                formatter,
                "DA-LUC tier quotas cover {actual_segments} segments; expected {expected_segments}"
            ),
            Self::AttentionMassLength { expected, actual } => write!(
                formatter,
                "DA-LUC attention-mass evidence has {actual} segments; expected {expected}"
            ),
            Self::InvalidAttentionMass { segment_index } => write!(
                formatter,
                "DA-LUC attention mass at segment {segment_index} must be finite and non-negative"
            ),
            Self::MalformedPlan(reason) => {
                write!(formatter, "malformed DA-LUC tier plan: {reason}")
            }
            Self::IncompatiblePlans(reason) => {
                write!(formatter, "incompatible DA-LUC tier plans: {reason}")
            }
            Self::ArithmeticOverflow(label) => {
                write!(
                    formatter,
                    "DA-LUC tier routing arithmetic overflow: {label}"
                )
            }
        }
    }
}

impl std::error::Error for DalucTierRoutingError {}

impl From<DalucKvViewError> for DalucTierRoutingError {
    fn from(value: DalucKvViewError) -> Self {
        Self::Contract(value)
    }
}

/// Route segments by deterministic recency.
///
/// `tiers` is ordered from first-selected to last-selected representation.
/// `quotas` must mention every tier exactly once and must cover every segment
/// exactly. Higher segment indices are newer. No representation is materialized
/// or mutated by this function.
pub fn route_by_recency(
    base_contract: DalucKvViewContract,
    segment_size: usize,
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
) -> Result<DalucTierRoutingPlan, DalucTierRoutingError> {
    let validated = validate_routing_inputs(base_contract, segment_size, tiers, quotas)?;
    let ranking = (0..validated.segment_count).rev().collect::<Vec<_>>();
    build_plan(
        base_contract,
        segment_size,
        tiers,
        quotas,
        DalucTierRoutingPolicy::Recency,
        &ranking,
    )
}

/// Route segments by deterministic attention-mass evidence.
///
/// Evidence is one finite, non-negative scalar per logical segment. Larger mass
/// ranks first; equal mass deterministically prefers the lower segment index.
/// The evidence is caller supplied and is not interpreted as a physical traffic
/// or performance measurement.
pub fn route_by_attention_mass(
    base_contract: DalucKvViewContract,
    segment_size: usize,
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
    attention_mass: &[f64],
) -> Result<DalucTierRoutingPlan, DalucTierRoutingError> {
    let validated = validate_routing_inputs(base_contract, segment_size, tiers, quotas)?;
    if attention_mass.len() != validated.segment_count {
        return Err(DalucTierRoutingError::AttentionMassLength {
            expected: validated.segment_count,
            actual: attention_mass.len(),
        });
    }
    if let Some((segment_index, _)) = attention_mass
        .iter()
        .enumerate()
        .find(|(_, mass)| !mass.is_finite() || **mass < 0.0)
    {
        return Err(DalucTierRoutingError::InvalidAttentionMass { segment_index });
    }

    let mut ranking = (0..validated.segment_count).collect::<Vec<_>>();
    ranking.sort_by(|left, right| {
        attention_mass[*right]
            .total_cmp(&attention_mass[*left])
            .then(left.cmp(right))
    });
    build_plan(
        base_contract,
        segment_size,
        tiers,
        quotas,
        DalucTierRoutingPolicy::AttentionMass,
        &ranking,
    )
}

impl DalucTierRoutingPlan {
    /// Validate this plan against the base logical KV contract and tier catalog.
    pub fn validate_against(
        &self,
        base_contract: DalucKvViewContract,
        tiers: &[DalucPrecisionTier],
    ) -> Result<(), DalucTierRoutingError> {
        if self.routing_version != DA_LUC_TIER_ROUTING_VERSION {
            return Err(DalucTierRoutingError::UnsupportedRoutingVersion {
                actual: self.routing_version,
                supported: DA_LUC_TIER_ROUTING_VERSION,
            });
        }
        base_contract.validate()?;
        validate_tiers(base_contract, tiers)?;
        if self.kv_view_schema_version != base_contract.schema_version {
            return Err(DalucTierRoutingError::MalformedPlan(
                "KV view schema version does not match base contract",
            ));
        }
        if self.kv_len != base_contract.shape.kv_len {
            return Err(DalucTierRoutingError::MalformedPlan(
                "KV length does not match base contract",
            ));
        }
        if self.segment_size == 0 {
            return Err(DalucTierRoutingError::InvalidSegmentSize);
        }
        let expected_segments = segment_count(self.kv_len, self.segment_size)?;
        if self.assignments.len() != expected_segments {
            return Err(DalucTierRoutingError::MalformedPlan(
                "assignment count does not cover every logical segment",
            ));
        }
        for (expected_index, assignment) in self.assignments.iter().enumerate() {
            if assignment.segment_index != expected_index {
                return Err(DalucTierRoutingError::MalformedPlan(
                    "assignments are not in canonical segment order",
                ));
            }
            let (start, end) = segment_bounds(self.kv_len, self.segment_size, expected_index)?;
            if assignment.start_token != start || assignment.end_token_exclusive != end {
                return Err(DalucTierRoutingError::MalformedPlan(
                    "assignment token bounds are not canonical",
                ));
            }
            if !tiers.iter().any(|tier| tier.id == assignment.tier_id) {
                return Err(DalucTierRoutingError::MalformedPlan(
                    "assignment references unknown tier",
                ));
            }
        }
        Ok(())
    }

    /// Return every representation change required to move from `previous` to
    /// this plan. The caller must apply these transitions explicitly; this
    /// method never mutates payloads or cache state.
    pub fn transitions_from(
        &self,
        previous: &Self,
    ) -> Result<Vec<DalucTierTransition>, DalucTierRoutingError> {
        if self.routing_version != previous.routing_version {
            return Err(DalucTierRoutingError::IncompatiblePlans(
                "routing versions differ",
            ));
        }
        if self.kv_view_schema_version != previous.kv_view_schema_version {
            return Err(DalucTierRoutingError::IncompatiblePlans(
                "KV view schema versions differ",
            ));
        }
        if self.kv_len != previous.kv_len || self.segment_size != previous.segment_size {
            return Err(DalucTierRoutingError::IncompatiblePlans(
                "logical segmentation differs",
            ));
        }
        if self.assignments.len() != previous.assignments.len() {
            return Err(DalucTierRoutingError::IncompatiblePlans(
                "assignment counts differ",
            ));
        }

        let mut transitions = Vec::new();
        for (next, prior) in self.assignments.iter().zip(&previous.assignments) {
            if next.segment_index != prior.segment_index
                || next.start_token != prior.start_token
                || next.end_token_exclusive != prior.end_token_exclusive
            {
                return Err(DalucTierRoutingError::IncompatiblePlans(
                    "canonical segment bounds differ",
                ));
            }
            if next.tier_id != prior.tier_id {
                transitions.push(DalucTierTransition {
                    segment_index: next.segment_index,
                    start_token: next.start_token,
                    end_token_exclusive: next.end_token_exclusive,
                    from_tier: prior.tier_id,
                    to_tier: next.tier_id,
                });
            }
        }
        Ok(transitions)
    }
}

struct ValidatedRoutingInputs {
    segment_count: usize,
}

fn validate_routing_inputs(
    base_contract: DalucKvViewContract,
    segment_size: usize,
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
) -> Result<ValidatedRoutingInputs, DalucTierRoutingError> {
    base_contract.validate()?;
    if segment_size == 0 {
        return Err(DalucTierRoutingError::InvalidSegmentSize);
    }
    validate_tiers(base_contract, tiers)?;
    let segment_count = segment_count(base_contract.shape.kv_len, segment_size)?;
    validate_quotas(tiers, quotas, segment_count)?;
    Ok(ValidatedRoutingInputs { segment_count })
}

fn validate_tiers(
    base_contract: DalucKvViewContract,
    tiers: &[DalucPrecisionTier],
) -> Result<(), DalucTierRoutingError> {
    if tiers.is_empty() {
        return Err(DalucTierRoutingError::EmptyTierCatalog);
    }
    for (index, tier) in tiers.iter().enumerate() {
        if tiers[..index].iter().any(|prior| prior.id == tier.id) {
            return Err(DalucTierRoutingError::DuplicateTierId(tier.id));
        }
        let mut tier_contract = base_contract;
        tier_contract.keys = tier.keys;
        tier_contract.values = tier.values;
        tier_contract.validate()?;
    }
    Ok(())
}

fn validate_quotas(
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
    segment_count: usize,
) -> Result<(), DalucTierRoutingError> {
    for (index, quota) in quotas.iter().enumerate() {
        if quotas[..index]
            .iter()
            .any(|prior| prior.tier_id == quota.tier_id)
        {
            return Err(DalucTierRoutingError::DuplicateQuotaTier(quota.tier_id));
        }
        if !tiers.iter().any(|tier| tier.id == quota.tier_id) {
            return Err(DalucTierRoutingError::UnknownQuotaTier(quota.tier_id));
        }
    }
    for tier in tiers {
        if !quotas.iter().any(|quota| quota.tier_id == tier.id) {
            return Err(DalucTierRoutingError::MissingQuotaTier(tier.id));
        }
    }
    let actual_segments = quotas.iter().try_fold(0usize, |sum, quota| {
        sum.checked_add(quota.segments)
            .ok_or(DalucTierRoutingError::ArithmeticOverflow("tier quota sum"))
    })?;
    if actual_segments != segment_count {
        return Err(DalucTierRoutingError::QuotaSumMismatch {
            expected_segments: segment_count,
            actual_segments,
        });
    }
    Ok(())
}

fn build_plan(
    base_contract: DalucKvViewContract,
    segment_size: usize,
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
    policy: DalucTierRoutingPolicy,
    ranking: &[usize],
) -> Result<DalucTierRoutingPlan, DalucTierRoutingError> {
    let count = segment_count(base_contract.shape.kv_len, segment_size)?;
    if ranking.len() != count {
        return Err(DalucTierRoutingError::MalformedPlan(
            "ranking does not cover every segment",
        ));
    }
    let mut assigned = vec![None; count];
    let mut cursor = 0usize;
    for tier in tiers {
        let quota = quotas
            .iter()
            .find(|quota| quota.tier_id == tier.id)
            .ok_or(DalucTierRoutingError::MissingQuotaTier(tier.id))?;
        let end =
            cursor
                .checked_add(quota.segments)
                .ok_or(DalucTierRoutingError::ArithmeticOverflow(
                    "tier ranking cursor",
                ))?;
        if end > ranking.len() {
            return Err(DalucTierRoutingError::MalformedPlan(
                "tier quota exceeds ranking length",
            ));
        }
        for &segment_index in &ranking[cursor..end] {
            if segment_index >= count || assigned[segment_index].is_some() {
                return Err(DalucTierRoutingError::MalformedPlan(
                    "ranking contains invalid or duplicate segment",
                ));
            }
            assigned[segment_index] = Some(tier.id);
        }
        cursor = end;
    }
    if cursor != ranking.len() || assigned.iter().any(Option::is_none) {
        return Err(DalucTierRoutingError::MalformedPlan(
            "routing left a segment unassigned",
        ));
    }

    let mut assignments = Vec::with_capacity(count);
    for (segment_index, tier_id) in assigned.into_iter().enumerate() {
        let (start_token, end_token_exclusive) =
            segment_bounds(base_contract.shape.kv_len, segment_size, segment_index)?;
        assignments.push(DalucTierAssignment {
            segment_index,
            start_token,
            end_token_exclusive,
            tier_id: tier_id.expect("validated complete assignment"),
        });
    }
    let plan = DalucTierRoutingPlan {
        routing_version: DA_LUC_TIER_ROUTING_VERSION,
        kv_view_schema_version: base_contract.schema_version,
        policy,
        kv_len: base_contract.shape.kv_len,
        segment_size,
        assignments,
    };
    plan.validate_against(base_contract, tiers)?;
    Ok(plan)
}

fn segment_count(kv_len: usize, segment_size: usize) -> Result<usize, DalucTierRoutingError> {
    if segment_size == 0 {
        return Err(DalucTierRoutingError::InvalidSegmentSize);
    }
    kv_len
        .checked_add(segment_size - 1)
        .map(|adjusted| adjusted / segment_size)
        .ok_or(DalucTierRoutingError::ArithmeticOverflow("segment count"))
}

fn segment_bounds(
    kv_len: usize,
    segment_size: usize,
    segment_index: usize,
) -> Result<(usize, usize), DalucTierRoutingError> {
    let start = segment_index
        .checked_mul(segment_size)
        .ok_or(DalucTierRoutingError::ArithmeticOverflow("segment start"))?;
    if start >= kv_len {
        return Err(DalucTierRoutingError::MalformedPlan(
            "segment starts outside logical KV length",
        ));
    }
    let end = start
        .checked_add(segment_size)
        .ok_or(DalucTierRoutingError::ArithmeticOverflow("segment end"))?
        .min(kv_len);
    Ok((start, end))
}
