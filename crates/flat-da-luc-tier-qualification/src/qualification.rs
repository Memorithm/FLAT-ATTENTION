use core::fmt;
use std::collections::BTreeSet;

use flat_attention::api::research_da_luc::{
    DalucCodebookScope, DalucFloatDType, DalucKvViewContract, DalucStorageTopology,
};
use flat_attention::api::research_da_luc_oracle::tiering::{
    route_by_attention_mass, route_by_recency, DalucPrecisionTier, DalucTierAssignment,
    DalucTierId, DalucTierQuota, DalucTierRoutingError, DalucTierRoutingPlan,
    DalucTierRoutingPolicy,
};
use flat_attention::api::research_da_luc_oracle::{
    DalucOracleError, DalucOracleErrorStats, DalucOraclePayload, DalucOracleReconstructionReport,
    DalucOracleStorageReport,
};
use flat_attention::FlatAttentionConfig;

pub const DA_LUC_TIER_QUALIFICATION_VERSION: u16 = 1;
pub const DA_LUC_RANDOM_CONTROL_VERSION: u16 = 1;
pub const DA_LUC_FIXED_CONTROL_VERSION: u16 = 1;
pub const DA_LUC_CODEBOOK_OWNERSHIP_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy)]
pub struct DalucTierMaterializationSpec<'a> {
    pub tier: DalucPrecisionTier,
    pub codebook: &'a [f32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DalucTierStorageOverhead {
    pub shared_metadata_bytes: usize,
    pub segment_metadata_bytes_per_segment: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalucTierBaselineControl {
    Recency,
    AttentionMass,
    DeterministicRandom { version: u16, seed: u64 },
    Fixed { version: u16 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DalucTierBaselinePlan {
    control: DalucTierBaselineControl,
    kv_view_schema_version: u16,
    kv_len: usize,
    segment_size: usize,
    assignments: Vec<DalucTierAssignment>,
    routed: Option<DalucTierRoutingPlan>,
}

impl DalucTierBaselinePlan {
    #[must_use]
    pub const fn control(&self) -> DalucTierBaselineControl {
        self.control
    }

    #[must_use]
    pub fn assignments(&self) -> &[DalucTierAssignment] {
        &self.assignments
    }

    #[must_use]
    pub const fn kv_len(&self) -> usize {
        self.kv_len
    }

    #[must_use]
    pub const fn segment_size(&self) -> usize {
        self.segment_size
    }

    #[must_use]
    pub fn routed_plan(&self) -> Option<&DalucTierRoutingPlan> {
        self.routed.as_ref()
    }

    pub fn validate(
        &self,
        base_contract: DalucKvViewContract,
        tiers: &[DalucPrecisionTier],
        quotas: &[DalucTierQuota],
    ) -> Result<(), DalucTierQualificationError> {
        base_contract.validate().map_err(DalucOracleError::from)?;
        validate_tier_catalog(base_contract, self.segment_size, tiers, quotas)?;
        if self.kv_view_schema_version != base_contract.schema_version {
            return Err(DalucTierQualificationError::MalformedPlan(
                "KV view schema version does not match base contract",
            ));
        }
        if self.kv_len != base_contract.shape.kv_len {
            return Err(DalucTierQualificationError::MalformedPlan(
                "KV length does not match base contract",
            ));
        }
        if self.segment_size == 0 {
            return Err(DalucTierQualificationError::MalformedPlan(
                "segment size must be non-zero",
            ));
        }
        let expected_segments = segment_count(self.kv_len, self.segment_size)?;
        if self.assignments.len() != expected_segments {
            return Err(DalucTierQualificationError::MalformedPlan(
                "assignment count does not cover every logical segment",
            ));
        }
        for (expected_index, assignment) in self.assignments.iter().enumerate() {
            if assignment.segment_index != expected_index {
                return Err(DalucTierQualificationError::MalformedPlan(
                    "missing, duplicate, or non-canonical segment index",
                ));
            }
            let (start, end) = segment_bounds(self.kv_len, self.segment_size, expected_index)?;
            if assignment.start_token != start || assignment.end_token_exclusive != end {
                return Err(DalucTierQualificationError::MalformedPlan(
                    "segment bounds are not canonical",
                ));
            }
            if !tiers.iter().any(|tier| tier.id == assignment.tier_id) {
                return Err(DalucTierQualificationError::UnknownTier(
                    assignment.tier_id,
                ));
            }
        }
        validate_quota_counts(&self.assignments, tiers, quotas)?;
        if let Some(routed) = &self.routed {
            routed.validate_against(base_contract, tiers)?;
            if routed.assignments != self.assignments {
                return Err(DalucTierQualificationError::MalformedPlan(
                    "retained FDAL5 routing plan disagrees with baseline assignments",
                ));
            }
            let expected = match routed.policy {
                DalucTierRoutingPolicy::Recency => DalucTierBaselineControl::Recency,
                DalucTierRoutingPolicy::AttentionMass => DalucTierBaselineControl::AttentionMass,
            };
            if self.control != expected {
                return Err(DalucTierQualificationError::MalformedPlan(
                    "baseline control does not match retained FDAL5 routing policy",
                ));
            }
        }
        Ok(())
    }

    pub fn fixed_from_assignments(
        base_contract: DalucKvViewContract,
        segment_size: usize,
        tiers: &[DalucPrecisionTier],
        quotas: &[DalucTierQuota],
        assignments: Vec<DalucTierAssignment>,
    ) -> Result<Self, DalucTierQualificationError> {
        validate_tier_catalog(base_contract, segment_size, tiers, quotas)?;
        let plan = Self {
            control: DalucTierBaselineControl::Fixed {
                version: DA_LUC_FIXED_CONTROL_VERSION,
            },
            kv_view_schema_version: base_contract.schema_version,
            kv_len: base_contract.shape.kv_len,
            segment_size,
            assignments,
            routed: None,
        };
        plan.validate(base_contract, tiers, quotas)?;
        Ok(plan)
    }
}

pub fn recency_baseline(
    base_contract: DalucKvViewContract,
    segment_size: usize,
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
) -> Result<DalucTierBaselinePlan, DalucTierQualificationError> {
    let routed = route_by_recency(base_contract, segment_size, tiers, quotas)?;
    Ok(DalucTierBaselinePlan {
        control: DalucTierBaselineControl::Recency,
        kv_view_schema_version: routed.kv_view_schema_version,
        kv_len: routed.kv_len,
        segment_size: routed.segment_size,
        assignments: routed.assignments.clone(),
        routed: Some(routed),
    })
}

pub fn attention_mass_baseline(
    base_contract: DalucKvViewContract,
    segment_size: usize,
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
    attention_mass: &[f64],
) -> Result<DalucTierBaselinePlan, DalucTierQualificationError> {
    let routed = route_by_attention_mass(
        base_contract,
        segment_size,
        tiers,
        quotas,
        attention_mass,
    )?;
    Ok(DalucTierBaselinePlan {
        control: DalucTierBaselineControl::AttentionMass,
        kv_view_schema_version: routed.kv_view_schema_version,
        kv_len: routed.kv_len,
        segment_size: routed.segment_size,
        assignments: routed.assignments.clone(),
        routed: Some(routed),
    })
}

pub fn deterministic_random_baseline(
    base_contract: DalucKvViewContract,
    segment_size: usize,
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
    version: u16,
    seed: u64,
) -> Result<DalucTierBaselinePlan, DalucTierQualificationError> {
    if version != DA_LUC_RANDOM_CONTROL_VERSION {
        return Err(DalucTierQualificationError::UnsupportedRandomVersion {
            actual: version,
            supported: DA_LUC_RANDOM_CONTROL_VERSION,
        });
    }
    let canonical = route_by_recency(base_contract, segment_size, tiers, quotas)?;
    let mut ranking = (0..canonical.assignments.len()).collect::<Vec<_>>();
    deterministic_shuffle_v1(&mut ranking, seed);
    let assignments = assignments_from_ranking(&canonical, tiers, quotas, &ranking)?;
    let plan = DalucTierBaselinePlan {
        control: DalucTierBaselineControl::DeterministicRandom { version, seed },
        kv_view_schema_version: canonical.kv_view_schema_version,
        kv_len: canonical.kv_len,
        segment_size: canonical.segment_size,
        assignments,
        routed: None,
    };
    plan.validate(base_contract, tiers, quotas)?;
    Ok(plan)
}

#[derive(Debug, Clone, Copy)]
pub struct DalucTierQualificationFixture<'a> {
    pub base_contract: DalucKvViewContract,
    pub tiers: &'a [DalucTierMaterializationSpec<'a>],
    pub quotas: &'a [DalucTierQuota],
    pub dense_keys: &'a [f32],
    pub dense_values: &'a [f32],
    pub query: &'a [f32],
    pub attention: FlatAttentionConfig,
    pub query_position: usize,
    pub storage_overhead: DalucTierStorageOverhead,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DalucTierSegmentEvidence {
    pub assignment: DalucTierAssignment,
    pub payload_storage: DalucOracleStorageReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DalucTierCompositeStorageReport {
    pub codebook_ownership_version: u16,
    pub logical_kv_scalar_count: usize,
    pub key_codebook_payload_bytes: usize,
    pub key_index_payload_bytes: usize,
    pub key_residual_value_payload_bytes: usize,
    pub key_residual_index_payload_bytes: usize,
    pub value_payload_bytes: usize,
    pub value_scale_payload_bytes: usize,
    pub value_zero_point_payload_bytes: usize,
    pub value_residual_value_payload_bytes: usize,
    pub value_residual_index_payload_bytes: usize,
    pub page_metadata_payload_bytes: usize,
    pub packing_tail_padding_bits: usize,
    pub alignment_padding_bytes: usize,
    pub shared_metadata_bytes: usize,
    pub segment_metadata_bytes: usize,
    pub total_representation_bytes: usize,
    pub effective_bits_per_value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DalucTierQualificationReport {
    pub qualification_version: u16,
    pub control: DalucTierBaselineControl,
    pub assignments: Vec<DalucTierAssignment>,
    pub segment_evidence: Vec<DalucTierSegmentEvidence>,
    pub storage: DalucTierCompositeStorageReport,
    pub reconstruction: DalucOracleReconstructionReport,
    pub q_len1_output_error: DalucOracleErrorStats,
    pub q_len1_lse_error: DalucOracleErrorStats,
}

pub fn qualify_baseline(
    fixture: DalucTierQualificationFixture<'_>,
    plan: &DalucTierBaselinePlan,
) -> Result<DalucTierQualificationReport, DalucTierQualificationError> {
    validate_fixture(fixture, plan.segment_size)?;
    let tier_descriptors = fixture
        .tiers
        .iter()
        .map(|spec| spec.tier)
        .collect::<Vec<_>>();
    plan.validate(fixture.base_contract, &tier_descriptors, fixture.quotas)?;

    let mut reconstructed_keys = vec![0.0f32; logical_side_len(fixture.base_contract, true)?];
    let mut reconstructed_values = vec![0.0f32; logical_side_len(fixture.base_contract, false)?];
    let mut segment_evidence = Vec::with_capacity(plan.assignments.len());
    let mut storage = CompositeStorageAccumulator::new(
        logical_kv_scalar_count(fixture.base_contract)?,
        fixture.storage_overhead,
        plan.assignments.len(),
    )?;
    let mut charged_codebooks = BTreeSet::new();

    for assignment in &plan.assignments {
        let spec = fixture
            .tiers
            .iter()
            .find(|spec| spec.tier.id == assignment.tier_id)
            .ok_or(DalucTierQualificationError::MissingTierMaterialization(
                assignment.tier_id,
            ))?;
        validate_codebook(fixture.base_contract, spec)?;
        let segment_contract = segment_contract(
            fixture.base_contract,
            spec.tier,
            assignment.start_token,
            assignment.end_token_exclusive,
        )?;
        let segment_keys = extract_segment(
            fixture.base_contract,
            fixture.dense_keys,
            assignment.start_token,
            assignment.end_token_exclusive,
            true,
        )?;
        let segment_values = extract_segment(
            fixture.base_contract,
            fixture.dense_values,
            assignment.start_token,
            assignment.end_token_exclusive,
            false,
        )?;
        let payload = DalucOraclePayload::encode(
            segment_contract,
            spec.codebook,
            &segment_keys,
            &segment_values,
        )?;
        let payload_storage = payload.storage_report(DalucFloatDType::F32, 0)?;
        let decoded_keys = payload.decode_keys()?;
        let decoded_values = payload.decode_values()?;
        insert_segment(
            fixture.base_contract,
            &mut reconstructed_keys,
            &decoded_keys,
            assignment.start_token,
            assignment.end_token_exclusive,
            true,
        )?;
        insert_segment(
            fixture.base_contract,
            &mut reconstructed_values,
            &decoded_values,
            assignment.start_token,
            assignment.end_token_exclusive,
            false,
        )?;
        storage.accumulate(
            &payload,
            &payload_storage,
            charged_codebooks.insert(assignment.tier_id),
        )?;
        segment_evidence.push(DalucTierSegmentEvidence {
            assignment: *assignment,
            payload_storage,
        });
    }

    let dense_reference = q_len1_dense_reference(
        fixture.base_contract,
        fixture.query,
        fixture.dense_keys,
        fixture.dense_values,
        fixture.attention,
        fixture.query_position,
    )?;
    let reconstructed_reference = q_len1_dense_reference(
        fixture.base_contract,
        fixture.query,
        &reconstructed_keys,
        &reconstructed_values,
        fixture.attention,
        fixture.query_position,
    )?;

    Ok(DalucTierQualificationReport {
        qualification_version: DA_LUC_TIER_QUALIFICATION_VERSION,
        control: plan.control,
        assignments: plan.assignments.clone(),
        segment_evidence,
        storage: storage.finish()?,
        reconstruction: DalucOracleReconstructionReport {
            keys: error_stats(fixture.dense_keys, &reconstructed_keys),
            values: error_stats(fixture.dense_values, &reconstructed_values),
        },
        q_len1_output_error: error_stats(&dense_reference.output, &reconstructed_reference.output),
        q_len1_lse_error: error_stats(&dense_reference.lse, &reconstructed_reference.lse),
    })
}

pub fn qualify_equal_budget(
    fixture: DalucTierQualificationFixture<'_>,
    plans: &[DalucTierBaselinePlan],
) -> Result<Vec<DalucTierQualificationReport>, DalucTierQualificationError> {
    if plans.is_empty() {
        return Err(DalucTierQualificationError::InvalidFixture(
            "at least one baseline plan is required",
        ));
    }
    let tiers = fixture
        .tiers
        .iter()
        .map(|spec| spec.tier)
        .collect::<Vec<_>>();
    let mut reports = Vec::with_capacity(plans.len());
    let mut reference_signature: Option<Vec<(DalucTierId, usize)>> = None;
    let mut reference_bytes = None;

    for (index, plan) in plans.iter().enumerate() {
        plan.validate(fixture.base_contract, &tiers, fixture.quotas)?;
        let signature = quota_signature(plan.assignments(), &tiers);
        let report = qualify_baseline(fixture, plan)?;
        if let Some(expected) = &reference_signature {
            if expected != &signature {
                return Err(DalucTierQualificationError::UnequalTierBudget {
                    candidate_index: index,
                });
            }
        } else {
            reference_signature = Some(signature);
        }
        if let Some(expected) = reference_bytes {
            if expected != report.storage.total_representation_bytes {
                return Err(DalucTierQualificationError::UnequalStorageBudget {
                    candidate_index: index,
                    expected_bytes: expected,
                    actual_bytes: report.storage.total_representation_bytes,
                });
            }
        } else {
            reference_bytes = Some(report.storage.total_representation_bytes);
        }
        reports.push(report);
    }
    Ok(reports)
}

#[derive(Debug, Clone, PartialEq)]
struct Qlen1DenseOutput {
    output: Vec<f32>,
    lse: Vec<f32>,
}

fn q_len1_dense_reference(
    contract: DalucKvViewContract,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    attention: FlatAttentionConfig,
    query_position: usize,
) -> Result<Qlen1DenseOutput, DalucTierQualificationError> {
    contract.validate().map_err(DalucOracleError::from)?;
    require_len(
        "query",
        query.len(),
        checked_product(&[
            contract.shape.batch,
            contract.shape.q_heads,
            contract.shape.key_head_dim,
        ])?,
    )?;
    require_len("keys", keys.len(), logical_side_len(contract, true)?)?;
    require_len("values", values.len(), logical_side_len(contract, false)?)?;
    validate_finite("query", query)?;
    validate_finite("keys", keys)?;
    validate_finite("values", values)?;
    if query_position >= contract.shape.kv_len {
        return Err(DalucTierQualificationError::InvalidFixture(
            "query position must address a live KV token",
        ));
    }
    let scale = attention
        .softmax_scale
        .unwrap_or_else(|| 1.0 / (contract.shape.key_head_dim as f32).sqrt());
    if !scale.is_finite() || scale <= 0.0 {
        return Err(DalucTierQualificationError::InvalidFixture(
            "attention scale must be finite and positive",
        ));
    }

    let mut output = vec![
        0.0f32;
        checked_product(&[
            contract.shape.batch,
            contract.shape.q_heads,
            contract.shape.value_head_dim,
        ])?
    ];
    let mut lse = vec![0.0f32; checked_product(&[contract.shape.batch, contract.shape.q_heads])?];
    let group_size = contract.shape.q_heads / contract.shape.kv_heads;

    for batch in 0..contract.shape.batch {
        for q_head in 0..contract.shape.q_heads {
            let kv_head = q_head / group_size;
            let q_base = head_vector_offset(
                batch,
                q_head,
                contract.shape.q_heads,
                contract.shape.key_head_dim,
            )?;
            let out_base = head_vector_offset(
                batch,
                q_head,
                contract.shape.q_heads,
                contract.shape.value_head_dim,
            )?;
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;

            for token in 0..contract.shape.kv_len {
                if attention.causal && token > query_position {
                    break;
                }
                let key_base = canonical_offset(contract, batch, kv_head, token, true)?;
                let value_base = canonical_offset(contract, batch, kv_head, token, false)?;
                let mut dot = 0.0f32;
                for feature in 0..contract.shape.key_head_dim {
                    dot += query[q_base + feature] * keys[key_base + feature];
                }
                let score = dot * scale;
                let new_max = running_max.max(score);
                let alpha = if running_max.is_infinite() {
                    0.0
                } else {
                    (running_max - new_max).exp()
                };
                let numerator = (score - new_max).exp();
                for feature in 0..contract.shape.value_head_dim {
                    let target = out_base + feature;
                    output[target] = output[target] * alpha + numerator * values[value_base + feature];
                }
                running_sum = running_sum * alpha + numerator;
                running_max = new_max;
            }
            let inv_sum = running_sum.recip();
            for feature in 0..contract.shape.value_head_dim {
                output[out_base + feature] *= inv_sum;
            }
            lse[batch * contract.shape.q_heads + q_head] = running_max + running_sum.ln();
        }
    }
    Ok(Qlen1DenseOutput { output, lse })
}

struct CompositeStorageAccumulator {
    logical_kv_scalar_count: usize,
    overhead: DalucTierStorageOverhead,
    segment_count: usize,
    key_codebook_payload_bytes: usize,
    key_index_payload_bytes: usize,
    key_residual_value_payload_bytes: usize,
    key_residual_index_payload_bytes: usize,
    value_payload_bytes: usize,
    value_scale_payload_bytes: usize,
    value_zero_point_payload_bytes: usize,
    value_residual_value_payload_bytes: usize,
    value_residual_index_payload_bytes: usize,
    page_metadata_payload_bytes: usize,
    packing_tail_padding_bits: usize,
    alignment_padding_bytes: usize,
    owned_payload_bytes: usize,
}

impl CompositeStorageAccumulator {
    fn new(
        logical_kv_scalar_count: usize,
        overhead: DalucTierStorageOverhead,
        segment_count: usize,
    ) -> Result<Self, DalucTierQualificationError> {
        overhead
            .segment_metadata_bytes_per_segment
            .checked_mul(segment_count)
            .ok_or(DalucTierQualificationError::ArithmeticOverflow(
                "segment metadata bytes",
            ))?;
        Ok(Self {
            logical_kv_scalar_count,
            overhead,
            segment_count,
            key_codebook_payload_bytes: 0,
            key_index_payload_bytes: 0,
            key_residual_value_payload_bytes: 0,
            key_residual_index_payload_bytes: 0,
            value_payload_bytes: 0,
            value_scale_payload_bytes: 0,
            value_zero_point_payload_bytes: 0,
            value_residual_value_payload_bytes: 0,
            value_residual_index_payload_bytes: 0,
            page_metadata_payload_bytes: 0,
            packing_tail_padding_bits: 0,
            alignment_padding_bytes: 0,
            owned_payload_bytes: 0,
        })
    }

    fn accumulate(
        &mut self,
        payload: &DalucOraclePayload,
        report: &DalucOracleStorageReport,
        charge_codebook: bool,
    ) -> Result<(), DalucTierQualificationError> {
        let codebook = payload.key_codebook_plane();
        if charge_codebook {
            add(&mut self.key_codebook_payload_bytes, report.key_codebook_payload_bytes, "K codebook payload bytes")?;
        }
        add(&mut self.key_index_payload_bytes, report.key_index_payload_bytes, "K index payload bytes")?;
        add(&mut self.key_residual_value_payload_bytes, report.key_residual_value_payload_bytes, "K residual value bytes")?;
        add(&mut self.key_residual_index_payload_bytes, report.key_residual_index_payload_bytes, "K residual index bytes")?;
        add(&mut self.value_payload_bytes, report.value_payload_bytes, "V payload bytes")?;
        add(&mut self.value_scale_payload_bytes, report.value_scale_payload_bytes, "V scale bytes")?;
        add(&mut self.value_zero_point_payload_bytes, report.value_zero_point_payload_bytes, "V zero-point bytes")?;
        add(&mut self.value_residual_value_payload_bytes, report.value_residual_value_payload_bytes, "V residual value bytes")?;
        add(&mut self.value_residual_index_payload_bytes, report.value_residual_index_payload_bytes, "V residual index bytes")?;
        add(&mut self.page_metadata_payload_bytes, report.page_metadata_payload_bytes, "page metadata bytes")?;

        let owned_bytes = if charge_codebook {
            report.total_representation_bytes
        } else {
            report
                .total_representation_bytes
                .checked_sub(codebook.total_bytes())
                .ok_or(DalucTierQualificationError::ArithmeticOverflow(
                    "shared codebook subtraction",
                ))?
        };
        add(&mut self.owned_payload_bytes, owned_bytes, "owned payload bytes")?;

        let owned_alignment = if charge_codebook {
            report.alignment_padding_bytes
        } else {
            report
                .alignment_padding_bytes
                .checked_sub(codebook.alignment_padding_bytes())
                .ok_or(DalucTierQualificationError::ArithmeticOverflow(
                    "shared codebook alignment subtraction",
                ))?
        };
        add(&mut self.alignment_padding_bytes, owned_alignment, "alignment padding bytes")?;

        let owned_tail = if charge_codebook {
            report.packing_tail_padding_bits
        } else {
            report
                .packing_tail_padding_bits
                .checked_sub(codebook.byte_tail_padding_bits())
                .ok_or(DalucTierQualificationError::ArithmeticOverflow(
                    "shared codebook tail subtraction",
                ))?
        };
        add(&mut self.packing_tail_padding_bits, owned_tail, "packing tail padding bits")
    }

    fn finish(self) -> Result<DalucTierCompositeStorageReport, DalucTierQualificationError> {
        let segment_metadata_bytes = self
            .overhead
            .segment_metadata_bytes_per_segment
            .checked_mul(self.segment_count)
            .ok_or(DalucTierQualificationError::ArithmeticOverflow(
                "segment metadata bytes",
            ))?;
        let total_representation_bytes = self
            .owned_payload_bytes
            .checked_add(self.overhead.shared_metadata_bytes)
            .and_then(|value| value.checked_add(segment_metadata_bytes))
            .ok_or(DalucTierQualificationError::ArithmeticOverflow(
                "composite representation bytes",
            ))?;
        Ok(DalucTierCompositeStorageReport {
            codebook_ownership_version: DA_LUC_CODEBOOK_OWNERSHIP_VERSION,
            logical_kv_scalar_count: self.logical_kv_scalar_count,
            key_codebook_payload_bytes: self.key_codebook_payload_bytes,
            key_index_payload_bytes: self.key_index_payload_bytes,
            key_residual_value_payload_bytes: self.key_residual_value_payload_bytes,
            key_residual_index_payload_bytes: self.key_residual_index_payload_bytes,
            value_payload_bytes: self.value_payload_bytes,
            value_scale_payload_bytes: self.value_scale_payload_bytes,
            value_zero_point_payload_bytes: self.value_zero_point_payload_bytes,
            value_residual_value_payload_bytes: self.value_residual_value_payload_bytes,
            value_residual_index_payload_bytes: self.value_residual_index_payload_bytes,
            page_metadata_payload_bytes: self.page_metadata_payload_bytes,
            packing_tail_padding_bits: self.packing_tail_padding_bits,
            alignment_padding_bytes: self.alignment_padding_bytes,
            shared_metadata_bytes: self.overhead.shared_metadata_bytes,
            segment_metadata_bytes,
            total_representation_bytes,
            effective_bits_per_value: total_representation_bytes as f64 * 8.0
                / self.logical_kv_scalar_count as f64,
        })
    }
}

fn validate_fixture(
    fixture: DalucTierQualificationFixture<'_>,
    segment_size: usize,
) -> Result<(), DalucTierQualificationError> {
    fixture
        .base_contract
        .validate()
        .map_err(DalucOracleError::from)?;
    if fixture.tiers.is_empty() {
        return Err(DalucTierQualificationError::InvalidFixture(
            "tier materialization catalog must not be empty",
        ));
    }
    let descriptors = fixture.tiers.iter().map(|spec| spec.tier).collect::<Vec<_>>();
    validate_tier_catalog(fixture.base_contract, segment_size, &descriptors, fixture.quotas)?;
    for (index, spec) in fixture.tiers.iter().enumerate() {
        if fixture.tiers[..index]
            .iter()
            .any(|prior| prior.tier.id == spec.tier.id)
        {
            return Err(DalucTierQualificationError::DuplicateTierMaterialization(
                spec.tier.id,
            ));
        }
        validate_codebook(fixture.base_contract, spec)?;
    }
    require_len("dense K", fixture.dense_keys.len(), logical_side_len(fixture.base_contract, true)?)?;
    require_len("dense V", fixture.dense_values.len(), logical_side_len(fixture.base_contract, false)?)?;
    require_len(
        "query",
        fixture.query.len(),
        checked_product(&[
            fixture.base_contract.shape.batch,
            fixture.base_contract.shape.q_heads,
            fixture.base_contract.shape.key_head_dim,
        ])?,
    )?;
    validate_finite("dense K", fixture.dense_keys)?;
    validate_finite("dense V", fixture.dense_values)?;
    validate_finite("query", fixture.query)?;
    if fixture.query_position >= fixture.base_contract.shape.kv_len {
        return Err(DalucTierQualificationError::InvalidFixture(
            "query position must address a live KV token",
        ));
    }
    Ok(())
}

fn validate_tier_catalog(
    base_contract: DalucKvViewContract,
    segment_size: usize,
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
) -> Result<(), DalucTierQualificationError> {
    route_by_recency(base_contract, segment_size, tiers, quotas)?;
    Ok(())
}

fn validate_codebook(
    contract: DalucKvViewContract,
    spec: &DalucTierMaterializationSpec<'_>,
) -> Result<(), DalucTierQualificationError> {
    if spec.tier.keys.subspace_dim == 0
        || !contract
            .shape
            .key_head_dim
            .is_multiple_of(spec.tier.keys.subspace_dim)
    {
        return Err(DalucTierQualificationError::InvalidFixture(
            "tier key subspace does not partition base key head dimension",
        ));
    }
    let scopes = match spec.tier.keys.codebook_scope {
        DalucCodebookScope::SharedAcrossKvHeads => 1,
        DalucCodebookScope::PerKvHead => contract.shape.kv_heads,
    };
    let expected = checked_product(&[
        scopes,
        contract.shape.key_head_dim / spec.tier.keys.subspace_dim,
        spec.tier.keys.codebook_entries,
        spec.tier.keys.subspace_dim,
    ])?;
    if spec.codebook.len() != expected {
        return Err(DalucTierQualificationError::CodebookLength {
            tier_id: spec.tier.id,
            expected,
            actual: spec.codebook.len(),
        });
    }
    if let Some((index, _)) = spec
        .codebook
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(DalucTierQualificationError::NonFiniteCodebook {
            tier_id: spec.tier.id,
            index,
        });
    }
    Ok(())
}

fn segment_contract(
    mut base: DalucKvViewContract,
    tier: DalucPrecisionTier,
    start: usize,
    end: usize,
) -> Result<DalucKvViewContract, DalucTierQualificationError> {
    let len = end
        .checked_sub(start)
        .ok_or(DalucTierQualificationError::ArithmeticOverflow(
            "segment length",
        ))?;
    if len == 0 {
        return Err(DalucTierQualificationError::MalformedPlan(
            "empty segment is not allowed",
        ));
    }
    base.shape.kv_len = len;
    base.keys = tier.keys;
    base.values = tier.values;
    base.layout.topology = match base.layout.topology {
        DalucStorageTopology::Contiguous { .. } => DalucStorageTopology::Contiguous {
            capacity_tokens: len,
        },
        DalucStorageTopology::Paged { page_size, .. } => {
            let pages = len
                .checked_add(page_size - 1)
                .and_then(|value| value.checked_div(page_size))
                .ok_or(DalucTierQualificationError::ArithmeticOverflow(
                    "segment page count",
                ))?;
            DalucStorageTopology::Paged {
                page_size,
                physical_pages_per_batch: pages,
            }
        }
    };
    base.validate().map_err(DalucOracleError::from)?;
    Ok(base)
}

fn extract_segment(
    contract: DalucKvViewContract,
    dense: &[f32],
    start: usize,
    end: usize,
    key: bool,
) -> Result<Vec<f32>, DalucTierQualificationError> {
    let dim = side_dim(contract, key);
    let segment_len = end
        .checked_sub(start)
        .ok_or(DalucTierQualificationError::ArithmeticOverflow(
            "segment extraction length",
        ))?;
    let mut output = Vec::with_capacity(checked_product(&[
        contract.shape.batch,
        contract.shape.kv_heads,
        segment_len,
        dim,
    ])?);
    for batch in 0..contract.shape.batch {
        for head in 0..contract.shape.kv_heads {
            for token in start..end {
                let base = canonical_offset(contract, batch, head, token, key)?;
                output.extend_from_slice(&dense[base..base + dim]);
            }
        }
    }
    Ok(output)
}

fn insert_segment(
    contract: DalucKvViewContract,
    target: &mut [f32],
    segment: &[f32],
    start: usize,
    end: usize,
    key: bool,
) -> Result<(), DalucTierQualificationError> {
    let dim = side_dim(contract, key);
    let segment_len = end
        .checked_sub(start)
        .ok_or(DalucTierQualificationError::ArithmeticOverflow(
            "segment insertion length",
        ))?;
    require_len(
        "decoded segment",
        segment.len(),
        checked_product(&[
            contract.shape.batch,
            contract.shape.kv_heads,
            segment_len,
            dim,
        ])?,
    )?;
    let mut source = 0;
    for batch in 0..contract.shape.batch {
        for head in 0..contract.shape.kv_heads {
            for token in start..end {
                let base = canonical_offset(contract, batch, head, token, key)?;
                target[base..base + dim].copy_from_slice(&segment[source..source + dim]);
                source += dim;
            }
        }
    }
    Ok(())
}

fn assignments_from_ranking(
    canonical: &DalucTierRoutingPlan,
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
    ranking: &[usize],
) -> Result<Vec<DalucTierAssignment>, DalucTierQualificationError> {
    if ranking.len() != canonical.assignments.len() {
        return Err(DalucTierQualificationError::MalformedPlan(
            "random ranking length does not cover every segment",
        ));
    }
    let mut seen = vec![false; ranking.len()];
    for &index in ranking {
        let slot = seen
            .get_mut(index)
            .ok_or(DalucTierQualificationError::MalformedPlan(
                "random ranking references an unknown segment",
            ))?;
        if *slot {
            return Err(DalucTierQualificationError::MalformedPlan(
                "random ranking contains a duplicate segment",
            ));
        }
        *slot = true;
    }
    let mut assignments = canonical.assignments.clone();
    let mut cursor = 0usize;
    for tier in tiers {
        let count = quotas
            .iter()
            .find(|quota| quota.tier_id == tier.id)
            .ok_or(DalucTierQualificationError::UnknownTier(tier.id))?
            .segments;
        let end = cursor
            .checked_add(count)
            .ok_or(DalucTierQualificationError::ArithmeticOverflow(
                "random quota cursor",
            ))?;
        for &segment_index in ranking
            .get(cursor..end)
            .ok_or(DalucTierQualificationError::MalformedPlan(
                "random quota range exceeds segment count",
            ))?
        {
            assignments[segment_index].tier_id = tier.id;
        }
        cursor = end;
    }
    Ok(assignments)
}

fn deterministic_shuffle_v1(values: &mut [usize], seed: u64) {
    let mut state = seed;
    for index in (1..values.len()).rev() {
        let target = (splitmix64(&mut state) % (index as u64 + 1)) as usize;
        values.swap(index, target);
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn validate_quota_counts(
    assignments: &[DalucTierAssignment],
    tiers: &[DalucPrecisionTier],
    quotas: &[DalucTierQuota],
) -> Result<(), DalucTierQualificationError> {
    for tier in tiers {
        let expected = quotas
            .iter()
            .find(|quota| quota.tier_id == tier.id)
            .ok_or(DalucTierQualificationError::UnknownTier(tier.id))?
            .segments;
        let actual = assignments
            .iter()
            .filter(|assignment| assignment.tier_id == tier.id)
            .count();
        if actual != expected {
            return Err(DalucTierQualificationError::TierQuotaMismatch {
                tier_id: tier.id,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn quota_signature(
    assignments: &[DalucTierAssignment],
    tiers: &[DalucPrecisionTier],
) -> Vec<(DalucTierId, usize)> {
    let mut signature = tiers
        .iter()
        .map(|tier| {
            (
                tier.id,
                assignments
                    .iter()
                    .filter(|assignment| assignment.tier_id == tier.id)
                    .count(),
            )
        })
        .collect::<Vec<_>>();
    signature.sort_by_key(|(id, _)| *id);
    signature
}

fn segment_count(
    kv_len: usize,
    segment_size: usize,
) -> Result<usize, DalucTierQualificationError> {
    if segment_size == 0 {
        return Err(DalucTierQualificationError::MalformedPlan(
            "segment size must be non-zero",
        ));
    }
    kv_len
        .checked_add(segment_size - 1)
        .and_then(|value| value.checked_div(segment_size))
        .ok_or(DalucTierQualificationError::ArithmeticOverflow(
            "segment count",
        ))
}

fn segment_bounds(
    kv_len: usize,
    segment_size: usize,
    index: usize,
) -> Result<(usize, usize), DalucTierQualificationError> {
    let start = index
        .checked_mul(segment_size)
        .ok_or(DalucTierQualificationError::ArithmeticOverflow(
            "segment start",
        ))?;
    let end = start
        .checked_add(segment_size)
        .ok_or(DalucTierQualificationError::ArithmeticOverflow(
            "segment end",
        ))?
        .min(kv_len);
    Ok((start, end))
}

fn side_dim(contract: DalucKvViewContract, key: bool) -> usize {
    if key {
        contract.shape.key_head_dim
    } else {
        contract.shape.value_head_dim
    }
}

fn logical_side_len(
    contract: DalucKvViewContract,
    key: bool,
) -> Result<usize, DalucTierQualificationError> {
    checked_product(&[
        contract.shape.batch,
        contract.shape.kv_heads,
        contract.shape.kv_len,
        side_dim(contract, key),
    ])
}

fn logical_kv_scalar_count(
    contract: DalucKvViewContract,
) -> Result<usize, DalucTierQualificationError> {
    logical_side_len(contract, true)?
        .checked_add(logical_side_len(contract, false)?)
        .ok_or(DalucTierQualificationError::ArithmeticOverflow(
            "logical KV scalar count",
        ))
}

fn canonical_offset(
    contract: DalucKvViewContract,
    batch: usize,
    head: usize,
    token: usize,
    key: bool,
) -> Result<usize, DalucTierQualificationError> {
    batch
        .checked_mul(contract.shape.kv_heads)
        .and_then(|value| value.checked_add(head))
        .and_then(|value| value.checked_mul(contract.shape.kv_len))
        .and_then(|value| value.checked_add(token))
        .and_then(|value| value.checked_mul(side_dim(contract, key)))
        .ok_or(DalucTierQualificationError::ArithmeticOverflow(
            "canonical KV offset",
        ))
}

fn head_vector_offset(
    batch: usize,
    head: usize,
    heads: usize,
    dim: usize,
) -> Result<usize, DalucTierQualificationError> {
    batch
        .checked_mul(heads)
        .and_then(|value| value.checked_add(head))
        .and_then(|value| value.checked_mul(dim))
        .ok_or(DalucTierQualificationError::ArithmeticOverflow(
            "head vector offset",
        ))
}

fn checked_product(values: &[usize]) -> Result<usize, DalucTierQualificationError> {
    values.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or(DalucTierQualificationError::ArithmeticOverflow(
                "dimension product",
            ))
    })
}

fn add(
    target: &mut usize,
    value: usize,
    label: &'static str,
) -> Result<(), DalucTierQualificationError> {
    *target = target
        .checked_add(value)
        .ok_or(DalucTierQualificationError::ArithmeticOverflow(label))?;
    Ok(())
}

fn require_len(
    what: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), DalucTierQualificationError> {
    if actual != expected {
        return Err(DalucTierQualificationError::LengthMismatch {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_finite(
    what: &'static str,
    values: &[f32],
) -> Result<(), DalucTierQualificationError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(DalucTierQualificationError::NonFinite { what, index });
    }
    Ok(())
}

fn error_stats(expected: &[f32], actual: &[f32]) -> DalucOracleErrorStats {
    debug_assert_eq!(expected.len(), actual.len());
    if expected.is_empty() {
        return DalucOracleErrorStats {
            samples: 0,
            max_abs: 0.0,
            mean_abs: 0.0,
            rmse: 0.0,
        };
    }
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for (&left, &right) in expected.iter().zip(actual) {
        let delta = f64::from(left) - f64::from(right);
        let absolute = delta.abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
        sum_sq += delta * delta;
    }
    DalucOracleErrorStats {
        samples: expected.len(),
        max_abs,
        mean_abs: sum_abs / expected.len() as f64,
        rmse: (sum_sq / expected.len() as f64).sqrt(),
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum DalucTierQualificationError {
    Routing(DalucTierRoutingError),
    Oracle(DalucOracleError),
    InvalidFixture(&'static str),
    MalformedPlan(&'static str),
    UnknownTier(DalucTierId),
    MissingTierMaterialization(DalucTierId),
    DuplicateTierMaterialization(DalucTierId),
    UnsupportedRandomVersion {
        actual: u16,
        supported: u16,
    },
    CodebookLength {
        tier_id: DalucTierId,
        expected: usize,
        actual: usize,
    },
    NonFiniteCodebook {
        tier_id: DalucTierId,
        index: usize,
    },
    TierQuotaMismatch {
        tier_id: DalucTierId,
        expected: usize,
        actual: usize,
    },
    LengthMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        what: &'static str,
        index: usize,
    },
    UnequalTierBudget {
        candidate_index: usize,
    },
    UnequalStorageBudget {
        candidate_index: usize,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    ArithmeticOverflow(&'static str),
}

impl fmt::Display for DalucTierQualificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Routing(error) => write!(formatter, "{error}"),
            Self::Oracle(error) => write!(formatter, "{error}"),
            Self::InvalidFixture(reason) => write!(formatter, "invalid FDAL5b fixture: {reason}"),
            Self::MalformedPlan(reason) => write!(formatter, "malformed FDAL5b plan: {reason}"),
            Self::UnknownTier(id) => write!(formatter, "unknown FDAL5b tier {}", id.0),
            Self::MissingTierMaterialization(id) => {
                write!(formatter, "missing materialization for FDAL5b tier {}", id.0)
            }
            Self::DuplicateTierMaterialization(id) => {
                write!(formatter, "duplicate materialization for FDAL5b tier {}", id.0)
            }
            Self::UnsupportedRandomVersion { actual, supported } => write!(
                formatter,
                "FDAL5b random control version {actual} is unsupported; expected {supported}"
            ),
            Self::CodebookLength {
                tier_id,
                expected,
                actual,
            } => write!(
                formatter,
                "FDAL5b tier {} codebook has {actual} scalars; expected {expected}",
                tier_id.0
            ),
            Self::NonFiniteCodebook { tier_id, index } => write!(
                formatter,
                "FDAL5b tier {} codebook scalar {index} is non-finite",
                tier_id.0
            ),
            Self::TierQuotaMismatch {
                tier_id,
                expected,
                actual,
            } => write!(
                formatter,
                "FDAL5b tier {} owns {actual} segments; expected {expected}",
                tier_id.0
            ),
            Self::LengthMismatch {
                what,
                expected,
                actual,
            } => write!(formatter, "{what} has length {actual}; expected {expected}"),
            Self::NonFinite { what, index } => {
                write!(formatter, "{what} scalar {index} is non-finite")
            }
            Self::UnequalTierBudget { candidate_index } => write!(
                formatter,
                "FDAL5b control {candidate_index} does not use the same tier quota budget"
            ),
            Self::UnequalStorageBudget {
                candidate_index,
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "FDAL5b control {candidate_index} uses {actual_bytes} bytes; equal-budget reference uses {expected_bytes} bytes"
            ),
            Self::ArithmeticOverflow(label) => {
                write!(formatter, "FDAL5b arithmetic overflow: {label}")
            }
        }
    }
}

impl std::error::Error for DalucTierQualificationError {}

impl From<DalucTierRoutingError> for DalucTierQualificationError {
    fn from(value: DalucTierRoutingError) -> Self {
        Self::Routing(value)
    }
}

impl From<DalucOracleError> for DalucTierQualificationError {
    fn from(value: DalucOracleError) -> Self {
        Self::Oracle(value)
    }
}
