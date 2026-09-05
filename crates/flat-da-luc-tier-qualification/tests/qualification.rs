use flat_attention::api::research_da_luc::{
    DalucBitOrder, DalucCodebookScope, DalucFloatDType, DalucKeyRepresentation,
    DalucKvViewContract, DalucLogicalKvShape, DalucPaddingRule, DalucPhysicalLayout,
    DalucResidualIndexing, DalucResidualSemantics, DalucRowOrder, DalucSparseResidual,
    DalucStorageTopology, DalucValueRepresentation, DalucZeroPointStorage,
    DA_LUC_KV_VIEW_SCHEMA_VERSION,
};
use flat_attention::api::research_da_luc_oracle::tiering::{
    DalucPrecisionTier, DalucTierId, DalucTierQuota,
};
use flat_attention::FlatAttentionConfig;
use flat_da_luc_tier_qualification::{
    attention_mass_baseline, deterministic_random_baseline, qualify_baseline, qualify_equal_budget,
    recency_baseline, DalucTierBaselinePlan, DalucTierMaterializationSpec,
    DalucTierQualificationError, DalucTierQualificationFixture, DalucTierStorageOverhead,
    DA_LUC_FIXED_CONTROL_VERSION, DA_LUC_RANDOM_CONTROL_VERSION,
};

#[test]
fn deterministic_controls_share_exact_budget_when_segments_are_equal() {
    let fixture_data = FixtureData::new(12, 2);
    let descriptors = fixture_data.descriptors();
    let quotas = fixture_data.quotas();
    let recency = recency_baseline(fixture_data.contract, 4, &descriptors, &quotas).unwrap();
    let attention = attention_mass_baseline(
        fixture_data.contract,
        4,
        &descriptors,
        &quotas,
        &[0.90, 0.10, 0.20],
    )
    .unwrap();
    let random_a = deterministic_random_baseline(
        fixture_data.contract,
        4,
        &descriptors,
        &quotas,
        DA_LUC_RANDOM_CONTROL_VERSION,
        0xC0FFEE,
    )
    .unwrap();
    let random_b = deterministic_random_baseline(
        fixture_data.contract,
        4,
        &descriptors,
        &quotas,
        DA_LUC_RANDOM_CONTROL_VERSION,
        0xC0FFEE,
    )
    .unwrap();
    assert_eq!(random_a, random_b);

    let fixed = DalucTierBaselinePlan::fixed_from_assignments(
        fixture_data.contract,
        4,
        &descriptors,
        &quotas,
        recency.assignments().to_vec(),
    )
    .unwrap();
    assert!(matches!(
        fixed.control(),
        flat_da_luc_tier_qualification::DalucTierBaselineControl::Fixed {
            version: DA_LUC_FIXED_CONTROL_VERSION
        }
    ));

    let specs = fixture_data.specs();
    let fixture = fixture_data.fixture(&specs, &quotas);
    let reports = qualify_equal_budget(fixture, &[recency, attention, random_a, fixed]).unwrap();
    assert_eq!(reports.len(), 4);
    let bytes = reports[0].storage.total_representation_bytes;
    assert!(reports
        .iter()
        .all(|report| report.storage.total_representation_bytes == bytes));
    assert!(reports.iter().all(|report| {
        report.storage.effective_bits_per_value.is_finite()
            && report.q_len1_output_error.rmse.is_finite()
            && report.q_len1_lse_error.rmse.is_finite()
    }));
}

#[test]
fn partial_segments_preserve_gqa_and_mqa_with_asymmetric_kv_representations() {
    for kv_heads in [1, 2] {
        let fixture_data = FixtureData::new(10, kv_heads);
        let descriptors = fixture_data.descriptors();
        let quotas = fixture_data.quotas();
        let plan = recency_baseline(fixture_data.contract, 4, &descriptors, &quotas).unwrap();
        assert_eq!(plan.assignments()[2].start_token, 8);
        assert_eq!(plan.assignments()[2].end_token_exclusive, 10);

        let specs = fixture_data.specs();
        let report = qualify_baseline(fixture_data.fixture(&specs, &quotas), &plan).unwrap();
        assert_eq!(report.assignments.len(), 3);
        assert_eq!(
            report.storage.logical_kv_scalar_count,
            fixture_data.contract.shape.batch
                * fixture_data.contract.shape.kv_heads
                * fixture_data.contract.shape.kv_len
                * (fixture_data.contract.shape.key_head_dim
                    + fixture_data.contract.shape.value_head_dim)
        );
        assert_eq!(
            report.q_len1_output_error.samples,
            fixture_data.contract.shape.batch
                * fixture_data.contract.shape.q_heads
                * fixture_data.contract.shape.value_head_dim
        );
        assert_eq!(
            report.q_len1_lse_error.samples,
            fixture_data.contract.shape.batch * fixture_data.contract.shape.q_heads
        );
        assert_eq!(report.storage.shared_metadata_bytes, 19);
        assert_eq!(report.storage.segment_metadata_bytes, 3 * 7);
        assert!(report.storage.key_residual_value_payload_bytes > 0);
        assert!(report.storage.key_residual_index_payload_bytes > 0);
        assert!(report.reconstruction.keys.rmse.is_finite());
        assert!(report.reconstruction.values.rmse.is_finite());
    }
}

#[test]
fn partial_segment_assignment_can_break_exact_equal_budget_and_is_rejected() {
    let fixture_data = FixtureData::new(10, 2);
    let descriptors = fixture_data.descriptors();
    let quotas = fixture_data.quotas();
    let recency = recency_baseline(fixture_data.contract, 4, &descriptors, &quotas).unwrap();
    let attention = attention_mass_baseline(
        fixture_data.contract,
        4,
        &descriptors,
        &quotas,
        &[0.90, 0.20, 0.10],
    )
    .unwrap();
    assert_ne!(recency.assignments(), attention.assignments());

    let specs = fixture_data.specs();
    let error = qualify_equal_budget(fixture_data.fixture(&specs, &quotas), &[recency, attention])
        .unwrap_err();
    assert!(matches!(
        error,
        DalucTierQualificationError::UnequalStorageBudget {
            candidate_index: 1,
            ..
        }
    ));
}

#[test]
fn malformed_fixed_assignments_and_payload_descriptor_mismatch_fail_closed() {
    let fixture_data = FixtureData::new(12, 2);
    let descriptors = fixture_data.descriptors();
    let quotas = fixture_data.quotas();
    let recency = recency_baseline(fixture_data.contract, 4, &descriptors, &quotas).unwrap();

    let mut missing = recency.assignments().to_vec();
    missing.pop();
    assert!(matches!(
        DalucTierBaselinePlan::fixed_from_assignments(
            fixture_data.contract,
            4,
            &descriptors,
            &quotas,
            missing,
        ),
        Err(DalucTierQualificationError::MalformedPlan(_))
    ));

    let mut duplicate = recency.assignments().to_vec();
    duplicate[1].segment_index = duplicate[0].segment_index;
    assert!(matches!(
        DalucTierBaselinePlan::fixed_from_assignments(
            fixture_data.contract,
            4,
            &descriptors,
            &quotas,
            duplicate,
        ),
        Err(DalucTierQualificationError::MalformedPlan(_))
    ));

    let mut short_codebook = fixture_data.codebook_high.clone();
    short_codebook.pop();
    let bad_specs = [
        DalucTierMaterializationSpec {
            tier: fixture_data.tier_high,
            codebook: &short_codebook,
        },
        DalucTierMaterializationSpec {
            tier: fixture_data.tier_low,
            codebook: &fixture_data.codebook_low,
        },
    ];
    let error = qualify_baseline(fixture_data.fixture(&bad_specs, &quotas), &recency).unwrap_err();
    assert!(matches!(
        error,
        DalucTierQualificationError::CodebookLength {
            tier_id: DalucTierId(10),
            ..
        }
    ));
}

#[test]
fn invalid_attention_mass_and_random_version_fail_closed() {
    let fixture_data = FixtureData::new(12, 2);
    let descriptors = fixture_data.descriptors();
    let quotas = fixture_data.quotas();
    assert!(attention_mass_baseline(
        fixture_data.contract,
        4,
        &descriptors,
        &quotas,
        &[0.5, f64::NAN, 0.2],
    )
    .is_err());
    assert!(matches!(
        deterministic_random_baseline(
            fixture_data.contract,
            4,
            &descriptors,
            &quotas,
            DA_LUC_RANDOM_CONTROL_VERSION + 1,
            1,
        ),
        Err(DalucTierQualificationError::UnsupportedRandomVersion { .. })
    ));
}

struct FixtureData {
    contract: DalucKvViewContract,
    tier_high: DalucPrecisionTier,
    tier_low: DalucPrecisionTier,
    codebook_high: Vec<f32>,
    codebook_low: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    query: Vec<f32>,
}

impl FixtureData {
    fn new(kv_len: usize, kv_heads: usize) -> Self {
        let q_heads = 4;
        assert_eq!(q_heads % kv_heads, 0);
        let high_key = DalucKeyRepresentation {
            subspace_dim: 4,
            codebook_entries: 8,
            codebook_dtype: DalucFloatDType::F32,
            codebook_scope: DalucCodebookScope::PerKvHead,
            index_bits: 3,
            index_bit_order: DalucBitOrder::Lsb0,
            residual: DalucResidualSemantics::Sparse(DalucSparseResidual {
                value_dtype: DalucFloatDType::F16,
                indexing: DalucResidualIndexing::Coordinates {
                    index_bits: 3,
                    bit_order: DalucBitOrder::Lsb0,
                },
                max_entries_per_vector: 2,
            }),
        };
        let high_value = DalucValueRepresentation::Dense {
            dtype: DalucFloatDType::F32,
        };
        let low_key = DalucKeyRepresentation {
            residual: DalucResidualSemantics::None,
            codebook_dtype: DalucFloatDType::F16,
            ..high_key
        };
        let low_value = DalucValueRepresentation::GroupwiseAffine {
            storage_bits: 4,
            group_size: 3,
            scale_dtype: DalucFloatDType::F16,
            zero_point: DalucZeroPointStorage::U8,
            bit_order: DalucBitOrder::Lsb0,
            residual: DalucResidualSemantics::None,
        };
        let contract = DalucKvViewContract {
            schema_version: DA_LUC_KV_VIEW_SCHEMA_VERSION,
            shape: DalucLogicalKvShape {
                batch: 1,
                q_heads,
                kv_heads,
                kv_len,
                key_head_dim: 8,
                value_head_dim: 6,
            },
            keys: high_key,
            values: high_value,
            layout: DalucPhysicalLayout {
                row_order: DalucRowOrder::BatchHeadToken,
                topology: DalucStorageTopology::Contiguous {
                    capacity_tokens: kv_len,
                },
                plane_alignment_bytes: 16,
                padding: DalucPaddingRule::ZeroFilledToAlignment,
            },
        };
        contract.validate().unwrap();
        let tier_high = DalucPrecisionTier {
            id: DalucTierId(10),
            keys: high_key,
            values: high_value,
        };
        let tier_low = DalucPrecisionTier {
            id: DalucTierId(20),
            keys: low_key,
            values: low_value,
        };
        let codebook_high = codebook(contract, tier_high, 7);
        let codebook_low = codebook(contract, tier_low, 11);
        let key_len = contract.shape.batch
            * contract.shape.kv_heads
            * contract.shape.kv_len
            * contract.shape.key_head_dim;
        let value_len = contract.shape.batch
            * contract.shape.kv_heads
            * contract.shape.kv_len
            * contract.shape.value_head_dim;
        let query_len = contract.shape.batch * contract.shape.q_heads * contract.shape.key_head_dim;
        Self {
            contract,
            tier_high,
            tier_low,
            codebook_high,
            codebook_low,
            keys: dense(key_len, 7),
            values: dense(value_len, 11),
            query: dense(query_len, 13),
        }
    }

    fn descriptors(&self) -> [DalucPrecisionTier; 2] {
        [self.tier_high, self.tier_low]
    }

    fn specs(&self) -> [DalucTierMaterializationSpec<'_>; 2] {
        [
            DalucTierMaterializationSpec {
                tier: self.tier_high,
                codebook: &self.codebook_high,
            },
            DalucTierMaterializationSpec {
                tier: self.tier_low,
                codebook: &self.codebook_low,
            },
        ]
    }

    fn quotas(&self) -> [DalucTierQuota; 2] {
        [
            DalucTierQuota {
                tier_id: DalucTierId(10),
                segments: 1,
            },
            DalucTierQuota {
                tier_id: DalucTierId(20),
                segments: 2,
            },
        ]
    }

    fn fixture<'a>(
        &'a self,
        specs: &'a [DalucTierMaterializationSpec<'a>],
        quotas: &'a [DalucTierQuota],
    ) -> DalucTierQualificationFixture<'a> {
        DalucTierQualificationFixture {
            base_contract: self.contract,
            tiers: specs,
            quotas,
            dense_keys: &self.keys,
            dense_values: &self.values,
            query: &self.query,
            attention: FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
            query_position: self.contract.shape.kv_len - 1,
            storage_overhead: DalucTierStorageOverhead {
                shared_metadata_bytes: 19,
                segment_metadata_bytes_per_segment: 7,
            },
        }
    }
}

fn codebook(contract: DalucKvViewContract, tier: DalucPrecisionTier, stride: usize) -> Vec<f32> {
    let scopes = match tier.keys.codebook_scope {
        DalucCodebookScope::SharedAcrossKvHeads => 1,
        DalucCodebookScope::PerKvHead => contract.shape.kv_heads,
    };
    let subspaces = contract.shape.key_head_dim / tier.keys.subspace_dim;
    let len = scopes * subspaces * tier.keys.codebook_entries * tier.keys.subspace_dim;
    dense(len, stride)
}

fn dense(len: usize, stride: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (((index * stride + 3) % 47) as f32 - 23.0) / 9.0)
        .collect()
}
