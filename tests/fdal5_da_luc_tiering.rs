use flat_attention::api::research_da_luc::{
    DalucBitOrder, DalucCodebookScope, DalucFloatDType, DalucKeyRepresentation,
    DalucKvViewContract, DalucLogicalKvShape, DalucPaddingRule, DalucPhysicalLayout,
    DalucResidualSemantics, DalucRowOrder, DalucStorageTopology, DalucValueRepresentation,
    DalucZeroPointStorage, DA_LUC_KV_VIEW_SCHEMA_VERSION,
};
use flat_attention::api::research_da_luc_oracle::tiering::{
    route_by_attention_mass, route_by_recency, DalucPrecisionTier, DalucTierId, DalucTierQuota,
    DalucTierRoutingError, DalucTierRoutingPolicy, DA_LUC_TIER_ROUTING_VERSION,
};

#[test]
fn recency_routes_newest_segments_first_and_is_reproducible() {
    let contract = base_contract();
    let tiers = tiers(contract);
    let quotas = quotas();

    let first = route_by_recency(contract, 4, &tiers, &quotas).unwrap();
    let second = route_by_recency(contract, 4, &tiers, &quotas).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.routing_version, DA_LUC_TIER_ROUTING_VERSION);
    assert_eq!(first.policy, DalucTierRoutingPolicy::Recency);
    assert_eq!(first.assignments.len(), 3);
    assert_eq!(first.assignments[0].start_token, 0);
    assert_eq!(first.assignments[0].end_token_exclusive, 4);
    assert_eq!(first.assignments[1].start_token, 4);
    assert_eq!(first.assignments[1].end_token_exclusive, 8);
    assert_eq!(first.assignments[2].start_token, 8);
    assert_eq!(first.assignments[2].end_token_exclusive, 10);
    assert_eq!(first.assignments[0].tier_id, DalucTierId(20));
    assert_eq!(first.assignments[1].tier_id, DalucTierId(20));
    assert_eq!(first.assignments[2].tier_id, DalucTierId(10));
    first.validate_against(contract, &tiers).unwrap();
}

#[test]
fn attention_mass_uses_stable_lower_segment_tie_break() {
    let contract = base_contract();
    let tiers = tiers(contract);
    let quotas = quotas();
    let masses = [0.75, 0.75, 0.10];

    let plan = route_by_attention_mass(contract, 4, &tiers, &quotas, &masses).unwrap();
    assert_eq!(plan.policy, DalucTierRoutingPolicy::AttentionMass);
    assert_eq!(plan.assignments[0].tier_id, DalucTierId(10));
    assert_eq!(plan.assignments[1].tier_id, DalucTierId(20));
    assert_eq!(plan.assignments[2].tier_id, DalucTierId(20));

    let repeated = route_by_attention_mass(contract, 4, &tiers, &quotas, &masses).unwrap();
    assert_eq!(plan, repeated);
}

#[test]
fn transitions_are_explicit_and_only_cover_changed_segments() {
    let contract = base_contract();
    let tiers = tiers(contract);
    let quotas = quotas();
    let recency = route_by_recency(contract, 4, &tiers, &quotas).unwrap();
    let attention =
        route_by_attention_mass(contract, 4, &tiers, &quotas, &[0.75, 0.75, 0.10]).unwrap();

    let transitions = attention.transitions_from(&recency).unwrap();
    assert_eq!(transitions.len(), 2);
    assert_eq!(transitions[0].segment_index, 0);
    assert_eq!(transitions[0].from_tier, DalucTierId(20));
    assert_eq!(transitions[0].to_tier, DalucTierId(10));
    assert_eq!(transitions[1].segment_index, 2);
    assert_eq!(transitions[1].from_tier, DalucTierId(10));
    assert_eq!(transitions[1].to_tier, DalucTierId(20));

    assert!(recency.transitions_from(&recency).unwrap().is_empty());
}

#[test]
fn routing_fails_closed_on_malformed_catalog_quota_and_evidence() {
    let contract = base_contract();
    let tiers = tiers(contract);

    let duplicate_tiers = [
        tiers[0],
        DalucPrecisionTier {
            id: tiers[0].id,
            ..tiers[1]
        },
    ];
    assert!(matches!(
        route_by_recency(contract, 4, &duplicate_tiers, &quotas()),
        Err(DalucTierRoutingError::DuplicateTierId(DalucTierId(10)))
    ));

    let bad_sum = [
        DalucTierQuota {
            tier_id: DalucTierId(10),
            segments: 1,
        },
        DalucTierQuota {
            tier_id: DalucTierId(20),
            segments: 1,
        },
    ];
    assert!(matches!(
        route_by_recency(contract, 4, &tiers, &bad_sum),
        Err(DalucTierRoutingError::QuotaSumMismatch {
            expected_segments: 3,
            actual_segments: 2,
        })
    ));

    assert!(matches!(
        route_by_attention_mass(contract, 4, &tiers, &quotas(), &[0.5, 0.5]),
        Err(DalucTierRoutingError::AttentionMassLength {
            expected: 3,
            actual: 2,
        })
    ));
    assert!(matches!(
        route_by_attention_mass(contract, 4, &tiers, &quotas(), &[0.5, f64::NAN, 0.5]),
        Err(DalucTierRoutingError::InvalidAttentionMass { segment_index: 1 })
    ));
    assert!(matches!(
        route_by_recency(contract, 0, &tiers, &quotas()),
        Err(DalucTierRoutingError::InvalidSegmentSize)
    ));
}

#[test]
fn tier_representations_are_validated_against_the_base_contract() {
    let contract = base_contract();
    let mut tiers = tiers(contract);
    tiers[1].keys.index_bits = 3;

    let error = route_by_recency(contract, 4, &tiers, &quotas()).unwrap_err();
    assert!(matches!(error, DalucTierRoutingError::Contract(_)));
}

fn quotas() -> [DalucTierQuota; 2] {
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

fn tiers(contract: DalucKvViewContract) -> [DalucPrecisionTier; 2] {
    let first = DalucPrecisionTier {
        id: DalucTierId(10),
        keys: contract.keys,
        values: contract.values,
    };
    let mut second = first;
    second.id = DalucTierId(20);
    second.keys.index_bits = 4;
    second.values = match second.values {
        DalucValueRepresentation::GroupwiseAffine {
            group_size,
            scale_dtype,
            zero_point,
            bit_order,
            residual,
            ..
        } => DalucValueRepresentation::GroupwiseAffine {
            storage_bits: 4,
            group_size,
            scale_dtype,
            zero_point,
            bit_order,
            residual,
        },
        DalucValueRepresentation::Dense { .. } => unreachable!(),
    };
    [first, second]
}

fn base_contract() -> DalucKvViewContract {
    DalucKvViewContract {
        schema_version: DA_LUC_KV_VIEW_SCHEMA_VERSION,
        shape: DalucLogicalKvShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            kv_len: 10,
            key_head_dim: 32,
            value_head_dim: 24,
        },
        keys: DalucKeyRepresentation {
            subspace_dim: 4,
            codebook_entries: 16,
            codebook_dtype: DalucFloatDType::F32,
            codebook_scope: DalucCodebookScope::PerKvHead,
            index_bits: 8,
            index_bit_order: DalucBitOrder::Lsb0,
            residual: DalucResidualSemantics::None,
        },
        values: DalucValueRepresentation::GroupwiseAffine {
            storage_bits: 8,
            group_size: 6,
            scale_dtype: DalucFloatDType::F32,
            zero_point: DalucZeroPointStorage::U8,
            bit_order: DalucBitOrder::Lsb0,
            residual: DalucResidualSemantics::None,
        },
        layout: DalucPhysicalLayout {
            row_order: DalucRowOrder::BatchHeadToken,
            topology: DalucStorageTopology::Contiguous {
                capacity_tokens: 12,
            },
            plane_alignment_bytes: 16,
            padding: DalucPaddingRule::ZeroFilledToAlignment,
        },
    }
}
