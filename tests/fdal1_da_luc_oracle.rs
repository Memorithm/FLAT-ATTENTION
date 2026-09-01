use flat_attention::api::research_da_luc::{
    DalucBitOrder, DalucCodebookScope, DalucFloatDType, DalucKeyRepresentation,
    DalucKvViewContract, DalucLogicalKvShape, DalucPaddingRule, DalucPhysicalLayout,
    DalucResidualIndexing, DalucResidualSemantics, DalucRowOrder, DalucSparseResidual,
    DalucStorageTopology, DalucValueRepresentation, DalucZeroPointStorage,
    DA_LUC_KV_VIEW_SCHEMA_VERSION,
};
use flat_attention::api::research_da_luc_oracle::DalucOraclePayload;

fn contract(order: DalucBitOrder, paged: bool) -> DalucKvViewContract {
    DalucKvViewContract {
        schema_version: DA_LUC_KV_VIEW_SCHEMA_VERSION,
        shape: DalucLogicalKvShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            kv_len: 5,
            key_head_dim: 8,
            value_head_dim: 8,
        },
        keys: DalucKeyRepresentation {
            subspace_dim: 4,
            codebook_entries: 8,
            codebook_dtype: DalucFloatDType::F16,
            codebook_scope: DalucCodebookScope::PerKvHead,
            index_bits: 3,
            index_bit_order: order,
            residual: DalucResidualSemantics::Sparse(DalucSparseResidual {
                value_dtype: DalucFloatDType::F16,
                indexing: DalucResidualIndexing::Coordinates {
                    index_bits: 3,
                    bit_order: order,
                },
                max_entries_per_vector: 2,
            }),
        },
        values: DalucValueRepresentation::GroupwiseAffine {
            storage_bits: 4,
            group_size: 4,
            scale_dtype: DalucFloatDType::F16,
            zero_point: DalucZeroPointStorage::U8,
            bit_order: order,
            residual: DalucResidualSemantics::Sparse(DalucSparseResidual {
                value_dtype: DalucFloatDType::F16,
                indexing: DalucResidualIndexing::Bitmap { bit_order: order },
                max_entries_per_vector: 2,
            }),
        },
        layout: DalucPhysicalLayout {
            row_order: if paged {
                DalucRowOrder::BatchHeadToken
            } else {
                DalucRowOrder::BatchTokenHead
            },
            topology: if paged {
                DalucStorageTopology::Paged {
                    page_size: 2,
                    physical_pages_per_batch: 4,
                }
            } else {
                DalucStorageTopology::Contiguous { capacity_tokens: 8 }
            },
            plane_alignment_bytes: 16,
            padding: DalucPaddingRule::ZeroFilledToAlignment,
        },
    }
}

fn codebook(contract: DalucKvViewContract) -> Vec<f32> {
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    let scopes = match contract.keys.codebook_scope {
        DalucCodebookScope::SharedAcrossKvHeads => 1,
        DalucCodebookScope::PerKvHead => contract.shape.kv_heads,
    };
    let len = scopes * subspaces * contract.keys.codebook_entries * contract.keys.subspace_dim;
    (0..len)
        .map(|index| ((index * 13 % 41) as f32 - 20.0) / 9.0)
        .collect()
}

fn dense(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (((index * salt + 3) % 53) as f32 - 26.0) / 11.0)
        .collect()
}

#[test]
fn public_oracle_is_deterministic_and_reconstructs_both_bit_orders() {
    for order in [DalucBitOrder::Lsb0, DalucBitOrder::Msb0] {
        let contract = contract(order, false);
        let side = contract.shape.batch * contract.shape.kv_heads * contract.shape.kv_len;
        let keys = dense(side * contract.shape.key_head_dim, 7);
        let values = dense(side * contract.shape.value_head_dim, 11);
        let codebook = codebook(contract);
        let first = DalucOraclePayload::encode(contract, &codebook, &keys, &values).unwrap();
        let second = DalucOraclePayload::encode(contract, &codebook, &keys, &values).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.decode_keys().unwrap().len(), keys.len());
        assert_eq!(first.decode_values().unwrap().len(), values.len());
        let error = first.reconstruction_report(&keys, &values).unwrap();
        assert!(error.keys.rmse.is_finite());
        assert!(error.values.rmse.is_finite());
    }
}

#[test]
fn exact_storage_report_counts_paging_residuals_and_declared_metadata() {
    let contract = contract(DalucBitOrder::Lsb0, true);
    let side = contract.shape.batch * contract.shape.kv_heads * contract.shape.kv_len;
    let keys = dense(side * contract.shape.key_head_dim, 5);
    let values = dense(side * contract.shape.value_head_dim, 17);
    let codebook = codebook(contract);
    let payload = DalucOraclePayload::encode_with_page_table(
        contract,
        &codebook,
        &keys,
        &values,
        Some(&[2, 0, 3]),
    )
    .unwrap();
    let report = payload.storage_report(DalucFloatDType::F16, 48).unwrap();
    assert_eq!(report.external_metadata_bytes, 48);
    assert_eq!(report.page_metadata_payload_bytes, 12);
    assert!(report.key_residual_value_payload_bytes > 0);
    assert!(report.key_residual_index_payload_bytes > 0);
    assert!(report.value_residual_value_payload_bytes > 0);
    assert!(report.value_residual_index_payload_bytes > 0);
    assert!(report.alignment_padding_bytes > 0);
    assert!(report.total_representation_bytes >= report.external_metadata_bytes);
    assert!(report.dense_baseline_bytes > 0);
    assert!(report.effective_bits_per_value.is_finite());
    assert!(report.compression_ratio_against_dense.is_finite());
}

#[test]
fn page_aliases_and_non_finite_inputs_fail_closed() {
    let contract = contract(DalucBitOrder::Lsb0, true);
    let side = contract.shape.batch * contract.shape.kv_heads * contract.shape.kv_len;
    let mut keys = dense(side * contract.shape.key_head_dim, 7);
    let values = dense(side * contract.shape.value_head_dim, 11);
    let codebook = codebook(contract);
    assert!(DalucOraclePayload::encode_with_page_table(
        contract,
        &codebook,
        &keys,
        &values,
        Some(&[0, 0, 1]),
    )
    .is_err());
    keys[3] = f32::INFINITY;
    assert!(DalucOraclePayload::encode(contract, &codebook, &keys, &values).is_err());
}
