use flat_attention::api::research_da_luc::{
    DalucBitOrder, DalucCodebookScope, DalucFloatDType, DalucKeyRepresentation,
    DalucKvViewContract, DalucLogicalKvShape, DalucPaddingRule, DalucPhysicalLayout,
    DalucResidualIndexing, DalucResidualSemantics, DalucRowOrder, DalucSparseResidual,
    DalucStorageTopology, DalucValueRepresentation, DalucZeroPointStorage,
    DA_LUC_KV_VIEW_SCHEMA_VERSION,
};
use flat_attention::api::research_da_luc_oracle::decode::DalucQlen1DecodeConfig;
use flat_attention::api::research_da_luc_oracle::DalucOraclePayload;
use flat_attention::FlatAttentionConfig;

fn contract(
    order: DalucBitOrder,
    paged: bool,
    key_bitmap: bool,
    quantized_values: bool,
) -> DalucKvViewContract {
    let key_indexing = if key_bitmap {
        DalucResidualIndexing::Bitmap { bit_order: order }
    } else {
        DalucResidualIndexing::Coordinates {
            index_bits: 3,
            bit_order: order,
        }
    };
    let values = if quantized_values {
        DalucValueRepresentation::GroupwiseAffine {
            storage_bits: 4,
            group_size: 3,
            scale_dtype: DalucFloatDType::F16,
            zero_point: DalucZeroPointStorage::U8,
            bit_order: order,
            residual: DalucResidualSemantics::Sparse(DalucSparseResidual {
                value_dtype: DalucFloatDType::F16,
                indexing: DalucResidualIndexing::Bitmap { bit_order: order },
                max_entries_per_vector: 2,
            }),
        }
    } else {
        DalucValueRepresentation::Dense {
            dtype: DalucFloatDType::F16,
        }
    };
    DalucKvViewContract {
        schema_version: DA_LUC_KV_VIEW_SCHEMA_VERSION,
        shape: DalucLogicalKvShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            kv_len: 5,
            key_head_dim: 8,
            value_head_dim: 6,
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
                indexing: key_indexing,
                max_entries_per_vector: 2,
            }),
        },
        values,
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

fn generated(len: usize, stride: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (((index * stride + 5) % 67) as f32 - 33.0) / 13.0)
        .collect()
}

fn codebook(contract: DalucKvViewContract) -> Vec<f32> {
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    let scopes = match contract.keys.codebook_scope {
        DalucCodebookScope::SharedAcrossKvHeads => 1,
        DalucCodebookScope::PerKvHead => contract.shape.kv_heads,
    };
    let len = scopes * subspaces * contract.keys.codebook_entries * contract.keys.subspace_dim;
    generated(len, 11)
}

fn payload(contract: DalucKvViewContract) -> DalucOraclePayload {
    let rows = contract.shape.batch * contract.shape.kv_heads * contract.shape.kv_len;
    let keys = generated(rows * contract.shape.key_head_dim, 7);
    let values = generated(rows * contract.shape.value_head_dim, 17);
    let codebook = codebook(contract);
    match contract.layout.topology {
        DalucStorageTopology::Paged { .. } => DalucOraclePayload::encode_with_page_table(
            contract,
            &codebook,
            &keys,
            &values,
            Some(&[2, 0, 3]),
        )
        .unwrap(),
        DalucStorageTopology::Contiguous { .. } => {
            DalucOraclePayload::encode(contract, &codebook, &keys, &values).unwrap()
        }
    }
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
        let error = (a - b).abs();
        assert!(
            error <= tolerance,
            "index {index}: {a} vs {b}, error={error}, tolerance={tolerance}"
        );
    }
}

#[test]
fn direct_compressed_decode_matches_dense_reference_across_layout_and_bit_order() {
    for order in [DalucBitOrder::Lsb0, DalucBitOrder::Msb0] {
        for paged in [false, true] {
            let contract = contract(order, paged, false, true);
            let payload = payload(contract);
            let query = generated(
                contract.shape.batch * contract.shape.q_heads * contract.shape.key_head_dim,
                19,
            );
            let config = DalucQlen1DecodeConfig {
                attention: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                query_position: 3,
            };
            let direct = payload.q_len1_attention_direct(&query, config).unwrap();
            let dense = payload
                .q_len1_attention_dense_reference(&query, config)
                .unwrap();
            assert_close(&direct.output, &dense.output, 1.0e-4);
            assert_close(&direct.lse, &dense.lse, 1.0e-4);
            assert_eq!(direct.trace.attended_kv_rows, 16);
            assert!(direct.trace.lut_entry_dot_products > 0);
            assert!(direct.trace.key_index_lookups > 0);
            assert!(direct.trace.key_residual_corrections > 0);
            assert!(direct.trace.value_primary_scalar_reads > 0);
            assert!(direct.trace.value_quantized_scalar_conversions > 0);
            assert!(direct.trace.value_residual_corrections > 0);
        }
    }
}

#[test]
fn direct_decode_handles_dense_values_and_bitmap_key_residuals() {
    let contract = contract(DalucBitOrder::Msb0, true, true, false);
    let payload = payload(contract);
    let query = generated(
        contract.shape.batch * contract.shape.q_heads * contract.shape.key_head_dim,
        23,
    );
    let config = DalucQlen1DecodeConfig::for_last_token(
        contract,
        FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(0.25),
        },
    )
    .unwrap();
    let direct = payload.q_len1_attention_direct(&query, config).unwrap();
    let dense = payload
        .q_len1_attention_dense_reference(&query, config)
        .unwrap();
    assert_close(&direct.output, &dense.output, 1.0e-4);
    assert_close(&direct.lse, &dense.lse, 1.0e-4);
    assert_eq!(
        direct.output.len(),
        contract.shape.q_heads * contract.shape.value_head_dim
    );
    assert_eq!(direct.trace.value_quantized_scalar_conversions, 0);
    assert!(direct.trace.value_primary_scalar_reads > 0);
    assert!(direct.trace.key_residual_corrections > 0);
}

#[test]
fn equivalence_report_is_deterministic_and_isolates_accumulation_error() {
    let contract = contract(DalucBitOrder::Lsb0, false, false, true);
    let payload = payload(contract);
    let query = generated(
        contract.shape.batch * contract.shape.q_heads * contract.shape.key_head_dim,
        29,
    );
    let config = DalucQlen1DecodeConfig::for_last_token(
        contract,
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        },
    )
    .unwrap();
    let first = payload
        .q_len1_attention_equivalence_report(&query, config)
        .unwrap();
    let second = payload
        .q_len1_attention_equivalence_report(&query, config)
        .unwrap();
    assert_eq!(first, second);
    assert!(first.output.max_abs <= 1.0e-4);
    assert!(first.lse.max_abs <= 1.0e-4);
}

#[test]
fn malformed_query_and_invalid_scale_fail_closed() {
    let contract = contract(DalucBitOrder::Lsb0, false, false, true);
    let payload = payload(contract);
    let expected = contract.shape.batch * contract.shape.q_heads * contract.shape.key_head_dim;
    let short = vec![0.0; expected - 1];
    let config =
        DalucQlen1DecodeConfig::for_last_token(contract, FlatAttentionConfig::default()).unwrap();
    assert!(payload.q_len1_attention_direct(&short, config).is_err());

    let mut query = vec![0.0; expected];
    query[3] = f32::NAN;
    assert!(payload.q_len1_attention_direct(&query, config).is_err());

    let query = vec![0.0; expected];
    let invalid = DalucQlen1DecodeConfig {
        attention: FlatAttentionConfig {
            causal: true,
            softmax_scale: Some(0.0),
        },
        query_position: contract.shape.kv_len - 1,
    };
    assert!(payload.q_len1_attention_direct(&query, invalid).is_err());
}
