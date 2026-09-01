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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("case,bit_order,topology,value_kind,output_max_abs,lse_max_abs,lut_entry_dot_products,key_index_lookups,key_residual_corrections,attended_kv_rows,value_primary_scalar_reads,value_quantized_scalar_conversions,value_residual_corrections,performance_claim");
    let mut case_index = 0usize;
    for order in [DalucBitOrder::Lsb0, DalucBitOrder::Msb0] {
        for paged in [false, true] {
            for quantized_values in [false, true] {
                case_index += 1;
                let contract = contract(order, paged, paged, quantized_values);
                let rows = contract.shape.batch * contract.shape.kv_heads * contract.shape.kv_len;
                let keys = generated(rows * contract.shape.key_head_dim, 7);
                let values = generated(rows * contract.shape.value_head_dim, 17);
                let codebook = codebook(contract);
                let payload = if paged {
                    DalucOraclePayload::encode_with_page_table(
                        contract,
                        &codebook,
                        &keys,
                        &values,
                        Some(&[2, 0, 3]),
                    )?
                } else {
                    DalucOraclePayload::encode(contract, &codebook, &keys, &values)?
                };
                let query = generated(
                    contract.shape.batch * contract.shape.q_heads * contract.shape.key_head_dim,
                    19 + case_index,
                );
                let config = DalucQlen1DecodeConfig::for_last_token(
                    contract,
                    FlatAttentionConfig {
                        causal: true,
                        softmax_scale: None,
                    },
                )?;
                let direct = payload.q_len1_attention_direct(&query, config)?;
                let report = payload.q_len1_attention_equivalence_report(&query, config)?;
                println!(
                    "case_{case_index},{},{},{},{:.9e},{:.9e},{},{},{},{},{},{},{},none",
                    order_name(order),
                    if paged { "paged" } else { "contiguous" },
                    if quantized_values { "groupwise_u8" } else { "dense_f16" },
                    report.output.max_abs,
                    report.lse.max_abs,
                    direct.trace.lut_entry_dot_products,
                    direct.trace.key_index_lookups,
                    direct.trace.key_residual_corrections,
                    direct.trace.attended_kv_rows,
                    direct.trace.value_primary_scalar_reads,
                    direct.trace.value_quantized_scalar_conversions,
                    direct.trace.value_residual_corrections,
                );
            }
        }
    }
    Ok(())
}

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

fn order_name(order: DalucBitOrder) -> &'static str {
    match order {
        DalucBitOrder::Lsb0 => "lsb0",
        DalucBitOrder::Msb0 => "msb0",
    }
}
