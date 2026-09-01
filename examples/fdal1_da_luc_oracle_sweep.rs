use flat_attention::api::research_da_luc_oracle::DalucOraclePayload;
use flat_attention::api::research_da_luc::{
    DalucBitOrder, DalucCodebookScope, DalucFloatDType, DalucKeyRepresentation,
    DalucKvViewContract, DalucLogicalKvShape, DalucPaddingRule, DalucPhysicalLayout,
    DalucResidualIndexing, DalucResidualSemantics, DalucRowOrder, DalucSparseResidual,
    DalucStorageTopology, DalucValueRepresentation, DalucZeroPointStorage,
    DA_LUC_KV_VIEW_SCHEMA_VERSION,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("case,bit_order,residual_indexing,value_bits,total_bytes,effective_bits_per_value,dense_baseline_bytes,compression_ratio,key_max_abs,key_rmse,value_max_abs,value_rmse,performance_claim");
    for bit_order in [DalucBitOrder::Lsb0, DalucBitOrder::Msb0] {
        for bitmap in [false, true] {
            for value_bits in [2u8, 4, 8] {
                run_case(bit_order, bitmap, value_bits)?;
            }
        }
    }
    Ok(())
}

fn run_case(
    bit_order: DalucBitOrder,
    bitmap: bool,
    value_bits: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let residual_indexing = if bitmap {
        DalucResidualIndexing::Bitmap { bit_order }
    } else {
        DalucResidualIndexing::Coordinates {
            index_bits: 4,
            bit_order,
        }
    };
    let contract = DalucKvViewContract {
        schema_version: DA_LUC_KV_VIEW_SCHEMA_VERSION,
        shape: DalucLogicalKvShape {
            batch: 1,
            q_heads: 8,
            kv_heads: 2,
            kv_len: 32,
            key_head_dim: 16,
            value_head_dim: 16,
        },
        keys: DalucKeyRepresentation {
            subspace_dim: 4,
            codebook_entries: 16,
            codebook_dtype: DalucFloatDType::F16,
            codebook_scope: DalucCodebookScope::PerKvHead,
            index_bits: 4,
            index_bit_order: bit_order,
            residual: DalucResidualSemantics::Sparse(DalucSparseResidual {
                value_dtype: DalucFloatDType::F16,
                indexing: residual_indexing,
                max_entries_per_vector: 2,
            }),
        },
        values: DalucValueRepresentation::GroupwiseAffine {
            storage_bits: value_bits,
            group_size: 8,
            scale_dtype: DalucFloatDType::F16,
            zero_point: DalucZeroPointStorage::U8,
            bit_order,
            residual: DalucResidualSemantics::Sparse(DalucSparseResidual {
                value_dtype: DalucFloatDType::F16,
                indexing: residual_indexing,
                max_entries_per_vector: 2,
            }),
        },
        layout: DalucPhysicalLayout {
            row_order: DalucRowOrder::BatchTokenHead,
            topology: DalucStorageTopology::Contiguous { capacity_tokens: 32 },
            plane_alignment_bytes: 16,
            padding: DalucPaddingRule::ZeroFilledToAlignment,
        },
    };
    let codebook_len = contract.shape.kv_heads
        * (contract.shape.key_head_dim / contract.keys.subspace_dim)
        * contract.keys.codebook_entries
        * contract.keys.subspace_dim;
    let codebook: Vec<f32> = (0..codebook_len)
        .map(|index| ((index * 29 % 127) as f32 - 63.0) / 23.0)
        .collect();
    let vectors = contract.shape.batch * contract.shape.kv_heads * contract.shape.kv_len;
    let keys: Vec<f32> = (0..vectors * contract.shape.key_head_dim)
        .map(|index| ((index * 17 % 101) as f32 - 50.0) / 19.0)
        .collect();
    let values: Vec<f32> = (0..vectors * contract.shape.value_head_dim)
        .map(|index| ((index * 31 % 109) as f32 - 54.0) / 17.0)
        .collect();
    let payload = DalucOraclePayload::encode(contract, &codebook, &keys, &values)?;
    let storage = payload.storage_report(DalucFloatDType::F16, 0)?;
    let error = payload.reconstruction_report(&keys, &values)?;
    let order = match bit_order {
        DalucBitOrder::Lsb0 => "lsb0",
        DalucBitOrder::Msb0 => "msb0",
    };
    let indexing = if bitmap { "bitmap" } else { "coordinates" };
    println!(
        "v{value_bits}_{order}_{indexing},{order},{indexing},{value_bits},{},{:.6},{},{:.6},{:.8},{:.8},{:.8},{:.8},none",
        storage.total_representation_bytes,
        storage.effective_bits_per_value,
        storage.dense_baseline_bytes,
        storage.compression_ratio_against_dense,
        error.keys.max_abs,
        error.keys.rmse,
        error.values.max_abs,
        error.values.rmse,
    );
    Ok(())
}
