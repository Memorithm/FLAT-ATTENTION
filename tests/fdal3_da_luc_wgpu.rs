#![cfg(feature = "wgpu")]

use flat_attention::api::research_da_luc::{
    DalucBitOrder, DalucCodebookScope, DalucFloatDType, DalucKeyRepresentation,
    DalucKvViewContract, DalucLogicalKvShape, DalucPaddingRule, DalucPhysicalLayout,
    DalucResidualSemantics, DalucRowOrder, DalucStorageTopology, DalucValueRepresentation,
    DalucZeroPointStorage, DA_LUC_KV_VIEW_SCHEMA_VERSION,
};
use flat_attention::api::research_da_luc_oracle::decode::DalucQlen1DecodeConfig;
use flat_attention::api::research_da_luc_oracle::wgpu::{
    DalucWgpuCandidateError, DalucWgpuPlan, WgpuDalucQlen1Candidate,
    DA_LUC_WGPU_CANDIDATE_VERSION,
};
use flat_attention::api::research_da_luc_oracle::DalucOraclePayload;
use flat_attention::FlatAttentionConfig;

const ATOL: f32 = 2.0e-4;
const RTOL: f32 = 2.0e-4;

#[test]
fn fdal3_plan_is_narrow_fail_closed_and_declares_no_dense_materialization() {
    assert_eq!(DA_LUC_WGPU_CANDIDATE_VERSION, 1);
    let contract = supported_contract();
    let config = decode_config(contract, true, 4);
    let plan = DalucWgpuPlan::new(contract, config).unwrap();
    assert!(!plan.materializes_dense_kv());
    assert_eq!(plan.query_elements(), 4 * 32);
    assert_eq!(plan.output_elements(), 4 * 24);
    assert_eq!(plan.lse_elements(), 4);

    let mut subbyte_keys = contract;
    subbyte_keys.keys.index_bits = 4;
    assert!(matches!(
        DalucWgpuPlan::new(subbyte_keys, config),
        Err(DalucWgpuCandidateError::UnsupportedCandidate(_))
    ));

    let mut paged = contract;
    paged.layout.topology = DalucStorageTopology::Paged {
        page_size: 4,
        physical_pages_per_batch: 2,
    };
    assert!(DalucWgpuPlan::new(paged, config).is_err());

    let mut dense_values = contract;
    dense_values.values = DalucValueRepresentation::Dense {
        dtype: DalucFloatDType::F32,
    };
    assert!(DalucWgpuPlan::new(dense_values, config).is_err());

    let mut msb0 = contract;
    msb0.keys.index_bit_order = DalucBitOrder::Msb0;
    assert!(DalucWgpuPlan::new(msb0, config).is_err());
}

#[test]
fn direct_compressed_wgpu_matches_fdal2_scalar_oracle() {
    let contract = supported_contract();
    let payload = payload(contract);
    let query = generated(
        contract.shape.batch * contract.shape.q_heads * contract.shape.key_head_dim,
        29,
    );
    let candidate = candidate_or_skip();
    eprintln!("FDAL3 WGPU adapter: {}", candidate.adapter_name());

    for (causal, query_position) in [(true, 4usize), (false, 0usize)] {
        let config = decode_config(contract, causal, query_position);
        let expected = payload.q_len1_attention_direct(&query, config).unwrap();
        let actual = candidate.forward(&payload, &query, config).unwrap();
        assert_close("output", &actual.output, &expected.output);
        assert_close("lse", &actual.lse, &expected.lse);
    }
}

fn candidate_or_skip() -> WgpuDalucQlen1Candidate {
    match WgpuDalucQlen1Candidate::new() {
        Ok(candidate) => candidate,
        Err(DalucWgpuCandidateError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional FDAL3 device test skipped");
            std::process::exit(0);
        }
        Err(error) => panic!("FDAL3 WGPU candidate creation failed: {error}"),
    }
}

fn supported_contract() -> DalucKvViewContract {
    DalucKvViewContract {
        schema_version: DA_LUC_KV_VIEW_SCHEMA_VERSION,
        shape: DalucLogicalKvShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            kv_len: 7,
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
            topology: DalucStorageTopology::Contiguous { capacity_tokens: 8 },
            plane_alignment_bytes: 16,
            padding: DalucPaddingRule::ZeroFilledToAlignment,
        },
    }
}

fn decode_config(
    contract: DalucKvViewContract,
    causal: bool,
    query_position: usize,
) -> DalucQlen1DecodeConfig {
    let config = DalucQlen1DecodeConfig {
        attention: FlatAttentionConfig {
            causal,
            softmax_scale: None,
        },
        query_position,
    };
    DalucWgpuPlan::new(contract, config).unwrap();
    config
}

fn payload(contract: DalucKvViewContract) -> DalucOraclePayload {
    let rows = contract.shape.batch * contract.shape.kv_heads * contract.shape.kv_len;
    let keys = generated(rows * contract.shape.key_head_dim, 7);
    let values = generated(rows * contract.shape.value_head_dim, 17);
    let codebook = codebook(contract);
    DalucOraclePayload::encode(contract, &codebook, &keys, &values).unwrap()
}

fn codebook(contract: DalucKvViewContract) -> Vec<f32> {
    let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
    let scopes = match contract.keys.codebook_scope {
        DalucCodebookScope::SharedAcrossKvHeads => 1,
        DalucCodebookScope::PerKvHead => contract.shape.kv_heads,
    };
    generated(
        scopes * subspaces * contract.keys.codebook_entries * contract.keys.subspace_dim,
        11,
    )
}

fn generated(len: usize, stride: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (((index * stride + 5) % 97) as f32 - 48.0) / 19.0)
        .collect()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            actual.is_finite() && error <= tolerance,
            "{label}[{index}] actual={actual} expected={expected} error={error} tolerance={tolerance}"
        );
    }
}
