//! Host-only smoke for the public oracle and `api::v1` contract.
//!
//! These tests require no GPU and no `wgpu` feature.

use flat_attention::api::v1::{
    AttentionConfig, AttentionShape as ApiShape, BorrowedAttentionRequest,
};
use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionError, NumericalExecutor,
    NumericalMode,
};

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i as f32 * 0.041 + phase;
            x.sin() * 1.5 + (x * 0.37).cos() * 0.25
        })
        .collect()
}

fn mha_shape() -> AttentionShape {
    AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 5,
        head_dim: 8,
    }
}

#[test]
fn exact_reference_matches_forward_reference_bit_exactly() {
    let shape = mha_shape();
    let q = fixture(shape.tensor_len().unwrap(), 0.2);
    let k = fixture(shape.tensor_len().unwrap(), 0.8);
    let v = fixture(shape.tensor_len().unwrap(), 1.4);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };

    let oracle = forward_reference(&q, &k, &v, shape, config).unwrap();
    let executor = NumericalExecutor::new(NumericalMode::ExactReference).unwrap();
    let via_policy = executor.forward(&q, &k, &v, shape, config).unwrap();

    assert_eq!(
        oracle.output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        via_policy
            .output
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        oracle.lse.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        via_policy.lse.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn oracle_emits_finite_o_and_lse_for_causal_and_non_causal() {
    let shape = mha_shape();
    let q = fixture(shape.tensor_len().unwrap(), 0.0);
    let k = fixture(shape.tensor_len().unwrap(), 0.5);
    let v = fixture(shape.tensor_len().unwrap(), 1.0);

    for causal in [false, true] {
        let out = forward_reference(
            &q,
            &k,
            &v,
            shape,
            FlatAttentionConfig {
                causal,
                softmax_scale: None,
            },
        )
        .unwrap();
        assert_eq!(out.output.len(), shape.tensor_len().unwrap());
        assert_eq!(out.lse.len(), shape.lse_len().unwrap());
        assert!(out.output.iter().all(|x| x.is_finite()), "O causal={causal}");
        assert!(out.lse.iter().all(|x| x.is_finite()), "LSE causal={causal}");
    }
}

#[test]
fn api_v1_borrowed_request_validates_and_rejects_bad_scale() {
    let shape = ApiShape {
        batch: 1,
        q_heads: 4,
        kv_heads: 2,
        query_len: 3,
        kv_len: 5,
        head_dim: 8,
        query_position_offset: 0,
    };
    let q = fixture(shape.q_elements().unwrap(), 0.1);
    let k = fixture(shape.kv_elements().unwrap(), 0.2);
    let v = fixture(shape.kv_elements().unwrap(), 0.3);

    let ok = BorrowedAttentionRequest {
        shape,
        config: AttentionConfig {
            causal: true,
            softmax_scale: None,
        },
        q: &q,
        k: &k,
        v: &v,
    };
    ok.validate().expect("valid borrowed request");

    let bad = BorrowedAttentionRequest {
        shape,
        config: AttentionConfig {
            causal: false,
            softmax_scale: Some(0.0),
        },
        q: &q,
        k: &k,
        v: &v,
    };
    assert!(bad.validate().is_err());
}

#[test]
fn forward_reference_rejects_invalid_scale() {
    let shape = mha_shape();
    let q = vec![0.0; shape.tensor_len().unwrap()];
    let error = forward_reference(
        &q,
        &q,
        &q,
        shape,
        FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(-1.0),
        },
    )
    .expect_err("negative scale must fail");
    assert!(matches!(error, FlatAttentionError::InvalidScale(_)));
}
