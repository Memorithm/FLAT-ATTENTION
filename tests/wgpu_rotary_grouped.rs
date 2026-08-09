#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference_grouped_rope, FlatAttentionConfig, GroupedAttentionShape,
    RotaryEmbeddingConfig, WgpuFlatAttentionError, WgpuRotaryGroupedAttention,
};

const ATOL: f32 = 1.5e-4;
const RTOL: f32 = 1.0e-3;

fn context() -> Option<WgpuRotaryGroupedAttention> {
    match WgpuRotaryGroupedAttention::new() {
        Ok(context) => Some(context),
        Err(WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional FLAT-R1 test skipped");
            None
        }
        Err(error) => panic!("FLAT-R1 context creation failed: {error}"),
    }
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i as f32 * 0.017 + phase;
            x.sin() * 2.25 + (x * 0.43).cos() * 0.4
        })
        .collect()
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "{name}[{index}] is not finite");
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

#[test]
fn fused_rope_grouped_wgpu_matches_oracle() {
    let Some(context) = context() else {
        return;
    };
    eprintln!("FLAT-R1 adapter: {}", context.adapter_name());

    for shape in [
        GroupedAttentionShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 4,
            seq_len: 17,
            head_dim: 32,
        },
        GroupedAttentionShape {
            batch: 1,
            q_heads: 8,
            kv_heads: 2,
            seq_len: 17,
            head_dim: 64,
        },
        GroupedAttentionShape {
            batch: 2,
            q_heads: 8,
            kv_heads: 1,
            seq_len: 9,
            head_dim: 80,
        },
        GroupedAttentionShape {
            batch: 1,
            q_heads: 16,
            kv_heads: 4,
            seq_len: 5,
            head_dim: 128,
        },
    ] {
        let q = fixture(shape.q_tensor_len().unwrap(), 0.15);
        let k = fixture(shape.kv_tensor_len().unwrap(), 0.85);
        let v = fixture(shape.kv_tensor_len().unwrap(), 1.55);
        for rotary in [
            RotaryEmbeddingConfig {
                theta: 10_000.0,
                position_offset: 0,
            },
            RotaryEmbeddingConfig {
                theta: 500_000.0,
                position_offset: 11,
            },
        ] {
            for causal in [false, true] {
                let config = FlatAttentionConfig {
                    causal,
                    softmax_scale: None,
                };
                let expected =
                    forward_reference_grouped_rope(&q, &k, &v, shape, config, rotary).unwrap();
                let actual = context.forward(&q, &k, &v, shape, config, rotary).unwrap();
                assert_close("R1 O", &actual.output, &expected.output);
                assert_close("R1 LSE", &actual.lse, &expected.lse);
            }
        }
    }
}

#[test]
fn resident_r1_keeps_raw_qkv_cardinality_and_no_rope_output_contract() {
    let Some(context) = context() else {
        return;
    };
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 12,
        kv_heads: 3,
        seq_len: 17,
        head_dim: 64,
    };
    let q = fixture(shape.q_tensor_len().unwrap(), 0.25);
    let k = fixture(shape.kv_tensor_len().unwrap(), 0.95);
    let v = fixture(shape.kv_tensor_len().unwrap(), 1.65);
    let q_gpu = context.upload(&q).unwrap();
    let k_gpu = context.upload(&k).unwrap();
    let v_gpu = context.upload(&v).unwrap();

    assert_eq!(q_gpu.len(), shape.q_tensor_len().unwrap());
    assert_eq!(k_gpu.len(), shape.kv_tensor_len().unwrap());
    assert_eq!(v_gpu.len(), shape.kv_tensor_len().unwrap());
    assert_eq!(q_gpu.len(), 4 * k_gpu.len());

    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: Some(0.125),
    };
    let rotary = RotaryEmbeddingConfig {
        theta: 10_000.0,
        position_offset: 23,
    };
    let resident = context
        .forward_resident(&q_gpu, &k_gpu, &v_gpu, shape, config, rotary)
        .unwrap();
    assert_eq!(resident.output_len(), shape.q_tensor_len().unwrap());
    assert_eq!(resident.lse_len(), shape.lse_len().unwrap());
    assert_eq!(
        resident.combined().len(),
        shape.q_tensor_len().unwrap() + shape.lse_len().unwrap()
    );

    let actual = context.download_attention(&resident).unwrap();
    let expected = forward_reference_grouped_rope(&q, &k, &v, shape, config, rotary).unwrap();
    assert_close("resident R1 O", &actual.output, &expected.output);
    assert_close("resident R1 LSE", &actual.lse, &expected.lse);
}
