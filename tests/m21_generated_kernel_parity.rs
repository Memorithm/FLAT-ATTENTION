//! M21 generated-kernel qualification: the WGSL produced by `emit_wgsl`
//! executes on a real WGPU device and matches the deterministic scalar
//! oracle.
//!
//! The logical attention problem is identical to the handwritten-kernel
//! parity cases; only the realization source changes (generated vs
//! handwritten). This test self-skips without an adapter unless
//! `FLAT_REQUIRE_WGPU` demands real execution.

#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference, AttentionProblem, AttentionShape, FlatAttentionConfig, FlatKernelIr,
    KernelVariantIdentity, PrecisionPolicy, ReductionStrategy,
};

const ATOL: f32 = 5.0e-5;
const RTOL: f32 = 5.0e-4;

fn canonical_ir(seq_len: usize, head_dim: usize) -> FlatKernelIr {
    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len,
        head_dim,
    };
    let problem = AttentionProblem::from_shape(&shape, FlatAttentionConfig::default())
        .expect("shape converts");
    let plan = flat_attention::ExecutionPlan::build(
        flat_attention::TileConfig {
            query_rows: 4,
            kv_tile: 8,
        },
        flat_attention::WorkgroupGeometry { invocations: 64 },
        ReductionStrategy::TreeInWorkgroup,
        PrecisionPolicy::F32StorageF32Accumulate,
        false,
    )
    .expect("canonical plan");
    FlatKernelIr::build(KernelVariantIdentity::portable_q4(), problem, plan).expect("canonical IR")
}

fn fixture(shape: &AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| ((i as f32) * 0.031 - 0.4).sin() * 0.9)
        .collect();
    let k = (0..len)
        .map(|i| ((i as f32) * 0.043 + 0.2).cos() * 0.8)
        .collect();
    let v = (0..len)
        .map(|i| ((i as f32) * 0.017 - 0.6).sin() * 1.3)
        .collect();
    (q, k, v)
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        let tolerance = ATOL + RTOL * expected_value.abs();
        let error = (actual_value - expected_value).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual_value}, expected={expected_value}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

#[test]
fn generated_kernel_matches_oracle_on_device() {
    let context = match flat_attention::WgpuFlatAttention::with_generated_portable_q4_kernel(
        &canonical_ir(17, 64),
    ) {
        Ok(context) => context,
        Err(flat_attention::WgpuFlatAttentionError::Unavailable)
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; generated-parity test skipped");
            return;
        }
        Err(error) => panic!("generated-kernel context creation failed: {error}"),
    };

    // The executing pipeline must really be the generated one.
    assert!(context.generated_kernel_cache_key().is_some());
    assert_eq!(
        context.kernel_variant(),
        flat_attention::WgpuKernelVariant::Q4Portable
    );

    for &(seq_len, head_dim) in &[(1usize, 64usize), (7, 64), (16, 64), (33, 64), (64, 128)] {
        let shape = AttentionShape {
            batch: 1,
            heads: 2,
            seq_len,
            head_dim,
        };
        let (q, k, v) = fixture(&shape);
        // Rebuild the IR per case so the problem geometry matches exactly;
        // the kernel specialization (tiles/workgroup/reduction) is constant.
        let context_for_shape =
            match flat_attention::WgpuFlatAttention::with_generated_portable_q4_kernel(
                &canonical_ir(seq_len, head_dim),
            ) {
                Ok(context) => Some(context),
                Err(flat_attention::WgpuFlatAttentionError::Unavailable)
                    if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
                {
                    eprintln!("WGPU adapter unavailable; generated-parity test skipped");
                    None
                }
                Err(error) => panic!("context creation failed at ({seq_len},{head_dim}): {error}"),
            };
        let Some(context) = context_for_shape else {
            return;
        };
        for causal in [false, true] {
            let config = FlatAttentionConfig {
                causal,
                softmax_scale: None,
            };
            let oracle = forward_reference(&q, &k, &v, shape, config).expect("oracle");
            let output = context.forward(&q, &k, &v, shape, config).expect("device");
            assert_close(
                &format!("output[n={seq_len},d={head_dim},causal={causal}]"),
                &output.output,
                &oracle.output,
            );
            assert_close(
                &format!("lse[n={seq_len},d={head_dim},causal={causal}]"),
                &output.lse,
                &oracle.lse,
            );
        }
    }
}
