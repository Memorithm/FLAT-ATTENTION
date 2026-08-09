#![cfg(feature = "wgpu")]

use flat_attention::{
    forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionError,
    NumericalBackendKind, NumericalError, NumericalExecutor, NumericalMode, WgpuFlatAttentionError,
    WgpuKernelVariant,
};

const ATOL: f32 = 5.0e-5;
const RTOL: f32 = 5.0e-4;

fn executor(mode: NumericalMode) -> Option<NumericalExecutor> {
    match NumericalExecutor::new(mode) {
        Ok(executor) => Some(executor),
        Err(NumericalError::Wgpu(WgpuFlatAttentionError::Unavailable))
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() =>
        {
            eprintln!("WGPU adapter unavailable; optional M9 device test skipped");
            None
        }
        Err(error) => panic!("M9 {mode:?} executor creation failed: {error}"),
    }
}

fn fixture(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    let q = (0..len)
        .map(|i| {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            sign * (((i as f32) * 0.027).sin() * 3.5 + 0.125)
        })
        .collect();
    let k = (0..len)
        .map(|i| {
            let sign = if (i / 5) % 2 == 0 { 1.0 } else { -1.0 };
            sign * (((i as f32) * 0.019).cos() * 2.75 - 0.0625)
        })
        .collect();
    let v = (0..len).map(|i| ((i as f32) * 0.013).sin() * 4.0).collect();
    (q, k, v)
}

fn assert_bits_equal(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{name}[{index}] bit mismatch: actual={actual:?}, expected={expected:?}"
        );
    }
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
fn deterministic_mode_forces_fixed_tree_kernel_family() {
    let Some(executor) = executor(NumericalMode::DeterministicPortable) else {
        return;
    };
    assert_eq!(executor.backend_kind(), NumericalBackendKind::Wgpu);
    assert_eq!(executor.mode(), NumericalMode::DeterministicPortable);
    assert!(executor.guarantees().repeatable_same_backend_device);
    assert!(!executor.guarantees().allows_subgroup);
    assert_eq!(
        executor.kernel_variant_for_head_dim(64),
        Some(WgpuKernelVariant::Q4Vec4Portable)
    );
    assert_eq!(
        executor.kernel_variant_for_head_dim(80),
        Some(WgpuKernelVariant::Q4Portable)
    );
}

#[test]
fn deterministic_mode_is_bit_repeatable_on_same_context() {
    let Some(executor) = executor(NumericalMode::DeterministicPortable) else {
        return;
    };
    eprintln!("M9 deterministic adapter: {:?}", executor.adapter_name());

    for (seq_len, head_dim) in [(17, 64), (33, 80), (9, 128)] {
        let shape = AttentionShape {
            batch: 1,
            heads: 2,
            seq_len,
            head_dim,
        };
        let (q, k, v) = fixture(shape);
        for causal in [false, true] {
            let config = FlatAttentionConfig {
                causal,
                softmax_scale: Some(0.125),
            };
            let first = executor.forward(&q, &k, &v, shape, config).unwrap();
            for iteration in 0..4 {
                let repeated = executor.forward(&q, &k, &v, shape, config).unwrap();
                assert_bits_equal(
                    &format!("O seq={seq_len} d={head_dim} causal={causal} iter={iteration}"),
                    &repeated.output,
                    &first.output,
                );
                assert_bits_equal(
                    &format!("LSE seq={seq_len} d={head_dim} causal={causal} iter={iteration}"),
                    &repeated.lse,
                    &first.lse,
                );
            }

            let reference = forward_reference(&q, &k, &v, shape, config).unwrap();
            assert_close("deterministic O parity", &first.output, &reference.output);
            assert_close("deterministic LSE parity", &first.lse, &reference.lse);
        }
    }
}

#[test]
fn fast_mode_remains_explicit_gpu_execution() {
    let Some(executor) = executor(NumericalMode::FastPortable) else {
        return;
    };
    assert_eq!(executor.backend_kind(), NumericalBackendKind::Wgpu);
    assert!(executor.guarantees().allows_subgroup);
    assert!(!executor.guarantees().repeatable_same_backend_device);

    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 17,
        head_dim: 64,
    };
    let (q, k, v) = fixture(shape);
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let actual = executor.forward(&q, &k, &v, shape, config).unwrap();
    let reference = forward_reference(&q, &k, &v, shape, config).unwrap();
    assert_close("fast O parity", &actual.output, &reference.output);
    assert_close("fast LSE parity", &actual.lse, &reference.lse);
}

#[test]
fn deterministic_mode_preserves_host_input_validation() {
    let Some(executor) = executor(NumericalMode::DeterministicPortable) else {
        return;
    };
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 1,
        head_dim: 64,
    };
    let mut q = vec![0.0; 64];
    let k = vec![0.0; 64];
    let v = vec![0.0; 64];
    q[7] = f32::NAN;
    let error = executor
        .forward(&q, &k, &v, shape, FlatAttentionConfig::default())
        .unwrap_err();
    assert!(matches!(
        error,
        NumericalError::Wgpu(WgpuFlatAttentionError::Core(
            FlatAttentionError::NonFiniteInput {
                tensor: "Q",
                index: 7
            }
        ))
    ));
}
