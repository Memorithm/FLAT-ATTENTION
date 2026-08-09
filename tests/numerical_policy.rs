use flat_attention::{
    AttentionShape, FlatAttentionConfig, FlatAttentionError, NumericalBackendKind, NumericalError,
    NumericalExecutor, NumericalMode,
};

fn assert_finite(values: &[f32], name: &str) {
    for (index, value) in values.iter().enumerate() {
        assert!(value.is_finite(), "{name}[{index}] is not finite: {value}");
    }
}

fn fixture(shape: AttentionShape, kind: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.batch * shape.heads * shape.seq_len * shape.head_dim;
    match kind {
        // Exactly equal QK scores: softmax should remain stable with ties.
        0 => (
            vec![0.0; len],
            vec![1.0; len],
            (0..len).map(|i| (i % 17) as f32 * 0.125 - 1.0).collect(),
        ),
        // Near-tied scores and small values around zero.
        1 => (
            (0..len)
                .map(|i| (i as f32 * 1.0e-5).sin() * 1.0e-2)
                .collect(),
            (0..len)
                .map(|i| (i as f32 * 1.0e-5).cos() * 1.0e-2)
                .collect(),
            (0..len)
                .map(|i| ((i as f32) * 0.031).sin())
                .collect(),
        ),
        // Alternating signs exercise cancellation in QK and P*V.
        2 => (
            (0..len)
                .map(|i| if i % 2 == 0 { 7.75 } else { -7.75 })
                .collect(),
            (0..len)
                .map(|i| if (i / 3) % 2 == 0 { -6.5 } else { 6.5 })
                .collect(),
            (0..len)
                .map(|i| if (i / 5) % 2 == 0 { 3.0 } else { -3.0 })
                .collect(),
        ),
        // Large but finite scores stress online max rescaling without dot overflow.
        3 => (
            (0..len)
                .map(|i| if i % 2 == 0 { 24.0 } else { -24.0 })
                .collect(),
            (0..len)
                .map(|i| if (i / 2) % 2 == 0 { 22.0 } else { -22.0 })
                .collect(),
            (0..len)
                .map(|i| ((i as f32) * 0.017).cos() * 4.0)
                .collect(),
        ),
        _ => unreachable!(),
    }
}

#[test]
fn exact_reference_corpus_is_finite_and_bit_repeatable() {
    let executor = NumericalExecutor::new(NumericalMode::ExactReference).unwrap();
    assert_eq!(executor.backend_kind(), NumericalBackendKind::ReferenceCpu);

    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 17,
        head_dim: 32,
    };
    for kind in 0..4 {
        let (q, k, v) = fixture(shape, kind);
        for causal in [false, true] {
            for softmax_scale in [None, Some(1.0e-3), Some(0.25), Some(2.0)] {
                let config = FlatAttentionConfig {
                    causal,
                    softmax_scale,
                };
                let first = executor.forward(&q, &k, &v, shape, config).unwrap();
                let second = executor.forward(&q, &k, &v, shape, config).unwrap();
                assert_finite(&first.output, "exact O");
                assert_finite(&first.lse, "exact LSE");
                assert_eq!(
                    first
                        .output
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    second
                        .output
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    first
                        .lse
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    second
                        .lse
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn exact_mode_preserves_validation_errors() {
    let executor = NumericalExecutor::new(NumericalMode::ExactReference).unwrap();
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 1,
        head_dim: 2,
    };
    let q = [f32::NAN, 0.0];
    let k = [0.0, 0.0];
    let v = [0.0, 0.0];
    let error = executor
        .forward(&q, &k, &v, shape, FlatAttentionConfig::default())
        .unwrap_err();
    assert!(matches!(
        error,
        NumericalError::Core(FlatAttentionError::NonFiniteInput {
            tensor: "Q",
            index: 0
        })
    ));

    let error = executor
        .forward(
            &[0.0, 0.0],
            &k,
            &v,
            shape,
            FlatAttentionConfig {
                causal: false,
                softmax_scale: Some(0.0),
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        NumericalError::Core(FlatAttentionError::InvalidScale(0.0))
    ));
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gpu_modes_fail_explicitly_without_wgpu_feature() {
    assert!(matches!(
        NumericalExecutor::new(NumericalMode::FastPortable),
        Err(NumericalError::GpuFeatureDisabled)
    ));
    assert!(matches!(
        NumericalExecutor::new(NumericalMode::DeterministicPortable),
        Err(NumericalError::GpuFeatureDisabled)
    ));
}
