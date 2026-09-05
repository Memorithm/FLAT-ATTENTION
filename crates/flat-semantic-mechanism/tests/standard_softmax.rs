use flat_attention::{forward_reference, AttentionShape, FlatAttentionConfig};
use flat_semantic::v1::{SemanticSavedState, StandardSoftmaxSemantic};
use flat_semantic_mechanism::{MechanismComponentKind, StandardSoftmaxMechanism};

#[test]
fn public_mechanism_metadata_preserves_legacy_standard_softmax_bits() {
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 3,
        head_dim: 2,
    };
    let q = vec![0.5, -1.0, 1.25, 0.75, -0.25, 0.5];
    let k = vec![1.0, 0.25, -0.5, 1.5, 0.75, -1.25];
    let v = vec![1.0, -2.0, 0.5, 0.25, -0.75, 1.5];
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: Some(0.625),
    };

    let legacy = forward_reference(&q, &k, &v, shape, config).unwrap();
    let semantic = StandardSoftmaxSemantic::from_flat_config(config, shape.head_dim).unwrap();
    let mechanism = StandardSoftmaxMechanism::new(semantic);
    let generic = mechanism
        .semantic()
        .forward_reference(&q, &k, &v, shape)
        .unwrap();
    let (output, saved) = generic.into_parts();

    assert_eq!(output, legacy.output);
    match saved {
        SemanticSavedState::LogSumExp(lse) => assert_eq!(lse, legacy.lse),
        _ => panic!("StandardSoftmax must retain its historical LSE state"),
    }

    let descriptor = mechanism.descriptor();
    assert_eq!(descriptor.projection().kind(), MechanismComponentKind::Projection);
    assert_eq!(descriptor.score().kind(), MechanismComponentKind::Score);
    assert_eq!(
        descriptor.normalization().kind(),
        MechanismComponentKind::Normalization
    );
    assert_eq!(descriptor.mixing().kind(), MechanismComponentKind::Mixing);
    assert_eq!(
        descriptor.numerical_policy().kind(),
        MechanismComponentKind::NumericalPolicy
    );

    let record = mechanism.canonical_record();
    assert!(record.contains("standard-softmax"));
    assert!(record.contains("scaled-dot-product"));
    assert!(record.contains("row-softmax"));
    assert!(!record.contains("wgpu"));
    assert!(!record.contains("device"));
    assert!(!record.contains("kernel"));
}
