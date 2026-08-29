use flat_semantic::v1::{NonlocalHistorySoftmaxSemantic, SemanticFamily, SemanticId};
use flat_semantic_control::{SemanticSelectionDecision, SemanticSelectionPolicy};
use flat_semantic_registry::SemanticRegistry;

fn nonlocal_id() -> SemanticId {
    NonlocalHistorySoftmaxSemantic::semantic_id()
}

#[test]
fn exact_nonlocal_identity_can_be_registered_and_selected() {
    let nonlocal = nonlocal_id();
    let registry = SemanticRegistry::new([nonlocal.clone()]).unwrap();
    let policy = SemanticSelectionPolicy::new([nonlocal.clone()]).unwrap();

    assert_eq!(
        policy.select(&registry),
        SemanticSelectionDecision::Selected {
            semantic: nonlocal,
            preference_rank: 0,
        }
    );
}

#[test]
fn missing_nonlocal_identity_does_not_fallback_to_standard_softmax() {
    let nonlocal = nonlocal_id();
    let standard = SemanticId::new(SemanticFamily::StandardSoftmax, "standard-softmax", 1).unwrap();
    let registry = SemanticRegistry::new([standard]).unwrap();
    let policy = SemanticSelectionPolicy::new([nonlocal]).unwrap();

    assert_eq!(
        policy.select(&registry),
        SemanticSelectionDecision::NoRegisteredPreference
    );
}
