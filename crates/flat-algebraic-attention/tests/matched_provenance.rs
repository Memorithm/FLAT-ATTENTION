use std::collections::BTreeSet;

use flat_algebraic_attention::cooperation::{
    route_attention_needs, AlgebraicRoute, AttentionAlgebraNeeds, M13bBooleanRoutingEvidence,
};
use flat_algebraic_attention::evidence::{
    ArmEvidence, EvidenceArm, EvidenceError, LatencyObservation, MatchedAttentionEvidence,
    MatchedWorkloadIdentity,
};
use flat_algebraic_attention::experiment::{derive_matched_host_evidence, MatchedExperimentError};
use flat_algebraic_attention::f2::{F2AffinePredicate, F2Vector};
use flat_algebraic_attention::qualification::{
    CandidateQualificationInputs, F2CandidateEvaluation, RecompositionPolicy,
};
use flat_algebraic_attention::survivor_set::{qualify_survivor_set, CandidateFrame, SurvivorSet};

const POLICY: RecompositionPolicy = RecompositionPolicy::AllSelectedMustQualify;

fn route(blocks: usize, words: Vec<u64>, generation: u64, signature: u64) -> AlgebraicRoute {
    let boolean = M13bBooleanRoutingEvidence::new(
        1,
        blocks,
        words,
        1,
        1,
        vec![signature],
        vec![1],
        Some(1),
        Some(generation),
    )
    .unwrap();
    route_attention_needs(
        AttentionAlgebraNeeds {
            eligibility_logic: true,
            parity_or_binary_linear: true,
            ..AttentionAlgebraNeeds::default()
        },
        Some(boolean),
    )
    .unwrap()
}

fn batch(route: &AlgebraicRoute, mapping: &[(usize, usize)], pass: &[bool]) -> SurvivorSet {
    let predicate = F2AffinePredicate::new(F2Vector::from_bools(&[true]).unwrap(), false);
    let yes = F2Vector::from_bools(&[true]).unwrap();
    let no = F2Vector::from_bools(&[false]).unwrap();
    let candidates: Vec<_> = mapping
        .iter()
        .map(|&(candidate_index, block)| {
            CandidateFrame::new(
                candidate_index,
                CandidateQualificationInputs {
                    boolean_block: Some(block),
                    f2: Some(F2CandidateEvaluation {
                        predicate: &predicate,
                        input: if pass[block] { &yes } else { &no },
                    }),
                    ..CandidateQualificationInputs::default()
                },
            )
        })
        .collect();
    qualify_survivor_set(route, &candidates, POLICY).unwrap()
}

fn identity(n: usize) -> MatchedWorkloadIdentity {
    MatchedWorkloadIdentity::new("maa-9", "synthetic-contract", "fixture:not-a-digest", n).unwrap()
}

fn derive(
    route: &AlgebraicRoute,
    survivors: &SurvivorSet,
    relevant: &[usize],
) -> Result<MatchedAttentionEvidence, MatchedExperimentError> {
    let n = route.boolean_evidence().unwrap().blocks();
    derive_matched_host_evidence(identity(n), relevant, route, POLICY, survivors, None)
}

fn canonical(n: usize) -> Vec<(usize, usize)> {
    (0..n).map(|i| (i, i)).collect()
}

#[test]
fn equal_cardinality_candidate_substitution_fails_closed() {
    let route = route(4, vec![0b0101], 7, 1);
    let mapping = [(1, 0), (0, 1), (3, 2), (2, 3)];
    let survivors = batch(&route, &mapping, &[true; 4]);
    assert_eq!(survivors.survivor_indices(), &[1, 3]);
    assert_eq!(survivors.survivor_count(), 2);
    assert_eq!(
        derive(&route, &survivors, &[]),
        Err(MatchedExperimentError::Evidence(
            EvidenceError::NonCanonicalBooleanMapping { candidate_index: 1 }
        ))
    );
}

#[test]
fn rejected_candidates_also_require_identity_mapping() {
    let route = route(4, vec![0b0101], 7, 1);
    let survivors = batch(&route, &[(1, 0), (0, 1), (2, 2), (3, 3)], &[false; 4]);
    assert!(survivors.survivor_indices().is_empty());
    assert_eq!(
        derive(&route, &survivors, &[]),
        Err(MatchedExperimentError::Evidence(
            EvidenceError::NonCanonicalBooleanMapping { candidate_index: 1 }
        ))
    );
}

#[test]
fn different_mask_with_equal_counts_cannot_relabel_an_oracle() {
    let source = route(4, vec![0b0101], 7, 1);
    let target = route(4, vec![0b1010], 7, 1);
    let survivors = batch(&source, &canonical(4), &[true; 4]);
    assert_eq!(
        derive(&target, &survivors, &[]),
        Err(MatchedExperimentError::Evidence(
            EvidenceError::OracleRouteMismatch
        ))
    );
}

#[test]
fn metadata_generation_and_signature_are_bound_even_when_masks_match() {
    let source = route(4, vec![0b0101], 7, 1);
    let survivors = batch(&source, &canonical(4), &[true; 4]);
    let targets = [route(4, vec![0b0101], 8, 1), route(4, vec![0b0101], 7, 0)];
    for target in targets {
        assert_eq!(
            derive(&target, &survivors, &[]),
            Err(MatchedExperimentError::Evidence(
                EvidenceError::OracleRouteMismatch
            ))
        );
    }
    assert_eq!(survivors.qualification_route(), &source);
    assert_eq!(survivors.qualification_policy(), POLICY);
    assert!(derive(&source.clone(), &survivors, &[]).is_ok());
}

#[test]
fn another_domain_selection_cannot_relabel_an_oracle() {
    let source = route(4, vec![0b0101], 7, 1);
    let survivors = batch(&source, &canonical(4), &[true; 4]);
    let target = route_attention_needs(
        AttentionAlgebraNeeds {
            eligibility_logic: true,
            parity_or_binary_linear: true,
            nonlinear_boolean_interaction: true,
            ..AttentionAlgebraNeeds::default()
        },
        source.boolean_evidence().cloned(),
    )
    .unwrap();
    assert_eq!(
        derive(&target, &survivors, &[]),
        Err(MatchedExperimentError::Evidence(
            EvidenceError::OracleRouteMismatch
        ))
    );
}

#[test]
fn input_order_may_change_without_changing_candidate_identity() {
    let route = route(4, vec![0b0101], 7, 1);
    let survivors = batch(&route, &[(3, 3), (1, 1), (2, 2), (0, 0)], &[true; 4]);
    assert_eq!(survivors.survivor_indices(), &[2, 0]);
    assert_eq!(survivors.first_noncanonical_boolean_candidate(), None);
    assert_eq!(
        derive(&route, &survivors, &[0, 2])
            .unwrap()
            .multi_algebra()
            .retained_relevant(),
        2
    );
}

#[test]
fn low_level_constructor_cannot_forge_boolean_control_count() {
    let route = route(4, vec![0b0101], 7, 1);
    let survivors = batch(&route, &canonical(4), &[true; 4]);
    let dense = ArmEvidence::new(EvidenceArm::DenseReference, 4, 4, 4, 0, 0, None).unwrap();
    let boolean = ArmEvidence::new(EvidenceArm::BooleanOnlyControl, 4, 3, 3, 0, 0, None).unwrap();
    let multi = ArmEvidence::new(EvidenceArm::MultiAlgebraCandidate, 4, 2, 2, 0, 0, None).unwrap();
    assert_eq!(
        MatchedAttentionEvidence::new(
            identity(4),
            dense,
            boolean,
            multi,
            &route,
            POLICY,
            &survivors
        ),
        Err(EvidenceError::OracleBooleanSurvivorMismatch {
            evidence: 3,
            oracle: 2
        })
    );
}

#[test]
fn nested_relevance_cannot_lose_more_labels_than_removed_candidates() {
    let route = route(4, vec![0b0111], 7, 1);
    let survivors = batch(&route, &canonical(4), &[true, true, false, true]);
    let dense = ArmEvidence::new(EvidenceArm::DenseReference, 4, 4, 4, 2, 2, None).unwrap();
    let boolean = ArmEvidence::new(EvidenceArm::BooleanOnlyControl, 4, 3, 3, 2, 2, None).unwrap();
    let multi = ArmEvidence::new(EvidenceArm::MultiAlgebraCandidate, 4, 2, 2, 2, 0, None).unwrap();
    assert_eq!(
        MatchedAttentionEvidence::new(
            identity(4),
            dense,
            boolean,
            multi,
            &route,
            POLICY,
            &survivors
        ),
        Err(EvidenceError::NonNestedRelevantCounts {
            boolean_only: 2,
            multi_algebra: 0,
            minimum: 1,
        })
    );
}

#[test]
fn all_four_bit_masks_predicates_and_labels_match_independent_sets() {
    let mut records = 0;
    for mask in 0u64..16 {
        let route = route(4, vec![mask], 7, 1);
        for predicate_mask in 0u64..16 {
            let pass: Vec<_> = (0..4).map(|i| predicate_mask & (1 << i) != 0).collect();
            let survivors = batch(&route, &canonical(4), &pass);
            let expected: Vec<_> = (0..4)
                .filter(|i| mask & predicate_mask & (1 << i) != 0)
                .collect();
            assert_eq!(survivors.survivor_indices(), expected);
            for relevant_mask in 0u64..16 {
                let relevant: Vec<_> = (0..4).filter(|i| relevant_mask & (1 << i) != 0).collect();
                let evidence = derive(&route, &survivors, &relevant).unwrap();
                assert_eq!(
                    evidence.boolean_only().survivor_count(),
                    mask.count_ones() as usize
                );
                assert_eq!(evidence.multi_algebra().survivor_count(), expected.len());
                assert_eq!(
                    evidence.multi_algebra().retained_relevant(),
                    (mask & predicate_mask & relevant_mask).count_ones() as usize
                );
                assert!(!evidence.has_matched_latency());
                records += 1;
            }
        }
    }
    assert_eq!(records, 4096);
}

#[test]
fn packed_word_boundary_preserves_exact_candidate_membership() {
    let route = route(65, vec![1u64 << 63, 1], 7, 1);
    let survivors = batch(&route, &canonical(65), &[true; 65]);
    assert_eq!(survivors.survivor_indices(), &[63, 64]);
    let evidence = derive(&route, &survivors, &[0, 63, 64]).unwrap();
    assert_eq!(evidence.multi_algebra().retained_relevant(), 2);
    assert_eq!(evidence.boolean_only().survivor_count(), 2);
}

#[test]
fn intersection_bounds_match_exhaustive_sets_not_just_inequalities() {
    for n in 1u32..=6 {
        let mut possible = BTreeSet::new();
        for a in 0u32..(1 << n) {
            for b in 0u32..(1 << n) {
                possible.insert((a.count_ones(), b.count_ones(), (a & b).count_ones()));
            }
        }
        for survivors in 0..=n {
            for relevant in 0..=n {
                for retained in 0..=n {
                    let evidence = ArmEvidence::new(
                        EvidenceArm::MultiAlgebraCandidate,
                        n as usize,
                        survivors as usize,
                        survivors as usize,
                        relevant as usize,
                        retained as usize,
                        None,
                    );
                    assert_eq!(
                        evidence.is_ok(),
                        possible.contains(&(survivors, relevant, retained))
                    );
                }
            }
        }
    }
}

#[test]
fn observed_latency_extrema_match_exhaustive_sample_sequences() {
    for count in 1u32..=4 {
        let mut possible = BTreeSet::new();
        for encoded in 0u64..4u64.pow(count) {
            let mut value = encoded;
            let mut samples = Vec::new();
            for _ in 0..count {
                samples.push(value % 4);
                value /= 4;
            }
            let total: u128 = samples.iter().map(|&x| u128::from(x)).sum();
            possible.insert((
                total,
                *samples.iter().min().unwrap(),
                *samples.iter().max().unwrap(),
            ));
        }
        for min in 0..4 {
            for max in 0..4 {
                for total in 0..=u128::from(count) * 3 {
                    let observation = LatencyObservation::new(u64::from(count), total, min, max);
                    assert_eq!(observation.is_ok(), possible.contains(&(total, min, max)));
                }
            }
        }
    }
}

#[test]
fn latency_extreme_integer_values_do_not_overflow() {
    let value = u64::MAX;
    let total = u128::from(value) * u128::from(value);
    assert!(LatencyObservation::new(value, total, value, value).is_ok());
    assert!(LatencyObservation::new(1, 10, 9, 11).is_err());
    assert!(LatencyObservation::new(2, 20, 9, 12).is_err());
    assert!(LatencyObservation::new(10, 1_000, 90, 200).is_err());
}
