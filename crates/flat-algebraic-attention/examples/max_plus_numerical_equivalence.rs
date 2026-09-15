//! MAA-10c final-readiness numerical equivalence; not a performance benchmark.
//! Protocol: docs/research/MAA_MAX_PLUS_NUMERICAL_EQUIVALENCE_PREREGISTRATION.md.

use std::error::Error;

use flat_algebraic_attention::cooperation::{
    route_attention_needs, AttentionAlgebraNeeds, M13bBooleanRoutingEvidence,
};
use flat_algebraic_attention::f2::{F2AffinePredicate, F2Vector};
use flat_algebraic_attention::max_plus::{MaxPlusEdge, MaxPlusSchedule, MaxPlusValue};
use flat_algebraic_attention::qualification::{
    CandidateQualificationInputs, F2CandidateEvaluation, RecompositionPolicy,
    ZhegalkinCandidateEvaluation,
};
use flat_algebraic_attention::readiness::{
    plan_survivor_readiness, ReadinessSnapshot, ScheduledSurvivor, SurvivorNodeBinding,
};
use flat_algebraic_attention::survivor_set::{qualify_survivor_set, CandidateFrame, SurvivorSet};
use flat_algebraic_attention::zhegalkin::ZhegalkinPolynomial;
use flat_attention::api::boolean_attention_mask::{
    BooleanAttentionMask, BOOLEAN_ATTENTION_MASK_SCHEMA_VERSION,
};
use flat_attention::api::boolean_attention_signature::{
    BooleanAttentionSignature, BOOLEAN_ATTENTION_SIGNATURE_SCHEMA_VERSION,
};
use flat_attention::{
    forward_reference_grouped_asymmetric, AsymmetricGroupedAttentionShape, FlatAttentionConfig,
    FlatAttentionOutput,
};

const PROTOCOL: &str = "maa-maxplus-numerical-equivalence/v1";
const N: usize = 6;
const D: usize = 2;
const POLICY: RecompositionPolicy = RecompositionPolicy::AllSelectedMustQualify;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn require(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn semantic_survivors() -> Result<SurvivorSet> {
    let admissions = [true, true, true, true, true, false];
    let mask = BooleanAttentionMask::from_admissions(&admissions)?;
    let query_policy = BooleanAttentionSignature::new(N, vec![(1u64 << N) - 1])?;
    let key_policy = BooleanAttentionSignature::new(N, mask.words().to_vec())?;
    let route = route_attention_needs(
        AttentionAlgebraNeeds {
            eligibility_logic: true,
            parity_or_binary_linear: true,
            nonlinear_boolean_interaction: true,
            precedence_or_critical_path: false,
        },
        Some(M13bBooleanRoutingEvidence::new(
            BOOLEAN_ATTENTION_MASK_SCHEMA_VERSION,
            N,
            mask.words().to_vec(),
            BOOLEAN_ATTENTION_SIGNATURE_SCHEMA_VERSION,
            N,
            query_policy.words().to_vec(),
            key_policy.words().to_vec(),
            None,
            None,
        )?),
    )?;

    // F2 qualifies iff feature 0 is true.
    let f2 = F2AffinePredicate::new(F2Vector::from_bools(&[true, false, false])?, false);
    // Zhegalkin qualifies iff features 1 and 2 are both true.
    let zhegalkin = ZhegalkinPolynomial::from_variable_sets(3, vec![vec![1, 2]])?;
    let features = [
        [true, true, true],
        [false, true, true], // rejected by F2
        [true, true, true],
        [true, true, true],
        [true, true, false], // rejected by Zhegalkin
        [true, true, true],  // rejected by Boolean
    ];
    let inputs = features
        .iter()
        .map(|feature| F2Vector::from_bools(feature))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let frames = inputs
        .iter()
        .enumerate()
        .map(|(candidate_index, input)| {
            CandidateFrame::new(
                candidate_index,
                CandidateQualificationInputs {
                    boolean_block: Some(candidate_index),
                    f2: Some(F2CandidateEvaluation {
                        predicate: &f2,
                        input,
                    }),
                    zhegalkin: Some(ZhegalkinCandidateEvaluation {
                        polynomial: &zhegalkin,
                        input,
                    }),
                    max_plus: None,
                },
            )
        })
        .collect::<Vec<_>>();

    let survivors = qualify_survivor_set(&route, &frames, POLICY)?;
    require(
        survivors.survivor_indices() == [0, 2, 3],
        "semantic survivor set drifted",
    )?;
    let counts = survivors.rejection_counts();
    require(
        counts.boolean() == 1 && counts.f2() == 1 && counts.zhegalkin() == 1,
        "semantic rejection attribution drifted",
    )?;
    require(
        counts.max_plus() == 0,
        "semantic survivor set must not be Max-Plus prequalified",
    )?;
    Ok(survivors)
}

fn evaluate(indices: &[usize]) -> Result<FlatAttentionOutput> {
    require(!indices.is_empty(), "final numerical selection must not be empty")?;
    require(
        indices.iter().all(|&candidate_index| candidate_index < N),
        "candidate index out of bounds",
    )?;
    require(
        indices.windows(2).all(|pair| pair[0] < pair[1]),
        "numerical candidate IDs must remain unique and ordered",
    )?;

    let q = [1.0f32, -0.5];
    let full_k = [
        0.5, -0.2, // 0
        1.5, 0.3, // 1
        -0.25, 1.0, // 2
        2.0, -1.0, // 3
        0.0, 0.75, // 4
        -1.0, -0.5, // 5
    ];
    let full_v = [
        1.0, 0.0, // 0
        0.0, 2.0, // 1
        -1.0, 0.5, // 2
        3.0, -2.0, // 3
        0.25, 1.5, // 4
        -2.0, 1.0, // 5
    ];
    let mut k = Vec::with_capacity(indices.len() * D);
    let mut v = Vec::with_capacity(indices.len() * D);
    for &candidate_index in indices {
        k.extend_from_slice(&full_k[candidate_index * D..(candidate_index + 1) * D]);
        v.extend_from_slice(&full_v[candidate_index * D..(candidate_index + 1) * D]);
    }

    Ok(forward_reference_grouped_asymmetric(
        &q,
        &k,
        &v,
        AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            query_len: 1,
            kv_len: indices.len(),
            head_dim: D,
            query_position_offset: N - 1,
        },
        FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(1.0),
        },
    )?)
}

fn complete_numerical_indices(snapshot: &ReadinessSnapshot) -> Option<&[usize]> {
    snapshot.is_complete().then(|| snapshot.ready_indices())
}

fn ids(indices: &[usize]) -> String {
    if indices.is_empty() {
        "-".to_owned()
    } else {
        indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("|")
    }
}

fn run() -> Result<Vec<String>> {
    let survivors = semantic_survivors()?;
    let original_indices = survivors.survivor_indices();
    let baseline = evaluate(original_indices)?;

    let schedule = MaxPlusSchedule::new(
        3,
        vec![MaxPlusEdge::new(0, 1, 2), MaxPlusEdge::new(1, 2, 3)],
    )?;
    let plan = plan_survivor_readiness(
        &survivors,
        &schedule,
        &[
            MaxPlusValue::Finite(0),
            MaxPlusValue::ZERO,
            MaxPlusValue::ZERO,
        ],
        &[
            SurvivorNodeBinding::new(3, 0),
            SurvivorNodeBinding::new(0, 1),
            SurvivorNodeBinding::new(2, 2),
        ],
    )?;

    let readiness_order = plan
        .readiness_order()
        .iter()
        .map(ScheduledSurvivor::candidate_index)
        .collect::<Vec<_>>();
    require(
        readiness_order == [3, 0, 2],
        "temporal readiness order drifted",
    )?;
    require(
        plan.critical_ready_time() == Some(5),
        "critical readiness coordinate drifted",
    )?;

    let mut rows = Vec::new();
    for cutoff in [0, 2, 4, 5] {
        let snapshot = plan.snapshot_at(cutoff);
        let complete = complete_numerical_indices(&snapshot);
        let (status, final_ids, output_equal, lse_equal) = match complete {
            None => ("deferred_nonfinal", "-".to_owned(), "", ""),
            Some(indices) => {
                require(
                    indices == original_indices,
                    "complete readiness snapshot changed numerical candidate identity/order",
                )?;
                let final_output = evaluate(indices)?;
                let output_equal = final_output.output == baseline.output;
                let lse_equal = final_output.lse == baseline.lse;
                require(
                    output_equal && lse_equal,
                    "complete readiness snapshot changed FLAT O/LSE",
                )?;
                (
                    "complete_final",
                    ids(indices),
                    if output_equal { "true" } else { "false" },
                    if lse_equal { "true" } else { "false" },
                )
            }
        };
        rows.push(format!(
            "{PROTOCOL},{cutoff},{status},{},{},{},{},{},{}",
            ids(snapshot.ready_indices()),
            ids(snapshot.deferred_indices()),
            ids(&readiness_order),
            final_ids,
            output_equal,
            lse_equal
        ));
    }

    require(
        rows[..3]
            .iter()
            .all(|row| row.contains(",deferred_nonfinal,")),
        "intermediate snapshot was published as final",
    )?;
    require(
        rows[3].contains(",complete_final,") && rows[3].ends_with(",true,true"),
        "critical snapshot did not close numerical equivalence",
    )?;
    Ok(rows)
}

fn main() -> Result<()> {
    println!(
        "protocol,cutoff,status,ready_ids,deferred_ids,readiness_order,final_candidate_ids,output_exact,lse_exact"
    );
    for row in run()? {
        println!("{row}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_readiness_is_exactly_numerically_equivalent() {
        let rows = run().unwrap();
        assert_eq!(rows.len(), 4);
        assert!(rows[0].contains(",0,deferred_nonfinal,3,0|2,"));
        assert!(rows[1].contains(",2,deferred_nonfinal,0|3,2,"));
        assert!(rows[2].contains(",4,deferred_nonfinal,0|3,2,"));
        assert!(rows[3].contains(",5,complete_final,0|2|3,-,"));
        assert!(rows[3].ends_with(",0|2|3,true,true"));
    }

    #[test]
    fn incomplete_snapshots_cannot_supply_final_numerical_indices() {
        let survivors = semantic_survivors().unwrap();
        let schedule = MaxPlusSchedule::new(
            3,
            vec![MaxPlusEdge::new(0, 1, 2), MaxPlusEdge::new(1, 2, 3)],
        )
        .unwrap();
        let plan = plan_survivor_readiness(
            &survivors,
            &schedule,
            &[
                MaxPlusValue::Finite(0),
                MaxPlusValue::ZERO,
                MaxPlusValue::ZERO,
            ],
            &[
                SurvivorNodeBinding::new(3, 0),
                SurvivorNodeBinding::new(0, 1),
                SurvivorNodeBinding::new(2, 2),
            ],
        )
        .unwrap();

        for cutoff in [0, 2, 4] {
            assert!(complete_numerical_indices(&plan.snapshot_at(cutoff)).is_none());
        }
        assert_eq!(
            complete_numerical_indices(&plan.snapshot_at(5)),
            Some(&[0, 2, 3][..])
        );
    }

    #[test]
    fn csv_rows_are_repeatable() {
        assert_eq!(run().unwrap(), run().unwrap());
    }
}
