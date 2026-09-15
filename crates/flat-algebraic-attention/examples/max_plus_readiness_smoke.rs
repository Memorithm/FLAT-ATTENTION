use flat_algebraic_attention::cooperation::{route_attention_needs, AttentionAlgebraNeeds};
use flat_algebraic_attention::f2::{F2AffinePredicate, F2Vector};
use flat_algebraic_attention::max_plus::{MaxPlusEdge, MaxPlusSchedule, MaxPlusValue};
use flat_algebraic_attention::qualification::{
    CandidateQualificationInputs, F2CandidateEvaluation, RecompositionPolicy,
};
use flat_algebraic_attention::readiness::{
    plan_survivor_readiness, MaxPlusReadinessPlan, SurvivorNodeBinding,
};
use flat_algebraic_attention::survivor_set::{
    qualify_survivor_set, CandidateFrame, SurvivorSet,
};

const PROTOCOL: &str = "maa-maxplus-readiness/v1";

fn survivor_set(candidate_ids: &[usize]) -> SurvivorSet {
    let route = route_attention_needs(
        AttentionAlgebraNeeds {
            parity_or_binary_linear: true,
            ..AttentionAlgebraNeeds::default()
        },
        None,
    )
    .unwrap();
    let predicate = F2AffinePredicate::new(F2Vector::from_bools(&[true]).unwrap(), false);
    let input = F2Vector::from_bools(&[true]).unwrap();
    let evaluation = F2CandidateEvaluation {
        predicate: &predicate,
        input: &input,
    };
    let frames = candidate_ids
        .iter()
        .copied()
        .map(|candidate_index| {
            CandidateFrame::new(
                candidate_index,
                CandidateQualificationInputs {
                    f2: Some(evaluation),
                    ..CandidateQualificationInputs::default()
                },
            )
        })
        .collect::<Vec<_>>();

    qualify_survivor_set(
        &route,
        &frames,
        RecompositionPolicy::AllSelectedMustQualify,
    )
    .unwrap()
}

fn ids(values: &[usize]) -> String {
    if values.is_empty() {
        "-".to_owned()
    } else {
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("|")
    }
}

fn readiness_order(plan: &MaxPlusReadinessPlan) -> String {
    let ordered = plan
        .readiness_order()
        .iter()
        .map(|survivor| survivor.candidate_index())
        .collect::<Vec<_>>();
    ids(&ordered)
}

fn emit_case(
    case: &str,
    survivors: &SurvivorSet,
    schedule: &MaxPlusSchedule,
    initial: &[MaxPlusValue],
    bindings: &[SurvivorNodeBinding],
    cutoffs: &[i64],
) {
    let plan = plan_survivor_readiness(survivors, schedule, initial, bindings).unwrap();
    let temporal_order = readiness_order(&plan);
    let critical = plan
        .critical_ready_time()
        .map_or_else(|| "none".to_owned(), |value| value.to_string());

    let mut previous_ready = 0;
    for &cutoff in cutoffs {
        let snapshot = plan.snapshot_at(cutoff);
        assert!(snapshot.ready_indices().len() >= previous_ready);
        previous_ready = snapshot.ready_indices().len();
        println!(
            "{PROTOCOL},{case},{cutoff},{},{},{temporal_order},{critical},{}",
            ids(snapshot.ready_indices()),
            ids(snapshot.deferred_indices()),
            snapshot.is_complete()
        );
    }

    match plan.critical_ready_time() {
        Some(cutoff) => {
            let final_snapshot = plan.snapshot_at(cutoff);
            assert_eq!(final_snapshot.ready_indices(), survivors.survivor_indices());
            assert!(final_snapshot.is_complete());
        }
        None => assert!(survivors.survivor_indices().is_empty()),
    }
}

fn main() {
    println!(
        "protocol,case,cutoff,ready_ids,deferred_ids,readiness_order,critical_ready_time,complete"
    );

    let chain_survivors = survivor_set(&[0, 1, 2]);
    let chain_schedule = MaxPlusSchedule::new(
        3,
        vec![MaxPlusEdge::new(0, 1, 3), MaxPlusEdge::new(1, 2, 4)],
    )
    .unwrap();
    emit_case(
        "chain",
        &chain_survivors,
        &chain_schedule,
        &[
            MaxPlusValue::Finite(0),
            MaxPlusValue::ZERO,
            MaxPlusValue::ZERO,
        ],
        &[
            SurvivorNodeBinding::new(0, 0),
            SurvivorNodeBinding::new(1, 1),
            SurvivorNodeBinding::new(2, 2),
        ],
        &[-1, 0, 2, 3, 6, 7],
    );

    let fork_join_survivors = survivor_set(&[0, 1, 2, 3]);
    let fork_join_schedule = MaxPlusSchedule::new(
        4,
        vec![
            MaxPlusEdge::new(0, 1, 2),
            MaxPlusEdge::new(0, 2, 5),
            MaxPlusEdge::new(1, 3, 7),
            MaxPlusEdge::new(2, 3, 1),
        ],
    )
    .unwrap();
    emit_case(
        "fork_join",
        &fork_join_survivors,
        &fork_join_schedule,
        &[
            MaxPlusValue::Finite(0),
            MaxPlusValue::ZERO,
            MaxPlusValue::ZERO,
            MaxPlusValue::ZERO,
        ],
        &[
            SurvivorNodeBinding::new(0, 0),
            SurvivorNodeBinding::new(1, 1),
            SurvivorNodeBinding::new(2, 2),
            SurvivorNodeBinding::new(3, 3),
        ],
        &[0, 2, 5, 8, 9],
    );

    let original_order_survivors = survivor_set(&[10, 5]);
    let original_order_schedule =
        MaxPlusSchedule::new(3, vec![MaxPlusEdge::new(0, 2, 4)]).unwrap();
    emit_case(
        "original_order",
        &original_order_survivors,
        &original_order_schedule,
        &[
            MaxPlusValue::Finite(0),
            MaxPlusValue::ZERO,
            MaxPlusValue::ZERO,
        ],
        &[
            SurvivorNodeBinding::new(10, 2),
            SurvivorNodeBinding::new(5, 0),
        ],
        &[0, 3, 4],
    );

    let shared_node_survivors = survivor_set(&[7, 4]);
    let shared_node_schedule =
        MaxPlusSchedule::new(2, vec![MaxPlusEdge::new(0, 1, 2)]).unwrap();
    emit_case(
        "shared_node",
        &shared_node_survivors,
        &shared_node_schedule,
        &[MaxPlusValue::Finite(0), MaxPlusValue::ZERO],
        &[
            SurvivorNodeBinding::new(7, 1),
            SurvivorNodeBinding::new(4, 1),
        ],
        &[1, 2],
    );

    let empty_survivors = survivor_set(&[]);
    let empty_schedule = MaxPlusSchedule::new(1, vec![]).unwrap();
    emit_case(
        "empty",
        &empty_survivors,
        &empty_schedule,
        &[MaxPlusValue::Finite(0)],
        &[],
        &[0],
    );
}
