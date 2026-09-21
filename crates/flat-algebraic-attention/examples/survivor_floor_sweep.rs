//! MAA-12a survivor-floor exploratory sweep; not confirmatory or performance evidence.
//! Preregistration: docs/research/MAA_SURVIVOR_FLOOR_EXPLORATORY_PREREGISTRATION.md.

use std::collections::BTreeSet;
use std::error::Error;

use flat_algebraic_attention::cooperation::{
    route_attention_needs, AttentionAlgebraNeeds, M13bBooleanRoutingEvidence,
};
use flat_algebraic_attention::f2::{F2AffinePredicate, F2Vector};
use flat_algebraic_attention::qualification::{
    qualify_candidate, CandidateQualificationDecision, CandidateQualificationInputs,
    F2CandidateEvaluation, RecompositionPolicy, ZhegalkinCandidateEvaluation,
};
use flat_algebraic_attention::survivor_floor::{
    apply_survivor_floor, SurvivorFloorCandidate,
};
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

const PROTOCOL: &str = "maa-survivor-floor-exploratory/v1";
const CASES: usize = 64;
const CANDIDATES: usize = 8;
const D: usize = 2;
const EXPLORATORY_SEED: u64 = 0x004d_4141_2d31_3261;
const FLOORS: [usize; 5] = [0, 1, 2, 3, 4];
const POLICY: RecompositionPolicy = RecompositionPolicy::AllSelectedMustQualify;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, Copy)]
struct Candidate {
    k: [f32; D],
    v: [f32; D],
    structural_admit: bool,
    x0: bool,
    x1: bool,
    x2: bool,
}

#[derive(Debug, Clone)]
struct Case {
    q: [f32; D],
    candidates: [Candidate; CANDIDATES],
}

#[derive(Debug, Clone, Copy)]
struct SparseMetrics {
    selected: usize,
    top2_hits: usize,
    false_negatives: usize,
    empty: usize,
    output_error: Option<f64>,
    lse_error: Option<f64>,
}

#[derive(Debug, Default, Clone, Copy)]
struct Aggregate {
    selected: usize,
    top2_hits: usize,
    false_negatives: usize,
    empty: usize,
    output_error_sum: f64,
    lse_error_sum: f64,
    rescued: usize,
}

impl Aggregate {
    fn record(&mut self, metrics: SparseMetrics, rescued: usize) {
        self.selected += metrics.selected;
        self.top2_hits += metrics.top2_hits;
        self.false_negatives += metrics.false_negatives;
        self.empty += metrics.empty;
        self.rescued += rescued;
        if let Some(value) = metrics.output_error {
            self.output_error_sum += value;
        }
        if let Some(value) = metrics.lse_error {
            self.lse_error_sum += value;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let top24 = (self.next_u64() >> 40) as u32;
        let unit = top24 as f64 / 0x00ff_ffffu32 as f64;
        (2.0 * unit - 1.0) as f32
    }
}

fn require(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn generate_cases() -> [Case; CASES] {
    let mut rng = SplitMix64::new(EXPLORATORY_SEED);
    std::array::from_fn(|_| {
        let q = [rng.next_f32(), rng.next_f32()];
        let candidates = std::array::from_fn(|_| {
            let k = [rng.next_f32(), rng.next_f32()];
            let v = [rng.next_f32(), rng.next_f32()];
            let feature_word = rng.next_u64();
            Candidate {
                k,
                v,
                structural_admit: feature_word & 1 != 0,
                x0: feature_word & 2 != 0,
                x1: feature_word & 4 != 0,
                x2: feature_word & 8 != 0,
            }
        });
        Case { q, candidates }
    })
}

fn policy_route(
    admissions: &[bool; CANDIDATES],
) -> Result<flat_algebraic_attention::cooperation::AlgebraicRoute> {
    let mask = BooleanAttentionMask::from_admissions(admissions)?;
    let query_signature = BooleanAttentionSignature::new(CANDIDATES, vec![0xff])?;
    let key_signature = BooleanAttentionSignature::new(CANDIDATES, mask.words().to_vec())?;
    Ok(route_attention_needs(
        AttentionAlgebraNeeds {
            eligibility_logic: true,
            parity_or_binary_linear: true,
            nonlinear_boolean_interaction: true,
            precedence_or_critical_path: false,
        },
        Some(M13bBooleanRoutingEvidence::new(
            BOOLEAN_ATTENTION_MASK_SCHEMA_VERSION,
            CANDIDATES,
            mask.words().to_vec(),
            BOOLEAN_ATTENTION_SIGNATURE_SCHEMA_VERSION,
            CANDIDATES,
            query_signature.words().to_vec(),
            key_signature.words().to_vec(),
            None,
            None,
        )?),
    )?)
}

fn decisions(case: &Case) -> Result<Vec<CandidateQualificationDecision>> {
    let admissions =
        std::array::from_fn(|candidate_id| case.candidates[candidate_id].structural_admit);
    let route = policy_route(&admissions)?;
    let f2 = F2AffinePredicate::new(F2Vector::from_bools(&[true, false, false])?, false);
    let zhegalkin = ZhegalkinPolynomial::from_variable_sets(3, vec![vec![1, 2]])?;

    case.candidates
        .iter()
        .enumerate()
        .map(|(candidate_id, candidate)| {
            let input = F2Vector::from_bools(&[candidate.x0, candidate.x1, candidate.x2])?;
            Ok(qualify_candidate(
                &route,
                CandidateQualificationInputs {
                    boolean_block: Some(candidate_id),
                    f2: Some(F2CandidateEvaluation {
                        predicate: &f2,
                        input: &input,
                    }),
                    zhegalkin: Some(ZhegalkinCandidateEvaluation {
                        polynomial: &zhegalkin,
                        input: &input,
                    }),
                    max_plus: None,
                },
                POLICY,
            )?)
        })
        .collect()
}

fn boolean_indices(case: &Case) -> Vec<usize> {
    case.candidates
        .iter()
        .enumerate()
        .filter_map(|(candidate_id, candidate)| candidate.structural_admit.then_some(candidate_id))
        .collect()
}

fn floor_indices(
    decisions: &[CandidateQualificationDecision],
    floor: usize,
) -> Result<(Vec<usize>, usize)> {
    let candidates = decisions
        .iter()
        .enumerate()
        .map(|(candidate_id, decision)| SurvivorFloorCandidate::new(candidate_id, decision))
        .collect::<Vec<_>>();
    let result = apply_survivor_floor(&candidates, floor)?;
    Ok((
        result.survivor_indices().to_vec(),
        result.rescued_count(),
    ))
}

fn dense_top2(case: &Case) -> [usize; 2] {
    let mut scores = (0..CANDIDATES)
        .map(|candidate_id| {
            let candidate = case.candidates[candidate_id];
            let score = case.q[0] * candidate.k[0] + case.q[1] * candidate.k[1];
            (candidate_id, score)
        })
        .collect::<Vec<_>>();
    scores.sort_by(|(left_id, left_score), (right_id, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_id.cmp(right_id))
    });
    [scores[0].0, scores[1].0]
}

fn evaluate(case: &Case, indices: &[usize]) -> Result<Option<FlatAttentionOutput>> {
    if indices.is_empty() {
        return Ok(None);
    }

    let mut k = Vec::with_capacity(indices.len() * D);
    let mut v = Vec::with_capacity(indices.len() * D);
    for &candidate_id in indices {
        require(candidate_id < CANDIDATES, "candidate index out of bounds")?;
        k.extend_from_slice(&case.candidates[candidate_id].k);
        v.extend_from_slice(&case.candidates[candidate_id].v);
    }

    Ok(Some(forward_reference_grouped_asymmetric(
        &case.q,
        &k,
        &v,
        AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            query_len: 1,
            kv_len: indices.len(),
            head_dim: D,
            query_position_offset: CANDIDATES - 1,
        },
        FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(1.0),
        },
    )?))
}

fn sparse_metrics(
    case: &Case,
    selected: &[usize],
    top2: &[usize; 2],
    dense: &FlatAttentionOutput,
) -> Result<SparseMetrics> {
    let relevance = top2.iter().copied().collect::<BTreeSet<_>>();
    let top2_hits = selected
        .iter()
        .filter(|candidate_id| relevance.contains(candidate_id))
        .count();
    let false_negatives = top2.len() - top2_hits;
    let sparse = evaluate(case, selected)?;
    let (output_error, lse_error) = match sparse {
        None => (None, None),
        Some(sparse) => {
            let output_error = dense
                .output
                .iter()
                .zip(&sparse.output)
                .map(|(dense_value, sparse_value)| (dense_value - sparse_value).abs() as f64)
                .fold(0.0f64, f64::max);
            let lse_error = (dense.lse[0] - sparse.lse[0]).abs() as f64;
            (Some(output_error), Some(lse_error))
        }
    };

    Ok(SparseMetrics {
        selected: selected.len(),
        top2_hits,
        false_negatives,
        empty: usize::from(selected.is_empty()),
        output_error,
        lse_error,
    })
}

fn run() -> Result<Vec<String>> {
    require(
        PROTOCOL == "maa-survivor-floor-exploratory/v1",
        "exploratory protocol identity drifted",
    )?;

    let cases = generate_cases();
    let dense_indices = (0..CANDIDATES).collect::<Vec<_>>();
    let mut boolean = Aggregate::default();
    let mut floor_aggregates = [Aggregate::default(); FLOORS.len()];

    for case in &cases {
        let top2 = dense_top2(case);
        let dense = evaluate(case, &dense_indices)?.ok_or("dense selection cannot be empty")?;

        let boolean_ids = boolean_indices(case);
        boolean.record(sparse_metrics(case, &boolean_ids, &top2, &dense)?, 0);

        let case_decisions = decisions(case)?;
        for (arm, floor) in FLOORS.iter().copied().enumerate() {
            let (selected, rescued) = floor_indices(&case_decisions, floor)?;
            require(
                selected.iter().all(|candidate_id| boolean_ids.contains(candidate_id)),
                "survivor floor resurrected a Boolean rejection",
            )?;
            floor_aggregates[arm].record(
                sparse_metrics(case, &selected, &top2, &dense)?,
                rescued,
            );
        }
    }

    let mut rows = Vec::with_capacity(FLOORS.len() + 1);
    rows.push(format!(
        "boolean,NA,{},{},{},{},{:.9},{:.9},{}",
        boolean.selected,
        boolean.top2_hits,
        boolean.false_negatives,
        boolean.empty,
        boolean.output_error_sum,
        boolean.lse_error_sum,
        boolean.rescued,
    ));
    for (index, floor) in FLOORS.iter().copied().enumerate() {
        let aggregate = floor_aggregates[index];
        rows.push(format!(
            "floor,{floor},{},{},{},{},{:.9},{:.9},{}",
            aggregate.selected,
            aggregate.top2_hits,
            aggregate.false_negatives,
            aggregate.empty,
            aggregate.output_error_sum,
            aggregate.lse_error_sum,
            aggregate.rescued,
        ));
    }
    Ok(rows)
}

fn main() -> Result<()> {
    println!(
        "arm,floor,score_evaluations,top2_hits,false_negatives,empty_selections,output_error_sum,lse_error_sum,rescued_candidates"
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
    fn generator_is_repeatable() {
        let first = generate_cases();
        let second = generate_cases();
        for case_id in 0..CASES {
            assert_eq!(first[case_id].q, second[case_id].q);
            for candidate_id in 0..CANDIDATES {
                assert_eq!(
                    first[case_id].candidates[candidate_id].k,
                    second[case_id].candidates[candidate_id].k
                );
                assert_eq!(
                    first[case_id].candidates[candidate_id].v,
                    second[case_id].candidates[candidate_id].v
                );
                assert_eq!(
                    first[case_id].candidates[candidate_id].structural_admit,
                    second[case_id].candidates[candidate_id].structural_admit
                );
                assert_eq!(
                    first[case_id].candidates[candidate_id].x0,
                    second[case_id].candidates[candidate_id].x0
                );
                assert_eq!(
                    first[case_id].candidates[candidate_id].x1,
                    second[case_id].candidates[candidate_id].x1
                );
                assert_eq!(
                    first[case_id].candidates[candidate_id].x2,
                    second[case_id].candidates[candidate_id].x2
                );
            }
        }
    }

    #[test]
    fn exploratory_rows_are_repeatable() {
        assert_eq!(run().unwrap(), run().unwrap());
    }

    #[test]
    fn floor_zero_matches_strict_candidate_admission() {
        let cases = generate_cases();
        for case in cases.iter().take(8) {
            let case_decisions = decisions(case).unwrap();
            let strict = case_decisions
                .iter()
                .enumerate()
                .filter_map(|(candidate_id, decision)| {
                    (decision.disposition()
                        == flat_algebraic_attention::qualification::CandidateDisposition::Admit)
                        .then_some(candidate_id)
                })
                .collect::<Vec<_>>();
            let (floor_zero, rescued) = floor_indices(&case_decisions, 0).unwrap();
            assert_eq!(floor_zero, strict);
            assert_eq!(rescued, 0);
        }
    }
}
