//! MAA-13a Q/K-derived Boolean signature exploration.
//! Preregistration: docs/research/MAA_QK_SIGNATURE_EXPLORATORY_PREREGISTRATION.md.

use std::error::Error;

use flat_algebraic_attention::f2::F2Vector;
use flat_algebraic_attention::qk_signature::SignedHyperplaneProjector;
use flat_attention::api::boolean_attention_signature::{
    BooleanAttentionSignature, HammingAdmissionRule,
};

const PROTOCOL: &str = "maa-qk-signature-exploratory/v1";
const DATASET_SEED: u64 = 0x004d_4141_2d31_3361;
const PROJECTOR_SEED: u64 = 0x514b_5349_474e_3133;
const SCALE_CONTROL_SEED: u64 = DATASET_SEED ^ 0x5ca1_e000_0000_0001;
const CASES: usize = 128;
const SCALE_CASES: usize = 64;
const CANDIDATES: usize = 16;
const D: usize = 16;
const WIDTHS: [usize; 5] = [8, 16, 32, 64, 128];
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Clone)]
struct Case {
    q: [f32; D],
    keys: [[f32; D]; CANDIDATES],
}

#[derive(Debug, Clone, Copy, Default)]
struct RankingAggregate {
    top1_matches: usize,
    top2_hits: usize,
    pair_agree: usize,
    pair_disagree: usize,
    pair_tied: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct AdmissionAggregate {
    selected: usize,
    top2_hits: usize,
    false_negatives: usize,
    empty: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct ScaleAggregate {
    signature_collisions: usize,
    strict_exact_orderings: usize,
    unresolved_hamming_ties: usize,
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

fn generated_vector(rng: &mut SplitMix64) -> [f32; D] {
    std::array::from_fn(|_| rng.next_f32())
}

fn normalize(mut vector: [f32; D]) -> Result<[f32; D]> {
    let squared_norm = vector
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>();
    require(
        squared_norm.is_finite() && squared_norm > 0.0,
        "generated vector has invalid norm",
    )?;
    let inverse_norm = squared_norm.sqrt().recip();
    for value in &mut vector {
        *value = (f64::from(*value) * inverse_norm) as f32;
    }
    Ok(vector)
}

fn generate_cases() -> Result<[Case; CASES]> {
    let mut rng = SplitMix64::new(DATASET_SEED);
    let mut cases = Vec::with_capacity(CASES);
    for _ in 0..CASES {
        let q = normalize(generated_vector(&mut rng))?;
        let mut keys = Vec::with_capacity(CANDIDATES);
        for _ in 0..CANDIDATES {
            keys.push(normalize(generated_vector(&mut rng))?);
        }
        let keys: [[f32; D]; CANDIDATES] = keys
            .try_into()
            .map_err(|_| "candidate-key array conversion failed")?;
        cases.push(Case { q, keys });
    }
    cases
        .try_into()
        .map_err(|_| "case array conversion failed".into())
}

fn dot(left: &[f32; D], right: &[f32; D]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

fn dense_order(case: &Case) -> Vec<usize> {
    let mut scored = case
        .keys
        .iter()
        .enumerate()
        .map(|(candidate_id, key)| (candidate_id, dot(&case.q, key)))
        .collect::<Vec<_>>();
    scored.sort_by(|(left_id, left_score), (right_id, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_id.cmp(right_id))
    });
    scored
        .into_iter()
        .map(|(candidate_id, _)| candidate_id)
        .collect()
}

fn to_m13b(vector: &F2Vector) -> Result<BooleanAttentionSignature> {
    Ok(BooleanAttentionSignature::new(
        vector.bit_len(),
        vector.words().to_vec(),
    )?)
}

fn thresholds(width: usize) -> Result<[usize; 3]> {
    let three_width = width
        .checked_mul(3)
        .ok_or("signature-width threshold arithmetic overflow")?;
    Ok([width / 4, three_width / 8, width / 2])
}

fn evaluate_width(
    cases: &[Case; CASES],
    width: usize,
) -> Result<(RankingAggregate, [AdmissionAggregate; 3])> {
    let projector = SignedHyperplaneProjector::new(D, width, PROJECTOR_SEED)?;
    let mut ranking = RankingAggregate::default();
    let mut admissions = [AdmissionAggregate::default(); 3];
    let thresholds = thresholds(width)?;
    let rules = [
        HammingAdmissionRule::new(thresholds[0], width)?,
        HammingAdmissionRule::new(thresholds[1], width)?,
        HammingAdmissionRule::new(thresholds[2], width)?,
    ];

    for case in cases {
        let dense = dense_order(case);
        let dense_top2 = [dense[0], dense[1]];
        let mut dense_rank = [0usize; CANDIDATES];
        for (rank, candidate_id) in dense.iter().copied().enumerate() {
            dense_rank[candidate_id] = rank;
        }

        let q_f2 = projector.project(&case.q)?;
        let q_signature = to_m13b(&q_f2)?;
        require(
            q_signature.words() == q_f2.words(),
            "Q packed bridge drifted",
        )?;

        let mut key_f2 = Vec::with_capacity(CANDIDATES);
        let mut key_signatures = Vec::with_capacity(CANDIDATES);
        let mut distances = Vec::with_capacity(CANDIDATES);

        for (candidate_id, key) in case.keys.iter().enumerate() {
            let f2 = projector.project(key)?;
            let signature = to_m13b(&f2)?;
            require(signature.words() == f2.words(), "K packed bridge drifted")?;
            let f2_distance = q_f2.hamming_distance(&f2)?;
            let m13b_distance = q_signature.hamming_distance(&signature)?;
            require(
                f2_distance == m13b_distance,
                "F2/M13B Hamming bridge drifted",
            )?;
            require(
                q_signature.xnor_match_count(&signature)? == width - m13b_distance,
                "M13B XNOR accounting drifted",
            )?;
            distances.push((candidate_id, m13b_distance));
            key_f2.push(f2);
            key_signatures.push(signature);
        }

        distances.sort_by(|(left_id, left_distance), (right_id, right_distance)| {
            left_distance
                .cmp(right_distance)
                .then_with(|| left_id.cmp(right_id))
        });
        if distances[0].0 == dense_top2[0] {
            ranking.top1_matches += 1;
        }
        ranking.top2_hits += distances
            .iter()
            .take(2)
            .filter(|(candidate_id, _)| dense_top2.contains(candidate_id))
            .count();

        let mut distance_by_id = [0usize; CANDIDATES];
        for (candidate_id, distance) in distances.iter().copied() {
            distance_by_id[candidate_id] = distance;
        }
        for left in 0..CANDIDATES {
            for right in (left + 1)..CANDIDATES {
                let left_distance = distance_by_id[left];
                let right_distance = distance_by_id[right];
                if left_distance == right_distance {
                    ranking.pair_tied += 1;
                } else {
                    let hamming_prefers_left = left_distance < right_distance;
                    let dense_prefers_left = dense_rank[left] < dense_rank[right];
                    if hamming_prefers_left == dense_prefers_left {
                        ranking.pair_agree += 1;
                    } else {
                        ranking.pair_disagree += 1;
                    }
                }
            }
        }

        for (rule_index, rule) in rules.iter().enumerate() {
            let mut selected = Vec::new();
            for (candidate_id, key_signature) in key_signatures.iter().enumerate() {
                if rule.admits(&q_signature, key_signature)? {
                    selected.push(candidate_id);
                }
            }
            let top2_hits = selected
                .iter()
                .filter(|candidate_id| dense_top2.contains(candidate_id))
                .count();
            admissions[rule_index].selected += selected.len();
            admissions[rule_index].top2_hits += top2_hits;
            admissions[rule_index].false_negatives += dense_top2.len() - top2_hits;
            admissions[rule_index].empty += usize::from(selected.is_empty());
        }

        // Keep the F2 vectors live through the full case evaluation so this
        // harness exercises the exact packed bridge for every candidate.
        require(
            key_f2.len() == CANDIDATES,
            "F2 candidate projection count drifted",
        )?;
    }

    Ok((ranking, admissions))
}

fn evaluate_scale_control(width: usize) -> Result<ScaleAggregate> {
    let projector = SignedHyperplaneProjector::new(D, width, PROJECTOR_SEED)?;
    let mut rng = SplitMix64::new(SCALE_CONTROL_SEED);
    let mut aggregate = ScaleAggregate::default();

    for _ in 0..SCALE_CASES {
        let q = normalize(generated_vector(&mut rng))?;
        let base = normalize(generated_vector(&mut rng))?;
        let low = base.map(|value| value * 0.5);
        let high = base.map(|value| value * 2.0);

        let q_signature = to_m13b(&projector.project(&q)?)?;
        let low_signature = to_m13b(&projector.project(&low)?)?;
        let high_signature = to_m13b(&projector.project(&high)?)?;

        if low_signature == high_signature {
            aggregate.signature_collisions += 1;
        }

        let low_score = dot(&q, &low);
        let high_score = dot(&q, &high);
        if low_score.total_cmp(&high_score) != std::cmp::Ordering::Equal {
            aggregate.strict_exact_orderings += 1;
            let low_distance = q_signature.hamming_distance(&low_signature)?;
            let high_distance = q_signature.hamming_distance(&high_signature)?;
            if low_distance == high_distance {
                aggregate.unresolved_hamming_ties += 1;
            }
        }
    }

    Ok(aggregate)
}

fn run() -> Result<Vec<String>> {
    require(
        PROTOCOL == "maa-qk-signature-exploratory/v1",
        "Q/K signature protocol identity drifted",
    )?;
    let cases = generate_cases()?;
    let mut rows = Vec::new();

    for width in WIDTHS {
        let (ranking, admissions) = evaluate_width(&cases, width)?;
        rows.push(format!(
            "ranking,{width},NA,{},{},{},{},{},NA,NA,NA,NA,NA,NA,NA",
            ranking.top1_matches,
            ranking.top2_hits,
            ranking.pair_agree,
            ranking.pair_disagree,
            ranking.pair_tied,
        ));

        let thresholds = thresholds(width)?;
        for index in 0..3 {
            let aggregate = admissions[index];
            rows.push(format!(
                "admission,{width},{},{},{},{},{},{},{},{},{},{},NA,NA,NA",
                thresholds[index],
                ranking.top1_matches,
                ranking.top2_hits,
                ranking.pair_agree,
                ranking.pair_disagree,
                ranking.pair_tied,
                aggregate.selected,
                aggregate.top2_hits,
                aggregate.false_negatives,
                aggregate.empty,
            ));
        }

        let scale = evaluate_scale_control(width)?;
        rows.push(format!(
            "scale_control,{width},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,{},{},{}",
            scale.signature_collisions, scale.strict_exact_orderings, scale.unresolved_hamming_ties,
        ));
    }

    Ok(rows)
}

fn main() -> Result<()> {
    println!(
        "row_type,width,threshold,top1_matches,top2_rank_hits,pair_agree,pair_disagree,pair_tied,selected,admission_top2_hits,false_negatives,empty_cases,scale_signature_collisions,scale_strict_exact_orderings,scale_unresolved_hamming_ties"
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
        let first = generate_cases().unwrap();
        let second = generate_cases().unwrap();
        for case_id in 0..CASES {
            assert_eq!(first[case_id].q, second[case_id].q);
            assert_eq!(first[case_id].keys, second[case_id].keys);
        }
    }

    #[test]
    fn f2_and_m13b_bridge_is_exact_on_fixed_fixture() {
        let projector = SignedHyperplaneProjector::new(D, 128, PROJECTOR_SEED).unwrap();
        let q = normalize([1.0; D]).unwrap();
        let k = normalize(std::array::from_fn(
            |index| {
                if index % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            },
        ))
        .unwrap();
        let q_f2 = projector.project(&q).unwrap();
        let k_f2 = projector.project(&k).unwrap();
        let q_signature = to_m13b(&q_f2).unwrap();
        let k_signature = to_m13b(&k_f2).unwrap();

        assert_eq!(q_signature.words(), q_f2.words());
        assert_eq!(k_signature.words(), k_f2.words());
        assert_eq!(
            q_signature.hamming_distance(&k_signature).unwrap(),
            q_f2.hamming_distance(&k_f2).unwrap()
        );
    }

    #[test]
    fn scale_control_preserves_all_positive_scale_collisions() {
        for width in WIDTHS {
            let aggregate = evaluate_scale_control(width).unwrap();
            assert_eq!(aggregate.signature_collisions, SCALE_CASES);
            assert_eq!(
                aggregate.unresolved_hamming_ties,
                aggregate.strict_exact_orderings
            );
        }
    }

    #[test]
    fn exploratory_rows_are_byte_repeatable() {
        assert_eq!(run().unwrap(), run().unwrap());
    }
}
