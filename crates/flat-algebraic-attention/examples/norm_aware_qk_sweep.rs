//! MAA-13b norm-aware Q/K multi-algebra exploration.
//! Preregistration: docs/research/MAA_NORM_AWARE_QK_EXPLORATORY_PREREGISTRATION.md.

use std::error::Error;

use flat_algebraic_attention::cooperation::split_zhegalkin_for_f2;
use flat_algebraic_attention::f2::F2Vector;
use flat_algebraic_attention::qk_signature::SignedHyperplaneProjector;
use flat_algebraic_attention::zhegalkin::ZhegalkinPolynomial;
use flat_attention::api::boolean_attention_signature::{
    BooleanAttentionSignature, HammingAdmissionRule,
};
use flat_attention::{
    forward_reference_grouped_asymmetric, AsymmetricGroupedAttentionShape, FlatAttentionConfig,
    FlatAttentionOutput,
};

const PROTOCOL: &str = "maa-norm-aware-qk-exploratory/v1";
const DATASET_SEED: u64 = 0x004d_4141_2d31_3362;
const RANDOM_CONTROL_SEED: u64 = 0x4d41_4131_3342_524e;
const PROJECTOR_SEED: u64 = 0x514b_5349_474e_3133;
const CASES: usize = 128;
const CANDIDATES: usize = 16;
const D: usize = 16;
const SIGNATURE_BITS: usize = 128;
const NEAR_THRESHOLD: usize = 64;
const MEDIUM_THRESHOLD: usize = 72;
const HIGH_NORM_THRESHOLD: f64 = 1.5;
const KEY_SCALES: [f32; 4] = [0.5, 1.0, 2.0, 4.0];
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    k: [f32; D],
    v: [f32; D],
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, Copy, Default)]
struct Aggregate {
    selected: usize,
    top2_hits: usize,
    false_negatives: usize,
    empty: usize,
    output_error_sum: f64,
    lse_error_sum: f64,
    comparable_cases: usize,
    additional_vs_near: usize,
    additional_top2_hits: usize,
}

impl Aggregate {
    fn record(&mut self, metrics: SparseMetrics) {
        self.selected += metrics.selected;
        self.top2_hits += metrics.top2_hits;
        self.false_negatives += metrics.false_negatives;
        self.empty += metrics.empty;
        if let (Some(output_error), Some(lse_error)) = (metrics.output_error, metrics.lse_error) {
            self.output_error_sum += output_error;
            self.lse_error_sum += lse_error;
            self.comparable_cases += 1;
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

fn generated_vector(rng: &mut SplitMix64) -> [f32; D] {
    std::array::from_fn(|_| rng.next_f32())
}

fn l2_norm(vector: &[f32; D]) -> Result<f64> {
    let squared = vector.iter().try_fold(0.0f64, |total, value| {
        require(
            value.is_finite(),
            "generated vector contains a non-finite value",
        )?;
        let value = f64::from(*value);
        Ok::<f64, Box<dyn Error>>(total + value * value)
    })?;
    require(
        squared.is_finite() && squared > 0.0,
        "generated vector has invalid norm",
    )?;
    Ok(squared.sqrt())
}

fn normalize(mut vector: [f32; D]) -> Result<[f32; D]> {
    let inverse_norm = l2_norm(&vector)?.recip();
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
        let mut candidates = Vec::with_capacity(CANDIDATES);

        for _ in 0..CANDIDATES {
            let direction = normalize(generated_vector(&mut rng))?;
            let scale = match rng.next_u64() & 0b11 {
                0 => KEY_SCALES[0],
                1 => KEY_SCALES[1],
                2 => KEY_SCALES[2],
                _ => KEY_SCALES[3],
            };
            let k = direction.map(|value| value * scale);
            let v = generated_vector(&mut rng);
            candidates.push(Candidate { k, v });
        }

        let candidates: [Candidate; CANDIDATES] = candidates
            .try_into()
            .map_err(|_| "candidate array conversion failed")?;
        cases.push(Case { q, candidates });
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

fn dense_top2(case: &Case) -> [usize; 2] {
    let mut scores = case
        .candidates
        .iter()
        .enumerate()
        .map(|(candidate_id, candidate)| (candidate_id, dot(&case.q, &candidate.k)))
        .collect::<Vec<_>>();
    scores.sort_by(|(left_id, left_score), (right_id, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_id.cmp(right_id))
    });
    [scores[0].0, scores[1].0]
}

fn to_m13b(vector: &F2Vector) -> Result<BooleanAttentionSignature> {
    BooleanAttentionSignature::new(vector.bit_len(), vector.words().to_vec())
        .map_err(|error| Box::new(error) as Box<dyn Error>)
}

fn policy_polynomial() -> Result<ZhegalkinPolynomial> {
    ZhegalkinPolynomial::from_variable_sets(3, vec![vec![0], vec![1, 2], vec![0, 1, 2]])
        .map_err(|error| Box::new(error) as Box<dyn Error>)
}

fn policy_verdict(
    near: bool,
    medium: bool,
    high_norm: bool,
    polynomial: &ZhegalkinPolynomial,
) -> Result<bool> {
    require(
        !near || medium,
        "near Hamming verdict must imply medium verdict",
    )?;

    let features = F2Vector::from_bools(&[near, medium, high_norm])?;
    let direct = near || (medium && high_norm);
    let polynomial_value = polynomial.evaluate(&features)?;
    let split = split_zhegalkin_for_f2(polynomial)?;
    let recomposed = split.affine().evaluate(&features)? ^ split.nonlinear().evaluate(&features)?;

    require(
        direct == polynomial_value,
        "direct Boolean and Zhegalkin norm-aware verdicts drifted",
    )?;
    require(
        direct == recomposed,
        "F2/Zhegalkin recomposition drifted from direct Boolean verdict",
    )?;

    Ok(direct)
}

fn matched_random_indices(case_id: usize, count: usize) -> Result<Vec<usize>> {
    require(
        count <= CANDIDATES,
        "matched-random count exceeds candidate geometry",
    )?;
    let case_key = u64::try_from(case_id).map_err(|_| "case ID exceeds u64")?;
    let mut priorities = (0..CANDIDATES)
        .map(|candidate_id| {
            let candidate_key =
                u64::try_from(candidate_id).map_err(|_| "candidate ID exceeds u64")?;
            let key = RANDOM_CONTROL_SEED
                ^ case_key.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ candidate_key.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            Ok((mix64(key), candidate_id))
        })
        .collect::<Result<Vec<_>>>()?;
    priorities.sort_unstable();
    let mut selected = priorities
        .into_iter()
        .take(count)
        .map(|(_, candidate_id)| candidate_id)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    Ok(selected)
}

#[must_use]
const fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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
            query_position_offset: 0,
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
    let top2_hits = selected
        .iter()
        .filter(|candidate_id| top2.contains(candidate_id))
        .count();
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
        false_negatives: top2.len() - top2_hits,
        empty: usize::from(selected.is_empty()),
        output_error,
        lse_error,
    })
}

fn run() -> Result<Vec<String>> {
    require(
        PROTOCOL == "maa-norm-aware-qk-exploratory/v1",
        "norm-aware Q/K protocol identity drifted",
    )?;

    let cases = generate_cases()?;
    let projector = SignedHyperplaneProjector::new(D, SIGNATURE_BITS, PROJECTOR_SEED)?;
    let near_rule = HammingAdmissionRule::new(NEAR_THRESHOLD, SIGNATURE_BITS)?;
    let medium_rule = HammingAdmissionRule::new(MEDIUM_THRESHOLD, SIGNATURE_BITS)?;
    let polynomial = policy_polynomial()?;

    let mut dense_aggregate = Aggregate::default();
    let mut near_aggregate = Aggregate::default();
    let mut medium_aggregate = Aggregate::default();
    let mut norm_aggregate = Aggregate::default();
    let mut maa_aggregate = Aggregate::default();
    let mut random_aggregate = Aggregate::default();

    let dense_indices = (0..CANDIDATES).collect::<Vec<_>>();

    for (case_id, case) in cases.iter().enumerate() {
        let top2 = dense_top2(case);
        let dense = evaluate(case, &dense_indices)?.ok_or("dense selection cannot be empty")?;
        dense_aggregate.record(sparse_metrics(case, &dense_indices, &top2, &dense)?);

        let q_signature = to_m13b(&projector.project(&case.q)?)?;
        let mut near = Vec::new();
        let mut medium = Vec::new();
        let mut norm_only = Vec::new();
        let mut norm_aware = Vec::new();

        for (candidate_id, candidate) in case.candidates.iter().enumerate() {
            let key_signature = to_m13b(&projector.project(&candidate.k)?)?;
            let near_value = near_rule.admits(&q_signature, &key_signature)?;
            let medium_value = medium_rule.admits(&q_signature, &key_signature)?;
            let high_norm = l2_norm(&candidate.k)? >= HIGH_NORM_THRESHOLD;
            let maa_value = policy_verdict(near_value, medium_value, high_norm, &polynomial)?;

            if near_value {
                near.push(candidate_id);
            }
            if medium_value {
                medium.push(candidate_id);
            }
            if high_norm {
                norm_only.push(candidate_id);
            }
            if maa_value {
                norm_aware.push(candidate_id);
            }
        }

        require(
            near.iter()
                .all(|candidate_id| norm_aware.contains(candidate_id)),
            "near selection is not a subset of norm-aware MAA",
        )?;
        require(
            norm_aware
                .iter()
                .all(|candidate_id| medium.contains(candidate_id)),
            "norm-aware MAA escaped the medium Hamming envelope",
        )?;

        let matched_random = matched_random_indices(case_id, norm_aware.len())?;
        require(
            matched_random.len() == norm_aware.len(),
            "matched-random cardinality drifted",
        )?;

        near_aggregate.record(sparse_metrics(case, &near, &top2, &dense)?);
        medium_aggregate.record(sparse_metrics(case, &medium, &top2, &dense)?);
        norm_aggregate.record(sparse_metrics(case, &norm_only, &top2, &dense)?);
        maa_aggregate.record(sparse_metrics(case, &norm_aware, &top2, &dense)?);
        random_aggregate.record(sparse_metrics(case, &matched_random, &top2, &dense)?);

        for candidate_id in norm_aware
            .iter()
            .copied()
            .filter(|candidate_id| !near.contains(candidate_id))
        {
            maa_aggregate.additional_vs_near += 1;
            maa_aggregate.additional_top2_hits += usize::from(top2.contains(&candidate_id));
        }
    }

    let aggregates = [
        ("dense", dense_aggregate),
        ("hamming_near", near_aggregate),
        ("hamming_medium", medium_aggregate),
        ("norm_only", norm_aggregate),
        ("norm_aware_maa", maa_aggregate),
        ("matched_random", random_aggregate),
    ];

    Ok(aggregates
        .into_iter()
        .map(|(arm, aggregate)| {
            format!(
                "{arm},{},{},{},{},{:.9},{:.9},{},{},{}",
                aggregate.selected,
                aggregate.top2_hits,
                aggregate.false_negatives,
                aggregate.empty,
                aggregate.output_error_sum,
                aggregate.lse_error_sum,
                aggregate.comparable_cases,
                aggregate.additional_vs_near,
                aggregate.additional_top2_hits,
            )
        })
        .collect())
}

fn main() -> Result<()> {
    println!(
        "arm,selected,top2_hits,false_negatives,empty_cases,output_error_sum,lse_error_sum,comparable_cases,additional_vs_near,additional_top2_hits"
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
        assert_eq!(generate_cases().unwrap(), generate_cases().unwrap());
    }

    #[test]
    fn frozen_boolean_zhegalkin_and_split_truth_tables_agree() {
        let polynomial = policy_polynomial().unwrap();
        for assignment in 0u8..8 {
            let near = assignment & 0b001 != 0;
            let medium = assignment & 0b010 != 0;
            let high_norm = assignment & 0b100 != 0;
            let features = F2Vector::from_bools(&[near, medium, high_norm]).unwrap();
            let direct = near || (medium && high_norm);
            let polynomial_value = polynomial.evaluate(&features).unwrap();
            let split = split_zhegalkin_for_f2(&polynomial).unwrap();
            let recomposed = split.affine().evaluate(&features).unwrap()
                ^ split.nonlinear().evaluate(&features).unwrap();

            assert_eq!(direct, polynomial_value);
            assert_eq!(direct, recomposed);
        }
    }

    #[test]
    fn matched_random_is_deterministic_and_cardinality_exact() {
        for count in 0..=CANDIDATES {
            let first = matched_random_indices(7, count).unwrap();
            let second = matched_random_indices(7, count).unwrap();
            assert_eq!(first, second);
            assert_eq!(first.len(), count);
            assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn exploratory_rows_are_byte_repeatable() {
        assert_eq!(run().unwrap(), run().unwrap());
    }
}
