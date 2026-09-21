//! MAA-13c retained softmax mass evidence.
//! Preregistration: docs/research/MAA_RETAINED_SOFTMAX_MASS_PREREGISTRATION.md.

use std::error::Error;

use flat_algebraic_attention::cooperation::split_zhegalkin_for_f2;
use flat_algebraic_attention::f2::F2Vector;
use flat_algebraic_attention::qk_signature::SignedHyperplaneProjector;
use flat_algebraic_attention::softmax_mass::{
    output_error_bound, retained_softmax_mass, RetainedSoftmaxMass,
};
use flat_algebraic_attention::zhegalkin::ZhegalkinPolynomial;
use flat_attention::api::boolean_attention_signature::{
    BooleanAttentionSignature, HammingAdmissionRule,
};
use flat_attention::{
    forward_reference_grouped_asymmetric, AsymmetricGroupedAttentionShape, FlatAttentionConfig,
    FlatAttentionOutput,
};

const PROTOCOL: &str = "maa-retained-softmax-mass/v1";
const DATASET_SEED: u64 = 0x004d_4141_2d31_3363;
const RANDOM_CONTROL_SEED: u64 = 0x4d41_4131_3343_524e;
const PROJECTOR_SEED: u64 = 0x514b_5349_474e_3133;
const CASES: usize = 256;
const CANDIDATES: usize = 32;
const D: usize = 16;
const SIGNATURE_BITS: usize = 128;
const NEAR_THRESHOLD: usize = 64;
const MEDIUM_THRESHOLD: usize = 72;
const HIGH_NORM_THRESHOLD: f64 = 1.5;
const KEY_SCALES: [f32; 4] = [0.5, 1.0, 2.0, 4.0];
const OUTPUT_BOUND_ABS_TOL: f64 = 1.0e-5;
const OUTPUT_BOUND_REL_TOL: f64 = 1.0e-5;
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
struct CaseMetrics {
    selected: usize,
    top2_hits: usize,
    false_negatives: usize,
    empty: usize,
    retained_mass: f64,
    dropped_mass: f64,
    output_error: Option<f64>,
    lse_error: Option<f64>,
    lse_identity_residual: Option<f64>,
    output_bound_excess: Option<f64>,
}

#[derive(Debug, Clone)]
struct Aggregate {
    selected: usize,
    top2_hits: usize,
    false_negatives: usize,
    empty: usize,
    retained_mass_sum: f64,
    dropped_mass_sum: f64,
    mass_values: Vec<f64>,
    cases_ge_90: usize,
    cases_ge_95: usize,
    cases_ge_99: usize,
    output_error_sum: f64,
    lse_error_sum: f64,
    max_lse_identity_residual: f64,
    max_output_bound_excess: f64,
}

impl Default for Aggregate {
    fn default() -> Self {
        Self {
            selected: 0,
            top2_hits: 0,
            false_negatives: 0,
            empty: 0,
            retained_mass_sum: 0.0,
            dropped_mass_sum: 0.0,
            mass_values: Vec::with_capacity(CASES),
            cases_ge_90: 0,
            cases_ge_95: 0,
            cases_ge_99: 0,
            output_error_sum: 0.0,
            lse_error_sum: 0.0,
            max_lse_identity_residual: 0.0,
            max_output_bound_excess: 0.0,
        }
    }
}

impl Aggregate {
    fn record(&mut self, metrics: CaseMetrics) {
        self.selected += metrics.selected;
        self.top2_hits += metrics.top2_hits;
        self.false_negatives += metrics.false_negatives;
        self.empty += metrics.empty;
        self.retained_mass_sum += metrics.retained_mass;
        self.dropped_mass_sum += metrics.dropped_mass;
        self.mass_values.push(metrics.retained_mass);
        self.cases_ge_90 += usize::from(metrics.retained_mass >= 0.90);
        self.cases_ge_95 += usize::from(metrics.retained_mass >= 0.95);
        self.cases_ge_99 += usize::from(metrics.retained_mass >= 0.99);

        if let Some(value) = metrics.output_error {
            self.output_error_sum += value;
        }
        if let Some(value) = metrics.lse_error {
            self.lse_error_sum += value;
        }
        if let Some(value) = metrics.lse_identity_residual {
            self.max_lse_identity_residual = self.max_lse_identity_residual.max(value);
        }
        if let Some(value) = metrics.output_bound_excess {
            self.max_output_bound_excess = self.max_output_bound_excess.max(value);
        }
    }

    fn finalized(mut self) -> Result<FinalAggregate> {
        require(
            self.mass_values.len() == CASES,
            "aggregate retained-mass sample count drifted",
        )?;
        self.mass_values.sort_by(f64::total_cmp);

        Ok(FinalAggregate {
            selected: self.selected,
            mean_density: self.selected as f64 / (CASES * CANDIDATES) as f64,
            top2_hits: self.top2_hits,
            false_negatives: self.false_negatives,
            retained_mass_sum: self.retained_mass_sum,
            retained_mass_mean: self.retained_mass_sum / CASES as f64,
            retained_mass_min: self.mass_values[0],
            retained_mass_p10: nearest_rank(&self.mass_values, 10, 100)?,
            retained_mass_p50: nearest_rank(&self.mass_values, 50, 100)?,
            retained_mass_p90: nearest_rank(&self.mass_values, 90, 100)?,
            cases_ge_90: self.cases_ge_90,
            cases_ge_95: self.cases_ge_95,
            cases_ge_99: self.cases_ge_99,
            dropped_mass_sum: self.dropped_mass_sum,
            output_error_sum: self.output_error_sum,
            lse_error_sum: self.lse_error_sum,
            max_lse_identity_residual: self.max_lse_identity_residual,
            max_output_bound_excess: self.max_output_bound_excess,
            empty: self.empty,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct FinalAggregate {
    selected: usize,
    mean_density: f64,
    top2_hits: usize,
    false_negatives: usize,
    retained_mass_sum: f64,
    retained_mass_mean: f64,
    retained_mass_min: f64,
    retained_mass_p10: f64,
    retained_mass_p50: f64,
    retained_mass_p90: f64,
    cases_ge_90: usize,
    cases_ge_95: usize,
    cases_ge_99: usize,
    dropped_mass_sum: f64,
    output_error_sum: f64,
    lse_error_sum: f64,
    max_lse_identity_residual: f64,
    max_output_bound_excess: f64,
    empty: usize,
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

fn nearest_rank(sorted: &[f64], numerator: usize, denominator: usize) -> Result<f64> {
    require(!sorted.is_empty(), "nearest-rank sample is empty")?;
    require(
        denominator > 0 && numerator > 0 && numerator <= denominator,
        "nearest-rank percentile fraction is invalid",
    )?;
    let scaled = sorted
        .len()
        .checked_mul(numerator)
        .ok_or("nearest-rank percentile multiplication overflow")?;
    let rank = scaled.div_ceil(denominator).max(1);
    Ok(sorted[rank - 1])
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

fn scores(case: &Case) -> Result<Vec<f64>> {
    case.candidates
        .iter()
        .map(|candidate| {
            let score = dot(&case.q, &candidate.k);
            require(score.is_finite(), "dense reference score is non-finite")?;
            Ok(score)
        })
        .collect()
}

fn dense_top2_from_scores(scores: &[f64]) -> Result<[usize; 2]> {
    require(
        scores.len() >= 2,
        "dense top-2 requires at least two candidate scores",
    )?;
    let mut indexed = scores.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(left_id, left_score), (right_id, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_id.cmp(right_id))
    });
    Ok([indexed[0].0, indexed[1].0])
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

fn mass_oracle_indices(scores: &[f64], count: usize) -> Result<Vec<usize>> {
    require(
        count <= scores.len(),
        "mass-oracle count exceeds candidate geometry",
    )?;
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|(left_id, left_score), (right_id, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_id.cmp(right_id))
    });
    let mut selected = ranked
        .into_iter()
        .take(count)
        .map(|(candidate_id, _)| candidate_id)
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

fn max_abs_value(case: &Case) -> Result<f64> {
    let mut max_value = 0.0f64;
    for candidate in &case.candidates {
        for value in candidate.v {
            require(value.is_finite(), "V contains a non-finite value")?;
            max_value = max_value.max(f64::from(value).abs());
        }
    }
    Ok(max_value)
}

fn case_metrics(
    case: &Case,
    selected: &[usize],
    dense_top2: &[usize; 2],
    dense: &FlatAttentionOutput,
    score_values: &[f64],
) -> Result<(CaseMetrics, RetainedSoftmaxMass)> {
    let mass = retained_softmax_mass(score_values, selected)?;
    let top2_hits = selected
        .iter()
        .filter(|candidate_id| dense_top2.contains(candidate_id))
        .count();
    let sparse = evaluate(case, selected)?;

    let (output_error, lse_error, lse_identity_residual, output_bound_excess) = match sparse {
        None => {
            require(
                mass.retained_mass().abs() <= f64::EPSILON,
                "empty sparse selection has non-zero retained mass",
            )?;
            (None, None, None, None)
        }
        Some(sparse) => {
            let output_error = dense
                .output
                .iter()
                .zip(&sparse.output)
                .map(|(dense_value, sparse_value)| (dense_value - sparse_value).abs() as f64)
                .fold(0.0f64, f64::max);
            let signed_lse_gap = f64::from(dense.lse[0] - sparse.lse[0]);
            let lse_error = signed_lse_gap.abs();
            let expected_gap = mass.lse_gap().ok_or("non-empty mass has no LSE gap")?;
            let lse_identity_residual = (signed_lse_gap - expected_gap).abs();
            require(
                lse_identity_residual.is_finite(),
                "LSE identity residual is non-finite",
            )?;

            let bound = output_error_bound(mass.retained_mass(), max_abs_value(case)?)?;
            let tolerance = OUTPUT_BOUND_ABS_TOL + OUTPUT_BOUND_REL_TOL * bound;
            require(
                output_error <= bound + tolerance,
                "FLAT output error exceeded retained-mass mathematical bound",
            )?;
            let output_bound_excess = (output_error - bound).max(0.0);

            (
                Some(output_error),
                Some(lse_error),
                Some(lse_identity_residual),
                Some(output_bound_excess),
            )
        }
    };

    Ok((
        CaseMetrics {
            selected: selected.len(),
            top2_hits,
            false_negatives: dense_top2.len() - top2_hits,
            empty: usize::from(selected.is_empty()),
            retained_mass: mass.retained_mass(),
            dropped_mass: mass.dropped_mass(),
            output_error,
            lse_error,
            lse_identity_residual,
            output_bound_excess,
        },
        mass,
    ))
}

fn run() -> Result<Vec<String>> {
    require(
        PROTOCOL == "maa-retained-softmax-mass/v1",
        "retained softmax mass protocol identity drifted",
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
    let mut oracle_aggregate = Aggregate::default();

    let dense_indices = (0..CANDIDATES).collect::<Vec<_>>();

    for (case_id, case) in cases.iter().enumerate() {
        let score_values = scores(case)?;
        let top2 = dense_top2_from_scores(&score_values)?;
        let dense = evaluate(case, &dense_indices)?.ok_or("dense selection cannot be empty")?;

        let (dense_metrics, dense_mass) =
            case_metrics(case, &dense_indices, &top2, &dense, &score_values)?;
        require(
            (dense_mass.retained_mass() - 1.0).abs() <= 16.0 * f64::EPSILON,
            "dense retained mass drifted from one",
        )?;
        dense_aggregate.record(dense_metrics);

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
        let mass_oracle = mass_oracle_indices(&score_values, norm_aware.len())?;
        require(
            matched_random.len() == norm_aware.len() && mass_oracle.len() == norm_aware.len(),
            "matched-control cardinality drifted",
        )?;

        let (near_metrics, near_mass) = case_metrics(case, &near, &top2, &dense, &score_values)?;
        let (medium_metrics, medium_mass) =
            case_metrics(case, &medium, &top2, &dense, &score_values)?;
        let (norm_metrics, norm_mass) =
            case_metrics(case, &norm_only, &top2, &dense, &score_values)?;
        let (maa_metrics, maa_mass) =
            case_metrics(case, &norm_aware, &top2, &dense, &score_values)?;
        let (random_metrics, random_mass) =
            case_metrics(case, &matched_random, &top2, &dense, &score_values)?;
        let (oracle_metrics, oracle_mass) =
            case_metrics(case, &mass_oracle, &top2, &dense, &score_values)?;

        for (selection, selection_mass) in [
            (&near, &near_mass),
            (&medium, &medium_mass),
            (&norm_only, &norm_mass),
            (&norm_aware, &maa_mass),
            (&matched_random, &random_mass),
        ] {
            if selection.len() == mass_oracle.len() {
                require(
                    oracle_mass.retained_mass() + 1.0e-15 >= selection_mass.retained_mass(),
                    "mass-oracle control failed to upper-bound equal-cardinality retained mass",
                )?;
            }
        }

        near_aggregate.record(near_metrics);
        medium_aggregate.record(medium_metrics);
        norm_aggregate.record(norm_metrics);
        maa_aggregate.record(maa_metrics);
        random_aggregate.record(random_metrics);
        oracle_aggregate.record(oracle_metrics);
    }

    let aggregates = [
        ("dense", dense_aggregate.finalized()?),
        ("hamming_near", near_aggregate.finalized()?),
        ("hamming_medium", medium_aggregate.finalized()?),
        ("norm_only", norm_aggregate.finalized()?),
        ("norm_aware_maa", maa_aggregate.finalized()?),
        ("matched_random", random_aggregate.finalized()?),
        ("mass_oracle_matched", oracle_aggregate.finalized()?),
    ];

    Ok(aggregates
        .into_iter()
        .map(|(arm, aggregate)| {
            format!(
                "{arm},{},{:.9},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{},{},{:.12},{:.9},{:.9},{:.12},{:.12},{}",
                aggregate.selected,
                aggregate.mean_density,
                aggregate.top2_hits,
                aggregate.false_negatives,
                aggregate.retained_mass_sum,
                aggregate.retained_mass_mean,
                aggregate.retained_mass_min,
                aggregate.retained_mass_p10,
                aggregate.retained_mass_p50,
                aggregate.retained_mass_p90,
                aggregate.cases_ge_90,
                aggregate.cases_ge_95,
                aggregate.cases_ge_99,
                aggregate.dropped_mass_sum,
                aggregate.output_error_sum,
                aggregate.lse_error_sum,
                aggregate.max_lse_identity_residual,
                aggregate.max_output_bound_excess,
                aggregate.empty,
            )
        })
        .collect())
}

fn main() -> Result<()> {
    println!(
        "arm,selected,mean_density,top2_hits,false_negatives,retained_mass_sum,retained_mass_mean,retained_mass_min,retained_mass_p10,retained_mass_p50,retained_mass_p90,cases_ge_0_90,cases_ge_0_95,cases_ge_0_99,dropped_mass_sum,output_error_sum,lse_error_sum,max_lse_identity_residual,max_output_bound_excess,empty_cases"
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
    fn nearest_rank_is_deterministic() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5];
        assert!((nearest_rank(&values, 10, 100).unwrap() - 0.1).abs() <= f64::EPSILON);
        assert!((nearest_rank(&values, 50, 100).unwrap() - 0.3).abs() <= f64::EPSILON);
        assert!((nearest_rank(&values, 90, 100).unwrap() - 0.5).abs() <= f64::EPSILON);
    }

    #[test]
    fn matched_controls_are_cardinality_exact() {
        let score_values = (0..CANDIDATES)
            .map(|index| index as f64)
            .collect::<Vec<_>>();
        for count in 0..=CANDIDATES {
            assert_eq!(matched_random_indices(17, count).unwrap().len(), count);
            assert_eq!(
                mass_oracle_indices(&score_values, count).unwrap().len(),
                count
            );
        }
    }

    #[test]
    fn mass_oracle_dominates_random_at_equal_cardinality() {
        let score_values = (0..CANDIDATES)
            .map(|index| index as f64 / 8.0)
            .collect::<Vec<_>>();
        for count in 1..=CANDIDATES {
            let random = matched_random_indices(3, count).unwrap();
            let oracle = mass_oracle_indices(&score_values, count).unwrap();
            let random_mass = retained_softmax_mass(&score_values, &random)
                .unwrap()
                .retained_mass();
            let oracle_mass = retained_softmax_mass(&score_values, &oracle)
                .unwrap()
                .retained_mass();
            assert!(oracle_mass + 1.0e-15 >= random_mass);
        }
    }

    #[test]
    fn exploratory_rows_are_byte_repeatable() {
        assert_eq!(run().unwrap(), run().unwrap());
    }
}
