//! MAA-11b frozen confirmatory synthetic holdout; not a performance benchmark.
//! Freeze: docs/research/MAA_CONFIRMATORY_HOLDOUT_FREEZE.md.

use std::collections::BTreeSet;
use std::error::Error;

use flat_algebraic_attention::cooperation::{
    route_attention_needs, AttentionAlgebraNeeds, M13bBooleanRoutingEvidence,
};
use flat_algebraic_attention::f2::{F2AffinePredicate, F2Vector};
use flat_algebraic_attention::holdout::{ConfirmatoryContext, FrozenPolicyManifest, Sha256Digest};
use flat_algebraic_attention::qualification::{
    CandidateQualificationInputs, F2CandidateEvaluation, RecompositionPolicy,
    ZhegalkinCandidateEvaluation,
};
use flat_algebraic_attention::survivor_set::{qualify_survivor_set, CandidateFrame};
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

const PROTOCOL: &str = "maa-confirmatory-holdout/v1";
const SOURCE_REVISION: &str = "4030b229a2ff329c1263cad677812d659422b448";
const POLICY_ID: &str = "maa-11b-structural-v1";
const POLICY_DIGEST: &str = "7c36df81b9d213fb5a198da5491211242d7db3ef4c02fa5be432885d03552194";
const FEATURE_SCHEMA_DIGEST: &str =
    "781cd7d2d78194b40083ce68705fe7e99a3130e1d175cdd1ce7d55c0ee8ea5fe";
const TUNING_DATASET_DIGEST: &str =
    "eb122c2d24751b7c2c81547755c69889e3cdf15fca7efd7a12f8447edf37aef4";
const CONFIRMATORY_DATASET_DIGEST: &str =
    "87585c258c254de0a2d4878581138c8a4a58bad07d446cf9876d9ab24622b3d9";
const CASES: usize = 16;
const CANDIDATES: usize = 8;
const D: usize = 2;
const HOLDOUT_SEED: u64 = 0x004d_4141_2d31_3162;
const EPSILON: f64 = 1.0e-6;
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
    output_error: Option<f64>,
    lse_error: Option<f64>,
}

#[derive(Debug, Default)]
struct Aggregate {
    boolean_scores: usize,
    multi_scores: usize,
    boolean_top2_hits: usize,
    multi_top2_hits: usize,
    boolean_empty: usize,
    multi_empty: usize,
    comparable_cases: usize,
    boolean_output_error_sum: f64,
    multi_output_error_sum: f64,
    boolean_lse_error_sum: f64,
    multi_lse_error_sum: f64,
}

impl Aggregate {
    fn record(&mut self, boolean: SparseMetrics, multi: SparseMetrics) {
        self.boolean_scores += boolean.selected;
        self.multi_scores += multi.selected;
        self.boolean_top2_hits += boolean.top2_hits;
        self.multi_top2_hits += multi.top2_hits;
        self.boolean_empty += usize::from(boolean.selected == 0);
        self.multi_empty += usize::from(multi.selected == 0);

        if let (Some(boolean_output), Some(multi_output), Some(boolean_lse), Some(multi_lse)) = (
            boolean.output_error,
            multi.output_error,
            boolean.lse_error,
            multi.lse_error,
        ) {
            self.comparable_cases += 1;
            self.boolean_output_error_sum += boolean_output;
            self.multi_output_error_sum += multi_output;
            self.boolean_lse_error_sum += boolean_lse;
            self.multi_lse_error_sum += multi_lse;
        }
    }

    fn classification(&self) -> &'static str {
        let pass = self.multi_scores < self.boolean_scores
            && self.multi_top2_hits >= self.boolean_top2_hits
            && self.multi_output_error_sum <= self.boolean_output_error_sum + EPSILON
            && self.multi_lse_error_sum <= self.boolean_lse_error_sum + EPSILON
            && self.multi_empty <= self.boolean_empty;
        if pass {
            "FRONTIER_PASS"
        } else {
            "FRONTIER_REJECT"
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
    let mut rng = SplitMix64::new(HOLDOUT_SEED);
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

fn bind_frozen_manifest() -> Result<()> {
    let route = policy_route(&[true; CANDIDATES])?;
    let predicate_digest = Sha256Digest::parse(POLICY_DIGEST)?;
    let feature_digest = Sha256Digest::parse(FEATURE_SCHEMA_DIGEST)?;
    let tuning_digest = Sha256Digest::parse(TUNING_DATASET_DIGEST)?;
    let confirmatory_digest = Sha256Digest::parse(CONFIRMATORY_DATASET_DIGEST)?;
    let manifest = FrozenPolicyManifest::new(
        POLICY_ID,
        SOURCE_REVISION,
        &route,
        POLICY,
        predicate_digest,
        feature_digest,
        tuning_digest,
        confirmatory_digest,
    )?;
    let context = ConfirmatoryContext::new(
        POLICY_ID,
        SOURCE_REVISION,
        &route,
        POLICY,
        predicate_digest,
        feature_digest,
        confirmatory_digest,
    );
    let binding = manifest.bind_confirmatory(&context)?;
    require(
        binding.manifest().confirmatory_dataset_digest() == confirmatory_digest,
        "confirmatory manifest binding drifted",
    )
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

fn boolean_indices(case: &Case) -> Vec<usize> {
    case.candidates
        .iter()
        .enumerate()
        .filter_map(|(candidate_id, candidate)| candidate.structural_admit.then_some(candidate_id))
        .collect()
}

fn multi_indices(case: &Case) -> Result<Vec<usize>> {
    let admissions =
        std::array::from_fn(|candidate_id| case.candidates[candidate_id].structural_admit);
    let route = policy_route(&admissions)?;
    let f2 = F2AffinePredicate::new(F2Vector::from_bools(&[true, false, false])?, false);
    let zhegalkin = ZhegalkinPolynomial::from_variable_sets(3, vec![vec![1, 2]])?;
    let inputs = case
        .candidates
        .iter()
        .map(|candidate| F2Vector::from_bools(&[candidate.x0, candidate.x1, candidate.x2]))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let frames = inputs
        .iter()
        .enumerate()
        .map(|(candidate_id, input)| {
            CandidateFrame::new(
                candidate_id,
                CandidateQualificationInputs {
                    boolean_block: Some(candidate_id),
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
    Ok(qualify_survivor_set(&route, &frames, POLICY)?
        .survivor_indices()
        .to_vec())
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
        output_error,
        lse_error,
    })
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

fn opt_float(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_owned(), |value| format!("{value:.9}"))
}

fn run() -> Result<Vec<String>> {
    bind_frozen_manifest()?;
    let cases = generate_cases();
    let dense_indices = (0..CANDIDATES).collect::<Vec<_>>();
    let mut aggregate = Aggregate::default();
    let mut rows = Vec::with_capacity(CASES + 1);

    for (case_id, case) in cases.iter().enumerate() {
        let top2 = dense_top2(case);
        let boolean = boolean_indices(case);
        let multi = multi_indices(case)?;
        require(
            multi
                .iter()
                .all(|candidate_id| boolean.contains(candidate_id)),
            "multi-algebra resurrected a Boolean rejection",
        )?;
        let dense = evaluate(case, &dense_indices)?.ok_or("dense selection cannot be empty")?;
        let boolean_metrics = sparse_metrics(case, &boolean, &top2, &dense)?;
        let multi_metrics = sparse_metrics(case, &multi, &top2, &dense)?;
        aggregate.record(boolean_metrics, multi_metrics);
        rows.push(format!(
            "case,{case_id},{},{},{},{},{},{},{},{},{},{},{},{},-,-,-",
            ids(&top2),
            ids(&boolean),
            ids(&multi),
            boolean_metrics.selected,
            multi_metrics.selected,
            boolean_metrics.top2_hits,
            multi_metrics.top2_hits,
            opt_float(boolean_metrics.output_error),
            opt_float(multi_metrics.output_error),
            opt_float(boolean_metrics.lse_error),
            opt_float(multi_metrics.lse_error),
            usize::from(boolean_metrics.selected == 0),
            usize::from(multi_metrics.selected == 0),
        ));
    }

    let additional_avoided = aggregate
        .boolean_scores
        .saturating_sub(aggregate.multi_scores);
    rows.push(format!(
        "aggregate,ALL,-,-,-,{},{},{},{},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{}",
        aggregate.boolean_scores,
        aggregate.multi_scores,
        aggregate.boolean_top2_hits,
        aggregate.multi_top2_hits,
        aggregate.boolean_output_error_sum,
        aggregate.multi_output_error_sum,
        aggregate.boolean_lse_error_sum,
        aggregate.multi_lse_error_sum,
        aggregate.boolean_empty,
        aggregate.multi_empty,
        aggregate.comparable_cases,
        additional_avoided,
        aggregate.classification(),
    ));
    Ok(rows)
}

fn main() -> Result<()> {
    println!(
        "row_type,case_id,dense_top2,boolean_ids,multi_ids,boolean_scores,multi_scores,boolean_top2_hits,multi_top2_hits,boolean_output_error,multi_output_error,boolean_lse_error,multi_lse_error,boolean_empty,multi_empty,comparable_cases,additional_scores_avoided,classification"
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
    fn frozen_manifest_binds_before_evaluation() {
        bind_frozen_manifest().unwrap();
    }

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
    fn confirmatory_rows_are_repeatable_without_retuning() {
        assert_eq!(run().unwrap(), run().unwrap());
    }
}
