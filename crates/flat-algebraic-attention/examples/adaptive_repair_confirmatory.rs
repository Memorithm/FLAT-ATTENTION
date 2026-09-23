//! MAA-14c coverage-7/8 confirmatory holdout.
//! Preregistration: docs/research/MAA_ADAPTIVE_REPAIR_CONFIRMATORY_PREREGISTRATION.md.

use std::error::Error;

use flat_algebraic_attention::adaptive_repair::StructuralRepairTrigger;
use flat_algebraic_attention::f2::F2Vector;
use flat_algebraic_attention::qk_signature::SignedHyperplaneProjector;
use flat_algebraic_attention::softmax_mass::retained_softmax_mass;
use flat_attention::api::boolean_attention_signature::{
    BooleanAttentionSignature, HammingAdmissionRule,
};
use flat_attention::{
    forward_reference_grouped_asymmetric, AsymmetricGroupedAttentionShape, FlatAttentionConfig,
    FlatAttentionOutput,
};

const PROTOCOL: &str = "maa-adaptive-repair-confirmatory/v1";
const DATASET_SEED: u64 = 0x004d_4141_2d31_3463;
const RANDOM_CONTROL_SEED: u64 = 0x4d41_4131_3463_524e;
const PROJECTOR_SEED: u64 = 0x514b_5349_474e_3133;
const CASES: usize = 1_024;
const CANDIDATES: usize = 32;
const D: usize = 16;
const SIGNATURE_BITS: usize = 128;
const NEAR_THRESHOLD: usize = 64;
const INNER_MEDIUM_THRESHOLD: usize = 68;
const MEDIUM_THRESHOLD: usize = 72;
const RECALL_NORM_THRESHOLD: f64 = 1.5;
const KEY_SCALES: [f32; 4] = [0.5, 1.0, 2.0, 4.0];
const DIAGNOSTIC_MASS_THRESHOLD: f64 = 0.90;

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
    retained_mass: f64,
    output_error: Option<f64>,
    lse_error: Option<f64>,
    empty: bool,
}

#[derive(Debug, Clone)]
struct Aggregate {
    rows: usize,
    rows_repaired: usize,
    initial_selected: usize,
    final_selected: usize,
    added: usize,
    top2_hits: usize,
    false_negatives: usize,
    masses: Vec<f64>,
    cases_ge_90: usize,
    cases_ge_95: usize,
    cases_ge_99: usize,
    output_error_sum: f64,
    lse_error_sum: f64,
    base_below_90: usize,
    trigger_true_positive: usize,
    trigger_false_positive: usize,
    trigger_false_negative: usize,
    empty: usize,
}

impl Default for Aggregate {
    fn default() -> Self {
        Self {
            rows: 0,
            rows_repaired: 0,
            initial_selected: 0,
            final_selected: 0,
            added: 0,
            top2_hits: 0,
            false_negatives: 0,
            masses: Vec::with_capacity(CASES),
            cases_ge_90: 0,
            cases_ge_95: 0,
            cases_ge_99: 0,
            output_error_sum: 0.0,
            lse_error_sum: 0.0,
            base_below_90: 0,
            trigger_true_positive: 0,
            trigger_false_positive: 0,
            trigger_false_negative: 0,
            empty: 0,
        }
    }
}

impl Aggregate {
    fn record(
        &mut self,
        initial_count: usize,
        metrics: CaseMetrics,
        repaired: bool,
        diagnostic_positive: bool,
    ) {
        self.rows += 1;
        self.rows_repaired += usize::from(repaired);
        self.initial_selected += initial_count;
        self.final_selected += metrics.selected;
        self.added += metrics.selected.saturating_sub(initial_count);
        self.top2_hits += metrics.top2_hits;
        self.false_negatives += 2usize.saturating_sub(metrics.top2_hits);
        self.masses.push(metrics.retained_mass);
        self.cases_ge_90 += usize::from(metrics.retained_mass >= 0.90);
        self.cases_ge_95 += usize::from(metrics.retained_mass >= 0.95);
        self.cases_ge_99 += usize::from(metrics.retained_mass >= 0.99);
        self.empty += usize::from(metrics.empty);
        if let Some(value) = metrics.output_error {
            self.output_error_sum += value;
        }
        if let Some(value) = metrics.lse_error {
            self.lse_error_sum += value;
        }
        if diagnostic_positive {
            self.base_below_90 += 1;
        }
        match (repaired, diagnostic_positive) {
            (true, true) => self.trigger_true_positive += 1,
            (true, false) => self.trigger_false_positive += 1,
            (false, true) => self.trigger_false_negative += 1,
            (false, false) => {}
        }
    }

    fn finalize(mut self) -> Result<FinalAggregate> {
        require(self.rows == CASES, "aggregate row count drifted")?;
        require(
            self.masses.len() == CASES,
            "aggregate mass sample count drifted",
        )?;
        self.masses.sort_by(f64::total_cmp);
        let precision_denominator = self
            .trigger_true_positive
            .checked_add(self.trigger_false_positive)
            .ok_or("precision denominator overflow")?;
        let recall_denominator = self
            .trigger_true_positive
            .checked_add(self.trigger_false_negative)
            .ok_or("recall denominator overflow")?;
        Ok(FinalAggregate {
            rows: self.rows,
            rows_repaired: self.rows_repaired,
            initial_selected: self.initial_selected,
            final_selected: self.final_selected,
            added: self.added,
            mean_density: self.final_selected as f64 / (CASES * CANDIDATES) as f64,
            top2_hits: self.top2_hits,
            false_negatives: self.false_negatives,
            retained_mass_mean: self.masses.iter().sum::<f64>() / CASES as f64,
            retained_mass_min: self.masses[0],
            retained_mass_p10: nearest_rank(&self.masses, 10, 100)?,
            retained_mass_p50: nearest_rank(&self.masses, 50, 100)?,
            retained_mass_p90: nearest_rank(&self.masses, 90, 100)?,
            cases_ge_90: self.cases_ge_90,
            cases_ge_95: self.cases_ge_95,
            cases_ge_99: self.cases_ge_99,
            output_error_sum: self.output_error_sum,
            lse_error_sum: self.lse_error_sum,
            base_below_90: self.base_below_90,
            trigger_true_positive: self.trigger_true_positive,
            trigger_false_positive: self.trigger_false_positive,
            trigger_false_negative: self.trigger_false_negative,
            trigger_precision: ratio(self.trigger_true_positive, precision_denominator),
            trigger_recall: ratio(self.trigger_true_positive, recall_denominator),
            empty: self.empty,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct FinalAggregate {
    rows: usize,
    rows_repaired: usize,
    initial_selected: usize,
    final_selected: usize,
    added: usize,
    mean_density: f64,
    top2_hits: usize,
    false_negatives: usize,
    retained_mass_mean: f64,
    retained_mass_min: f64,
    retained_mass_p10: f64,
    retained_mass_p50: f64,
    retained_mass_p90: f64,
    cases_ge_90: usize,
    cases_ge_95: usize,
    cases_ge_99: usize,
    output_error_sum: f64,
    lse_error_sum: f64,
    base_below_90: usize,
    trigger_true_positive: usize,
    trigger_false_positive: usize,
    trigger_false_negative: usize,
    trigger_precision: Option<f64>,
    trigger_recall: Option<f64>,
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
        mix64(self.state)
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

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then_some(numerator as f64 / denominator as f64)
}

fn format_ratio(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_owned(), |value| format!("{value:.12}"))
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
        .ok_or("nearest-rank multiplication overflow")?;
    let rank = scaled.div_ceil(denominator).max(1);
    Ok(sorted[rank - 1])
}

fn generated_vector(rng: &mut SplitMix64) -> [f32; D] {
    std::array::from_fn(|_| rng.next_f32())
}

fn l2_norm(vector: &[f32; D]) -> Result<f64> {
    let squared = vector.iter().try_fold(0.0f64, |total, value| {
        require(value.is_finite(), "generated vector is non-finite")?;
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

fn generate_cases() -> Result<Vec<Case>> {
    let mut rng = SplitMix64::new(DATASET_SEED);
    let mut cases = Vec::with_capacity(CASES);
    for _ in 0..CASES {
        let q = normalize(generated_vector(&mut rng))?;
        let mut candidates = Vec::with_capacity(CANDIDATES);
        for _ in 0..CANDIDATES {
            let direction = normalize(generated_vector(&mut rng))?;
            let scale = KEY_SCALES[(rng.next_u64() & 0b11) as usize];
            candidates.push(Candidate {
                k: direction.map(|value| value * scale),
                v: generated_vector(&mut rng),
            });
        }
        cases.push(Case {
            q,
            candidates: candidates
                .try_into()
                .map_err(|_| "candidate array conversion failed")?,
        });
    }
    Ok(cases)
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

fn dense_top2(scores: &[f64]) -> Result<[usize; 2]> {
    require(scores.len() >= 2, "dense top-2 requires two candidates")?;
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|(left_id, left_score), (right_id, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_id.cmp(right_id))
    });
    Ok([ranked[0].0, ranked[1].0])
}

fn to_m13b(vector: &F2Vector) -> Result<BooleanAttentionSignature> {
    BooleanAttentionSignature::new(vector.bit_len(), vector.words().to_vec())
        .map_err(|error| Box::new(error) as Box<dyn Error>)
}

fn base_and_medium(
    case: &Case,
    projector: &SignedHyperplaneProjector,
    near_rule: &HammingAdmissionRule,
    inner_rule: &HammingAdmissionRule,
    medium_rule: &HammingAdmissionRule,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let q_signature = to_m13b(&projector.project(&case.q)?)?;
    let mut base = Vec::new();
    let mut medium = Vec::new();

    for (candidate_id, candidate) in case.candidates.iter().enumerate() {
        let key_signature = to_m13b(&projector.project(&candidate.k)?)?;
        let near = near_rule.admits(&q_signature, &key_signature)?;
        let inner = inner_rule.admits(&q_signature, &key_signature)?;
        let in_medium = medium_rule.admits(&q_signature, &key_signature)?;
        require(!near || inner, "near route escaped inner-medium envelope")?;
        require(
            !inner || in_medium,
            "inner-medium route escaped medium envelope",
        )?;
        if in_medium {
            medium.push(candidate_id);
        }
        if inner || (in_medium && l2_norm(&candidate.k)? >= RECALL_NORM_THRESHOLD) {
            base.push(candidate_id);
        }
    }

    require(
        base.iter()
            .all(|candidate_id| medium.contains(candidate_id)),
        "tiered-recall base escaped hamming-medium envelope",
    )?;
    Ok((base, medium))
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

fn case_metrics(
    case: &Case,
    selected: &[usize],
    top2: &[usize; 2],
    dense: &FlatAttentionOutput,
    score_values: &[f64],
) -> Result<CaseMetrics> {
    let mass = retained_softmax_mass(score_values, selected)?;
    let top2_hits = selected
        .iter()
        .filter(|candidate_id| top2.contains(candidate_id))
        .count();
    match evaluate(case, selected)? {
        None => Ok(CaseMetrics {
            selected: 0,
            top2_hits,
            retained_mass: mass.retained_mass(),
            output_error: None,
            lse_error: None,
            empty: true,
        }),
        Some(sparse) => {
            let output_error = dense
                .output
                .iter()
                .zip(&sparse.output)
                .map(|(left, right)| f64::from((left - right).abs()))
                .fold(0.0f64, f64::max);
            let lse_error = f64::from((dense.lse[0] - sparse.lse[0]).abs());
            Ok(CaseMetrics {
                selected: selected.len(),
                top2_hits,
                retained_mass: mass.retained_mass(),
                output_error: Some(output_error),
                lse_error: Some(lse_error),
                empty: false,
            })
        }
    }
}

fn adaptive_selection<'a>(
    trigger: StructuralRepairTrigger,
    base: &'a [usize],
    medium: &'a [usize],
) -> Result<(&'a [usize], bool)> {
    let decision = trigger.decide(base.len(), medium.len())?;
    Ok(if decision.repair {
        (medium, true)
    } else {
        (base, false)
    })
}

fn matched_random_expansion(
    case_id: usize,
    trigger: StructuralRepairTrigger,
    base: &[usize],
    target_count: usize,
) -> Result<Vec<usize>> {
    require(target_count >= base.len(), "random target is below base")?;
    require(
        target_count <= CANDIDATES,
        "random target exceeds candidate geometry",
    )?;
    let trigger_id = trigger
        .random_control_id()
        .ok_or("endpoint trigger has no random control ID")?;
    let mut selected = base.to_vec();
    let add_count = target_count - base.len();
    let mut complement = (0..CANDIDATES)
        .filter(|candidate_id| !base.contains(candidate_id))
        .map(|candidate_id| {
            let case_key = u64::try_from(case_id).map_err(|_| "case ID exceeds u64")?;
            let candidate_key =
                u64::try_from(candidate_id).map_err(|_| "candidate ID exceeds u64")?;
            let priority = mix64(
                RANDOM_CONTROL_SEED
                    ^ case_key.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ trigger_id.wrapping_mul(0x94d0_49bb_1331_11eb)
                    ^ candidate_key.wrapping_mul(0xbf58_476d_1ce4_e5b9),
            );
            Ok((priority, candidate_id))
        })
        .collect::<Result<Vec<_>>>()?;
    complement.sort_unstable();
    selected.extend(
        complement
            .into_iter()
            .take(add_count)
            .map(|(_, candidate_id)| candidate_id),
    );
    selected.sort_unstable();
    require(
        selected.len() == target_count,
        "random control cardinality drifted",
    )?;
    require(
        base.iter()
            .all(|candidate_id| selected.contains(candidate_id)),
        "random control removed a base candidate",
    )?;
    Ok(selected)
}

#[must_use]
const fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct CaseEvaluation<'a> {
    case: &'a Case,
    base: &'a [usize],
    medium: &'a [usize],
    top2: &'a [usize; 2],
    dense: &'a FlatAttentionOutput,
    scores: &'a [f64],
    diagnostic_positive: bool,
}

fn record_trigger_arm(
    aggregate: &mut Aggregate,
    trigger: StructuralRepairTrigger,
    context: &CaseEvaluation<'_>,
) -> Result<()> {
    let (selection, repaired) = adaptive_selection(trigger, context.base, context.medium)?;
    aggregate.record(
        context.base.len(),
        case_metrics(
            context.case,
            selection,
            context.top2,
            context.dense,
            context.scores,
        )?,
        repaired,
        context.diagnostic_positive,
    );
    Ok(())
}

fn record_random_arm(
    aggregate: &mut Aggregate,
    trigger: StructuralRepairTrigger,
    case_id: usize,
    context: &CaseEvaluation<'_>,
) -> Result<()> {
    let (_, repaired) = adaptive_selection(trigger, context.base, context.medium)?;
    let target_count = if repaired {
        context.medium.len()
    } else {
        context.base.len()
    };
    let random = matched_random_expansion(case_id, trigger, context.base, target_count)?;
    aggregate.record(
        context.base.len(),
        case_metrics(
            context.case,
            &random,
            context.top2,
            context.dense,
            context.scores,
        )?,
        repaired,
        context.diagnostic_positive,
    );
    Ok(())
}

fn run() -> Result<(Vec<String>, bool, f64, f64)> {
    require(
        PROTOCOL == "maa-adaptive-repair-confirmatory/v1",
        "MAA-14c protocol identity drifted",
    )?;

    let cases = generate_cases()?;
    let projector = SignedHyperplaneProjector::new(D, SIGNATURE_BITS, PROJECTOR_SEED)?;
    let near_rule = HammingAdmissionRule::new(NEAR_THRESHOLD, SIGNATURE_BITS)?;
    let inner_rule = HammingAdmissionRule::new(INNER_MEDIUM_THRESHOLD, SIGNATURE_BITS)?;
    let medium_rule = HammingAdmissionRule::new(MEDIUM_THRESHOLD, SIGNATURE_BITS)?;

    let mut dense_aggregate = Aggregate::default();
    let mut base_aggregate = Aggregate::default();
    let mut candidate_aggregate = Aggregate::default();
    let mut random_aggregate = Aggregate::default();
    let mut always_aggregate = Aggregate::default();
    let mut oracle_aggregate = Aggregate::default();
    let dense_indices = (0..CANDIDATES).collect::<Vec<_>>();

    for (case_id, case) in cases.iter().enumerate() {
        let score_values = scores(case)?;
        let top2 = dense_top2(&score_values)?;
        let dense = evaluate(case, &dense_indices)?.ok_or("dense selection cannot be empty")?;
        let (base, medium) =
            base_and_medium(case, &projector, &near_rule, &inner_rule, &medium_rule)?;
        let base_mass = retained_softmax_mass(&score_values, &base)?;
        let diagnostic_positive = base_mass.retained_mass() < DIAGNOSTIC_MASS_THRESHOLD;

        dense_aggregate.record(
            CANDIDATES,
            case_metrics(case, &dense_indices, &top2, &dense, &score_values)?,
            false,
            diagnostic_positive,
        );

        let context = CaseEvaluation {
            case,
            base: &base,
            medium: &medium,
            top2: &top2,
            dense: &dense,
            scores: &score_values,
            diagnostic_positive,
        };

        record_trigger_arm(
            &mut base_aggregate,
            StructuralRepairTrigger::NeverRepair,
            &context,
        )?;
        record_trigger_arm(
            &mut candidate_aggregate,
            StructuralRepairTrigger::CoverageBelowSevenEighths,
            &context,
        )?;
        record_random_arm(
            &mut random_aggregate,
            StructuralRepairTrigger::CoverageBelowSevenEighths,
            case_id,
            &context,
        )?;
        record_trigger_arm(
            &mut always_aggregate,
            StructuralRepairTrigger::AlwaysRepair,
            &context,
        )?;

        let oracle_repair = diagnostic_positive;
        let oracle_selection = if oracle_repair { &medium } else { &base };
        oracle_aggregate.record(
            base.len(),
            case_metrics(case, oracle_selection, &top2, &dense, &score_values)?,
            oracle_repair,
            diagnostic_positive,
        );
    }

    let dense = dense_aggregate.finalize()?;
    let base = base_aggregate.finalize()?;
    let candidate = candidate_aggregate.finalize()?;
    let random = random_aggregate.finalize()?;
    let always = always_aggregate.finalize()?;
    let oracle = oracle_aggregate.finalize()?;

    require(always.added > 0, "confirmatory always-repair added no candidates")?;
    let added_fraction = candidate.added as f64 / always.added as f64;
    require(
        added_fraction.is_finite(),
        "confirmatory added-candidate fraction is non-finite",
    )?;

    let mass_gap = always.retained_mass_mean - base.retained_mass_mean;
    require(
        mass_gap.is_finite() && mass_gap > 0.0,
        "confirmatory retained-mass endpoint gap is invalid",
    )?;
    let mass_recovery =
        (candidate.retained_mass_mean - base.retained_mass_mean) / mass_gap;
    require(
        mass_recovery.is_finite(),
        "confirmatory retained-mass recovery is non-finite",
    )?;

    let confirmatory_pass = candidate.empty == 0
        && candidate.false_negatives <= base.false_negatives
        && candidate.rows_repaired > 0
        && candidate.rows_repaired < CASES
        && candidate.final_selected > base.final_selected
        && candidate.final_selected < always.final_selected
        && candidate.retained_mass_mean > base.retained_mass_mean
        && candidate.retained_mass_p10 > base.retained_mass_p10
        && candidate.output_error_sum < base.output_error_sum
        && candidate.lse_error_sum < base.lse_error_sum
        && candidate.retained_mass_mean > random.retained_mass_mean
        && added_fraction <= 0.60
        && mass_recovery >= 0.20;

    let rows = vec![
        format_row("dense", dense),
        format_row("never_repair", base),
        format_row("repair_if_coverage_below_7_8", candidate),
        format_row("matched_random_repair_if_coverage_below_7_8", random),
        format_row("always_repair", always),
        format_row("oracle_repair_if_base_mass_below_0_90", oracle),
    ];

    Ok((rows, confirmatory_pass, added_fraction, mass_recovery))
}

fn format_row(label: &str, aggregate: FinalAggregate) -> String {
    format!(
        "{label},{},{},{},{},{},{:.9},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{},{},{},{:.9},{:.9},{},{},{},{},{},{},{},{}",
        aggregate.rows,
        aggregate.rows_repaired,
        aggregate.initial_selected,
        aggregate.final_selected,
        aggregate.added,
        aggregate.mean_density,
        aggregate.top2_hits,
        aggregate.false_negatives,
        aggregate.retained_mass_mean,
        aggregate.retained_mass_min,
        aggregate.retained_mass_p10,
        aggregate.retained_mass_p50,
        aggregate.retained_mass_p90,
        aggregate.cases_ge_90,
        aggregate.cases_ge_95,
        aggregate.cases_ge_99,
        aggregate.output_error_sum,
        aggregate.lse_error_sum,
        aggregate.base_below_90,
        aggregate.trigger_true_positive,
        aggregate.trigger_false_positive,
        aggregate.trigger_false_negative,
        format_ratio(aggregate.trigger_precision),
        format_ratio(aggregate.trigger_recall),
        aggregate.empty,
        PROTOCOL,
    )
}

fn main() -> Result<()> {
    println!(
        "arm,rows,rows_repaired,initial_selected,final_selected,added,mean_density,top2_hits,false_negatives,retained_mass_mean,retained_mass_min,retained_mass_p10,retained_mass_p50,retained_mass_p90,cases_ge_0_90,cases_ge_0_95,cases_ge_0_99,output_error_sum,lse_error_sum,base_below_0_90,trigger_tp,trigger_fp,trigger_fn,trigger_precision,trigger_recall,empty_cases,protocol"
    );
    let (rows, pass, added_fraction, mass_recovery) = run()?;
    for row in rows {
        println!("{row}");
    }
    println!(
        "decision,{},added_fraction={:.12},mass_recovery={:.12},{}",
        if pass {
            "CONFIRMATORY_PASS"
        } else {
            "CONFIRMATORY_FAIL"
        },
        added_fraction,
        mass_recovery,
        PROTOCOL,
    );
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
    fn matched_random_control_is_exact_and_repeatable() {
        let base = vec![1, 4, 7, 11];
        let trigger = StructuralRepairTrigger::CoverageBelowSevenEighths;
        for target in base.len()..=CANDIDATES {
            let first = matched_random_expansion(19, trigger, &base, target).unwrap();
            let second = matched_random_expansion(19, trigger, &base, target).unwrap();
            assert_eq!(first, second);
            assert_eq!(first.len(), target);
            assert!(base.iter().all(|candidate_id| first.contains(candidate_id)));
            assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn evidence_rows_are_byte_repeatable() {
        assert_eq!(run().unwrap(), run().unwrap());
    }

    #[test]
    fn decision_metrics_are_finite() {
        let (_, _, added_fraction, mass_recovery) = run().unwrap();
        assert!(added_fraction.is_finite());
        assert!(mass_recovery.is_finite());
    }
}
