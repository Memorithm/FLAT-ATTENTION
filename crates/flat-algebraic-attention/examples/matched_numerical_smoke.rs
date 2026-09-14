//! MAA-10a synthetic numerical diagnostics; not a performance benchmark.
//! Protocol: docs/research/MAA_NUMERICAL_SMOKE_PREREGISTRATION.md.

use std::error::Error;
use std::io::{self, Write};

use flat_algebraic_attention::cooperation::{
    route_attention_needs, AttentionAlgebraNeeds, M13bBooleanRoutingEvidence,
};
use flat_algebraic_attention::evidence::MatchedWorkloadIdentity;
use flat_algebraic_attention::experiment::derive_matched_host_evidence;
use flat_algebraic_attention::f2::{F2AffinePredicate, F2Vector};
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

const PROTOCOL: &str = "maa-numerical-smoke/v1";
const N: usize = 8;
const D: usize = 2;
const SEED: u64 = 0x4d41410a;
const POLICY: RecompositionPolicy = RecompositionPolicy::AllSelectedMustQualify;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone)]
struct Case {
    id: &'static str,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    boolean: Vec<bool>,
    features: Vec<[bool; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
struct Row {
    case: &'static str,
    arm: &'static str,
    indices: Vec<usize>,
    retained_top1: usize,
    output_error: Option<f32>,
    lse_error: Option<f32>,
}

fn require(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn frozen_cases() -> Vec<Case> {
    [
        "all_accept",
        "constant_values",
        "dominant_key_dropped",
        "empty_selection",
    ]
    .into_iter()
    .map(|id| {
        let mut k = Vec::with_capacity(N * D);
        let mut v = Vec::with_capacity(N * D);
        for i in 0..N {
            if id == "dominant_key_dropped" {
                k.extend_from_slice(&[if i == 2 { 12.0 } else { 0.0 }, 0.0]);
                let value = if i == 2 { 10.0 } else { 0.0 };
                v.extend_from_slice(&[value, -value]);
            } else {
                k.extend_from_slice(&[i as f32 / 4.0 - 1.0, (i % 3) as f32 - 1.0]);
                if id == "constant_values" {
                    v.extend_from_slice(&[2.0, -3.0]);
                } else {
                    v.extend_from_slice(&[i as f32 / 8.0, 1.0 - i as f32 / 8.0]);
                }
            }
        }
        let boolean = (0..N)
            .map(|i| id == "all_accept" || (i != 3 && i != 7))
            .collect();
        let features = (0..N)
            .map(|i| {
                if id == "all_accept" {
                    [true; 3]
                } else {
                    [id != "empty_selection" && i % 2 == 0, i < 6, i != 2]
                }
            })
            .collect();
        Case {
            id,
            q: vec![1.0, 0.0],
            k,
            v,
            boolean,
            features,
        }
    })
    .collect()
}

fn validate_case(case: &Case) -> Result<()> {
    require(case.q.len() == D, "wrong query geometry")?;
    require(
        case.k.len() == N * D && case.v.len() == N * D,
        "wrong KV geometry",
    )?;
    require(
        case.boolean.len() == N && case.features.len() == N,
        "wrong predicate geometry",
    )?;
    require(
        case.q
            .iter()
            .chain(&case.k)
            .chain(&case.v)
            .all(|value| value.is_finite()),
        "non-finite input, including rejected keys",
    )
}

fn validate_selection(indices: &[usize]) -> Result<()> {
    require(
        indices.iter().all(|&i| i < N),
        "candidate index out of bounds",
    )?;
    require(
        indices.windows(2).all(|pair| pair[0] < pair[1]),
        "candidate IDs must be unique and ordered",
    )
}

/// No positional terms are present; all original keys are in the completed prefix.
/// This compaction is deliberately NOT a general positional attention adapter.
fn evaluate(case: &Case, indices: &[usize]) -> Result<Option<FlatAttentionOutput>> {
    validate_case(case)?;
    validate_selection(indices)?;
    if indices.is_empty() {
        return Ok(None);
    }
    let mut k = Vec::with_capacity(indices.len() * D);
    let mut v = Vec::with_capacity(indices.len() * D);
    for &i in indices {
        k.extend_from_slice(&case.k[i * D..(i + 1) * D]);
        v.extend_from_slice(&case.v[i * D..(i + 1) * D]);
    }
    let output = forward_reference_grouped_asymmetric(
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
            query_position_offset: N - 1,
        },
        FlatAttentionConfig {
            causal: false,
            softmax_scale: Some(1.0),
        },
    )?;
    require(
        output.output.len() == D && output.lse.len() == 1,
        "unexpected oracle output geometry",
    )?;
    require(
        output
            .output
            .iter()
            .chain(&output.lse)
            .all(|x| x.is_finite()),
        "non-finite oracle result",
    )?;
    Ok(Some(output))
}

/// Dense scoring is offline label construction, not an input to the policies.
fn dense_top1(case: &Case) -> Result<usize> {
    validate_case(case)?;
    let mut scores = Vec::with_capacity(N);
    for i in 0..N {
        let mut score = 0.0f32;
        for d in 0..D {
            score += case.q[d] * case.k[i * D + d];
        }
        require(score.is_finite(), "non-finite dense reference score")?;
        scores.push((i, score));
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(scores[0].0)
}

fn algebraic_selection(case: &Case, use_f2: bool, use_zhegalkin: bool) -> Result<Vec<usize>> {
    validate_case(case)?;
    let mask = BooleanAttentionMask::from_admissions(&case.boolean)?;
    // Structural eligibility feature vectors, NOT numerical Q/K sign signatures.
    // The dependency-free MAA adapter currently holds one such pair per route.
    let query_policy = BooleanAttentionSignature::new(N, vec![(1u64 << N) - 1])?;
    let key_policy = BooleanAttentionSignature::new(N, mask.words().to_vec())?;
    let route = route_attention_needs(
        AttentionAlgebraNeeds {
            eligibility_logic: true,
            parity_or_binary_linear: use_f2,
            nonlinear_boolean_interaction: use_zhegalkin,
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
    let predicate = F2AffinePredicate::new(F2Vector::from_bools(&[true, false, false])?, false);
    let polynomial = ZhegalkinPolynomial::from_variable_sets(3, vec![vec![1, 2]])?;
    let inputs = case
        .features
        .iter()
        .map(|x| F2Vector::from_bools(x))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let frames: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            CandidateFrame::new(
                i,
                CandidateQualificationInputs {
                    boolean_block: Some(i),
                    f2: use_f2.then_some(F2CandidateEvaluation {
                        predicate: &predicate,
                        input,
                    }),
                    zhegalkin: use_zhegalkin.then_some(ZhegalkinCandidateEvaluation {
                        polynomial: &polynomial,
                        input,
                    }),
                    max_plus: None,
                },
            )
        })
        .collect();
    let survivors = qualify_survivor_set(&route, &frames, POLICY)?;
    // These fixed fixtures are versioned in source; this identifier is not a hash.
    let identity =
        MatchedWorkloadIdentity::new(PROTOCOL, case.id, "source-fixture:v1:not-a-digest", N)?;
    let evidence = derive_matched_host_evidence(
        identity,
        &[dense_top1(case)?],
        &route,
        POLICY,
        &survivors,
        None,
    )?;
    require(
        !evidence.has_matched_latency(),
        "synthetic smoke must not carry latency",
    )?;
    require(
        evidence.multi_algebra().survivor_count() == survivors.survivor_count(),
        "evidence mismatch",
    )?;
    let indices = survivors.survivor_indices().to_vec();
    validate_selection(&indices)?;
    Ok(indices)
}

fn mixed_priority(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn density_control(boolean: &[usize], count: usize, arm: &'static str) -> Result<Vec<usize>> {
    validate_selection(boolean)?;
    require(
        count <= boolean.len(),
        "control exceeds available candidates",
    )?;
    let mut indices = boolean.to_vec();
    match arm {
        "first" => {}
        "last" => indices.reverse(),
        "seeded_priority" => indices.sort_by_key(|&i| (mixed_priority(i as u64 ^ SEED), i)),
        _ => return Err("unknown density control".into()),
    }
    indices.truncate(count);
    indices.sort_unstable();
    Ok(indices)
}

fn case_rows(case: &Case) -> Result<Vec<Row>> {
    validate_case(case)?;
    let all: Vec<_> = (0..N).collect();
    let dense = evaluate(case, &all)?.ok_or("dense case unexpectedly empty")?;
    let relevant = dense_top1(case)?;
    let boolean = BooleanAttentionMask::from_admissions(&case.boolean)?.admitted_blocks();
    let f2 = algebraic_selection(case, true, false)?;
    let zhegalkin = algebraic_selection(case, false, true)?;
    let combined = algebraic_selection(case, true, true)?;
    let mut arms = vec![
        ("dense", all),
        ("boolean", boolean.clone()),
        ("f2", f2),
        ("zhegalkin", zhegalkin),
        ("combined", combined.clone()),
    ];
    for name in ["first", "last", "seeded_priority"] {
        arms.push((name, density_control(&boolean, combined.len(), name)?));
    }
    let mut rows = Vec::new();
    for (arm, indices) in arms {
        let result = evaluate(case, &indices)?;
        let (output_error, lse_error) = match result {
            None => (None, None),
            Some(output) => {
                let error = output
                    .output
                    .iter()
                    .zip(&dense.output)
                    .map(|(&actual, &reference)| (actual - reference).abs())
                    .fold(0.0f32, f32::max);
                let lse = (output.lse[0] - dense.lse[0]).abs();
                require(
                    error.is_finite() && lse.is_finite(),
                    "non-finite diagnostic",
                )?;
                (Some(error), Some(lse))
            }
        };
        rows.push(Row {
            case: case.id,
            arm,
            retained_top1: usize::from(indices.contains(&relevant)),
            indices,
            output_error,
            lse_error,
        });
    }
    Ok(rows)
}

fn row<'a>(rows: &'a [Row], case: &str, arm: &str) -> Result<&'a Row> {
    rows.iter()
        .find(|r| r.case == case && r.arm == arm)
        .ok_or_else(|| "missing frozen arm".into())
}

fn validate_acceptance(rows: &[Row]) -> Result<()> {
    require(rows.len() == 32, "wrong number of frozen observations")?;
    for arm in [
        "dense",
        "boolean",
        "f2",
        "zhegalkin",
        "combined",
        "first",
        "last",
        "seeded_priority",
    ] {
        let r = row(rows, "all_accept", arm)?;
        require(
            r.indices.len() == N
                && r.output_error.is_some_and(|x| x <= 1e-6)
                && r.lse_error.is_some_and(|x| x <= 1e-6),
            "all-accept parity failed",
        )?;
    }
    let constant = row(rows, "constant_values", "combined")?;
    require(
        constant.output_error.is_some_and(|x| x <= 1e-6)
            && constant.lse_error.is_some_and(|x| x > 1e-3),
        "constant-value O/LSE distinction failed",
    )?;
    let negative = row(rows, "dominant_key_dropped", "combined")?;
    require(
        negative.retained_top1 == 0 && negative.output_error.is_some_and(|x| x > 9.0),
        "required negative control failed",
    )?;
    require(
        row(rows, "dominant_key_dropped", "f2")?.retained_top1 == 1,
        "F2 ablation lost the dominant key",
    )?;
    for r in rows {
        validate_selection(&r.indices)?;
        if r.indices.is_empty() {
            require(
                r.output_error.is_none() && r.lse_error.is_none() && r.retained_top1 == 0,
                "empty arm fabricated output",
            )?;
        }
    }
    let empty = row(rows, "empty_selection", "combined")?;
    require(
        empty.indices.is_empty(),
        "empty-selection fixture is not empty",
    )?;
    for case in frozen_cases() {
        let boolean = row(rows, case.id, "boolean")?;
        let combined = row(rows, case.id, "combined")?;
        for name in ["first", "last", "seeded_priority"] {
            let control = row(rows, case.id, name)?;
            require(
                control.indices.len() == combined.indices.len()
                    && control.indices.iter().all(|i| boolean.indices.contains(i)),
                "density control mismatch",
            )?;
        }
    }
    Ok(())
}

fn run_suite() -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    for case in frozen_cases() {
        rows.extend(case_rows(&case)?);
    }
    validate_acceptance(&rows)?;
    Ok(rows)
}

fn write_csv(writer: &mut impl Write, rows: &[Row]) -> Result<()> {
    writeln!(writer, "protocol,case,arm,candidates,score_evaluations,retained_top1,status,output_max_abs_error,lse_abs_error,selected_ids")?;
    for r in rows {
        let status = if r.indices.is_empty() {
            "no_survivors"
        } else {
            "observed"
        };
        let output = r
            .output_error
            .map_or_else(String::new, |x| format!("{x:.9e}"));
        let lse = r.lse_error.map_or_else(String::new, |x| format!("{x:.9e}"));
        let ids = r
            .indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("|");
        writeln!(
            writer,
            "{PROTOCOL},{},{},{N},{},{},{status},{output},{lse},{ids}",
            r.case,
            r.arm,
            r.indices.len(),
            r.retained_top1
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let rows = run_suite()?;
    write_csv(&mut io::stdout().lock(), &rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_frozen_criteria_pass_and_all_arms_are_reported() {
        let rows = run_suite().unwrap();
        assert_eq!(rows.len(), 32);
        assert_eq!(
            row(&rows, "dominant_key_dropped", "combined")
                .unwrap()
                .indices,
            vec![0, 4]
        );
    }

    #[test]
    fn all_accept_and_single_key_reuse_the_numerical_oracle() {
        let case = &frozen_cases()[0];
        let all: Vec<_> = (0..N).collect();
        let dense = evaluate(case, &all).unwrap().unwrap();
        let selection = algebraic_selection(case, true, true).unwrap();
        assert_eq!(evaluate(case, &selection).unwrap().unwrap(), dense);
        let single = evaluate(case, &[2]).unwrap().unwrap();
        assert_eq!(single.output, case.v[2 * D..3 * D]);
        assert_eq!(single.lse[0], -0.5);
    }

    #[test]
    fn empty_selection_never_substitutes_dense_or_zero_output() {
        let case = &frozen_cases()[3];
        let selected = algebraic_selection(case, true, true).unwrap();
        assert!(selected.is_empty());
        assert_eq!(evaluate(case, &selected).unwrap(), None);
    }

    #[test]
    fn malformed_selections_and_inputs_fail_closed_even_for_empty_output() {
        let case = &frozen_cases()[0];
        for ids in [vec![0, 0], vec![2, 1], vec![N]] {
            assert!(evaluate(case, &ids).is_err());
        }
        let mut bad = case.clone();
        bad.k[7 * D] = f32::NAN;
        assert!(evaluate(&bad, &[]).is_err());
        assert!(evaluate(&bad, &[0]).is_err());
        bad = case.clone();
        bad.v.pop();
        assert!(evaluate(&bad, &[0]).is_err());
        bad = case.clone();
        bad.features.pop();
        assert!(algebraic_selection(&bad, true, true).is_err());
    }

    #[test]
    fn finite_inputs_that_overflow_are_not_published_as_success() {
        let mut case = frozen_cases()[0].clone();
        case.q[0] = f32::MAX;
        case.k[0] = f32::MAX;
        assert!(evaluate(&case, &[0]).is_err());
        assert!(dense_top1(&case).is_err());
    }

    #[test]
    fn density_controls_are_reproducible_nested_and_label_independent() {
        let boolean = vec![0, 1, 2, 4, 5, 6];
        for count in 0..=boolean.len() {
            for arm in ["first", "last", "seeded_priority"] {
                let ids = density_control(&boolean, count, arm).unwrap();
                assert_eq!(ids.len(), count);
                assert!(ids.iter().all(|i| boolean.contains(i)));
                assert!(validate_selection(&ids).is_ok());
                assert_eq!(ids, density_control(&boolean, count, arm).unwrap());
            }
        }
        let rows = run_suite().unwrap();
        for arm in ["first", "last", "seeded_priority"] {
            assert_eq!(
                row(&rows, "constant_values", arm).unwrap().indices,
                row(&rows, "dominant_key_dropped", arm).unwrap().indices
            );
        }
        assert!(density_control(&boolean, boolean.len() + 1, "first").is_err());
        assert!(density_control(&boolean, 1, "unknown").is_err());
    }

    #[test]
    fn dense_top1_ties_choose_original_smallest_id() {
        let mut case = frozen_cases()[0].clone();
        case.k.fill(0.0);
        assert_eq!(dense_top1(&case).unwrap(), 0);
    }

    #[test]
    fn csv_is_repeatable_and_contains_negative_and_empty_controls() {
        let rows = run_suite().unwrap();
        let mut first = Vec::new();
        let mut second = Vec::new();
        write_csv(&mut first, &rows).unwrap();
        write_csv(&mut second, &run_suite().unwrap()).unwrap();
        assert_eq!(first, second);
        let csv = String::from_utf8(first).unwrap();
        assert_eq!(csv.lines().count(), 33);
        assert!(csv.contains("dominant_key_dropped,combined,8,2,0,observed,"));
        assert!(csv.contains("empty_selection,combined,8,0,0,no_survivors,,,"));
    }
}
