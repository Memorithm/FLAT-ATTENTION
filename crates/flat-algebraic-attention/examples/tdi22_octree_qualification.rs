//! Deterministic, host-only numerical qualification for the TDI-22 octree.
//!
//! This is an exploratory CPU harness, NOT real-hardware performance evidence.
//! Invoke via Memorithm/RemoteOps for isolated physical runs; no GPU is used.

use flat_algebraic_attention::tdi22_torsor_projection::{TorsorKey, TorsorQuery};
use flat_algebraic_attention::torsor_octree::{
    PreparedTorsorOctree, TorsorAnalyticalErrorBounds, TorsorOutputErrorBudget, TorsorTreeOutput,
    MAX_OCTREE_KEYS, TDI22_TORSOR_OCTREE_LOCAL_BUDGET_V2, TDI22_TORSOR_OCTREE_OUTPUT_BUDGET_V1,
};
use std::hint::black_box;
use std::time::Instant;

const SCALE: f64 = 0.125;
const THETA: f64 = 0.5;
const MAX_SCORE_SPREAD: f64 = 0.25;
const BUDGET: TorsorOutputErrorBudget = TorsorOutputErrorBudget {
    resultant_abs: [0.05; 3],
    moment_at_query_abs: [0.05; 3],
};

type Vec3 = [f64; 3];

#[derive(Clone, Copy, Debug)]
enum Panel {
    UniformProjectedScore,
    SmoothSemantic,
    FarAdversarial,
    CoincidentCoordinates,
    SeparatedClusters,
}

impl Panel {
    const ALL: [Self; 4] = [
        Self::UniformProjectedScore,
        Self::SmoothSemantic,
        Self::FarAdversarial,
        Self::CoincidentCoordinates,
    ];
    const COMPARE_LOCAL: [Self; 5] = [
        Self::UniformProjectedScore,
        Self::SmoothSemantic,
        Self::FarAdversarial,
        Self::CoincidentCoordinates,
        Self::SeparatedClusters,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::UniformProjectedScore => "uniform_projected_score",
            Self::SmoothSemantic => "smooth_semantic",
            Self::FarAdversarial => "far_adversarial",
            Self::CoincidentCoordinates => "coincident_coordinates",
            Self::SeparatedClusters => "separated_clusters",
        }
    }

    fn keys(self, count: usize) -> Vec<TorsorKey> {
        if matches!(self, Self::SeparatedClusters) {
            let split = count / 2;
            return (0..count)
                .map(|i| {
                    let (cluster, within) = if i < split {
                        (0.0_f32, i)
                    } else {
                        (1.0_f32, i - split)
                    };
                    TorsorKey {
                        reference: [cluster * 100.0 + within as f32 * 0.01, 0.0, 0.0],
                        resultant: [cluster * 10.0 + within as f32 * 0.0005, 0.0, 0.0],
                        moment_at_reference: [0.0; 3],
                    }
                })
                .collect();
        }
        (0..count)
            .map(|i| {
                let t = i as f32;
                let reference = match self {
                    Self::CoincidentCoordinates => [0.0; 3],
                    Self::SeparatedClusters => unreachable!("handled above"),
                    _ => [
                        t * 0.04,
                        ((i * 17) % 37) as f32 * 0.03,
                        ((i * 11) % 29) as f32 * 0.02,
                    ],
                };
                let resultant = match self {
                    Self::UniformProjectedScore => [1.0, 0.02 * t, (0.1 * t).sin()],
                    Self::SmoothSemantic => [
                        0.5 + 0.004 * t,
                        -0.2 + 0.02 * (t * 0.03).sin(),
                        0.1 * (t * 0.07).cos(),
                    ],
                    Self::FarAdversarial => [if i % 2 == 0 { 12.0 } else { -12.0 }, 0.01 * t, 0.0],
                    Self::CoincidentCoordinates => [0.1 + 0.02 * t, 0.1 * (t * 0.1).sin(), -0.15],
                    Self::SeparatedClusters => unreachable!("handled above"),
                };
                TorsorKey {
                    reference,
                    resultant,
                    moment_at_reference: [
                        0.01 * t,
                        0.15 * (t * 0.03).cos(),
                        -0.06 + 0.02 * (t * 0.05).sin(),
                    ],
                }
            })
            .collect()
    }
}

fn queries() -> [(&'static str, TorsorQuery); 2] {
    [
        (
            "near_twist",
            TorsorQuery {
                linear: [0.6, -0.1, 0.4],
                angular: [0.03, -0.01, 0.02],
                evaluation_point: [0.0, 0.0, 0.0],
            },
        ),
        (
            "distant_linear",
            TorsorQuery {
                linear: [1.0, 0.0, 0.0],
                angular: [0.0; 3],
                evaluation_point: [1000.0, -100.0, 50.0],
            },
        ),
    ]
}

fn convert(v: [f32; 3]) -> Vec3 {
    [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Clone, Copy)]
struct DenseRow {
    resultant: Vec3,
    moment_at_query: Vec3,
    lse: f64,
}

/// Direct per-key Varignon calculation, without octree or stored envelopes.
fn independent_dense(keys: &[TorsorKey], query: TorsorQuery) -> Result<DenseRow, String> {
    let point = convert(query.evaluation_point);
    let mut scored = Vec::with_capacity(keys.len());
    for key in keys {
        let r = convert(key.resultant);
        let p = convert(key.reference);
        let base = convert(key.moment_at_reference);
        let offset = [p[0] - point[0], p[1] - point[1], p[2] - point[2]];
        let arm = cross(offset, r);
        let m = [base[0] + arm[0], base[1] + arm[1], base[2] + arm[2]];
        let score = SCALE * (dot(convert(query.linear), r) + dot(convert(query.angular), m));
        if !score.is_finite() {
            return Err("non-finite independent oracle score".into());
        }
        scored.push((score, r, m));
    }
    let max_score = scored
        .iter()
        .map(|(score, _, _)| *score)
        .fold(f64::NEG_INFINITY, f64::max);
    let normalizer: f64 = scored.iter().map(|(s, _, _)| (s - max_score).exp()).sum();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err("invalid independent oracle normalizer".into());
    }
    let mut r = [0.0; 3];
    let mut m = [0.0; 3];
    for (score, kr, km) in scored {
        let weight = (score - max_score).exp() / normalizer;
        for axis in 0..3 {
            r[axis] += weight * kr[axis];
            m[axis] += weight * km[axis];
        }
    }
    Ok(DenseRow {
        resultant: r,
        moment_at_query: m,
        lse: max_score + normalizer.ln(),
    })
}

#[derive(Clone, Copy)]
enum Mode {
    Exact,
    SpatialOnly,
    ScoreGuard,
    SixOutputBudget,
    SubtreeBudget,
}

impl Mode {
    const ALL: [Self; 4] = [
        Self::Exact,
        Self::SpatialOnly,
        Self::ScoreGuard,
        Self::SixOutputBudget,
    ];
    const COMPARE_LOCAL: [Self; 5] = [
        Self::Exact,
        Self::SpatialOnly,
        Self::ScoreGuard,
        Self::SixOutputBudget,
        Self::SubtreeBudget,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact_tree",
            Self::SpatialOnly => "spatial_only",
            Self::ScoreGuard => "score_guard",
            Self::SixOutputBudget => "six_output_budget",
            Self::SubtreeBudget => "subtree_local_budget",
        }
    }
}

fn evaluate(
    tree: &PreparedTorsorOctree,
    query: TorsorQuery,
    mode: Mode,
) -> Result<(TorsorTreeOutput, Option<TorsorAnalyticalErrorBounds>), String> {
    let (attention, bound) = match mode {
        Mode::Exact => (
            tree.forward(query, SCALE, 0.0)
                .map_err(|error| format!("{error:?}"))?,
            None,
        ),
        Mode::SpatialOnly => (
            tree.forward(query, SCALE, THETA)
                .map_err(|error| format!("{error:?}"))?,
            None,
        ),
        Mode::ScoreGuard => (
            tree.forward_with_score_spread_guard(query, SCALE, THETA, MAX_SCORE_SPREAD)
                .map_err(|error| format!("{error:?}"))?,
            None,
        ),
        Mode::SixOutputBudget => {
            let output = tree
                .forward_with_output_budgets(query, SCALE, THETA, MAX_SCORE_SPREAD, BUDGET)
                .map_err(|error| format!("{error:?}"))?;
            (output.attention, Some(output.analytical_bounds))
        }
        Mode::SubtreeBudget => {
            let output = tree
                .forward_with_local_output_budgets(query, SCALE, THETA, MAX_SCORE_SPREAD, BUDGET)
                .map_err(|error| format!("{error:?}"))?;
            (output.attention, Some(output.analytical_bounds))
        }
    };
    Ok((attention, bound))
}

/// One value and median of multiple timings, to avoid claiming a speedup
/// from a single noisy host-only invocation.
fn median_timing<T, F>(samples: usize, mut f: F) -> Result<(T, u128), String>
where
    F: FnMut() -> Result<T, String>,
{
    let mut times = Vec::with_capacity(samples);
    let mut last = None;
    for _ in 0..samples {
        let start = Instant::now();
        let result = f()?;
        let nanos = start.elapsed().as_nanos();
        black_box(&result);
        times.push(nanos);
        last = Some(result);
    }
    times.sort_unstable();
    Ok((last.expect("samples validated nonzero"), times[samples / 2]))
}

fn numerical_slack(left: f64, right: f64) -> f64 {
    1.0e-9 * (1.0 + left.abs() + right.abs())
}

fn maximum(a: Vec3) -> f64 {
    a.into_iter().fold(0.0_f64, f64::max)
}

fn check(
    exact: DenseRow,
    approximation: TorsorTreeOutput,
    bounds: Option<TorsorAnalyticalErrorBounds>,
    mode: Mode,
    count: usize,
) -> Result<(Vec3, Vec3, f64), String> {
    if approximation.covered_keys != count
        || approximation.exact_keys + approximation.summarized_nodes > count
    {
        return Err("octree does not cover every key exactly once".into());
    }
    let mut r_delta = [0.0; 3];
    let mut m_delta = [0.0; 3];
    for axis in 0..3 {
        r_delta[axis] = (exact.resultant[axis] - approximation.resultant[axis]).abs();
        m_delta[axis] = (exact.moment_at_query[axis] - approximation.moment_at_query[axis]).abs();
        if let Some(b) = bounds {
            if r_delta[axis]
                > b.resultant_abs[axis]
                    + numerical_slack(exact.resultant[axis], approximation.resultant[axis])
                || m_delta[axis]
                    > b.moment_at_query_abs[axis]
                        + numerical_slack(
                            exact.moment_at_query[axis],
                            approximation.moment_at_query[axis],
                        )
                || b.resultant_abs[axis] > BUDGET.resultant_abs[axis]
                || b.moment_at_query_abs[axis] > BUDGET.moment_at_query_abs[axis]
            {
                return Err(format!(
                    "six-output analytical budget exceeded on axis {axis}"
                ));
            }
        }
        if matches!(mode, Mode::Exact)
            && (r_delta[axis]
                > numerical_slack(exact.resultant[axis], approximation.resultant[axis])
                || m_delta[axis]
                    > numerical_slack(
                        exact.moment_at_query[axis],
                        approximation.moment_at_query[axis],
                    ))
        {
            return Err(format!(
                "exact tree differs from independent dense on axis {axis}"
            ));
        }
    }
    let lse_delta = (exact.lse - approximation.log_sum_exp).abs();
    if let Some(b) = bounds {
        if lse_delta > b.log_sum_exp_abs + numerical_slack(exact.lse, approximation.log_sum_exp) {
            return Err("six-output analytical log-sum-exp bound exceeded".into());
        }
    }
    if matches!(mode, Mode::ScoreGuard)
        && lse_delta > MAX_SCORE_SPREAD + numerical_slack(exact.lse, approximation.log_sum_exp)
    {
        return Err("score-spread guard log-sum-exp budget exceeded".into());
    }
    if matches!(mode, Mode::Exact)
        && lse_delta > numerical_slack(exact.lse, approximation.log_sum_exp)
    {
        return Err("exact tree log-sum-exp differs from independent dense".into());
    }
    Ok((r_delta, m_delta, lse_delta))
}

fn parse_args() -> Result<(usize, usize, bool), String> {
    let mut count = 256;
    let mut samples = 5;
    let mut compare_local = false;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--compare-local" {
            if compare_local {
                return Err("duplicate --compare-local".into());
            }
            compare_local = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        let parsed: usize = value
            .parse()
            .map_err(|_| format!("invalid integer for {flag}: {value}"))?;
        match flag.as_str() {
            "--keys" => count = parsed,
            "--samples" => samples = parsed,
            _ => return Err(format!("unsupported flag: {flag}")),
        }
    }
    if !(16..=MAX_OCTREE_KEYS).contains(&count) {
        return Err(format!("--keys must be between 16 and {MAX_OCTREE_KEYS}"));
    }
    if !(1..=25).contains(&samples) {
        return Err("--samples must be between 1 and 25".into());
    }
    Ok((count, samples, compare_local))
}

fn run(count: usize, samples: usize, compare_local: bool) -> Result<(), String> {
    let contract = if compare_local {
        TDI22_TORSOR_OCTREE_LOCAL_BUDGET_V2
    } else {
        TDI22_TORSOR_OCTREE_OUTPUT_BUDGET_V1
    };
    eprintln!(
        "research_only=true;contract={contract};host_only=true;real_device_qualified=false;paired_mode={compare_local}"
    );
    println!(
        "panel,query,mode,keys,nodes,reserved_buffer_bytes,build_ns,dense_median_ns,mode_median_ns,exact_keys,summarized_nodes,summarized_keys,max_resultant_error,max_moment_error,lse_error,bounds_present,max_resultant_bound,max_moment_bound,lse_bound"
    );
    let panels: &[Panel] = if compare_local {
        &Panel::COMPARE_LOCAL
    } else {
        &Panel::ALL
    };
    let modes: &[Mode] = if compare_local {
        &Mode::COMPARE_LOCAL
    } else {
        &Mode::ALL
    };
    for &panel in panels {
        let keys = panel.keys(count);
        let started = Instant::now();
        let tree =
            PreparedTorsorOctree::build(&keys, 4, 20).map_err(|error| format!("{error:?}"))?;
        let build_ns = started.elapsed().as_nanos();
        if tree.reserved_buffer_bytes() < count * std::mem::size_of::<TorsorKey>() {
            return Err("tree reserved capacity is smaller than key storage".into());
        }
        for (query_name, query) in queries() {
            // Align the distant cluster fixture with the R-axis so its
            // moment does not inadvertently dominate the six-output budget.
            let query =
                if matches!(panel, Panel::SeparatedClusters) && query_name == "distant_linear" {
                    TorsorQuery {
                        evaluation_point: [1000.0, 0.0, 0.0],
                        ..query
                    }
                } else {
                    query
                };
            let (reference, dense_ns) = median_timing(samples, || independent_dense(&keys, query))?;
            for &mode in modes {
                let ((out, bound), mode_ns) =
                    median_timing(samples, || evaluate(&tree, query, mode))?;
                let (r, m, lse) = check(reference, out, bound, mode, count).map_err(|error| {
                    format!(
                        "panel={} query={query_name} mode={}: {error}",
                        panel.name(),
                        mode.name()
                    )
                })?;
                let (r_bound, m_bound, lse_bound) = bound
                    .map(|b| {
                        (
                            maximum(b.resultant_abs),
                            maximum(b.moment_at_query_abs),
                            b.log_sum_exp_abs,
                        )
                    })
                    .unwrap_or((0.0, 0.0, 0.0));
                println!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{:.12e},{:.12e},{:.12e},{},{:.12e},{:.12e},{:.12e}",
                    panel.name(),
                    query_name,
                    mode.name(),
                    count,
                    tree.node_count(),
                    tree.reserved_buffer_bytes(),
                    build_ns,
                    dense_ns,
                    mode_ns,
                    out.exact_keys,
                    out.summarized_nodes,
                    out.covered_keys - out.exact_keys,
                    maximum(r),
                    maximum(m),
                    lse,
                    bound.is_some(),
                    r_bound,
                    m_bound,
                    lse_bound
                );
            }
        }
    }
    Ok(())
}

fn main() {
    let result =
        parse_args().and_then(|(count, samples, compare_local)| run(count, samples, compare_local));
    if let Err(error) = result {
        eprintln!("tdi22 octree qualification failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_panels_and_independent_oracle_hold_declared_budgets() {
        run(64, 1, false).unwrap();
    }

    #[test]
    fn opt_in_subtree_panel_adds_50_oracle_checked_rows_without_changing_v1() {
        // The frozen default still prints 32 data rows; compare-local
        // explicitly adds the fifth panel and fifth mode (5 * 2 * 5 = 50).
        run(64, 1, false).unwrap();
        run(64, 1, true).unwrap();
    }

    #[test]
    fn separated_clusters_are_more_selective_under_local_budgets() {
        for count in [64, 256, 1024] {
            let keys = Panel::SeparatedClusters.keys(count);
            let tree = PreparedTorsorOctree::build(&keys, 4, 20).unwrap();
            let q = TorsorQuery {
                linear: [1.0, 0.0, 0.0],
                angular: [0.0; 3],
                evaluation_point: [1000.0, 0.0, 0.0],
            };
            let v1 = tree
                .forward_with_output_budgets(q, SCALE, THETA, MAX_SCORE_SPREAD, BUDGET)
                .unwrap();
            let v2 = tree
                .forward_with_local_output_budgets(q, SCALE, THETA, MAX_SCORE_SPREAD, BUDGET)
                .unwrap();
            assert_eq!(v1.attention.covered_keys, count);
            assert_eq!(v2.attention.covered_keys, count);
            assert_eq!(v2.attention.exact_keys, 0);
            assert_eq!(v2.attention.summarized_nodes, 2);
            // On 64 keys V1 already accepts both groups; larger groups
            // expose V1's root-range over-conservatism.
            if count >= 256 {
                assert!(
                    v1.attention.exact_keys > v2.attention.exact_keys,
                    "expected strictly fewer exact key evaluations for {count} keys"
                );
            }
            let oracle = independent_dense(&keys, q).unwrap();
            check(
                oracle,
                v1.attention,
                Some(v1.analytical_bounds),
                Mode::SixOutputBudget,
                count,
            )
            .unwrap();
            check(
                oracle,
                v2.attention,
                Some(v2.analytical_bounds),
                Mode::SubtreeBudget,
                count,
            )
            .unwrap();
        }
    }

    #[test]
    fn far_uniform_projected_scores_are_compressible_at_zero_spread() {
        let keys = Panel::UniformProjectedScore.keys(64);
        let tree = PreparedTorsorOctree::build(&keys, 1, 20).unwrap();
        let (_, query) = queries()[1];
        let output = tree
            .forward_with_output_budgets(
                query,
                SCALE,
                THETA,
                0.0,
                TorsorOutputErrorBudget {
                    resultant_abs: [0.0; 3],
                    moment_at_query_abs: [0.0; 3],
                },
            )
            .unwrap();
        assert_eq!(output.attention.summarized_nodes, 1);
        assert_eq!(output.attention.exact_keys, 0);
        let dense = independent_dense(&keys, query).unwrap();
        for axis in 0..3 {
            assert!(
                (output.attention.resultant[axis] - dense.resultant[axis]).abs()
                    <= numerical_slack(output.attention.resultant[axis], dense.resultant[axis])
            );
            assert!(
                (output.attention.moment_at_query[axis] - dense.moment_at_query[axis]).abs()
                    <= numerical_slack(
                        output.attention.moment_at_query[axis],
                        dense.moment_at_query[axis],
                    )
            );
        }
    }

    #[test]
    fn distant_adversarial_logits_reject_root_summary_under_tight_budget() {
        let keys = Panel::FarAdversarial.keys(64);
        let tree = PreparedTorsorOctree::build(&keys, 1, 20).unwrap();
        let (_, query) = queries()[1];
        let spatial = tree.forward(query, SCALE, THETA).unwrap();
        assert_eq!(spatial.summarized_nodes, 1);
        let bounded = tree
            .forward_with_output_budgets(
                query,
                SCALE,
                THETA,
                0.0,
                TorsorOutputErrorBudget {
                    resultant_abs: [0.0; 3],
                    moment_at_query_abs: [0.0; 3],
                },
            )
            .unwrap();
        // The root cannot collapse, but zero-span *children* may still
        // collapse safely; do not conflate exact coverage with exact leaves.
        assert!(bounded.attention.exact_keys > 0);
        assert_eq!(bounded.attention.covered_keys, keys.len());
    }

    #[test]
    fn coincident_positions_cannot_compress_without_subdivision() {
        let keys = Panel::CoincidentCoordinates.keys(64);
        let tree = PreparedTorsorOctree::build(&keys, 1, 20).unwrap();
        assert_eq!(tree.node_count(), 1);
        let (_, query) = queries()[1];
        let output = tree
            .forward_with_output_budgets(query, SCALE, THETA, MAX_SCORE_SPREAD, BUDGET)
            .unwrap();
        assert_eq!(output.attention.exact_keys, keys.len());
        assert_eq!(output.attention.summarized_nodes, 0);
    }
}
