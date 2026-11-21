//! Q4-TOPO-4: frozen matched signed-cycle panel with no threshold tuning.
//!
//! This deterministic finite panel tests whether signed-cycle frustration
//! covaries with exact communication heights and post-perturbation Q4 recall.
//! Six balanced and four frustrated sign assignments share the SAME graph,
//! degrees, coupling magnitudes and exactly two negative edges. The only
//! independent intervention is which two of five edges carry negative signs.
//!
//! There is a single square cycle and one pendant edge. A sign assignment is
//! frustrated iff the square has odd negative parity; the pendant edge can
//! shift negative count without changing cycle parity. This is a synthetic
//! topology control, not real connectome evidence, trained memory, or speedup.
//! The four-state Q4 barrier is compared with a 2-state always-active binary
//! barrier only if both measured endpoints are fully active.

use core::fmt;

use crate::ising_binary_barrier_control::{
    exact_active_binary_barrier, BinaryBarrierError, ExactBinaryBarrierReport,
};
use crate::ising_energy_barriers::{
    exact_quaternary_barrier, single_bit_perturbation_recall, ExactBarrierReport, Q4Landscape,
    Q4LandscapeError, Q4RecoveryPanel,
};
use crate::ising_quaternary::{BooleanQuaternary, Spin4, Spin4Error};
use crate::ising_sparse_state::SparseIsingEdge;
use crate::ising_topological_defects::{analyze_topological_defects, TopologicalDefectError};

pub const MATCHED_PANEL_SCHEMA: &str = "Q4-TOPO-4-v1";
pub const MATCHED_PANEL_VERTICES: usize = 5;
pub const MATCHED_PANEL_EDGES: usize = 5;
pub const MATCHED_PANEL_NEGATIVE_EDGES: u32 = 2;
pub const MATCHED_PANEL_COUPLING_ABS: i16 = 2;
pub const MATCHED_PANEL_SWEEP_BUDGET: usize = 64;
pub const MATCHED_PANEL_COSTS: [i32; 3] = [-1, 0, 2];

/// Edges: 0-1, 0-3, 1-2, 2-3 (square), and 3-4 (pendant bridge).
/// Edge index 3 is the sole non-tree fundamental-cycle closure under BFS.
const LINKS: [(usize, usize); MATCHED_PANEL_EDGES] = [(0, 1), (0, 3), (1, 2), (2, 3), (3, 4)];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SignedCycleFamily {
    Balanced,
    Frustrated,
}

impl SignedCycleFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Frustrated => "frustrated",
        }
    }
}

#[derive(Debug)]
pub enum MatchedPanelError {
    Spin(Spin4Error),
    Topology(TopologicalDefectError),
    Landscape(Q4LandscapeError),
    Binary(BinaryBarrierError),
    NotConverged {
        negative_edge_mask: u8,
        activation_cost: i32,
        initial_root_positive: bool,
    },
}

impl fmt::Display for MatchedPanelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MatchedPanelError {}

impl From<Spin4Error> for MatchedPanelError {
    fn from(value: Spin4Error) -> Self {
        Self::Spin(value)
    }
}

impl From<TopologicalDefectError> for MatchedPanelError {
    fn from(value: TopologicalDefectError) -> Self {
        Self::Topology(value)
    }
}

impl From<Q4LandscapeError> for MatchedPanelError {
    fn from(value: Q4LandscapeError) -> Self {
        Self::Landscape(value)
    }
}

impl From<BinaryBarrierError> for MatchedPanelError {
    fn from(value: BinaryBarrierError) -> Self {
        Self::Binary(value)
    }
}

/// One signed graph at one fixed activation penalty. Zero biases for all runs.
/// Both baselines start with all vertices active and tree-aligned spin signs,
/// with the root sign flipped as a second preregistered initialization.
/// A strict-descent fixed point is found before any recovery measurement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedCycleCase {
    pub negative_edge_mask: u8,
    pub activation_cost: i32,
    pub family: SignedCycleFamily,
    pub cycle_rank: usize,
    pub negative_fundamental_cycles: usize,
    pub positive_root_attractor: Vec<Spin4>,
    pub negative_root_attractor: Vec<Spin4>,
    pub positive_root_settle_sweeps: usize,
    pub negative_root_settle_sweeps: usize,
    pub exact_q4: ExactBarrierReport,
    /// Not defined if either strict-descent fixed point contains dormant sites.
    pub exact_binary: Option<ExactBinaryBarrierReport>,
    pub positive_root_recall: Q4RecoveryPanel,
    pub negative_root_recall: Q4RecoveryPanel,
}

/// All aggregations retain counts and denominators. These small, exhaustive
/// synthetic families are not independent statistical samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedCycleAggregate {
    pub family: SignedCycleFamily,
    pub activation_cost: i32,
    pub graph_count: usize,
    pub perturbation_trials: usize,
    pub exact_bit_recalls: usize,
    pub effective_spin_recalls: usize,
    pub fixed_point_trials: usize,
    pub exact_q4_forward_barrier_sum: i128,
    pub exact_q4_backward_barrier_sum: i128,
    pub binary_comparable_graphs: usize,
    pub exact_binary_forward_barrier_sum: i128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedCyclePanel {
    pub schema: &'static str,
    pub cases: Vec<MatchedCycleCase>,
    pub aggregates: Vec<MatchedCycleAggregate>,
}

fn signed_edges(negative_edge_mask: u8) -> [SparseIsingEdge; MATCHED_PANEL_EDGES] {
    core::array::from_fn(|index| {
        let (left, right) = LINKS[index];
        let coupling = if negative_edge_mask & (1_u8 << index) != 0 {
            -MATCHED_PANEL_COUPLING_ABS
        } else {
            MATCHED_PANEL_COUPLING_ABS
        };
        SparseIsingEdge {
            left,
            right,
            coupling,
        }
    })
}

/// The reference initialization satisfies all four BFS-tree edges. It
/// transports the spin convention covariantly with sign switches on the tree;
/// only the square-cycle chord can remain unsatisfied at initialization.
/// This avoids using an all-positive initialization which would not be
/// comparable across different gauge/sign arrangements.
fn tree_aligned_seed(negative_edge_mask: u8, root_positive: bool) -> [Spin4; 5] {
    let is_negative = |edge: usize| negative_edge_mask & (1_u8 << edge) != 0;
    let mut signs = [false; 5];
    signs[0] = root_positive;
    signs[1] = signs[0] ^ is_negative(0);
    signs[3] = signs[0] ^ is_negative(1);
    signs[2] = signs[1] ^ is_negative(2);
    signs[4] = signs[3] ^ is_negative(4);
    signs.map(|positive| {
        if positive {
            Spin4::ActivePositive
        } else {
            Spin4::ActiveNegative
        }
    })
}

fn settle(
    problem: Q4Landscape<'_>,
    starting: &[Spin4],
    negative_edge_mask: u8,
    activation_cost: i32,
    initial_root_positive: bool,
) -> Result<(Vec<Spin4>, usize), MatchedPanelError> {
    let mut activity = 0_u64;
    let mut polarity = 0_u64;
    for (vertex, state) in starting.iter().enumerate() {
        activity |= u64::from(state.active()) << vertex;
        polarity |= u64::from(state.positive()) << vertex;
    }
    let mut controller = BooleanQuaternary::new(
        problem.biases,
        problem.activation_costs,
        problem.eligibility,
        &[activity],
        &[polarity],
        problem.edges,
    )?;
    for sweep in 1..=MATCHED_PANEL_SWEEP_BUDGET {
        if controller.sweep() == 0 {
            let states = (0..MATCHED_PANEL_VERTICES)
                .map(|i| controller.state(i).map(|s| s.expect("all sites eligible")))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok((states, sweep));
        }
    }
    Err(MatchedPanelError::NotConverged {
        negative_edge_mask,
        activation_cost,
        initial_root_positive,
    })
}

/// Exhaustively enumerate all 10 ways to place 2 negative signs on 5 fixed
/// edges, then cross each signed graph with three fixed Q4 activity costs.
///
/// There are 6 balanced and 4 frustrated graphs for each cost, i.e. 30 cases.
/// Positive/negative tree-root initializations are paired and no sign placement
/// is selected or discarded after observing the energetic or recovery results.
pub fn run_matched_signed_cycle_panel() -> Result<MatchedCyclePanel, MatchedPanelError> {
    let biases = [0_i32; MATCHED_PANEL_VERTICES];
    let eligibility = [0b1_1111_u64];
    let mut cases = Vec::with_capacity(30);
    for negative_edge_mask in 0_u8..(1_u8 << MATCHED_PANEL_EDGES) {
        if negative_edge_mask.count_ones() != MATCHED_PANEL_NEGATIVE_EDGES {
            continue;
        }
        let edges = signed_edges(negative_edge_mask);
        let topology = analyze_topological_defects(
            MATCHED_PANEL_VERTICES,
            &eligibility,
            &eligibility,
            &eligibility,
            &edges,
        )?;
        let family = if topology.structurally_balanced() {
            SignedCycleFamily::Balanced
        } else {
            SignedCycleFamily::Frustrated
        };
        for activation_cost in MATCHED_PANEL_COSTS {
            let costs = [activation_cost; MATCHED_PANEL_VERTICES];
            let problem = Q4Landscape {
                biases: &biases,
                activation_costs: &costs,
                eligibility: &eligibility,
                edges: &edges,
            };
            let (positive_root_attractor, positive_root_settle_sweeps) = settle(
                problem,
                &tree_aligned_seed(negative_edge_mask, true),
                negative_edge_mask,
                activation_cost,
                true,
            )?;
            let (negative_root_attractor, negative_root_settle_sweeps) = settle(
                problem,
                &tree_aligned_seed(negative_edge_mask, false),
                negative_edge_mask,
                activation_cost,
                false,
            )?;
            let exact_q4 = exact_quaternary_barrier(
                problem,
                &positive_root_attractor,
                &negative_root_attractor,
            )?;
            let exact_binary = if positive_root_attractor.iter().all(|s| s.active())
                && negative_root_attractor.iter().all(|s| s.active())
            {
                Some(exact_active_binary_barrier(
                    problem,
                    &positive_root_attractor,
                    &negative_root_attractor,
                )?)
            } else {
                None
            };
            let positive_root_recall = single_bit_perturbation_recall(
                problem,
                &positive_root_attractor,
                MATCHED_PANEL_SWEEP_BUDGET,
            )?;
            let negative_root_recall = single_bit_perturbation_recall(
                problem,
                &negative_root_attractor,
                MATCHED_PANEL_SWEEP_BUDGET,
            )?;
            cases.push(MatchedCycleCase {
                negative_edge_mask,
                activation_cost,
                family,
                cycle_rank: topology.cycle_rank,
                negative_fundamental_cycles: topology.negative_fundamental_cycles,
                positive_root_attractor,
                negative_root_attractor,
                positive_root_settle_sweeps,
                negative_root_settle_sweeps,
                exact_q4,
                exact_binary,
                positive_root_recall,
                negative_root_recall,
            });
        }
    }
    let aggregates = aggregate_matched_cycle_cases(&cases);
    Ok(MatchedCyclePanel {
        schema: MATCHED_PANEL_SCHEMA,
        cases,
        aggregates,
    })
}

/// Aggregate counts per activation cost and structural-balance family.
///
/// No normalization, p-value, causal interpretation, holdout claim or hardware
/// measurement is calculated by this accounting helper.
pub fn aggregate_matched_cycle_cases(cases: &[MatchedCycleCase]) -> Vec<MatchedCycleAggregate> {
    let mut groups = Vec::with_capacity(6);
    for activation_cost in MATCHED_PANEL_COSTS {
        for family in [SignedCycleFamily::Balanced, SignedCycleFamily::Frustrated] {
            let mut record = MatchedCycleAggregate {
                family,
                activation_cost,
                graph_count: 0,
                perturbation_trials: 0,
                exact_bit_recalls: 0,
                effective_spin_recalls: 0,
                fixed_point_trials: 0,
                exact_q4_forward_barrier_sum: 0,
                exact_q4_backward_barrier_sum: 0,
                binary_comparable_graphs: 0,
                exact_binary_forward_barrier_sum: 0,
            };
            for case in cases
                .iter()
                .filter(|case| case.activation_cost == activation_cost && case.family == family)
            {
                record.graph_count += 1;
                record.exact_q4_forward_barrier_sum += case.exact_q4.barrier_above_initial;
                record.exact_q4_backward_barrier_sum +=
                    case.exact_q4.communication_height - case.exact_q4.target_energy;
                if let Some(binary) = &case.exact_binary {
                    record.binary_comparable_graphs += 1;
                    record.exact_binary_forward_barrier_sum += binary.barrier_above_initial;
                }
                for recall in [&case.positive_root_recall, &case.negative_root_recall] {
                    record.perturbation_trials += recall.trials.len();
                    record.exact_bit_recalls += recall.exact_recall_count;
                    record.effective_spin_recalls += recall.effective_state_recall_count;
                    record.fixed_point_trials += recall.fixed_point_count;
                }
            }
            groups.push(record);
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_is_degree_magnitude_and_negative_count_matched() {
        let panel = run_matched_signed_cycle_panel().unwrap();
        assert_eq!(panel.schema, MATCHED_PANEL_SCHEMA);
        assert_eq!(panel.cases.len(), 30);
        assert_eq!(panel.aggregates.len(), 6);
        for cost in MATCHED_PANEL_COSTS {
            let matched: Vec<_> = panel
                .cases
                .iter()
                .filter(|r| r.activation_cost == cost)
                .collect();
            assert_eq!(matched.len(), 10);
            assert_eq!(
                matched
                    .iter()
                    .filter(|r| r.family == SignedCycleFamily::Balanced)
                    .count(),
                6
            );
            assert_eq!(
                matched
                    .iter()
                    .filter(|r| r.family == SignedCycleFamily::Frustrated)
                    .count(),
                4
            );
        }
        for row in &panel.cases {
            assert_eq!(row.negative_edge_mask.count_ones(), 2);
            assert_eq!(row.cycle_rank, 1);
            assert_eq!(
                row.negative_fundamental_cycles,
                usize::from(row.family == SignedCycleFamily::Frustrated),
            );
            assert_eq!(row.positive_root_recall.trials.len(), 10);
            assert_eq!(row.negative_root_recall.trials.len(), 10);
            assert_eq!(row.exact_q4.enumerated_states, 1024);
            assert_eq!(row.exact_q4.eligible_vertices, 5);
            assert!(row.positive_root_settle_sweeps <= MATCHED_PANEL_SWEEP_BUDGET);
            assert!(row.negative_root_settle_sweeps <= MATCHED_PANEL_SWEEP_BUDGET);
            if let Some(binary) = &row.exact_binary {
                assert_eq!(binary.enumerated_states, 32);
                assert_eq!(binary.initial_energy, row.exact_q4.initial_energy);
                assert_eq!(binary.target_energy, row.exact_q4.target_energy);
                // Binary paths are a proper subset of the allowed Q4 paths.
                assert!(row.exact_q4.communication_height <= binary.communication_height);
            }
        }
        for summary in &panel.aggregates {
            assert_eq!(
                summary.graph_count,
                if summary.family == SignedCycleFamily::Balanced {
                    6
                } else {
                    4
                },
            );
            assert_eq!(summary.perturbation_trials, summary.graph_count * 20);
            assert!(summary.exact_bit_recalls <= summary.effective_spin_recalls);
            assert!(summary.effective_spin_recalls <= summary.perturbation_trials);
            assert!(summary.fixed_point_trials <= summary.perturbation_trials);
        }
    }

    #[test]
    fn rerunning_frozen_panel_yields_identical_reports() {
        assert_eq!(
            run_matched_signed_cycle_panel().unwrap(),
            run_matched_signed_cycle_panel().unwrap(),
        );
    }

    #[test]
    fn frozen_reference_aggregates_match_independent_scalar_recomputation() {
        // Golden values from an independent, explicit scalar enumeration of
        // the fixed 5-vertex signed panel. These are regression fixtures,
        // NOT training targets, tuned thresholds, or statistical inference.
        let expected = [
            (-1, SignedCycleFamily::Balanced, 6, 120, 120, 48, 6, 48),
            (-1, SignedCycleFamily::Frustrated, 4, 64, 64, 12, 4, 16),
            (0, SignedCycleFamily::Balanced, 6, 120, 120, 48, 6, 48),
            (0, SignedCycleFamily::Frustrated, 4, 56, 56, 8, 4, 16),
            (2, SignedCycleFamily::Balanced, 6, 60, 60, 12, 6, 48),
            (2, SignedCycleFamily::Frustrated, 4, 32, 40, 0, 0, 0),
        ];
        let panel = run_matched_signed_cycle_panel().unwrap();
        for (cost, family, graphs, exact, effective, q4_sum, binary_graphs, binary_sum) in expected {
            let actual = panel
                .aggregates
                .iter()
                .find(|r| r.activation_cost == cost && r.family == family)
                .unwrap();
            assert_eq!(actual.graph_count, graphs);
            assert_eq!(actual.perturbation_trials, graphs * 20);
            assert_eq!(actual.exact_bit_recalls, exact);
            assert_eq!(actual.effective_spin_recalls, effective);
            assert_eq!(actual.fixed_point_trials, graphs * 20);
            assert_eq!(actual.exact_q4_forward_barrier_sum, q4_sum);
            assert_eq!(actual.exact_q4_backward_barrier_sum, q4_sum);
            assert_eq!(actual.binary_comparable_graphs, binary_graphs);
            assert_eq!(actual.exact_binary_forward_barrier_sum, binary_sum);
        }
    }

    #[test]
    fn rooted_tree_signs_are_gauge_covariant() {
        // A switch at vertex 1 toggles both incident edge signs.
        // The resulting tree-aligned spin at vertex 1 switches, but other
        // vertex conventions are unchanged.
        let original = tree_aligned_seed(0b00011, true);
        let switched = tree_aligned_seed(0b00110, true);
        // The first edge toggles from negative to positive, and edge 2
        // toggles from positive to negative: both are incident to vertex 1.
        assert_eq!(original[0], switched[0]);
        assert_ne!(original[1], switched[1]);
        for i in [2, 3, 4] {
            assert_eq!(original[i], switched[i]);
        }
    }
}
