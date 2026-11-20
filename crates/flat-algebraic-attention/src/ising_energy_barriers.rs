//! Q4-TOPO-3: bounded exact energy barriers and deterministic perturbation recall.
//!
//! The communication height between two Q4 configurations is the minimum,
//! across all paths changing ONE ELIGIBLE VERTEX to any other Q4 state per step,
//! of the maximum intermediate Ising energy. The height is calculated by an
//! exact bottleneck shortest-path search, with an explicit 8-vertex ceiling.
//! This transition graph matches the local alternatives of update_one() but
//! not its strictly downhill acceptance criterion: barrier paths may climb.
//!
//! Dormant negative and dormant positive have identical instantaneous energy
//! but distinct memory bits; both states are kept in the exhaustive graph.
//! Neither a positive communication barrier nor perfect deterministic recall
//! establishes stochastic retention times or topologically protected memory.

use core::cmp::{max, Reverse};
use core::collections::BinaryHeap;
use core::fmt;

use crate::ising_quaternary::{BooleanQuaternary, Spin4, Spin4Error};
use crate::ising_sparse_state::SparseIsingEdge;

/// Hard ceiling for exhaustive 4^N state enumeration, including masked sites.
pub const MAX_EXACT_Q4_VERTICES: usize = 8;
/// Hard ceiling on deterministic post-perturbation sweeps.
pub const MAX_RECALL_SWEEPS: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Q4LandscapeError {
    Empty,
    TooManyVertices { actual: usize, limit: usize },
    NoEligibleVertices,
    EndpointLength,
    NonneutralMaskedState { vertex: usize },
    InvalidSweepBudget,
    NotLocalAttractor,
    Spin(Spin4Error),
}

impl From<Spin4Error> for Q4LandscapeError {
    fn from(value: Spin4Error) -> Self {
        Self::Spin(value)
    }
}

impl fmt::Display for Q4LandscapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Q4LandscapeError {}

/// Frozen signed graph and local Q4 costs. All couplings must be in canonical
/// strictly increasing (left, right) order, with left < right.
#[derive(Debug, Clone, Copy)]
pub struct Q4Landscape<'a> {
    pub biases: &'a [i32],
    pub activation_costs: &'a [i32],
    pub eligibility: &'a [u64],
    pub edges: &'a [SparseIsingEdge],
}

/// A witness path realizes the exact minimax height under single-site Q4
/// changes. Every intermediate state is preserved, including dormant polarity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactBarrierReport {
    pub eligible_vertices: usize,
    pub explored_states: usize,
    pub enumerated_states: usize,
    pub initial_energy: i128,
    pub target_energy: i128,
    pub communication_height: i128,
    pub barrier_above_initial: i128,
    pub witness: Vec<Vec<Spin4>>,
}

fn bit(bits: &[u64], i: usize) -> bool {
    bits[i / 64] & (1_u64 << (i % 64)) != 0
}

fn validate(problem: Q4Landscape<'_>) -> Result<Vec<usize>, Q4LandscapeError> {
    let n = problem.biases.len();
    if n == 0 {
        return Err(Q4LandscapeError::Empty);
    }
    if n > MAX_EXACT_Q4_VERTICES {
        return Err(Q4LandscapeError::TooManyVertices {
            actual: n,
            limit: MAX_EXACT_Q4_VERTICES,
        });
    }
    // The existing model checks bitplane padding, shape and canonical edges.
    let zeros = [0_u64];
    BooleanQuaternary::new(
        problem.biases,
        problem.activation_costs,
        problem.eligibility,
        &zeros,
        &zeros,
        problem.edges,
    )?;
    let positions: Vec<usize> = (0..n)
        .filter(|&vertex| bit(problem.eligibility, vertex))
        .collect();
    if positions.is_empty() {
        return Err(Q4LandscapeError::NoEligibleVertices);
    }
    Ok(positions)
}

fn check_state(
    problem: Q4Landscape<'_>,
    states: &[Spin4],
) -> Result<(), Q4LandscapeError> {
    if states.len() != problem.biases.len() {
        return Err(Q4LandscapeError::EndpointLength);
    }
    for (vertex, &state) in states.iter().enumerate() {
        if !bit(problem.eligibility, vertex) && state != Spin4::DormantNegative {
            return Err(Q4LandscapeError::NonneutralMaskedState { vertex });
        }
    }
    Ok(())
}

fn code_from_states(states: &[Spin4], positions: &[usize]) -> usize {
    positions
        .iter()
        .enumerate()
        .fold(0usize, |word, (slot, &vertex)| {
            word | ((states[vertex] as usize) << (slot * 2))
        })
}

fn states_from_code(code: usize, positions: &[usize], n: usize) -> Vec<Spin4> {
    let mut states = vec![Spin4::DormantNegative; n];
    for (slot, &vertex) in positions.iter().enumerate() {
        states[vertex] = Spin4::ALL[(code >> (slot * 2)) & 3];
    }
    states
}

fn spin_value(state: Spin4) -> i128 {
    i128::from(state.contribution())
}

fn energy_for_code(
    problem: Q4Landscape<'_>,
    code: usize,
    positions: &[usize],
    vertex_to_slot: &[Option<usize>],
) -> i128 {
    let mut energy = 0i128;
    for (slot, &vertex) in positions.iter().enumerate() {
        let state = Spin4::ALL[(code >> (2 * slot)) & 3];
        if state.active() {
            energy += i128::from(problem.activation_costs[vertex]);
        }
        energy -= i128::from(problem.biases[vertex]) * spin_value(state);
    }
    for edge in problem.edges {
        if let (Some(left), Some(right)) =
            (vertex_to_slot[edge.left], vertex_to_slot[edge.right])
        {
            let z_left = spin_value(Spin4::ALL[(code >> (2 * left)) & 3]);
            let z_right = spin_value(Spin4::ALL[(code >> (2 * right)) & 3]);
            energy -= i128::from(edge.coupling) * z_left * z_right;
        }
    }
    energy
}

/// Exact minimax barrier for a fixed, small signed Q4 graph.
///
/// A step changes just one eligible vertex to ANY of its other three Q4 states,
/// the same legal local proposals as the existing greedy Q4 controller.
/// Edges and eligibility remain frozen along the path. The algorithm minimizes
/// the path's maximal energy, not the sum of transition costs. It retains a
/// deterministic witness using a min-heap over (height, encoded configuration).
/// Results are exact for this declared transition graph, not for all possible
/// continuous, synchronized, nonlocal, or thermally activated dynamics.
pub fn exact_quaternary_barrier(
    problem: Q4Landscape<'_>,
    initial: &[Spin4],
    target: &[Spin4],
) -> Result<ExactBarrierReport, Q4LandscapeError> {
    let positions = validate(problem)?;
    check_state(problem, initial)?;
    check_state(problem, target)?;
    let n = problem.biases.len();
    let enumerated_states = 1usize << (2 * positions.len());
    let mut vertex_to_slot = vec![None; n];
    for (slot, &vertex) in positions.iter().enumerate() {
        vertex_to_slot[vertex] = Some(slot);
    }
    let energies: Vec<i128> = (0..enumerated_states)
        .map(|code| energy_for_code(problem, code, &positions, &vertex_to_slot))
        .collect();
    let origin = code_from_states(initial, &positions);
    let destination = code_from_states(target, &positions);
    let mut heights = vec![i128::MAX; enumerated_states];
    let mut parent = vec![None; enumerated_states];
    let mut heap = BinaryHeap::new();
    heights[origin] = energies[origin];
    heap.push(Reverse((heights[origin], origin)));
    let mut explored_states = 0usize;

    while let Some(Reverse((height, code))) = heap.pop() {
        if height != heights[code] {
            continue;
        }
        explored_states += 1;
        if code == destination {
            break;
        }
        for slot in 0..positions.len() {
            let shift = slot * 2;
            let current_digit = (code >> shift) & 3;
            for digit in 0..4 {
                if digit == current_digit {
                    continue;
                }
                let neighbor = (code & !(3usize << shift)) | (digit << shift);
                let candidate_height = max(height, energies[neighbor]);
                if candidate_height < heights[neighbor] {
                    heights[neighbor] = candidate_height;
                    parent[neighbor] = Some(code);
                    heap.push(Reverse((candidate_height, neighbor)));
                }
            }
        }
    }
    // Every Q4 configuration is connected under single-vertex transitions.
    let communication_height = heights[destination];
    let mut witness_codes = vec![destination];
    while *witness_codes.last().expect("nonempty witness") != origin {
        let previous = parent[*witness_codes.last().expect("nonempty witness")]
            .expect("full Q4 graph is connected");
        witness_codes.push(previous);
    }
    witness_codes.reverse();
    let witness = witness_codes
        .into_iter()
        .map(|code| states_from_code(code, &positions, n))
        .collect();
    Ok(ExactBarrierReport {
        eligible_vertices: positions.len(),
        explored_states,
        enumerated_states,
        initial_energy: energies[origin],
        target_energy: energies[destination],
        communication_height,
        barrier_above_initial: communication_height - energies[origin],
        witness,
    })
}

/// A controlled one-bit external perturbation, distinct from Q4's local
/// optimizer which may change one vertex to any other state in one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q4Perturbation {
    ActivityBit,
    PolarityBit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q4RecoveryTrial {
    pub vertex: usize,
    pub perturbation: Q4Perturbation,
    pub energy_after_perturbation: i128,
    pub final_energy: i128,
    pub sweeps: usize,
    pub reached_fixed_point: bool,
    /// Exact equality of all eligible activity AND dormant polarity bits.
    pub exact_recall: bool,
    /// Equality of energy-contributing z values; dormant polarity is ignored.
    /// This is only an instantaneous functional-state check, not task recall.
    pub effective_state_recall: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q4RecoveryPanel {
    pub baseline_energy: i128,
    pub trials: Vec<Q4RecoveryTrial>,
    pub exact_recall_count: usize,
    pub effective_state_recall_count: usize,
    pub fixed_point_count: usize,
}

fn controller_for(
    problem: Q4Landscape<'_>,
    states: &[Spin4],
) -> Result<BooleanQuaternary, Q4LandscapeError> {
    let mut activity = [0_u64];
    let mut polarity = [0_u64];
    for (i, state) in states.iter().enumerate() {
        if state.active() {
            activity[0] |= 1_u64 << i;
        }
        if state.positive() {
            polarity[0] |= 1_u64 << i;
        }
    }
    Ok(BooleanQuaternary::new(
        problem.biases,
        problem.activation_costs,
        problem.eligibility,
        &activity,
        &polarity,
        problem.edges,
    )?)
}

/// Reconstruct a bounded deterministic attractor after single-bit perturbations.
///
/// The supplied baseline must already be a fixed point for one complete
/// ascending-index Q4 sweep, or the experiment fails closed. For each eligible
/// vertex, independently flip its activity bit or its polarity bit in a fresh
/// model, then apply the unchanged strictly-energy-decreasing Q4 sweeps.
/// Report both exact stored-bit recall and effective z-state recall.
/// Convergence is reported explicitly; exhausted budgets never imply recall
/// stability. No thermal, noise-averaged or stochastic reliability is measured.
pub fn single_bit_perturbation_recall(
    problem: Q4Landscape<'_>,
    baseline: &[Spin4],
    max_sweeps: usize,
) -> Result<Q4RecoveryPanel, Q4LandscapeError> {
    let positions = validate(problem)?;
    check_state(problem, baseline)?;
    if max_sweeps == 0 || max_sweeps > MAX_RECALL_SWEEPS {
        return Err(Q4LandscapeError::InvalidSweepBudget);
    }
    let mut baseline_model = controller_for(problem, baseline)?;
    let baseline_energy = baseline_model.energy();
    if baseline_model.sweep() != 0 {
        return Err(Q4LandscapeError::NotLocalAttractor);
    }

    let mut trials = Vec::with_capacity(2 * positions.len());
    let mut exact_recall_count = 0usize;
    let mut effective_state_recall_count = 0usize;
    let mut fixed_point_count = 0usize;

    for &vertex in &positions {
        for perturbation in [Q4Perturbation::ActivityBit, Q4Perturbation::PolarityBit] {
            let mut perturbed = baseline.to_vec();
            let toggled = (baseline[vertex] as usize)
                ^ match perturbation {
                    Q4Perturbation::ActivityBit => 2,
                    Q4Perturbation::PolarityBit => 1,
                };
            perturbed[vertex] = Spin4::ALL[toggled];
            let mut controller = controller_for(problem, &perturbed)?;
            let energy_after_perturbation = controller.energy();
            let mut sweeps = 0usize;
            let mut reached_fixed_point = false;
            for _ in 0..max_sweeps {
                sweeps += 1;
                if controller.sweep() == 0 {
                    reached_fixed_point = true;
                    break;
                }
            }
            let mut exact_recall = true;
            let mut effective_state_recall = true;
            for &i in &positions {
                let current = controller.state(i)?.expect("eligible vertex");
                exact_recall &= current == baseline[i];
                effective_state_recall &=
                    current.contribution() == baseline[i].contribution();
            }
            exact_recall_count += usize::from(exact_recall);
            effective_state_recall_count += usize::from(effective_state_recall);
            fixed_point_count += usize::from(reached_fixed_point);
            trials.push(Q4RecoveryTrial {
                vertex,
                perturbation,
                energy_after_perturbation,
                final_energy: controller.energy(),
                sweeps,
                reached_fixed_point,
                exact_recall,
                effective_state_recall,
            });
        }
    }
    Ok(Q4RecoveryPanel {
        baseline_energy,
        trials,
        exact_recall_count,
        effective_state_recall_count,
        fixed_point_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising_topological_defects::analyze_topological_defects;

    #[test]
    fn two_ferromagnetic_spins_have_exact_dormancy_assisted_barrier() {
        let edges = [SparseIsingEdge { left: 0, right: 1, coupling: 3 }];
        let problem = Q4Landscape {
            biases: &[0, 0], activation_costs: &[0, 0],
            eligibility: &[0b11], edges: &edges,
        };
        let initial = [Spin4::ActivePositive; 2];
        let target = [Spin4::ActiveNegative; 2];
        let result = exact_quaternary_barrier(problem, &initial, &target).unwrap();
        assert_eq!(result.initial_energy, -3);
        assert_eq!(result.target_energy, -3);
        assert_eq!(result.communication_height, 0);
        assert_eq!(result.barrier_above_initial, 3);
        assert_eq!(result.enumerated_states, 16);
        assert_eq!(result.witness.first().unwrap(), &initial);
        assert_eq!(result.witness.last().unwrap(), &target);
        let mut witness_peak = i128::MIN;
        for state in &result.witness {
            let mut controller = controller_for(problem, state).unwrap();
            witness_peak = witness_peak.max(controller.energy());
            // Every witness vertex assignment is already a valid controller.
            assert_eq!(controller.len(), 2);
        }
        assert_eq!(witness_peak, result.communication_height);
        for pair in result.witness.windows(2) {
            let differences = pair[0].iter().zip(&pair[1])
                .filter(|(a, b)| a != b).count();
            assert_eq!(differences, 1);
        }
    }

    #[test]
    fn energy_barrier_is_gauge_invariant_with_switched_coupling() {
        let positive = [SparseIsingEdge { left: 0, right: 1, coupling: 3 }];
        let negative = [SparseIsingEdge { left: 0, right: 1, coupling: -3 }];
        let a = Q4Landscape {
            biases: &[0, 0], activation_costs: &[0, 0],
            eligibility: &[0b11], edges: &positive,
        };
        let b = Q4Landscape { edges: &negative, ..a };
        let x = [Spin4::ActivePositive; 2];
        let y = [Spin4::ActiveNegative; 2];
        let x_switched = [Spin4::ActiveNegative, Spin4::ActivePositive];
        let y_switched = [Spin4::ActivePositive, Spin4::ActiveNegative];
        let left = exact_quaternary_barrier(a, &x, &y).unwrap();
        let right = exact_quaternary_barrier(b, &x_switched, &y_switched).unwrap();
        assert_eq!(left.initial_energy, right.initial_energy);
        assert_eq!(left.target_energy, right.target_energy);
        assert_eq!(left.communication_height, right.communication_height);
    }

    #[test]
    fn dormant_polarity_retains_latent_bit_without_exact_autorecall() {
        let p = Q4Landscape {
            biases: &[0], activation_costs: &[5],
            eligibility: &[1], edges: &[],
        };
        let base = [Spin4::DormantPositive];
        let panel = single_bit_perturbation_recall(p, &base, 4).unwrap();
        assert_eq!(panel.trials.len(), 2);
        assert_eq!(panel.exact_recall_count, 1);
        assert_eq!(panel.effective_state_recall_count, 2);
        assert_eq!(panel.fixed_point_count, 2);
        assert!(panel.trials[0].exact_recall);
        assert!(!panel.trials[1].exact_recall);
        assert!(panel.trials[1].effective_state_recall);
    }

    #[test]
    fn biased_active_spin_recovers_both_bit_perturbations() {
        let p = Q4Landscape {
            biases: &[5], activation_costs: &[0],
            eligibility: &[1], edges: &[],
        };
        let panel = single_bit_perturbation_recall(p, &[Spin4::ActivePositive], 4).unwrap();
        assert_eq!(panel.baseline_energy, -5);
        assert_eq!(panel.exact_recall_count, 2);
        assert_eq!(panel.effective_state_recall_count, 2);
        assert_eq!(panel.fixed_point_count, 2);
    }

    #[test]
    fn negative_triangle_has_structural_defect_but_no_recall_claim() {
        let balanced = [
            SparseIsingEdge { left: 0, right: 1, coupling: 2 },
            SparseIsingEdge { left: 0, right: 2, coupling: 2 },
            SparseIsingEdge { left: 1, right: 2, coupling: 2 },
        ];
        let frustrated = [
            SparseIsingEdge { left: 0, right: 1, coupling: 2 },
            SparseIsingEdge { left: 0, right: 2, coupling: 2 },
            SparseIsingEdge { left: 1, right: 2, coupling: -2 },
        ];
        let b = Q4Landscape {
            biases: &[0; 3], activation_costs: &[0; 3],
            eligibility: &[0b111], edges: &balanced,
        };
        let f = Q4Landscape { edges: &frustrated, ..b };
        let source = [Spin4::ActivePositive; 3];
        let target = [Spin4::ActiveNegative; 3];
        let br = exact_quaternary_barrier(b, &source, &target).unwrap();
        let fr = exact_quaternary_barrier(f, &source, &target).unwrap();
        assert_eq!(br.enumerated_states, 64);
        assert_eq!(fr.enumerated_states, 64);
        assert!(br.barrier_above_initial >= 0);
        assert!(fr.barrier_above_initial >= 0);
        let balanced_topology = analyze_topological_defects(
            3, &[0b111], &[0b111], &[0b111], &balanced,
        ).unwrap();
        let frustrated_topology = analyze_topological_defects(
            3, &[0b111], &[0b111], &[0b111], &frustrated,
        ).unwrap();
        assert!(balanced_topology.structurally_balanced());
        assert!(!frustrated_topology.structurally_balanced());
        // This fixture intentionally does not assert one topology has better recall.
    }

    #[test]
    fn invalid_inputs_and_non_attractors_fail_closed() {
        let invalid = Q4Landscape {
            biases: &[0; 9], activation_costs: &[0; 9],
            eligibility: &[0x1ff], edges: &[],
        };
        assert!(matches!(
            exact_quaternary_barrier(invalid, &[], &[]),
            Err(Q4LandscapeError::TooManyVertices { .. })
        ));
        let p = Q4Landscape {
            biases: &[2], activation_costs: &[0], eligibility: &[1], edges: &[],
        };
        assert_eq!(
            single_bit_perturbation_recall(p, &[Spin4::ActiveNegative], 3).unwrap_err(),
            Q4LandscapeError::NotLocalAttractor
        );
        assert_eq!(
            single_bit_perturbation_recall(p, &[Spin4::ActivePositive], 0).unwrap_err(),
            Q4LandscapeError::InvalidSweepBudget
        );
        let masked = Q4Landscape {
            biases: &[0, 0], activation_costs: &[0, 0],
            eligibility: &[0b01], edges: &[],
        };
        assert_eq!(
            exact_quaternary_barrier(masked, &[Spin4::DormantNegative, Spin4::ActivePositive],
                                      &[Spin4::DormantNegative; 2]).unwrap_err(),
            Q4LandscapeError::NonneutralMaskedState { vertex: 1 }
        );
    }
}
