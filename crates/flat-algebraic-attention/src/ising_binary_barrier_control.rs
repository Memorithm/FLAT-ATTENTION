//! Small, exact binary-Ising control for Q4 energy-barrier experiments.
//!
//! Uses the identical fixed Q4 Hamiltonian, mask, signed couplings and local
//! costs, but restricts every eligible vertex to ACTIVE +/- spins. Each edge
//! in the transition graph flips exactly one polarity. This establishes an
//! energy-barrier control against Q4's additional dormant states, not a speed
//! or neural-network performance benchmark.

use core::cmp::{max, Reverse};
use core::fmt;
use std::collections::BinaryHeap;

use crate::ising_energy_barriers::{Q4Landscape, Q4LandscapeError, MAX_EXACT_Q4_VERTICES};
use crate::ising_quaternary::{BooleanQuaternary, Spin4};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryBarrierError {
    Landscape(Q4LandscapeError),
    InactiveEndpoint { vertex: usize },
}

impl From<Q4LandscapeError> for BinaryBarrierError {
    fn from(value: Q4LandscapeError) -> Self {
        Self::Landscape(value)
    }
}

impl fmt::Display for BinaryBarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for BinaryBarrierError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactBinaryBarrierReport {
    pub eligible_vertices: usize,
    pub enumerated_states: usize,
    pub explored_states: usize,
    pub initial_energy: i128,
    pub target_energy: i128,
    pub communication_height: i128,
    pub barrier_above_initial: i128,
    pub witness: Vec<Vec<Spin4>>,
}

fn bit(bits: &[u64], i: usize) -> bool {
    (bits[i / 64] >> (i % 64)) & 1 != 0
}

fn validate(
    problem: Q4Landscape<'_>,
    initial: &[Spin4],
    target: &[Spin4],
) -> Result<Vec<usize>, BinaryBarrierError> {
    let n = problem.biases.len();
    if n == 0 {
        return Err(Q4LandscapeError::Empty.into());
    }
    if n > MAX_EXACT_Q4_VERTICES {
        return Err(Q4LandscapeError::TooManyVertices {
            actual: n,
            limit: MAX_EXACT_Q4_VERTICES,
        }
        .into());
    }
    if initial.len() != n || target.len() != n {
        return Err(Q4LandscapeError::EndpointLength.into());
    }
    let zeros = [0_u64];
    BooleanQuaternary::new(
        problem.biases,
        problem.activation_costs,
        problem.eligibility,
        problem.eligibility,
        &zeros,
        problem.edges,
    )
    .map_err(Q4LandscapeError::from)?;
    let mut positions = Vec::new();
    for vertex in 0..n {
        if bit(problem.eligibility, vertex) {
            for state in [initial[vertex], target[vertex]] {
                if !state.active() {
                    return Err(BinaryBarrierError::InactiveEndpoint { vertex });
                }
            }
            positions.push(vertex);
        } else if initial[vertex] != Spin4::DormantNegative
            || target[vertex] != Spin4::DormantNegative
        {
            return Err(Q4LandscapeError::NonneutralMaskedState { vertex }.into());
        }
    }
    if positions.is_empty() {
        return Err(Q4LandscapeError::NoEligibleVertices.into());
    }
    Ok(positions)
}

fn code_from_states(states: &[Spin4], positions: &[usize]) -> usize {
    positions
        .iter()
        .enumerate()
        .fold(0_usize, |acc, (slot, &vertex)| {
            acc | (usize::from(states[vertex].positive()) << slot)
        })
}

fn states_from_code(code: usize, positions: &[usize], n: usize) -> Vec<Spin4> {
    let mut states = vec![Spin4::DormantNegative; n];
    for (slot, &vertex) in positions.iter().enumerate() {
        states[vertex] = if (code >> slot) & 1 == 1 {
            Spin4::ActivePositive
        } else {
            Spin4::ActiveNegative
        };
    }
    states
}

fn energy(
    problem: Q4Landscape<'_>,
    code: usize,
    positions: &[usize],
    slots: &[Option<usize>],
) -> i128 {
    let mut result = 0_i128;
    for (slot, &vertex) in positions.iter().enumerate() {
        let z = if (code >> slot) & 1 == 1 {
            1_i128
        } else {
            -1_i128
        };
        result += i128::from(problem.activation_costs[vertex]);
        result -= i128::from(problem.biases[vertex]) * z;
    }
    for edge in problem.edges {
        if let (Some(i), Some(j)) = (slots[edge.left], slots[edge.right]) {
            let zi = if (code >> i) & 1 == 1 {
                1_i128
            } else {
                -1_i128
            };
            let zj = if (code >> j) & 1 == 1 {
                1_i128
            } else {
                -1_i128
            };
            result -= i128::from(edge.coupling) * zi * zj;
        }
    }
    result
}

/// Exhaustive binary-active minimax barrier, with a validated witness.
///
/// Source and target must be fully active on the eligible vertices. Dormant
/// Q4 endpoints must NOT be silently coerced into active endpoints. This
/// control explores 2^k states, with no more than eight eligible vertices.
pub fn exact_active_binary_barrier(
    problem: Q4Landscape<'_>,
    initial: &[Spin4],
    target: &[Spin4],
) -> Result<ExactBinaryBarrierReport, BinaryBarrierError> {
    let positions = validate(problem, initial, target)?;
    let n = problem.biases.len();
    let count = 1_usize << positions.len();
    let mut slots = vec![None; n];
    for (slot, &vertex) in positions.iter().enumerate() {
        slots[vertex] = Some(slot);
    }
    let energies: Vec<i128> = (0..count)
        .map(|code| energy(problem, code, &positions, &slots))
        .collect();
    let origin = code_from_states(initial, &positions);
    let destination = code_from_states(target, &positions);
    let mut heights = vec![i128::MAX; count];
    let mut parents = vec![None; count];
    let mut frontier = BinaryHeap::new();
    heights[origin] = energies[origin];
    frontier.push(Reverse((heights[origin], origin)));
    let mut explored_states = 0_usize;

    while let Some(Reverse((height, code))) = frontier.pop() {
        if height != heights[code] {
            continue;
        }
        explored_states += 1;
        if code == destination {
            break;
        }
        for bit in 0..positions.len() {
            let adjacent = code ^ (1_usize << bit);
            let candidate = max(height, energies[adjacent]);
            if candidate < heights[adjacent] {
                heights[adjacent] = candidate;
                parents[adjacent] = Some(code);
                frontier.push(Reverse((candidate, adjacent)));
            }
        }
    }

    let communication_height = heights[destination];
    let mut codes = vec![destination];
    while *codes.last().expect("nonempty path") != origin {
        let previous =
            parents[*codes.last().expect("nonempty path")].expect("binary hypercube is connected");
        codes.push(previous);
    }
    codes.reverse();

    Ok(ExactBinaryBarrierReport {
        eligible_vertices: positions.len(),
        enumerated_states: count,
        explored_states,
        initial_energy: energies[origin],
        target_energy: energies[destination],
        communication_height,
        barrier_above_initial: communication_height - energies[origin],
        witness: codes
            .into_iter()
            .map(|code| states_from_code(code, &positions, n))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising_energy_barriers::exact_quaternary_barrier;
    use crate::ising_sparse_state::SparseIsingEdge;

    #[test]
    fn binary_ferromagnetic_barrier_is_six_q4_barrier_is_three() {
        let edges = [SparseIsingEdge {
            left: 0,
            right: 1,
            coupling: 3,
        }];
        let problem = Q4Landscape {
            biases: &[0, 0],
            activation_costs: &[0, 0],
            eligibility: &[0b11],
            edges: &edges,
        };
        let start = [Spin4::ActivePositive; 2];
        let end = [Spin4::ActiveNegative; 2];
        let binary = exact_active_binary_barrier(problem, &start, &end).unwrap();
        let q4 = exact_quaternary_barrier(problem, &start, &end).unwrap();
        assert_eq!(binary.communication_height, 3);
        assert_eq!(binary.barrier_above_initial, 6);
        assert_eq!(q4.communication_height, 0);
        assert_eq!(q4.barrier_above_initial, 3);
        assert_eq!(binary.enumerated_states, 4);
        assert_eq!(q4.enumerated_states, 16);
        for adjacent in binary.witness.windows(2) {
            assert_eq!(
                adjacent[0]
                    .iter()
                    .zip(&adjacent[1])
                    .filter(|(a, b)| a != b)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn binary_rejects_dormant_endpoint_instead_of_reinterpreting_polarity() {
        let problem = Q4Landscape {
            biases: &[0],
            activation_costs: &[0],
            eligibility: &[0b1],
            edges: &[],
        };
        assert_eq!(
            exact_active_binary_barrier(
                problem,
                &[Spin4::DormantPositive],
                &[Spin4::ActiveNegative],
            )
            .unwrap_err(),
            BinaryBarrierError::InactiveEndpoint { vertex: 0 },
        );
    }

    #[test]
    fn binary_gauge_transform_preserves_height() {
        use crate::ising_sparse_state::SparseIsingEdge;
        let positive = [SparseIsingEdge {
            left: 0,
            right: 1,
            coupling: 2,
        }];
        let negative = [SparseIsingEdge {
            left: 0,
            right: 1,
            coupling: -2,
        }];
        let a = Q4Landscape {
            biases: &[0; 2],
            activation_costs: &[1; 2],
            eligibility: &[0b11],
            edges: &positive,
        };
        let b = Q4Landscape {
            edges: &negative,
            ..a
        };
        let start = [Spin4::ActivePositive, Spin4::ActivePositive];
        let end = [Spin4::ActiveNegative, Spin4::ActiveNegative];
        let transformed_start = [Spin4::ActiveNegative, Spin4::ActivePositive];
        let transformed_end = [Spin4::ActivePositive, Spin4::ActiveNegative];
        assert_eq!(
            exact_active_binary_barrier(a, &start, &end)
                .unwrap()
                .communication_height,
            exact_active_binary_barrier(b, &transformed_start, &transformed_end)
                .unwrap()
                .communication_height,
        );
    }
}
