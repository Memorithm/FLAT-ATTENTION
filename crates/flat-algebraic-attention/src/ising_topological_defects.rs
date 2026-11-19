//! Research-only topological diagnostics for Boolean-quaternary Ising graphs.
//!
//! A negative product of coupling signs along a cycle is gauge invariant:
//! it cannot be satisfied by any assignment of binary spin orientations.
//! The number of negative *fundamental* cycles depends on the chosen spanning
//! forest; its zero/nonzero status is invariant. This is distinct from a
//! state-dependent unsatisfied edge, activity boundary or polarity wall.
//! A standard Ising domain wall is NOT automatically a protected memory.
//!
//! Masked vertices and zero-weight couplings do not belong to the effective
//! interaction graph. The observation is read-only and has no learning, repair,
//! numerical-attention or performance-promotion side effect.

use core::fmt;
use std::collections::VecDeque;

use crate::ising_quaternary_routing::QuaternaryHeadPlan;
use crate::ising_sparse_state::SparseIsingEdge;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologicalDefectError {
    EmptyGraph,
    TooManyVertices,
    InvalidBitplaneLength,
    InvalidPaddingBits,
    ActiveOutsideEligibility,
    NoncanonicalEdge { index: usize },
    QueryOutOfRange,
}

impl fmt::Display for TopologicalDefectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for TopologicalDefectError {}

/// Flags for one effective, nonzero-coupling, structurally eligible edge.
///
/// A fundamental-cycle closure depends on the deterministic spanning forest.
/// The sign of the corresponding complete cycle, not its chosen closure edge,
/// is the gauge-invariant object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologicalDefectEdge {
    pub left: usize,
    pub right: usize,
    pub coupling: i16,
    pub activity_boundary: bool,
    pub active_polarity_wall: bool,
    pub latent_polarity_boundary: bool,
    pub unsatisfied_active_coupling: bool,
    pub negative_basis_cycle_closure: bool,
}

/// Exact structural and state-conditioned counters over the effective graph.
///
/// All counts refer to eligible vertices and nonzero eligible-eligible edges.
/// Domain-wall and activation-boundary counts depend on current Q4 bits;
/// frustrated cycle existence depends on coupling signs and graph topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologicalDefectReport {
    pub eligible_vertices: usize,
    pub active_vertices: usize,
    pub effective_edges: usize,
    pub connected_components: usize,
    pub isolated_vertices: usize,
    pub cycle_rank: usize,
    pub negative_fundamental_cycles: usize,
    pub activity_boundaries: usize,
    pub active_polarity_walls: usize,
    pub latent_polarity_boundaries: usize,
    pub unsatisfied_active_couplings: usize,
    /// Exactly twice the absolute coupling magnitude for each unsatisfied
    /// active pair. Excludes biases and activity costs; not total system energy.
    pub active_coupling_excess_energy: u128,
    pub defect_edges: Vec<TopologicalDefectEdge>,
}

impl TopologicalDefectReport {
    /// A signed graph is balanced iff every cycle has positive sign product.
    pub fn structurally_balanced(&self) -> bool {
        self.negative_fundamental_cycles == 0
    }
}

fn bit(plane: &[u64], position: usize) -> bool {
    plane[position / 64] & (1_u64 << (position % 64)) != 0
}

/// Diagnose structural frustration and Q4 state boundaries on a frozen graph.
///
/// Edges must be canonical (left < right, strictly increasing pair order).
/// Couplings of zero are ignored: they do not constrain the spin state or
/// connect components. Isolated eligible vertices count as components.
/// The deterministic ascending-index BFS forest defines the reported cycle
/// basis. Cycle rank equals effective_edges - eligible_vertices + components.
/// The negative fundamental-cycle count is basis-dependent; only its
/// zero/nonzero status is a complete invariant for structural balance.
pub fn analyze_topological_defects(
    vertices: usize,
    eligibility: &[u64],
    active: &[u64],
    polarity: &[u64],
    edges: &[SparseIsingEdge],
) -> Result<TopologicalDefectReport, TopologicalDefectError> {
    if vertices == 0 {
        return Err(TopologicalDefectError::EmptyGraph);
    }
    if vertices > u32::MAX as usize {
        return Err(TopologicalDefectError::TooManyVertices);
    }
    let words = vertices.div_ceil(64);
    if eligibility.len() != words || active.len() != words || polarity.len() != words {
        return Err(TopologicalDefectError::InvalidBitplaneLength);
    }
    if vertices % 64 != 0 {
        let invalid = !((1_u64 << (vertices % 64)) - 1);
        if (eligibility[words - 1] | active[words - 1] | polarity[words - 1]) & invalid != 0 {
            return Err(TopologicalDefectError::InvalidPaddingBits);
        }
    }
    for (&allowed, &activated) in eligibility.iter().zip(active) {
        if activated & !allowed != 0 {
            return Err(TopologicalDefectError::ActiveOutsideEligibility);
        }
    }

    let mut adjacency = vec![Vec::<(usize, usize)>::new(); vertices];
    let mut live = vec![false; edges.len()];
    let mut previous: Option<(usize, usize)> = None;
    let mut effective_edges = 0usize;
    for (index, edge) in edges.iter().enumerate() {
        let pair = (edge.left, edge.right);
        if edge.left >= edge.right
            || edge.right >= vertices
            || previous.is_some_and(|last| pair <= last)
        {
            return Err(TopologicalDefectError::NoncanonicalEdge { index });
        }
        previous = Some(pair);
        if edge.coupling == 0 || !bit(eligibility, edge.left) || !bit(eligibility, edge.right) {
            continue;
        }
        live[index] = true;
        effective_edges += 1;
        adjacency[edge.left].push((edge.right, index));
        adjacency[edge.right].push((edge.left, index));
    }

    let eligible_vertices = (0..vertices).filter(|&i| bit(eligibility, i)).count();
    let active_vertices = (0..vertices).filter(|&i| bit(active, i)).count();
    let isolated_vertices = (0..vertices)
        .filter(|&i| bit(eligibility, i) && adjacency[i].is_empty())
        .count();

    let mut gauge = vec![0_i8; vertices];
    let mut tree_edge = vec![false; edges.len()];
    let mut queue = VecDeque::new();
    let mut connected_components = 0usize;

    for root in 0..vertices {
        if !bit(eligibility, root) || gauge[root] != 0 {
            continue;
        }
        connected_components += 1;
        gauge[root] = 1;
        queue.push_back(root);
        while let Some(u) = queue.pop_front() {
            for &(v, edge_index) in &adjacency[u] {
                if gauge[v] != 0 {
                    continue;
                }
                let sign = if edges[edge_index].coupling > 0 {
                    1
                } else {
                    -1
                };
                gauge[v] = gauge[u] * sign;
                tree_edge[edge_index] = true;
                queue.push_back(v);
            }
        }
    }

    let mut cycle_rank = 0usize;
    let mut negative_fundamental_cycles = 0usize;
    let mut activity_boundaries = 0usize;
    let mut active_polarity_walls = 0usize;
    let mut latent_polarity_boundaries = 0usize;
    let mut unsatisfied_active_couplings = 0usize;
    let mut active_coupling_excess_energy = 0u128;
    let mut defect_edges = Vec::new();

    for (index, edge) in edges.iter().enumerate() {
        if !live[index] {
            continue;
        }
        let (u, v) = (edge.left, edge.right);
        let (active_u, active_v) = (bit(active, u), bit(active, v));
        let unequal_polarity = bit(polarity, u) != bit(polarity, v);
        let activity_boundary = active_u != active_v;
        let active_polarity_wall = active_u && active_v && unequal_polarity;
        let latent_polarity_boundary = (!active_u || !active_v) && unequal_polarity;
        let sign = if edge.coupling > 0 { 1_i8 } else { -1_i8 };
        let negative_basis_cycle_closure = !tree_edge[index] && gauge[u] * gauge[v] * sign < 0;
        if !tree_edge[index] {
            cycle_rank += 1;
            if negative_basis_cycle_closure {
                negative_fundamental_cycles += 1;
            }
        }
        let spin_product = if unequal_polarity { -1 } else { 1 };
        let unsatisfied_active_coupling = active_u && active_v && spin_product != sign;
        if activity_boundary {
            activity_boundaries += 1;
        }
        if active_polarity_wall {
            active_polarity_walls += 1;
        }
        if latent_polarity_boundary {
            latent_polarity_boundaries += 1;
        }
        if unsatisfied_active_coupling {
            unsatisfied_active_couplings += 1;
            active_coupling_excess_energy +=
                2 * u128::from(i32::from(edge.coupling).unsigned_abs());
        }
        if activity_boundary
            || active_polarity_wall
            || latent_polarity_boundary
            || unsatisfied_active_coupling
            || negative_basis_cycle_closure
        {
            defect_edges.push(TopologicalDefectEdge {
                left: u,
                right: v,
                coupling: edge.coupling,
                activity_boundary,
                active_polarity_wall,
                latent_polarity_boundary,
                unsatisfied_active_coupling,
                negative_basis_cycle_closure,
            });
        }
    }

    debug_assert_eq!(
        cycle_rank + eligible_vertices,
        effective_edges + connected_components
    );
    Ok(TopologicalDefectReport {
        eligible_vertices,
        active_vertices,
        effective_edges,
        connected_components,
        isolated_vertices,
        cycle_rank,
        negative_fundamental_cycles,
        activity_boundaries,
        active_polarity_walls,
        latent_polarity_boundaries,
        unsatisfied_active_couplings,
        active_coupling_excess_energy,
        defect_edges,
    })
}

/// Inspect the exact causal eligibility induced by one FLAT Q4 query row.
///
/// This is a read-only observer of the same per-head initial-state inputs used
/// by the routing bridge. It does not call the routing energy optimizer and
/// therefore reports topology BEFORE local descent, not after selection.
pub fn analyze_quaternary_head_at_query(
    plan: &QuaternaryHeadPlan<'_>,
    vertices: usize,
    query_position: usize,
    causal: bool,
) -> Result<TopologicalDefectReport, TopologicalDefectError> {
    if vertices == 0 {
        return Err(TopologicalDefectError::EmptyGraph);
    }
    if query_position >= vertices {
        return Err(TopologicalDefectError::QueryOutOfRange);
    }
    if plan.biases.len() != vertices || plan.activation_costs.len() != vertices {
        return Err(TopologicalDefectError::InvalidBitplaneLength);
    }
    let words = vertices.div_ceil(64);
    if plan.eligible.len() != words || plan.active.len() != words || plan.polarity.len() != words {
        return Err(TopologicalDefectError::InvalidBitplaneLength);
    }
    // Validate original masking before applying the causal intersection.
    for (&a, &e) in plan.active.iter().zip(plan.eligible) {
        if a & !e != 0 {
            return Err(TopologicalDefectError::ActiveOutsideEligibility);
        }
    }
    let mut eligible = plan.eligible.to_vec();
    if causal {
        for (word_index, word) in eligible.iter_mut().enumerate() {
            let start = word_index * 64;
            if start > query_position {
                *word = 0;
            } else {
                let remaining = query_position - start + 1;
                if remaining < 64 {
                    *word &= (1_u64 << remaining) - 1;
                }
            }
        }
    }
    let active: Vec<u64> = plan
        .active
        .iter()
        .zip(&eligible)
        .map(|(&a, &allowed)| a & allowed)
        .collect();
    analyze_topological_defects(vertices, &eligible, &active, plan.polarity, plan.couplings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(u: usize, v: usize, w: i16) -> SparseIsingEdge {
        SparseIsingEdge {
            left: u,
            right: v,
            coupling: w,
        }
    }

    #[test]
    fn negative_triangle_is_structurally_frustrated_in_every_spin_state() {
        let edges = [edge(0, 1, 1), edge(0, 2, 1), edge(1, 2, -1)];
        for polarity in 0_u64..8 {
            let report =
                analyze_topological_defects(3, &[0b111], &[0b111], &[polarity], &edges).unwrap();
            assert_eq!(report.eligible_vertices, 3);
            assert_eq!(report.cycle_rank, 1);
            assert_eq!(report.negative_fundamental_cycles, 1);
            assert_eq!(
                report.unsatisfied_active_couplings,
                1 + 2 * usize::from(report.unsatisfied_active_couplings == 3)
            );
            assert!(!report.structurally_balanced());
        }
    }

    #[test]
    fn graph_balance_is_gauge_invariant() {
        let graph = [edge(0, 1, 2), edge(0, 2, -3), edge(1, 2, 4)];
        // Flip gauge at vertex 0: both incident coupling signs flip.
        let switched = [edge(0, 1, -2), edge(0, 2, 3), edge(1, 2, 4)];
        let a = analyze_topological_defects(3, &[0b111], &[0b111], &[0b111], &graph).unwrap();
        let b = analyze_topological_defects(3, &[0b111], &[0b111], &[0b110], &switched).unwrap();
        assert_eq!(a.negative_fundamental_cycles, b.negative_fundamental_cycles);
        assert_eq!(
            a.unsatisfied_active_couplings,
            b.unsatisfied_active_couplings
        );
        assert_eq!(
            a.active_coupling_excess_energy,
            b.active_coupling_excess_energy
        );
        assert_eq!(a.cycle_rank, b.cycle_rank);
    }

    #[test]
    fn balanced_cycle_with_unsatisfied_edge_is_not_a_structural_defect() {
        let graph = [edge(0, 1, 1), edge(0, 2, 1), edge(1, 2, 1)];
        let report = analyze_topological_defects(3, &[0b111], &[0b111], &[0b001], &graph).unwrap();
        assert!(report.structurally_balanced());
        assert_eq!(report.cycle_rank, 1);
        assert_eq!(report.unsatisfied_active_couplings, 2);
        assert_eq!(report.active_polarity_walls, 2);
        assert_eq!(report.active_coupling_excess_energy, 4);
    }

    #[test]
    fn masks_and_zero_couplings_remove_edges_before_topology() {
        let graph = [edge(0, 1, -2), edge(0, 2, 1), edge(1, 2, -2), edge(2, 3, 0)];
        let report =
            analyze_topological_defects(4, &[0b1011], &[0b0011], &[0b1010], &graph).unwrap();
        assert_eq!(report.eligible_vertices, 3);
        assert_eq!(report.effective_edges, 1);
        assert_eq!(report.connected_components, 2);
        assert_eq!(report.isolated_vertices, 1);
        assert_eq!(report.cycle_rank, 0);
        assert_eq!(report.defect_edges.len(), 1);
        assert_eq!(report.activity_boundaries, 0);
    }

    #[test]
    fn dormant_boundaries_are_not_energetic_frustration() {
        let graph = [edge(0, 1, 2)];
        let report = analyze_topological_defects(2, &[0b11], &[0b00], &[0b10], &graph).unwrap();
        assert_eq!(report.latent_polarity_boundaries, 1);
        assert_eq!(report.active_polarity_walls, 0);
        assert_eq!(report.unsatisfied_active_couplings, 0);
        assert_eq!(report.active_coupling_excess_energy, 0);
    }

    #[test]
    fn causal_mask_excludes_future_negative_cycle_before_diagnosis() {
        let graph = [edge(0, 1, 1), edge(0, 2, 1), edge(1, 2, -1)];
        let plan = QuaternaryHeadPlan {
            biases: &[0; 3],
            activation_costs: &[0; 3],
            eligible: &[0b111],
            active: &[0b111],
            polarity: &[0b111],
            couplings: &graph,
        };
        let before = analyze_quaternary_head_at_query(&plan, 3, 1, true).unwrap();
        let after = analyze_quaternary_head_at_query(&plan, 3, 2, true).unwrap();
        assert!(before.structurally_balanced());
        assert_eq!(before.effective_edges, 1);
        assert_eq!(before.active_vertices, 2);
        assert_eq!(after.negative_fundamental_cycles, 1);
        assert_eq!(after.active_vertices, 3);
    }

    #[test]
    fn rejects_malformed_topology() {
        assert_eq!(
            analyze_topological_defects(0, &[], &[], &[], &[]).unwrap_err(),
            TopologicalDefectError::EmptyGraph
        );
        assert_eq!(
            analyze_topological_defects(2, &[0b11], &[0b100], &[0], &[]).unwrap_err(),
            TopologicalDefectError::InvalidPaddingBits
        );
        assert_eq!(
            analyze_topological_defects(2, &[0b01], &[0b11], &[0], &[]).unwrap_err(),
            TopologicalDefectError::ActiveOutsideEligibility
        );
        assert_eq!(
            analyze_topological_defects(2, &[0b11], &[0], &[0], &[edge(1, 0, 1)]).unwrap_err(),
            TopologicalDefectError::NoncanonicalEdge { index: 0 }
        );
    }
}
