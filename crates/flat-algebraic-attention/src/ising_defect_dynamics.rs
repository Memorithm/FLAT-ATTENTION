//! Temporal comparison of signed-graph and Q4 Boolean defect snapshots.
//!
//! A negative coupling cycle is a structural property, not a transient spin
//! defect. State-boundary births/resolutions are counted ONLY on effective
//! edges shared between snapshots (same endpoints AND same coupling).
//! Edges gained or lost due to masks or rewiring are reported separately.
//! Changing the effective graph invalidates direct signed-cycle comparisons.
//!
//! Snapshots are read-only. They may represent consecutive Q4 descent sweeps,
//! causal query positions, or controlled perturbations. No energy barrier,
//! dynamical stability, memory retention or wall-clock speedup is inferred.

use core::cmp::Ordering;

use crate::ising_sparse_state::SparseIsingEdge;
use crate::ising_topological_defects::{
    analyze_topological_defects, TopologicalDefectError, TopologicalDefectReport,
};

/// Frozen bitplanes and signed couplings at one observed instant.
#[derive(Debug, Clone, Copy)]
pub struct QuaternaryDefectFrame<'a> {
    pub eligibility: &'a [u64],
    pub activity: &'a [u64],
    pub polarity: &'a [u64],
    pub edges: &'a [SparseIsingEdge],
}

/// Event classification restricted to effective edges common to both frames.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DefectTransitions {
    pub appeared: usize,
    pub resolved: usize,
    pub persisted: usize,
}

impl DefectTransitions {
    fn observe(&mut self, previous: bool, current: bool) {
        match (previous, current) {
            (false, true) => self.appeared += 1,
            (true, false) => self.resolved += 1,
            (true, true) => self.persisted += 1,
            (false, false) => {}
        }
    }
}

/// Exact temporal delta with masks and effective graph changes separated.
///
/// Index [old_state][new_state] in the 4x4 matrix uses the Q4 encoding:
/// dormant-negative=0, dormant-positive=1, active-negative=2,
/// active-positive=3. Only vertices eligible in BOTH frames are counted.
/// Masked vertices never implicitly become dormant or negative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuaternaryDefectDynamics {
    pub before: TopologicalDefectReport,
    pub after: TopologicalDefectReport,
    /// True iff both masks and the signed effective edge lists are identical.
    /// A direct structural-frustration comparison is valid only if true.
    pub same_effective_topology: bool,
    pub newly_eligible_vertices: usize,
    pub newly_masked_vertices: usize,
    pub shared_eligible_vertices: usize,
    pub state_transitions: [[usize; 4]; 4],
    pub shared_effective_edges: usize,
    pub entered_effective_edges: usize,
    pub exited_effective_edges: usize,
    pub activity_boundaries: DefectTransitions,
    pub active_polarity_walls: DefectTransitions,
    pub latent_polarity_boundaries: DefectTransitions,
    pub unsatisfied_active_couplings: DefectTransitions,
}

fn bit(plane: &[u64], i: usize) -> bool {
    (plane[i / 64] >> (i % 64)) & 1 != 0
}

fn q4_code(frame: QuaternaryDefectFrame<'_>, i: usize) -> usize {
    usize::from(bit(frame.activity, i)) * 2 + usize::from(bit(frame.polarity, i))
}

/// Per-effective-edge dynamic classification, independent of cycle basis.
fn flags(frame: QuaternaryDefectFrame<'_>, edge: &SparseIsingEdge) -> [bool; 4] {
    let au = bit(frame.activity, edge.left);
    let av = bit(frame.activity, edge.right);
    let polarity_differs = bit(frame.polarity, edge.left) != bit(frame.polarity, edge.right);
    [
        au != av,
        au && av && polarity_differs,
        (!au || !av) && polarity_differs,
        au && av && (polarity_differs == (edge.coupling > 0)),
    ]
}

fn effective<'a>(
    frame: QuaternaryDefectFrame<'a>,
) -> impl Iterator<Item = &'a SparseIsingEdge> + 'a {
    frame.edges.iter().filter(move |edge| {
        edge.coupling != 0
            && bit(frame.eligibility, edge.left)
            && bit(frame.eligibility, edge.right)
    })
}

/// Compare two Q4 graph states with explicit structural-change accounting.
///
/// Both inputs are independently validated, including canonical signed edges,
/// mask coverage and bitplane padding. Changed connectivity or mask does not
/// create fictitious state-defect appearance/disappearance: only common edges
/// contribute to the four defect transition counters. An edge with changed
/// coupling is an exited edge plus an entered edge, NOT an unchanged edge.
///
/// This does not estimate an energy barrier: an energy barrier requires a
/// specified transition path and its energy maximum above the initial energy.
pub fn compare_quaternary_defects(
    vertices: usize,
    previous: QuaternaryDefectFrame<'_>,
    current: QuaternaryDefectFrame<'_>,
) -> Result<QuaternaryDefectDynamics, TopologicalDefectError> {
    let before = analyze_topological_defects(
        vertices, previous.eligibility, previous.activity,
        previous.polarity, previous.edges,
    )?;
    let after = analyze_topological_defects(
        vertices, current.eligibility, current.activity,
        current.polarity, current.edges,
    )?;

    let mut state_transitions = [[0usize; 4]; 4];
    let mut newly_eligible_vertices = 0;
    let mut newly_masked_vertices = 0;
    let mut shared_eligible_vertices = 0;
    let mut identical_masks = true;
    for i in 0..vertices {
        match (bit(previous.eligibility, i), bit(current.eligibility, i)) {
            (false, true) => {
                identical_masks = false;
                newly_eligible_vertices += 1;
            }
            (true, false) => {
                identical_masks = false;
                newly_masked_vertices += 1;
            }
            (true, true) => {
                shared_eligible_vertices += 1;
                state_transitions[q4_code(previous, i)][q4_code(current, i)] += 1;
            }
            (false, false) => {}
        }
    }

    let mut shared_effective_edges = 0;
    let mut entered_effective_edges = 0;
    let mut exited_effective_edges = 0;
    let mut activity_boundaries = DefectTransitions::default();
    let mut active_polarity_walls = DefectTransitions::default();
    let mut latent_polarity_boundaries = DefectTransitions::default();
    let mut unsatisfied_active_couplings = DefectTransitions::default();

    let mut a = effective(previous).peekable();
    let mut b = effective(current).peekable();
    loop {
        match (a.peek(), b.peek()) {
            (Some(left), Some(right)) => {
                let lhs = (left.left, left.right, left.coupling);
                let rhs = (right.left, right.right, right.coupling);
                match lhs.cmp(&rhs) {
                    Ordering::Less => {
                        exited_effective_edges += 1;
                        a.next();
                    }
                    Ordering::Greater => {
                        entered_effective_edges += 1;
                        b.next();
                    }
                    Ordering::Equal => {
                        let old_flags = flags(previous, left);
                        let new_flags = flags(current, right);
                        activity_boundaries.observe(old_flags[0], new_flags[0]);
                        active_polarity_walls.observe(old_flags[1], new_flags[1]);
                        latent_polarity_boundaries.observe(old_flags[2], new_flags[2]);
                        unsatisfied_active_couplings.observe(old_flags[3], new_flags[3]);
                        shared_effective_edges += 1;
                        a.next();
                        b.next();
                    }
                }
            }
            (Some(_), None) => {
                exited_effective_edges += a.count();
                break;
            }
            (None, Some(_)) => {
                entered_effective_edges += b.count();
                break;
            }
            (None, None) => break,
        }
    }
    let same_effective_topology = identical_masks
        && entered_effective_edges == 0 && exited_effective_edges == 0;
    debug_assert_eq!(
        shared_effective_edges + exited_effective_edges,
        before.effective_edges
    );
    debug_assert_eq!(
        shared_effective_edges + entered_effective_edges,
        after.effective_edges
    );
    if same_effective_topology {
        debug_assert_eq!(before.cycle_rank, after.cycle_rank);
        debug_assert_eq!(
            before.negative_fundamental_cycles,
            after.negative_fundamental_cycles
        );
    }

    Ok(QuaternaryDefectDynamics {
        before,
        after,
        same_effective_topology,
        newly_eligible_vertices,
        newly_masked_vertices,
        shared_eligible_vertices,
        state_transitions,
        shared_effective_edges,
        entered_effective_edges,
        exited_effective_edges,
        activity_boundaries,
        active_polarity_walls,
        latent_polarity_boundaries,
        unsatisfied_active_couplings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising_quaternary::BooleanQuaternary;

    fn edge(left: usize, right: usize, coupling: i16) -> SparseIsingEdge {
        SparseIsingEdge { left, right, coupling }
    }

    #[test]
    fn frustrated_cycle_persists_as_state_defect_moves() {
        let edges = [edge(0, 1, 2), edge(0, 2, 2), edge(1, 2, -3)];
        let old = QuaternaryDefectFrame {
            eligibility: &[0b111], activity: &[0b111],
            polarity: &[0b111], edges: &edges,
        };
        let next = QuaternaryDefectFrame { polarity: &[0b011], ..old };
        let result = compare_quaternary_defects(3, old, next).unwrap();
        assert!(result.same_effective_topology);
        assert_eq!(result.before.negative_fundamental_cycles, 1);
        assert_eq!(result.after.negative_fundamental_cycles, 1);
        assert_eq!(result.before.unsatisfied_active_couplings, 1);
        assert_eq!(result.after.unsatisfied_active_couplings, 1);
        assert_eq!(result.unsatisfied_active_couplings.appeared, 1);
        assert_eq!(result.unsatisfied_active_couplings.resolved, 1);
        assert_eq!(result.unsatisfied_active_couplings.persisted, 0);
        assert_eq!(result.state_transitions[3][2], 1);
        assert_eq!(result.state_transitions[3][3], 2);
        assert_eq!(result.active_polarity_walls.appeared, 2);
    }

    #[test]
    fn dormant_polarity_survives_activation_boundary_transition() {
        let edges = [edge(0, 1, 3)];
        let first = QuaternaryDefectFrame {
            eligibility: &[0b11], activity: &[0b11],
            polarity: &[0b11], edges: &edges,
        };
        let second = QuaternaryDefectFrame { activity: &[0b01], ..first };
        let outcome = compare_quaternary_defects(2, first, second).unwrap();
        assert!(outcome.same_effective_topology);
        assert_eq!(outcome.state_transitions[3][1], 1);
        assert_eq!(outcome.state_transitions[3][3], 1);
        assert_eq!(outcome.activity_boundaries.appeared, 1);
        assert_eq!(outcome.latent_polarity_boundaries.appeared, 0);
        assert!(outcome.after.structurally_balanced());
    }

    #[test]
    fn causal_mask_growth_is_not_counted_as_defect_creation_on_shared_edges() {
        let edges = [edge(0, 1, 1), edge(0, 2, 1), edge(1, 2, -1)];
        let early = QuaternaryDefectFrame {
            eligibility: &[0b011], activity: &[0b011],
            polarity: &[0b111], edges: &edges,
        };
        let late = QuaternaryDefectFrame {
            eligibility: &[0b111], activity: &[0b111],
            polarity: &[0b111], edges: &edges,
        };
        let report = compare_quaternary_defects(3, early, late).unwrap();
        assert!(!report.same_effective_topology);
        assert_eq!(report.newly_eligible_vertices, 1);
        assert_eq!(report.shared_eligible_vertices, 2);
        assert_eq!(report.shared_effective_edges, 1);
        assert_eq!(report.entered_effective_edges, 2);
        assert_eq!(report.exited_effective_edges, 0);
        assert_eq!(report.unsatisfied_active_couplings.appeared, 0);
        assert!(report.before.structurally_balanced());
        assert!(!report.after.structurally_balanced());
    }

    #[test]
    fn coupling_change_is_topology_change_not_state_transition() {
        let original = [edge(0, 1, -2)];
        let modified = [edge(0, 1, 2)];
        let first = QuaternaryDefectFrame {
            eligibility: &[0b11], activity: &[0b11],
            polarity: &[0b11], edges: &original,
        };
        let second = QuaternaryDefectFrame { edges: &modified, ..first };
        let comparison = compare_quaternary_defects(2, first, second).unwrap();
        assert!(!comparison.same_effective_topology);
        assert_eq!(comparison.exited_effective_edges, 1);
        assert_eq!(comparison.entered_effective_edges, 1);
        assert_eq!(comparison.unsatisfied_active_couplings.appeared, 0);
    }

    #[test]
    fn monitor_can_consume_real_quaternary_descent_snapshots() {
        let edges = [edge(0, 1, 2), edge(0, 2, 2), edge(1, 2, -3)];
        let mut controller = BooleanQuaternary::new(
            &[0; 3], &[0; 3], &[0b111], &[0b111], &[0b111], &edges,
        ).unwrap();
        let baseline = controller.energy();
        let (mask0, activity0, polarity0) = controller.bitplanes();
        let first_mask = mask0.to_vec();
        let first_activity = activity0.to_vec();
        let first_polarity = polarity0.to_vec();
        let first = QuaternaryDefectFrame {
            eligibility: &first_mask, activity: &first_activity,
            polarity: &first_polarity, edges: &edges,
        };

        assert!(controller.sweep() > 0);
        assert!(controller.energy() < baseline);
        let (mask1, activity1, polarity1) = controller.bitplanes();
        let second = QuaternaryDefectFrame {
            eligibility: mask1, activity: activity1, polarity: polarity1, edges: &edges,
        };
        let comparison = compare_quaternary_defects(3, first, second).unwrap();
        assert!(comparison.same_effective_topology);
        assert_eq!(comparison.before.negative_fundamental_cycles, 1);
        assert_eq!(comparison.after.negative_fundamental_cycles, 1);
        assert_eq!(comparison.before.unsatisfied_active_couplings, 1);
        assert_eq!(comparison.after.unsatisfied_active_couplings, 1);
        assert_eq!(comparison.unsatisfied_active_couplings.appeared, 1);
        assert_eq!(comparison.unsatisfied_active_couplings.resolved, 1);
    }

    #[test]
    fn invalid_frame_fails_closed() {
        let edges = [edge(0, 1, 2)];
        let first = QuaternaryDefectFrame {
            eligibility: &[0b11], activity: &[0b11],
            polarity: &[0b11], edges: &edges,
        };
        let broken = QuaternaryDefectFrame { activity: &[0b100], ..first };
        assert!(matches!(
            compare_quaternary_defects(2, first, broken),
            Err(TopologicalDefectError::InvalidPaddingBits)
        ));
    }
}
