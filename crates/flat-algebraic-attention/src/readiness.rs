use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use crate::max_plus::{MaxPlusError, MaxPlusSchedule, MaxPlusValue};
use crate::survivor_set::SurvivorSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurvivorNodeBinding {
    candidate_index: usize,
    node: usize,
}

impl SurvivorNodeBinding {
    #[must_use]
    pub const fn new(candidate_index: usize, node: usize) -> Self {
        Self {
            candidate_index,
            node,
        }
    }

    #[must_use]
    pub const fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    #[must_use]
    pub const fn node(&self) -> usize {
        self.node
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalDisposition {
    Ready,
    DeferredUntil(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledSurvivor {
    candidate_index: usize,
    node: usize,
    feasible_time: i64,
}

impl ScheduledSurvivor {
    #[must_use]
    pub const fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    #[must_use]
    pub const fn node(&self) -> usize {
        self.node
    }

    #[must_use]
    pub const fn feasible_time(&self) -> i64 {
        self.feasible_time
    }

    #[must_use]
    pub const fn disposition_at(&self, cutoff: i64) -> TemporalDisposition {
        if self.feasible_time <= cutoff {
            TemporalDisposition::Ready
        } else {
            TemporalDisposition::DeferredUntil(self.feasible_time)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadinessSnapshot {
    cutoff: i64,
    ready_indices: Vec<usize>,
    deferred_indices: Vec<usize>,
}

impl ReadinessSnapshot {
    #[must_use]
    pub const fn cutoff(&self) -> i64 {
        self.cutoff
    }

    #[must_use]
    pub fn ready_indices(&self) -> &[usize] {
        &self.ready_indices
    }

    #[must_use]
    pub fn deferred_indices(&self) -> &[usize] {
        &self.deferred_indices
    }

    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.deferred_indices.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxPlusReadinessPlan {
    survivors_in_original_order: Vec<ScheduledSurvivor>,
    readiness_order: Vec<ScheduledSurvivor>,
    critical_ready_time: Option<i64>,
}

impl MaxPlusReadinessPlan {
    #[must_use]
    pub fn survivors_in_original_order(&self) -> &[ScheduledSurvivor] {
        &self.survivors_in_original_order
    }

    #[must_use]
    pub fn readiness_order(&self) -> &[ScheduledSurvivor] {
        &self.readiness_order
    }

    #[must_use]
    pub const fn critical_ready_time(&self) -> Option<i64> {
        self.critical_ready_time
    }

    #[must_use]
    pub fn snapshot_at(&self, cutoff: i64) -> ReadinessSnapshot {
        let mut ready_indices = Vec::with_capacity(self.survivors_in_original_order.len());
        let mut deferred_indices = Vec::new();

        for survivor in &self.survivors_in_original_order {
            match survivor.disposition_at(cutoff) {
                TemporalDisposition::Ready => ready_indices.push(survivor.candidate_index()),
                TemporalDisposition::DeferredUntil(_) => {
                    deferred_indices.push(survivor.candidate_index());
                }
            }
        }

        ReadinessSnapshot {
            cutoff,
            ready_indices,
            deferred_indices,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReadinessError {
    DuplicateBinding {
        candidate_index: usize,
    },
    BindingForRejectedCandidate {
        candidate_index: usize,
    },
    MissingBinding {
        candidate_index: usize,
    },
    NodeOutOfBounds {
        candidate_index: usize,
        node: usize,
        node_count: usize,
    },
    UnreachableSurvivor {
        candidate_index: usize,
        node: usize,
    },
    MaxPlus(MaxPlusError),
}

impl fmt::Display for ReadinessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateBinding { candidate_index } => write!(
                formatter,
                "candidate {candidate_index} has more than one max-plus readiness binding"
            ),
            Self::BindingForRejectedCandidate { candidate_index } => write!(
                formatter,
                "candidate {candidate_index} is not in the qualified survivor set"
            ),
            Self::MissingBinding { candidate_index } => write!(
                formatter,
                "qualified survivor {candidate_index} has no max-plus readiness binding"
            ),
            Self::NodeOutOfBounds {
                candidate_index,
                node,
                node_count,
            } => write!(
                formatter,
                "candidate {candidate_index} references max-plus node {node} outside 0..{node_count}"
            ),
            Self::UnreachableSurvivor {
                candidate_index,
                node,
            } => write!(
                formatter,
                "qualified survivor {candidate_index} is bound to unreachable max-plus node {node}"
            ),
            Self::MaxPlus(error) => write!(formatter, "max-plus readiness planning failed: {error}"),
        }
    }
}

impl std::error::Error for ReadinessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MaxPlus(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MaxPlusError> for ReadinessError {
    fn from(error: MaxPlusError) -> Self {
        Self::MaxPlus(error)
    }
}

/// Attach temporal readiness to an already-qualified survivor set.
///
/// Max-Plus is deliberately not used here as a relevance filter. Every survivor
/// must have exactly one reachable schedule node. A late survivor is deferred
/// until its feasible time and remains part of the eventual numerical candidate
/// set. An unreachable survivor fails closed instead of being silently dropped.
pub fn plan_survivor_readiness(
    survivor_set: &SurvivorSet,
    schedule: &MaxPlusSchedule,
    initial: &[MaxPlusValue],
    bindings: &[SurvivorNodeBinding],
) -> Result<MaxPlusReadinessPlan, ReadinessError> {
    let feasible_times = schedule.earliest_feasible_times(initial)?;
    let survivor_ids = survivor_set
        .survivor_indices()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut node_by_candidate = BTreeMap::new();

    for binding in bindings {
        if !survivor_ids.contains(&binding.candidate_index()) {
            return Err(ReadinessError::BindingForRejectedCandidate {
                candidate_index: binding.candidate_index(),
            });
        }
        if binding.node() >= schedule.node_count() {
            return Err(ReadinessError::NodeOutOfBounds {
                candidate_index: binding.candidate_index(),
                node: binding.node(),
                node_count: schedule.node_count(),
            });
        }
        if node_by_candidate
            .insert(binding.candidate_index(), binding.node())
            .is_some()
        {
            return Err(ReadinessError::DuplicateBinding {
                candidate_index: binding.candidate_index(),
            });
        }
    }

    let mut survivors_in_original_order = Vec::with_capacity(survivor_set.survivor_count());
    for &candidate_index in survivor_set.survivor_indices() {
        let node = *node_by_candidate
            .get(&candidate_index)
            .ok_or(ReadinessError::MissingBinding { candidate_index })?;
        let feasible_time =
            feasible_times[node]
                .as_finite()
                .ok_or(ReadinessError::UnreachableSurvivor {
                    candidate_index,
                    node,
                })?;
        survivors_in_original_order.push(ScheduledSurvivor {
            candidate_index,
            node,
            feasible_time,
        });
    }

    let mut readiness_order = survivors_in_original_order.clone();
    readiness_order
        .sort_unstable_by_key(|survivor| (survivor.feasible_time(), survivor.candidate_index()));
    let critical_ready_time = readiness_order.last().map(ScheduledSurvivor::feasible_time);

    Ok(MaxPlusReadinessPlan {
        survivors_in_original_order,
        readiness_order,
        critical_ready_time,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cooperation::{route_attention_needs, AttentionAlgebraNeeds};
    use crate::f2::{F2AffinePredicate, F2Vector};
    use crate::max_plus::MaxPlusEdge;
    use crate::qualification::{
        CandidateQualificationInputs, F2CandidateEvaluation, RecompositionPolicy,
    };
    use crate::survivor_set::{qualify_survivor_set, CandidateFrame};

    fn survivor_set(candidate_ids: &[usize]) -> SurvivorSet {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                parity_or_binary_linear: true,
                ..AttentionAlgebraNeeds::default()
            },
            None,
        )
        .unwrap();
        let predicate = F2AffinePredicate::new(F2Vector::from_bools(&[true]).unwrap(), false);
        let input = F2Vector::from_bools(&[true]).unwrap();
        let evaluation = F2CandidateEvaluation {
            predicate: &predicate,
            input: &input,
        };
        let frames = candidate_ids
            .iter()
            .copied()
            .map(|candidate_index| {
                CandidateFrame::new(
                    candidate_index,
                    CandidateQualificationInputs {
                        f2: Some(evaluation),
                        ..CandidateQualificationInputs::default()
                    },
                )
            })
            .collect::<Vec<_>>();
        qualify_survivor_set(&route, &frames, RecompositionPolicy::AllSelectedMustQualify).unwrap()
    }

    #[test]
    fn late_survivors_are_deferred_without_changing_the_eventual_set() {
        let survivors = survivor_set(&[0, 1, 2]);
        let schedule = MaxPlusSchedule::new(
            3,
            vec![MaxPlusEdge::new(0, 1, 3), MaxPlusEdge::new(1, 2, 4)],
        )
        .unwrap();
        let initial = [
            MaxPlusValue::Finite(0),
            MaxPlusValue::ZERO,
            MaxPlusValue::ZERO,
        ];
        let plan = plan_survivor_readiness(
            &survivors,
            &schedule,
            &initial,
            &[
                SurvivorNodeBinding::new(0, 0),
                SurvivorNodeBinding::new(1, 1),
                SurvivorNodeBinding::new(2, 2),
            ],
        )
        .unwrap();

        assert_eq!(plan.critical_ready_time(), Some(7));
        let early = plan.snapshot_at(2);
        assert_eq!(early.ready_indices(), &[0]);
        assert_eq!(early.deferred_indices(), &[1, 2]);
        assert!(!early.is_complete());

        let final_snapshot = plan.snapshot_at(7);
        assert_eq!(final_snapshot.ready_indices(), survivors.survivor_indices());
        assert!(final_snapshot.deferred_indices().is_empty());
        assert!(final_snapshot.is_complete());
    }

    #[test]
    fn readiness_order_is_separate_from_numerical_candidate_order() {
        let survivors = survivor_set(&[10, 5]);
        let schedule = MaxPlusSchedule::new(3, vec![MaxPlusEdge::new(0, 2, 4)]).unwrap();
        let initial = [
            MaxPlusValue::Finite(0),
            MaxPlusValue::ZERO,
            MaxPlusValue::ZERO,
        ];
        let plan = plan_survivor_readiness(
            &survivors,
            &schedule,
            &initial,
            &[
                SurvivorNodeBinding::new(10, 2),
                SurvivorNodeBinding::new(5, 0),
            ],
        )
        .unwrap();

        assert_eq!(
            plan.readiness_order()
                .iter()
                .map(ScheduledSurvivor::candidate_index)
                .collect::<Vec<_>>(),
            vec![5, 10]
        );
        assert_eq!(
            plan.survivors_in_original_order()
                .iter()
                .map(ScheduledSurvivor::candidate_index)
                .collect::<Vec<_>>(),
            vec![10, 5]
        );
        assert_eq!(plan.snapshot_at(4).ready_indices(), &[10, 5]);
    }

    #[test]
    fn duplicate_missing_and_foreign_bindings_fail_closed() {
        let survivors = survivor_set(&[0, 1]);
        let schedule = MaxPlusSchedule::new(2, vec![MaxPlusEdge::new(0, 1, 1)]).unwrap();
        let initial = [MaxPlusValue::Finite(0), MaxPlusValue::ZERO];

        assert_eq!(
            plan_survivor_readiness(
                &survivors,
                &schedule,
                &initial,
                &[
                    SurvivorNodeBinding::new(0, 0),
                    SurvivorNodeBinding::new(0, 1),
                ],
            ),
            Err(ReadinessError::DuplicateBinding { candidate_index: 0 })
        );
        assert_eq!(
            plan_survivor_readiness(
                &survivors,
                &schedule,
                &initial,
                &[SurvivorNodeBinding::new(0, 0)],
            ),
            Err(ReadinessError::MissingBinding { candidate_index: 1 })
        );
        assert_eq!(
            plan_survivor_readiness(
                &survivors,
                &schedule,
                &initial,
                &[
                    SurvivorNodeBinding::new(0, 0),
                    SurvivorNodeBinding::new(9, 1),
                ],
            ),
            Err(ReadinessError::BindingForRejectedCandidate { candidate_index: 9 })
        );
    }

    #[test]
    fn unreachable_survivor_is_an_error_not_a_relevance_rejection() {
        let survivors = survivor_set(&[0]);
        let schedule = MaxPlusSchedule::new(2, vec![]).unwrap();
        let initial = [MaxPlusValue::Finite(0), MaxPlusValue::ZERO];

        assert_eq!(
            plan_survivor_readiness(
                &survivors,
                &schedule,
                &initial,
                &[SurvivorNodeBinding::new(0, 1)],
            ),
            Err(ReadinessError::UnreachableSurvivor {
                candidate_index: 0,
                node: 1,
            })
        );
    }

    #[test]
    fn node_and_initial_geometry_fail_closed() {
        let survivors = survivor_set(&[0]);
        let schedule = MaxPlusSchedule::new(1, vec![]).unwrap();

        assert_eq!(
            plan_survivor_readiness(
                &survivors,
                &schedule,
                &[MaxPlusValue::Finite(0)],
                &[SurvivorNodeBinding::new(0, 1)],
            ),
            Err(ReadinessError::NodeOutOfBounds {
                candidate_index: 0,
                node: 1,
                node_count: 1,
            })
        );
        assert_eq!(
            plan_survivor_readiness(
                &survivors,
                &schedule,
                &[],
                &[SurvivorNodeBinding::new(0, 0)],
            ),
            Err(ReadinessError::MaxPlus(MaxPlusError::DimensionMismatch {
                expected: 1,
                actual: 0,
            }))
        );
    }

    #[test]
    fn empty_survivor_set_has_no_critical_time() {
        let survivors = survivor_set(&[]);
        let schedule = MaxPlusSchedule::new(1, vec![]).unwrap();
        let plan = plan_survivor_readiness(&survivors, &schedule, &[MaxPlusValue::Finite(0)], &[])
            .unwrap();

        assert_eq!(plan.critical_ready_time(), None);
        assert!(plan.survivors_in_original_order().is_empty());
        assert!(plan.readiness_order().is_empty());
        assert!(plan.snapshot_at(0).is_complete());
    }
}
