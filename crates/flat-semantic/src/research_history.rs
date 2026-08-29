//! Typed control-plane configuration for research-only history semantics.
//!
//! This module defines semantic identity and configuration only. It deliberately
//! provides no scalar oracle, kernel, backend route, or implicit fallback into
//! StandardSoftmax. Executable support must be added and qualified separately.

use core::num::NonZeroUsize;

use crate::v1::{
    MaskSemantics, SavedStateContract, SemanticDescriptor, SemanticFamily, SemanticId,
    StateSemantics, WeightSemantics,
};

/// Retention semantics of the research history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistoryMode {
    /// Keep the complete available history as the reference retention policy.
    CompleteReference,
    /// Keep at most `max_entries`; this is always an explicit approximation.
    Bounded {
        /// Maximum retained entries. Non-zero by construction.
        max_entries: NonZeroUsize,
    },
}

/// Deterministic schedule selecting which logical positions are admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistorySchedule {
    /// Admit every causally available logical position.
    EveryToken,
    /// Admit positions whose token index is an exact multiple of `every`.
    ///
    /// Selected entries must still carry their true original/absolute position;
    /// this stride is a selection rule, not a replacement coordinate system.
    Stride {
        /// Positive token-index stride.
        every: NonZeroUsize,
    },
}

/// Contextual weighting rule attached to retained history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HistoryWeighting {
    /// Exact multiplicative identity for every admitted history entry.
    Identity,
}

/// Explicit resource bound on one history evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistoryBudgetPolicy {
    /// No semantic contribution-count budget beyond retention/schedule rules.
    Unbounded,
    /// Evaluate at most this many retained contributions.
    ///
    /// Hitting this bound is an approximation/event that a future executable
    /// implementation must report explicitly rather than silently changing the rule.
    MaxContributions(NonZeroUsize),
}

/// Research-only nonlocal/recurrent attention configuration.
///
/// This type is intentionally absent from `flat_attention::api::v1::AttentionConfig`;
/// callers must explicitly choose the research semantic family to use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NonlocalAttentionConfig {
    /// Retention/reference-vs-approximation policy.
    pub history_mode: HistoryMode,
    /// Deterministic admission schedule over true logical positions.
    pub history_schedule: HistorySchedule,
    /// Contextual history weighting rule.
    pub history_weighting: HistoryWeighting,
    /// Explicit evaluation resource budget.
    pub history_budget_policy: HistoryBudgetPolicy,
}

impl Default for NonlocalAttentionConfig {
    fn default() -> Self {
        Self {
            history_mode: HistoryMode::CompleteReference,
            history_schedule: HistorySchedule::EveryToken,
            history_weighting: HistoryWeighting::Identity,
            history_budget_policy: HistoryBudgetPolicy::Unbounded,
        }
    }
}

impl NonlocalAttentionConfig {
    /// Whether this configuration requests any explicit history reduction.
    ///
    /// `false` means only that retention/schedule/budget are unreduced; it does
    /// not assert equivalence with StandardSoftmax or any other semantic rule.
    #[must_use]
    pub const fn is_history_reduced(self) -> bool {
        !matches!(self.history_mode, HistoryMode::CompleteReference)
            || !matches!(self.history_schedule, HistorySchedule::EveryToken)
            || !matches!(self.history_budget_policy, HistoryBudgetPolicy::Unbounded)
    }

    /// Deterministic semantic-instance record excluding kernel/device identity.
    #[must_use]
    pub fn canonical_record(self) -> String {
        let mode = match self.history_mode {
            HistoryMode::CompleteReference => "complete-reference".to_owned(),
            HistoryMode::Bounded { max_entries } => format!("bounded:{}", max_entries.get()),
        };
        let schedule = match self.history_schedule {
            HistorySchedule::EveryToken => "every-token".to_owned(),
            HistorySchedule::Stride { every } => format!("stride:{}", every.get()),
        };
        let weighting = match self.history_weighting {
            HistoryWeighting::Identity => "identity",
        };
        let budget = match self.history_budget_policy {
            HistoryBudgetPolicy::Unbounded => "unbounded".to_owned(),
            HistoryBudgetPolicy::MaxContributions(max) => {
                format!("max-contributions:{}", max.get())
            }
        };
        format!(
            "flat-research-history-v1;mode={mode};schedule={schedule};weighting={weighting};budget={budget}"
        )
    }
}

/// Non-executable semantic descriptor for the research history family.
///
/// Constructing this value does not register, select, or execute an implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NonlocalHistorySemantic {
    config: NonlocalAttentionConfig,
}

impl NonlocalHistorySemantic {
    /// Construct the research semantic from an explicit typed configuration.
    #[must_use]
    pub const fn new(config: NonlocalAttentionConfig) -> Self {
        Self { config }
    }

    /// Return the exact typed configuration.
    #[must_use]
    pub const fn config(self) -> NonlocalAttentionConfig {
        self.config
    }

    /// Stable descriptor of the mathematical/control-plane family.
    ///
    /// The descriptor is deliberately recurrent and causal. Weight semantics
    /// are state-dependent because future executable rules may depend on carried
    /// history state even when `HistoryWeighting::Identity` is selected.
    #[must_use]
    pub fn descriptor(self) -> SemanticDescriptor {
        SemanticDescriptor::new(
            SemanticId::new(SemanticFamily::RecurrentMemory, "nonlocal-history-research", 1)
                .expect("static research semantic identity is valid"),
            MaskSemantics::Causal,
            StateSemantics::Recurrent,
            WeightSemantics::StateDependent,
            SavedStateContract::None,
        )
    }

    /// Deterministic semantic-instance provenance record.
    #[must_use]
    pub fn canonical_record(self) -> String {
        format!(
            "family=recurrent-memory;name=nonlocal-history-research;revision=1;{}",
            self.config.canonical_record()
        )
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroUsize;

    use super::*;

    #[test]
    fn default_is_explicit_reference_configuration_not_standard_attention() {
        let config = NonlocalAttentionConfig::default();
        assert!(!config.is_history_reduced());
        let semantic = NonlocalHistorySemantic::new(config);
        let descriptor = semantic.descriptor();
        assert_eq!(descriptor.id().family(), SemanticFamily::RecurrentMemory);
        assert_eq!(descriptor.id().name(), "nonlocal-history-research");
        assert_eq!(descriptor.mask(), MaskSemantics::Causal);
        assert_eq!(descriptor.state(), StateSemantics::Recurrent);
    }

    #[test]
    fn every_reduction_is_visible_in_classification_and_record() {
        let one = NonZeroUsize::new(1).unwrap();
        let config = NonlocalAttentionConfig {
            history_mode: HistoryMode::Bounded { max_entries: one },
            history_schedule: HistorySchedule::Stride { every: one },
            history_weighting: HistoryWeighting::Identity,
            history_budget_policy: HistoryBudgetPolicy::MaxContributions(one),
        };
        assert!(config.is_history_reduced());
        assert_eq!(
            config.canonical_record(),
            "flat-research-history-v1;mode=bounded:1;schedule=stride:1;weighting=identity;budget=max-contributions:1"
        );
    }

    #[test]
    fn configuration_does_not_create_executable_support() {
        let semantic = NonlocalHistorySemantic::new(NonlocalAttentionConfig::default());
        assert_eq!(semantic.descriptor().saved_state(), SavedStateContract::None);
        assert!(semantic.canonical_record().contains("nonlocal-history-research"));
    }
}
