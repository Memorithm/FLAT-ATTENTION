//! RAE-3: paired, deterministic counterfactual page-selection audit.
//!
//! This experiment does not execute numerical attention or establish speedup.
//! Independently supplied required-page masks are the reference for omissions.
//! All five policies see identical sampled states, epochs and reference masks.

use core::fmt;

pub const RAE_POLICY_COUNT: usize = 5;
pub const RAE_MAX_STEPS: usize = 10_000;
pub const RAE_MAX_BITS: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaePolicy {
    AllPages,
    RisingOnly,
    AnyTransition,
    ActiveState,
    RisingWithContextInvalidation,
}

pub const RAE_POLICIES: [RaePolicy; RAE_POLICY_COUNT] = [
    RaePolicy::AllPages,
    RaePolicy::RisingOnly,
    RaePolicy::AnyTransition,
    RaePolicy::ActiveState,
    RaePolicy::RisingWithContextInvalidation,
];

#[derive(Debug, Clone, Copy)]
pub struct RaeStep<'a> {
    pub tick: u64,
    pub feature_state: &'a [u64],
    /// Reference page set supplied by a distinct, authoritative oracle.
    pub required_pages: &'a [u64],
    /// An external query/context invalidation; NEVER inferred from rising bits.
    pub context_invalidated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaeInputError {
    InvalidDimensions,
    EmptyOrExcessiveTrace,
    WrongMappingLength,
    InvalidMapping { feature: usize, page: usize },
    WrongWordCount { step: usize, actual: usize, expected: usize },
    NonZeroPadding { step: usize },
    NonIncreasingTick { step: usize },
}

impl fmt::Display for RaeInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RaeInputError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RaePolicyCounts {
    pub selected_pages: u128,
    pub required_pages: u128,
    pub missed_required_pages: u128,
    pub extra_selected_pages: u128,
}

impl RaePolicyCounts {
    fn accumulate(&mut self, other: Self) {
        self.selected_pages += other.selected_pages;
        self.required_pages += other.required_pages;
        self.missed_required_pages += other.missed_required_pages;
        self.extra_selected_pages += other.extra_selected_pages;
    }

    #[must_use]
    pub const fn has_omissions(self) -> bool {
        self.missed_required_pages != 0
    }
}

/// One row of policy masks, relative to exactly one independently supplied
/// required-page mask. Masks contain ORIGINAL page positions (not renumbered).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaeAuditRow {
    pub tick: u64,
    pub rising_features: u64,
    pub falling_features: u64,
    pub context_invalidated: bool,
    pub policy_masks: [Vec<u64>; RAE_POLICY_COUNT],
    pub counts: [RaePolicyCounts; RAE_POLICY_COUNT],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaeAudit {
    pub feature_width: usize,
    pub page_count: usize,
    pub rows: Vec<RaeAuditRow>,
    pub totals: [RaePolicyCounts; RAE_POLICY_COUNT],
    pub total_rising_features: u128,
    pub total_falling_features: u128,
}

impl RaeAudit {
    #[must_use]
    pub const fn policy_counts(&self, policy: RaePolicy) -> RaePolicyCounts {
        self.totals[policy_index(policy)]
    }
}

const fn policy_index(policy: RaePolicy) -> usize {
    match policy {
        RaePolicy::AllPages => 0,
        RaePolicy::RisingOnly => 1,
        RaePolicy::AnyTransition => 2,
        RaePolicy::ActiveState => 3,
        RaePolicy::RisingWithContextInvalidation => 4,
    }
}

/// Compare five policies on the same sampled bit stream. A "miss" counts a
/// page required by the external reference but not selected by a policy.
/// It is not a next-token error or numerical attention error. No policy is
/// authorized for production work elision by this host-only audit.
///
/// The hybrid policy wakes on any transition and re-examines currently
/// active pages on external context invalidation. It is NOT guaranteed
/// safe: a required page with no currently active feature may still be lost.
pub fn audit_paired_page_policies(
    feature_width: usize,
    page_count: usize,
    feature_pages: &[usize],
    baseline_tick: u64,
    baseline_state: &[u64],
    steps: &[RaeStep<'_>],
) -> Result<RaeAudit, RaeInputError> {
    if feature_width == 0
        || page_count == 0
        || feature_width > RAE_MAX_BITS
        || page_count > RAE_MAX_BITS
    {
        return Err(RaeInputError::InvalidDimensions);
    }
    if steps.is_empty() || steps.len() > RAE_MAX_STEPS {
        return Err(RaeInputError::EmptyOrExcessiveTrace);
    }
    if feature_pages.len() != feature_width {
        return Err(RaeInputError::WrongMappingLength);
    }
    for (feature, page) in feature_pages.iter().copied().enumerate() {
        if page >= page_count {
            return Err(RaeInputError::InvalidMapping { feature, page });
        }
    }
    validate_mask(feature_width, baseline_state, 0)?;

    let feature_words = words_for(feature_width);
    let page_words = words_for(page_count);
    let mut previous = baseline_state.to_vec();
    let mut last_tick = baseline_tick;
    let mut all_mask = vec![u64::MAX; page_words];
    if page_count % 64 != 0 {
        all_mask[page_words - 1] = (1_u64 << (page_count % 64)) - 1;
    }

    let mut rows = Vec::with_capacity(steps.len());
    let mut totals = [RaePolicyCounts::default(); RAE_POLICY_COUNT];
    let mut total_rising_features = 0_u128;
    let mut total_falling_features = 0_u128;

    for (index, step) in steps.iter().enumerate() {
        let step_number = index + 1;
        if step.tick <= last_tick {
            return Err(RaeInputError::NonIncreasingTick { step: step_number });
        }
        validate_mask(feature_width, step.feature_state, step_number)?;
        validate_mask(page_count, step.required_pages, step_number)?;

        let mut rises = vec![0_u64; feature_words];
        let mut falls = vec![0_u64; feature_words];
        for (word, ((previous_word, current), (rise, fall))) in previous
            .iter()
            .zip(step.feature_state)
            .zip(rises.iter_mut().zip(falls.iter_mut()))
            .enumerate()
        {
            debug_assert!(word < feature_words);
            *rise = *current & !*previous_word;
            *fall = *previous_word & !*current;
        }
        let rising_features = count_set(&rises);
        let falling_features = count_set(&falls);
        total_rising_features += u128::from(rising_features);
        total_falling_features += u128::from(falling_features);

        let rising_pages = project_pages(&rises, page_words, feature_pages);
        let falling_pages = project_pages(&falls, page_words, feature_pages);
        let active_pages = project_pages(step.feature_state, page_words, feature_pages);
        let mut any_transition = rising_pages.clone();
        for (destination, falling) in any_transition.iter_mut().zip(&falling_pages) {
            *destination |= *falling;
        }

        let mut hybrid = any_transition.clone();
        if step.context_invalidated {
            for (destination, active) in hybrid.iter_mut().zip(&active_pages) {
                *destination |= *active;
            }
        }
        let masks = [
            all_mask.clone(),
            rising_pages,
            any_transition,
            active_pages,
            hybrid,
        ];

        let counts = masks.map(|selected| calculate_counts(&selected, step.required_pages));
        for (total, row) in totals.iter_mut().zip(counts) {
            total.accumulate(row);
        }
        rows.push(RaeAuditRow {
            tick: step.tick,
            rising_features,
            falling_features,
            context_invalidated: step.context_invalidated,
            policy_masks: masks,
            counts,
        });
        previous.copy_from_slice(step.feature_state);
        last_tick = step.tick;
    }

    Ok(RaeAudit {
        feature_width,
        page_count,
        rows,
        totals,
        total_rising_features,
        total_falling_features,
    })
}

fn words_for(width: usize) -> usize {
    width / 64 + usize::from(width % 64 != 0)
}

fn validate_mask(width: usize, words: &[u64], step: usize) -> Result<(), RaeInputError> {
    let expected = words_for(width);
    if words.len() != expected {
        return Err(RaeInputError::WrongWordCount {
            step,
            actual: words.len(),
            expected,
        });
    }
    if width % 64 != 0 {
        let mask = (1_u64 << (width % 64)) - 1;
        if words[expected - 1] & !mask != 0 {
            return Err(RaeInputError::NonZeroPadding { step });
        }
    }
    Ok(())
}

fn project_pages(state: &[u64], page_words: usize, feature_pages: &[usize]) -> Vec<u64> {
    let mut pages = vec![0_u64; page_words];
    for (word_index, word) in state.iter().copied().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let feature = word_index * 64 + remaining.trailing_zeros() as usize;
            let page = feature_pages[feature];
            pages[page / 64] |= 1_u64 << (page % 64);
            remaining &= remaining - 1;
        }
    }
    pages
}

fn count_set(words: &[u64]) -> u64 {
    words.iter().map(|word| u64::from(word.count_ones())).sum()
}

fn calculate_counts(selected: &[u64], required: &[u64]) -> RaePolicyCounts {
    let mut counts = RaePolicyCounts::default();
    for (&chosen, &needed) in selected.iter().zip(required) {
        counts.selected_pages += u128::from(chosen.count_ones());
        counts.required_pages += u128::from(needed.count_ones());
        counts.missed_required_pages += u128::from((needed & !chosen).count_ones());
        counts.extra_selected_pages += u128::from((chosen & !needed).count_ones());
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_steps() -> [RaeStep<'static>; 4] {
        [
            RaeStep {
                tick: 1,
                feature_state: &[0b01],
                required_pages: &[0b01],
                context_invalidated: false,
            },
            RaeStep {
                tick: 2,
                feature_state: &[0b01],
                required_pages: &[0b01],
                context_invalidated: false,
            },
            RaeStep {
                tick: 3,
                feature_state: &[0b01],
                required_pages: &[0b10],
                context_invalidated: true,
            },
            RaeStep {
                tick: 4,
                feature_state: &[0b00],
                required_pages: &[0],
                context_invalidated: false,
            },
        ]
    }

    #[test]
    fn four_arms_show_unchanged_state_hazard_and_query_invalidation() {
        let audit = audit_paired_page_policies(2, 2, &[0, 1], 0, &[0], &sample_steps()).unwrap();
        assert_eq!(audit.rows.len(), 4);
        assert_eq!(audit.total_rising_features, 1);
        assert_eq!(audit.total_falling_features, 1);

        assert_eq!(audit.policy_counts(RaePolicy::AllPages).missed_required_pages, 0);
        assert_eq!(audit.rows[0].policy_masks[1], vec![0b01]);
        assert_eq!(audit.rows[1].policy_masks[1], vec![0]);
        assert_eq!(audit.rows[1].counts[1].missed_required_pages, 1);
        assert_eq!(audit.rows[2].policy_masks[4], vec![0b01]);
        assert_eq!(audit.rows[2].counts[4].missed_required_pages, 1);
        assert_eq!(audit.rows[2].counts[3].missed_required_pages, 1);
        assert_eq!(audit.rows[3].policy_masks[2], vec![0b01]);
        assert_eq!(audit.rows[3].policy_masks[1], vec![0]);
    }

    #[test]
    fn common_destination_coalesces_and_65_bit_tail_is_exact() {
        let mut map = vec![0; 65];
        map[63] = 64;
        map[64] = 64;
        let steps = [RaeStep {
            tick: 8,
            feature_state: &[1_u64 << 63, 1],
            required_pages: &[0, 1],
            context_invalidated: false,
        }];
        let audit = audit_paired_page_policies(65, 65, &map, 7, &[0, 0], &steps).unwrap();
        assert_eq!(audit.total_rising_features, 2);
        assert_eq!(audit.rows[0].policy_masks[1], vec![0, 1]);
        assert_eq!(audit.rows[0].counts[1].selected_pages, 1);
        assert_eq!(audit.rows[0].counts[1].missed_required_pages, 0);
    }

    #[test]
    fn malformed_padding_and_nonmonotonic_ticks_fail_closed() {
        let invalid_padding = [RaeStep {
            tick: 1,
            feature_state: &[0b100],
            required_pages: &[1],
            context_invalidated: false,
        }];
        assert_eq!(
            audit_paired_page_policies(2, 2, &[0, 1], 0, &[0], &invalid_padding),
            Err(RaeInputError::NonZeroPadding { step: 1 })
        );
        let backward = [RaeStep {
            tick: 1,
            feature_state: &[0],
            required_pages: &[0],
            context_invalidated: false,
        }];
        assert_eq!(
            audit_paired_page_policies(2, 2, &[0, 1], 1, &[0], &backward),
            Err(RaeInputError::NonIncreasingTick { step: 1 })
        );
        assert_eq!(
            audit_paired_page_policies(2, 2, &[0, 2], 0, &[0], &sample_steps()),
            Err(RaeInputError::InvalidMapping { feature: 1, page: 2 })
        );
    }

    #[test]
    fn invalid_reference_page_padding_is_not_silently_accepted() {
        let steps = [RaeStep {
            tick: 1,
            feature_state: &[1],
            required_pages: &[0b10],
            context_invalidated: false,
        }];
        assert_eq!(
            audit_paired_page_policies(1, 1, &[0], 0, &[0], &steps),
            Err(RaeInputError::NonZeroPadding { step: 1 })
        );
    }
}
