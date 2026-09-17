//! Research-only target-page recall evidence for BKV-K6 Boolean selection.
//!
//! This module compares one already validated Boolean page-selection decision
//! with an independently declared set of logical pages that a dense reference
//! says must be retained for the experiment. It records exact set counts and
//! rational metrics only. It does not infer model quality, choose a threshold,
//! authorize promotion, or turn logical page recall into a performance claim.

use core::fmt;
use std::fmt::Write as _;

use crate::api::boolean_kv_paged_selection::{
    BooleanIndexedKvSelection, BooleanKvSelectionEvidenceError,
};
use crate::fingerprint::fnv1a64;

pub const BKV6_SELECTION_QUALITY_SCHEMA: &str = "flat.bikv-selection-quality.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bkv6SelectionQualityEvidence {
    selection_json: String,
    mapped_pages: usize,
    selected_pages: Vec<usize>,
    declared_dense_target_pages: Vec<usize>,
    true_positive_pages: usize,
    false_negative_pages: usize,
    false_positive_pages: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Bkv6SelectionQualityError {
    Selection(BooleanKvSelectionEvidenceError),
    EmptyMappedPageSet,
    EmptyDenseTarget,
    TargetOutOfRange {
        logical_page: usize,
        mapped_pages: usize,
    },
    TargetOrderOrDuplicate {
        previous: usize,
        current: usize,
    },
}

impl fmt::Display for Bkv6SelectionQualityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Selection(error) => write!(f, "invalid Boolean-KV selection evidence: {error}"),
            Self::EmptyMappedPageSet => write!(f, "selection quality requires at least one mapped page"),
            Self::EmptyDenseTarget => write!(
                f,
                "selection quality requires a non-empty independently declared dense target"
            ),
            Self::TargetOutOfRange {
                logical_page,
                mapped_pages,
            } => write!(
                f,
                "dense-target logical page {logical_page} is outside mapped page count {mapped_pages}"
            ),
            Self::TargetOrderOrDuplicate { previous, current } => write!(
                f,
                "dense-target pages must be strictly increasing and duplicate-free: {previous} then {current}"
            ),
        }
    }
}

impl std::error::Error for Bkv6SelectionQualityError {}

impl From<BooleanKvSelectionEvidenceError> for Bkv6SelectionQualityError {
    fn from(error: BooleanKvSelectionEvidenceError) -> Self {
        Self::Selection(error)
    }
}

impl Bkv6SelectionQualityEvidence {
    pub fn new(
        selection: &BooleanIndexedKvSelection,
        declared_dense_target_pages: Vec<usize>,
    ) -> Result<Self, Bkv6SelectionQualityError> {
        selection.validate_evidence()?;
        if selection.mapped_pages == 0 {
            return Err(Bkv6SelectionQualityError::EmptyMappedPageSet);
        }
        validate_target_pages(selection.mapped_pages, &declared_dense_target_pages)?;

        let selected_pages = selection.selected_page_ids();
        let true_positive_pages = intersection_count(&selected_pages, &declared_dense_target_pages);
        let false_negative_pages = declared_dense_target_pages.len() - true_positive_pages;
        let false_positive_pages = selected_pages.len() - true_positive_pages;

        Ok(Self {
            selection_json: selection.canonical_evidence_json_v2()?,
            mapped_pages: selection.mapped_pages,
            selected_pages,
            declared_dense_target_pages,
            true_positive_pages,
            false_negative_pages,
            false_positive_pages,
        })
    }

    /// Canonical embedded FLAT selection evidence used to derive this record.
    ///
    /// This accessor exists for fail-closed provenance binding. It does not
    /// authorize routing or reinterpret the quality record as model quality.
    #[must_use]
    pub fn selection_json(&self) -> &str {
        &self.selection_json
    }

    #[must_use]
    pub fn mapped_pages(&self) -> usize {
        self.mapped_pages
    }

    #[must_use]
    pub fn selected_pages(&self) -> &[usize] {
        &self.selected_pages
    }

    #[must_use]
    pub fn declared_dense_target_pages(&self) -> &[usize] {
        &self.declared_dense_target_pages
    }

    #[must_use]
    pub fn true_positive_pages(&self) -> usize {
        self.true_positive_pages
    }

    #[must_use]
    pub fn false_negative_pages(&self) -> usize {
        self.false_negative_pages
    }

    #[must_use]
    pub fn false_positive_pages(&self) -> usize {
        self.false_positive_pages
    }

    #[must_use]
    pub fn recall_fraction(&self) -> (usize, usize) {
        (
            self.true_positive_pages,
            self.declared_dense_target_pages.len(),
        )
    }

    #[must_use]
    pub fn false_negative_rate_fraction(&self) -> (usize, usize) {
        (
            self.false_negative_pages,
            self.declared_dense_target_pages.len(),
        )
    }

    #[must_use]
    pub fn candidate_density_fraction(&self) -> (usize, usize) {
        (self.selected_pages.len(), self.mapped_pages)
    }

    #[must_use]
    pub fn recall(&self) -> f64 {
        self.true_positive_pages as f64 / self.declared_dense_target_pages.len() as f64
    }

    #[must_use]
    pub fn false_negative_rate(&self) -> f64 {
        self.false_negative_pages as f64 / self.declared_dense_target_pages.len() as f64
    }

    /// Deterministic evidence encoding that retains the exact router decision
    /// and the independently declared target set.
    ///
    /// The target set is an experimental input, not a result inferred by FLAT.
    #[must_use]
    pub fn canonical_json(&self) -> String {
        let mut payload = String::with_capacity(self.selection_json.len() + 384);
        write!(
            payload,
            "{{\"schema\":\"{}\",\"selection\":{},\"declared_dense_target_pages\":",
            BKV6_SELECTION_QUALITY_SCHEMA, self.selection_json
        )
        .expect("writing to String cannot fail");
        write_usize_array(&mut payload, &self.declared_dense_target_pages);
        let target_pages = self.declared_dense_target_pages.len();
        let selected_pages = self.selected_pages.len();
        write!(
            payload,
            ",\"metrics\":{{\"mapped_pages\":{},\"selected_pages\":{},\"target_pages\":{},\"true_positive_pages\":{},\"false_negative_pages\":{},\"false_positive_pages\":{},\"recall\":{{\"numerator\":{},\"denominator\":{}}},\"false_negative_rate\":{{\"numerator\":{},\"denominator\":{}}},\"candidate_density\":{{\"numerator\":{},\"denominator\":{}}}}}}}",
            self.mapped_pages,
            selected_pages,
            target_pages,
            self.true_positive_pages,
            self.false_negative_pages,
            self.false_positive_pages,
            self.true_positive_pages,
            target_pages,
            self.false_negative_pages,
            target_pages,
            selected_pages,
            self.mapped_pages,
        )
        .expect("writing to String cannot fail");
        let closing_brace = payload.pop();
        debug_assert_eq!(closing_brace, Some('}'));
        let checksum = fnv1a64(payload.as_bytes());
        write!(
            payload,
            ",\"quality_checksum\":{{\"algorithm\":\"fnv1a64\",\"value\":\"{checksum:016x}\"}}}}"
        )
        .expect("writing to String cannot fail");
        payload
    }
}

fn validate_target_pages(
    mapped_pages: usize,
    target_pages: &[usize],
) -> Result<(), Bkv6SelectionQualityError> {
    if target_pages.is_empty() {
        return Err(Bkv6SelectionQualityError::EmptyDenseTarget);
    }
    let mut previous = None;
    for &logical_page in target_pages {
        if logical_page >= mapped_pages {
            return Err(Bkv6SelectionQualityError::TargetOutOfRange {
                logical_page,
                mapped_pages,
            });
        }
        if let Some(previous) = previous {
            if logical_page <= previous {
                return Err(Bkv6SelectionQualityError::TargetOrderOrDuplicate {
                    previous,
                    current: logical_page,
                });
            }
        }
        previous = Some(logical_page);
    }
    Ok(())
}

fn intersection_count(left: &[usize], right: &[usize]) -> usize {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut count = 0;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            core::cmp::Ordering::Less => left_index += 1,
            core::cmp::Ordering::Greater => right_index += 1,
            core::cmp::Ordering::Equal => {
                count += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    count
}

fn write_usize_array(output: &mut String, values: &[usize]) {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").expect("writing to String cannot fail");
    }
    output.push(']');
}

#[cfg(test)]
mod tests {
    use super::{
        Bkv6SelectionQualityError, Bkv6SelectionQualityEvidence, BKV6_SELECTION_QUALITY_SCHEMA,
    };
    use crate::api::boolean_kv::{BooleanKvCache, PackedBooleanSignature};
    use crate::api::boolean_kv_paged_selection::{
        build_boolean_indexed_kv_selection, BooleanIndexedKvSelection, NumericalKvPageGeometry,
    };
    use crate::paged_kv::{PagedKvConfig, PagedKvTable};

    fn signature(byte: u8) -> PackedBooleanSignature {
        let bits = (0..8).map(|bit| byte & (1 << bit) != 0).collect::<Vec<_>>();
        PackedBooleanSignature::from_bools(&bits).unwrap()
    }

    fn selection() -> BooleanIndexedKvSelection {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 4,
        })
        .unwrap();
        table.append(10).unwrap();

        let mut cache = BooleanKvCache::new(8).unwrap();
        cache.append(signature(0), None).unwrap();
        cache.append(signature(0xff), None).unwrap();
        cache.append(signature(0x01), None).unwrap();

        build_boolean_indexed_kv_selection(
            &cache,
            &table,
            &signature(0),
            1,
            None,
            NumericalKvPageGeometry {
                kv_heads: 2,
                head_dim: 4,
                scalar_bytes: 4,
            },
        )
        .unwrap()
    }

    #[test]
    fn computes_exact_target_page_recall_without_inference() {
        let selection = selection();
        assert_eq!(selection.selected_page_ids(), vec![0, 2]);

        let evidence = Bkv6SelectionQualityEvidence::new(&selection, vec![0, 1]).unwrap();
        assert_eq!(evidence.mapped_pages(), 3);
        assert_eq!(evidence.selected_pages(), &[0, 2]);
        assert_eq!(evidence.declared_dense_target_pages(), &[0, 1]);
        assert_eq!(evidence.true_positive_pages(), 1);
        assert_eq!(evidence.false_negative_pages(), 1);
        assert_eq!(evidence.false_positive_pages(), 1);
        assert_eq!(evidence.recall_fraction(), (1, 2));
        assert_eq!(evidence.false_negative_rate_fraction(), (1, 2));
        assert_eq!(evidence.candidate_density_fraction(), (2, 3));
        assert_eq!(evidence.recall(), 0.5);
        assert_eq!(evidence.false_negative_rate(), 0.5);

        let json = evidence.canonical_json();
        assert!(json.contains(BKV6_SELECTION_QUALITY_SCHEMA));
        assert!(json.contains("\"declared_dense_target_pages\":[0,1]"));
        assert!(json.contains("\"true_positive_pages\":1"));
        assert!(json.contains("\"false_negative_pages\":1"));
        assert!(json.contains("\"false_positive_pages\":1"));
        assert!(json.contains("\"candidate_density\":{\"numerator\":2,\"denominator\":3}"));
        assert!(json.contains("\"quality_checksum\":{\"algorithm\":\"fnv1a64\",\"value\":"));
    }

    #[test]
    fn outer_checksum_covers_target_and_derived_metrics() {
        fn checksum_value(json: &str) -> &str {
            let marker = "\"quality_checksum\":{\"algorithm\":\"fnv1a64\",\"value\":\"";
            let start = json.find(marker).unwrap() + marker.len();
            &json[start..start + 16]
        }

        let selection = selection();
        let one_target = Bkv6SelectionQualityEvidence::new(&selection, vec![0]).unwrap();
        let two_targets = Bkv6SelectionQualityEvidence::new(&selection, vec![0, 1]).unwrap();
        let one_json = one_target.canonical_json();
        let two_json = two_targets.canonical_json();
        assert_ne!(checksum_value(&one_json), checksum_value(&two_json));
    }

    #[test]
    fn all_target_hits_are_explicit_even_with_false_positives() {
        let selection = selection();
        let evidence = Bkv6SelectionQualityEvidence::new(&selection, vec![0]).unwrap();
        assert_eq!(evidence.recall_fraction(), (1, 1));
        assert_eq!(evidence.false_negative_rate_fraction(), (0, 1));
        assert_eq!(evidence.false_positive_pages(), 1);
    }

    #[test]
    fn empty_selection_can_record_complete_miss() {
        let mut selection = selection();
        selection.selected_pages.clear();
        selection.selected_numerical_kv_bytes = 0;
        selection.avoided_numerical_kv_bytes = selection.full_numerical_kv_bytes;
        let evidence = Bkv6SelectionQualityEvidence::new(&selection, vec![0, 2]).unwrap();
        assert_eq!(evidence.recall_fraction(), (0, 2));
        assert_eq!(evidence.false_negative_rate_fraction(), (2, 2));
        assert_eq!(evidence.candidate_density_fraction(), (0, 3));
    }

    #[test]
    fn malformed_or_undefined_targets_fail_closed() {
        let selection = selection();
        assert_eq!(
            Bkv6SelectionQualityEvidence::new(&selection, Vec::new()),
            Err(Bkv6SelectionQualityError::EmptyDenseTarget)
        );
        assert_eq!(
            Bkv6SelectionQualityEvidence::new(&selection, vec![1, 1]),
            Err(Bkv6SelectionQualityError::TargetOrderOrDuplicate {
                previous: 1,
                current: 1,
            })
        );
        assert_eq!(
            Bkv6SelectionQualityEvidence::new(&selection, vec![2, 1]),
            Err(Bkv6SelectionQualityError::TargetOrderOrDuplicate {
                previous: 2,
                current: 1,
            })
        );
        assert_eq!(
            Bkv6SelectionQualityEvidence::new(&selection, vec![3]),
            Err(Bkv6SelectionQualityError::TargetOutOfRange {
                logical_page: 3,
                mapped_pages: 3,
            })
        );
    }
}
