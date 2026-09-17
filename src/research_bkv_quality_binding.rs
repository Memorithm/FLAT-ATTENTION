//! Fail-closed provenance binding for BKV-K6 selection-quality evidence.
//!
//! `flat.bikv-selection-quality.v1` records target-page recall/FNR for one
//! exact Boolean-KV page selection, while `flat.bikv-selection-binding.v1`
//! binds that same selection to the measured BKV-K6 qualification envelope.
//! This module joins those two evidence surfaces without inventing a quality
//! threshold or changing the independent `quality_gate_passed` semantics.
//! It changes no runtime routing and carries no performance or model-quality
//! claim.

use core::fmt;
use std::fmt::Write as _;

use crate::api::bkv6_selection_quality::Bkv6SelectionQualityEvidence;
use crate::api::boolean_kv_paged_selection::{
    BooleanIndexedKvSelection, BooleanKvSelectionEvidenceError,
};
use crate::research_bkv_evidence::BikvEvidenceManifest;
use crate::research_bkv_selection_binding::{
    BikvSelectionBindingError, BikvSelectionEvidenceBinding,
};

pub const BIKV_SELECTION_QUALITY_BINDING_SCHEMA: &str = "flat.bikv-selection-quality-binding.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BikvSelectionQualityBinding {
    selection_quality_json: String,
    selection_qualification_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BikvSelectionQualityBindingError {
    Selection(BooleanKvSelectionEvidenceError),
    SelectionQualification(BikvSelectionBindingError),
    QualitySelectionMismatch,
}

impl fmt::Display for BikvSelectionQualityBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Selection(error) => write!(f, "invalid Boolean-KV selection evidence: {error}"),
            Self::SelectionQualification(error) => {
                write!(f, "invalid selection/qualification binding: {error}")
            }
            Self::QualitySelectionMismatch => write!(
                f,
                "BKV-K6 selection-quality evidence does not embed the exact selection being qualified"
            ),
        }
    }
}

impl std::error::Error for BikvSelectionQualityBindingError {}

impl From<BooleanKvSelectionEvidenceError> for BikvSelectionQualityBindingError {
    fn from(error: BooleanKvSelectionEvidenceError) -> Self {
        Self::Selection(error)
    }
}

impl From<BikvSelectionBindingError> for BikvSelectionQualityBindingError {
    fn from(error: BikvSelectionBindingError) -> Self {
        Self::SelectionQualification(error)
    }
}

impl BikvSelectionQualityBinding {
    pub fn new(
        selection: &BooleanIndexedKvSelection,
        quality: &Bkv6SelectionQualityEvidence,
        qualification: &BikvEvidenceManifest,
    ) -> Result<Self, BikvSelectionQualityBindingError> {
        selection.validate_evidence()?;
        let selection_json = selection.canonical_evidence_json_v2()?;
        if quality.selection_json() != selection_json {
            return Err(BikvSelectionQualityBindingError::QualitySelectionMismatch);
        }

        let selection_qualification = BikvSelectionEvidenceBinding::new(selection, qualification)?;

        Ok(Self {
            selection_quality_json: quality.canonical_json(),
            selection_qualification_json: selection_qualification.canonical_json(),
        })
    }

    #[must_use]
    pub fn selection_quality_json(&self) -> &str {
        &self.selection_quality_json
    }

    #[must_use]
    pub fn selection_qualification_json(&self) -> &str {
        &self.selection_qualification_json
    }

    #[must_use]
    pub fn canonical_json(&self) -> String {
        let mut payload = String::with_capacity(
            self.selection_quality_json.len() + self.selection_qualification_json.len() + 224,
        );
        write!(
            payload,
            "{{\"schema\":\"{}\",\"selection_quality\":{},\"selection_qualification\":{}",
            BIKV_SELECTION_QUALITY_BINDING_SCHEMA,
            self.selection_quality_json,
            self.selection_qualification_json
        )
        .expect("writing to String cannot fail");
        let checksum = fnv1a64(payload.as_bytes());
        write!(
            payload,
            ",\"binding_checksum\":{{\"algorithm\":\"fnv1a64\",\"value\":\"{checksum:016x}\"}}}}"
        )
        .expect("writing to String cannot fail");
        payload
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
