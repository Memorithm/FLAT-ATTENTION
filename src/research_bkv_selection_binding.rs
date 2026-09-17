//! Fail-closed binding between one Boolean-KV page selection and BKV-K6 evidence.
//!
//! The aggregate BKV-K6 qualification record is not sufficient provenance for
//! the exact page decision that produced it. This research-only layer binds the
//! canonical `flat.boolean-kv-selection.v2` envelope to the canonical BKV-K6
//! qualification envelope and rejects any accounting drift between them.
//! It changes no runtime routing and carries no performance claim.

use core::fmt;
use std::fmt::Write as _;

use crate::api::boolean_kv_paged_selection::{
    BooleanIndexedKvSelection, BooleanKvSelectionEvidenceError,
};
use crate::research_bkv_evidence::{BikvEvidenceError, BikvEvidenceManifest};

pub const BIKV_SELECTION_BINDING_SCHEMA: &str = "flat.bikv-selection-binding.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BikvSelectionEvidenceBinding {
    selection_json: String,
    qualification_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BikvSelectionBindingError {
    Selection(BooleanKvSelectionEvidenceError),
    Qualification(BikvEvidenceError),
    AccountingMismatch(&'static str),
    SelectedLiveTokenOverflow,
    IntegerConversionOverflow(&'static str),
}

impl fmt::Display for BikvSelectionBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Selection(error) => write!(f, "invalid Boolean-KV selection evidence: {error}"),
            Self::Qualification(error) => {
                write!(f, "invalid BKV-K6 qualification evidence: {error}")
            }
            Self::AccountingMismatch(field) => write!(
                f,
                "Boolean-KV selection does not match BKV-K6 qualification field {field}"
            ),
            Self::SelectedLiveTokenOverflow => {
                write!(f, "Boolean-KV selected live-token accounting overflow")
            }
            Self::IntegerConversionOverflow(field) => write!(
                f,
                "Boolean-KV selection field {field} does not fit BKV-K6 u64 accounting"
            ),
        }
    }
}

impl std::error::Error for BikvSelectionBindingError {}

impl From<BooleanKvSelectionEvidenceError> for BikvSelectionBindingError {
    fn from(error: BooleanKvSelectionEvidenceError) -> Self {
        Self::Selection(error)
    }
}

impl From<BikvEvidenceError> for BikvSelectionBindingError {
    fn from(error: BikvEvidenceError) -> Self {
        Self::Qualification(error)
    }
}

impl BikvSelectionEvidenceBinding {
    pub fn new(
        selection: &BooleanIndexedKvSelection,
        qualification: &BikvEvidenceManifest,
    ) -> Result<Self, BikvSelectionBindingError> {
        selection.validate_evidence()?;
        qualification.validate()?;
        validate_accounting_binding(selection, qualification)?;

        Ok(Self {
            selection_json: selection.canonical_evidence_json_v2()?,
            qualification_json: qualification.canonical_json()?,
        })
    }

    #[must_use]
    pub fn selection_json(&self) -> &str {
        &self.selection_json
    }

    #[must_use]
    pub fn qualification_json(&self) -> &str {
        &self.qualification_json
    }
    pub fn canonical_json(&self) -> String {
        let mut payload =
            String::with_capacity(self.selection_json.len() + self.qualification_json.len() + 192);
        write!(
            payload,
            "{{\"schema\":\"{}\",\"selection\":{},\"qualification\":{}",
            BIKV_SELECTION_BINDING_SCHEMA, self.selection_json, self.qualification_json
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

fn validate_accounting_binding(
    selection: &BooleanIndexedKvSelection,
    qualification: &BikvEvidenceManifest,
) -> Result<(), BikvSelectionBindingError> {
    let accounting = qualification.qualification.accounting();
    if selection.signature_bits != qualification.signature_bits {
        return Err(BikvSelectionBindingError::AccountingMismatch(
            "signature_bits",
        ));
    }
    if selection.max_distance != qualification.max_distance {
        return Err(BikvSelectionBindingError::AccountingMismatch(
            "max_distance",
        ));
    }
    if selection.live_tokens != accounting.live_tokens {
        return Err(BikvSelectionBindingError::AccountingMismatch("live_tokens"));
    }
    if selection.mapped_pages != accounting.mapped_pages {
        return Err(BikvSelectionBindingError::AccountingMismatch(
            "mapped_pages",
        ));
    }
    if selection.selected_pages.len() != accounting.selected_pages {
        return Err(BikvSelectionBindingError::AccountingMismatch(
            "selected_pages",
        ));
    }
    let selected_live_tokens = selection
        .selected_pages
        .iter()
        .try_fold(0usize, |sum, page| {
            sum.checked_add(page.live_tokens)
                .ok_or(BikvSelectionBindingError::SelectedLiveTokenOverflow)
        })?;
    if selected_live_tokens != accounting.selected_live_tokens {
        return Err(BikvSelectionBindingError::AccountingMismatch(
            "selected_live_tokens",
        ));
    }
    if usize_to_u64(selection.boolean_key_bytes_read, "boolean_key_bytes_read")?
        != accounting.boolean_index_bytes_read
    {
        return Err(BikvSelectionBindingError::AccountingMismatch(
            "boolean_index_bytes_read",
        ));
    }
    if usize_to_u64(
        selection.numerical_kv_bytes_per_token,
        "numerical_kv_bytes_per_token",
    )? != qualification.qualification.kv_bytes_per_token()
    {
        return Err(BikvSelectionBindingError::AccountingMismatch(
            "kv_bytes_per_token",
        ));
    }
    for (field, selection_value, qualification_value) in [
        (
            "full_numerical_kv_bytes",
            selection.full_numerical_kv_bytes,
            qualification.qualification.dense_numerical_kv_bytes(),
        ),
        (
            "selected_numerical_kv_bytes",
            selection.selected_numerical_kv_bytes,
            qualification.qualification.selected_numerical_kv_bytes(),
        ),
        (
            "avoided_numerical_kv_bytes",
            selection.avoided_numerical_kv_bytes,
            qualification.qualification.avoided_numerical_kv_bytes(),
        ),
    ] {
        if usize_to_u64(selection_value, field)? != qualification_value {
            return Err(BikvSelectionBindingError::AccountingMismatch(field));
        }
    }
    Ok(())
}

fn usize_to_u64(value: usize, field: &'static str) -> Result<u64, BikvSelectionBindingError> {
    u64::try_from(value).map_err(|_| BikvSelectionBindingError::IntegerConversionOverflow(field))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
