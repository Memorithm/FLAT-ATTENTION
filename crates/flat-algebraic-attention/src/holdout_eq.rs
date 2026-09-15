//! Equality support for confirmatory holdout bindings.
//!
//! Kept separate from the holdout contract so the binding stays a lightweight
//! borrowed capability while tests can compare `Result<ConfirmatoryBinding, _>`
//! values without changing any confirmation semantics.

use crate::holdout::ConfirmatoryBinding;

impl<'a> PartialEq for ConfirmatoryBinding<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.manifest() == other.manifest()
    }
}

impl Eq for ConfirmatoryBinding<'_> {}
