//! Explicit comparison tolerances for FLAT numerical modes.
//!
//! These constants describe **how a test or qualifier may compare** a result
//! against the scalar oracle. They do not change kernel arithmetic.
//!
//! - [`EXACT_REFERENCE_BITS`]: `NumericalMode::ExactReference` repeats must
//!   match `f32::to_bits()` on the same build/runtime/platform contract.
//! - [`FAST_PORTABLE_ABS_ATOL`] / [`FAST_PORTABLE_REL_RTOL`]: comparison
//!   against the oracle when a GPU path may change reduction order.
//! - Packed-f16 I/O keeps a wider absolute floor because values are rounded
//!   before they re-enter FP32 accumulation.

/// Bit-exact comparison. Used only for ExactReference and same-device
/// DeterministicPortable repeats.
pub const EXACT_REFERENCE_BITS: bool = true;

/// Absolute floor for FastPortable f32 vs ExactReference comparisons.
pub const FAST_PORTABLE_ABS_ATOL: f32 = 1.0e-4;

/// Relative tolerance for FastPortable f32 vs ExactReference comparisons.
pub const FAST_PORTABLE_REL_RTOL: f32 = 1.0e-3;

/// Absolute floor for packed-binary16 I/O paths that promote to FP32.
pub const PACKED_F16_ABS_ATOL: f32 = 2.0e-3;

/// Relative tolerance for packed-binary16 I/O paths.
pub const PACKED_F16_REL_RTOL: f32 = 5.0e-3;

/// Compare two finite f32 values with an absolute+relative band.
#[must_use]
pub fn within_tol(actual: f32, expected: f32, atol: f32, rtol: f32) -> bool {
    if actual.to_bits() == expected.to_bits() {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    let diff = (actual - expected).abs();
    diff <= atol.max(rtol * expected.abs())
}

/// Compare two slices with [`within_tol`].
#[must_use]
pub fn slices_within_tol(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&a, &e)| within_tol(a, e, atol, rtol))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_identical_values_pass_any_band() {
        assert!(within_tol(1.0, 1.0, 0.0, 0.0));
        assert!(slices_within_tol(&[0.5, -2.0], &[0.5, -2.0], 0.0, 0.0));
    }

    #[test]
    fn fast_portable_band_accepts_small_reduction_noise() {
        let expected = 1.0f32;
        let actual = expected + FAST_PORTABLE_ABS_ATOL * 0.5;
        assert!(within_tol(
            actual,
            expected,
            FAST_PORTABLE_ABS_ATOL,
            FAST_PORTABLE_REL_RTOL
        ));
    }

    #[test]
    fn non_finite_never_matches_a_finite_expected() {
        assert!(!within_tol(f32::NAN, 0.0, 1.0, 1.0));
        assert!(!within_tol(f32::INFINITY, 1.0, 1.0, 1.0));
    }
}
