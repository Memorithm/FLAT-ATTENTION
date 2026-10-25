//! Host-only bfloat16 storage contract and scalar oracle (FML1 bf16 slice).
//!
//! `Bf16` is an interchange/storage type exactly like [`crate::F16`]: values
//! are decoded exactly to `f32` before any arithmetic, Q·K, online-softmax
//! state, P·V and LSE all stay in FP32, and only O is re-encoded with an
//! explicit round-to-nearest-even policy. No GPU executor is exposed and no
//! performance claim is made; see `docs/m8-precision.md`.

use core::fmt;

use crate::{
    forward_reference, forward_reference_grouped, AttentionShape, FlatAttentionConfig,
    FlatAttentionError, FlatAttentionOutput, GroupedAttentionShape,
};

/// Dependency-free bfloat16 storage value (upper 16 bits of an IEEE binary32).
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Bf16(u16);

impl Bf16 {
    pub const ZERO: Self = Self(0);

    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert an `f32` to bfloat16 using round-to-nearest, ties-to-even.
    ///
    /// Finite values that round past the largest bfloat16 become infinity.
    /// NaN stays NaN (sign and leading payload preserved, quiet bit forced) so
    /// that rounding can never turn a NaN into an infinity or a zero.
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if value.is_nan() {
            return Self(((bits >> 16) as u16) | 0x0040);
        }
        let lsb = (bits >> 16) & 1;
        Self((bits.wrapping_add(0x7fff + lsb) >> 16) as u16)
    }

    /// Convert bfloat16 to `f32` exactly.
    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits(u32::from(self.0) << 16)
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.0 & 0x7f80 != 0x7f80
    }
}

impl From<f32> for Bf16 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<Bf16> for f32 {
    fn from(value: Bf16) -> Self {
        value.to_f32()
    }
}

impl fmt::Debug for Bf16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Bf16")
            .field(&format_args!("0x{:04x}", self.0))
            .field(&self.to_f32())
            .finish()
    }
}

/// bfloat16 forward result: bfloat16 context output and FP32 LSE.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatAttentionBf16Output {
    /// Attention context tensor rounded to bfloat16 (round-to-nearest-even).
    pub output: Vec<Bf16>,
    /// Per-query log-sum-exp statistics kept in FP32 for backward recomputation.
    pub lse: Vec<f32>,
}

fn decode(values: &[Bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn encode(output: FlatAttentionOutput) -> FlatAttentionBf16Output {
    FlatAttentionBf16Output {
        output: output.output.into_iter().map(Bf16::from_f32).collect(),
        lse: output.lse,
    }
}

/// Deterministic bfloat16-I/O MHA oracle.
///
/// Defined as exact decode → [`forward_reference`] → RNE encode of O, so the
/// FP32 core is bit-identical to the qualified scalar oracle on the decoded
/// inputs. Length, finiteness, shape and score-overflow rejections are the
/// same typed errors as the FP32 oracle.
pub fn forward_reference_bf16(
    q: &[Bf16],
    k: &[Bf16],
    v: &[Bf16],
    shape: AttentionShape,
    config: FlatAttentionConfig,
) -> Result<FlatAttentionBf16Output, FlatAttentionError> {
    forward_reference(&decode(q), &decode(k), &decode(v), shape, config).map(encode)
}

/// Deterministic bfloat16-I/O GQA/MQA oracle without K/V head expansion.
///
/// Defined as exact decode → [`forward_reference_grouped`] → RNE encode of O.
pub fn forward_reference_grouped_bf16(
    q: &[Bf16],
    k: &[Bf16],
    v: &[Bf16],
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
) -> Result<FlatAttentionBf16Output, FlatAttentionError> {
    forward_reference_grouped(&decode(q), &decode(k), &decode(v), shape, config).map(encode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_bfloat16_encodings() {
        assert_eq!(Bf16::from_f32(0.0).to_bits(), 0x0000);
        assert_eq!(Bf16::from_f32(-0.0).to_bits(), 0x8000);
        assert_eq!(Bf16::from_f32(1.0).to_bits(), 0x3f80);
        assert_eq!(Bf16::from_f32(-2.0).to_bits(), 0xc000);
        assert_eq!(Bf16::from_f32(f32::INFINITY).to_bits(), 0x7f80);
        assert_eq!(Bf16::from_f32(f32::NEG_INFINITY).to_bits(), 0xff80);
        // Largest finite bfloat16.
        assert_eq!(Bf16::from_bits(0x7f7f).to_f32(), 3.389_531_4e38);
    }

    #[test]
    fn rounding_is_nearest_even() {
        // 1 + 2^-8 is exactly halfway between 1.0 (even) and 1 + 2^-7.
        assert_eq!(Bf16::from_f32(1.0 + 2f32.powi(-8)).to_bits(), 0x3f80);
        // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 (odd) and 1 + 2^-6 (even).
        assert_eq!(Bf16::from_f32(1.0 + 3.0 * 2f32.powi(-8)).to_bits(), 0x3f82);
        // Just above halfway rounds up.
        assert_eq!(
            Bf16::from_f32(1.0 + 2f32.powi(-8) + 2f32.powi(-20)).to_bits(),
            0x3f81
        );
        // f32::MAX rounds past the largest finite bfloat16.
        assert_eq!(Bf16::from_f32(f32::MAX).to_bits(), 0x7f80);
    }

    #[test]
    fn nan_never_rounds_to_infinity_or_zero() {
        for bits in [0x7f80_0001u32, 0x7fff_ffff, 0xffff_ffff, 0x7fc0_0000] {
            let encoded = Bf16::from_f32(f32::from_bits(bits));
            assert!(encoded.to_f32().is_nan(), "{bits:#x} -> {encoded:?}");
            assert!(!encoded.is_finite());
            assert_eq!(encoded.to_bits() & 0x8000, ((bits >> 16) as u16) & 0x8000);
        }
    }

    #[test]
    fn every_finite_bfloat16_roundtrips_exactly() {
        for bits in 0..=u16::MAX {
            let value = Bf16::from_bits(bits);
            if value.to_f32().is_nan() {
                continue;
            }
            assert_eq!(Bf16::from_f32(value.to_f32()), value, "{bits:#06x}");
        }
    }

    #[test]
    fn rounding_error_is_within_half_ulp() {
        for value in [
            -17.75f32,
            -1.0,
            -0.333_25,
            0.1,
            1.0,
            core::f32::consts::PI,
            1024.5,
            6.02e23,
            1.0e-30,
        ] {
            let decoded = Bf16::from_f32(value).to_f32();
            let relative = ((decoded - value) / value).abs();
            assert!(relative <= 2f32.powi(-8), "{value} -> {decoded}");
        }
    }
}
