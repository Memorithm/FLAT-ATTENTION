use core::fmt;

/// Dependency-free IEEE-754 binary16 storage value.
///
/// `F16` is intentionally an interchange/storage type. FLAT-ATTENTION promotes
/// values to `f32` before dot products, online-softmax updates and output
/// accumulation.
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct F16(u16);

impl F16 {
    pub const ZERO: Self = Self(0);

    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert an `f32` to IEEE binary16 using round-to-nearest, ties-to-even.
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x007f_ffff;

        if exponent == 0xff {
            if mantissa == 0 {
                return Self(sign | 0x7c00);
            }
            let mut payload = (mantissa >> 13) as u16;
            if payload == 0 {
                payload = 1;
            }
            return Self(sign | 0x7c00 | payload);
        }

        let mut half_exponent = exponent - 127 + 15;
        if half_exponent >= 31 {
            return Self(sign | 0x7c00);
        }

        if half_exponent <= 0 {
            if half_exponent < -10 {
                return Self(sign);
            }

            let mantissa = mantissa | 0x0080_0000;
            let shift = (14 - half_exponent) as u32;
            let mut half_mantissa = mantissa >> shift;
            let remainder_mask = (1u32 << shift) - 1;
            let remainder = mantissa & remainder_mask;
            let halfway = 1u32 << (shift - 1);
            if remainder > halfway || (remainder == halfway && half_mantissa & 1 != 0) {
                half_mantissa += 1;
            }
            return Self(sign | half_mantissa as u16);
        }

        let mut half_mantissa = mantissa >> 13;
        let remainder = mantissa & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && half_mantissa & 1 != 0) {
            half_mantissa += 1;
            if half_mantissa == 0x400 {
                half_mantissa = 0;
                half_exponent += 1;
                if half_exponent >= 31 {
                    return Self(sign | 0x7c00);
                }
            }
        }

        Self(sign | ((half_exponent as u16) << 10) | half_mantissa as u16)
    }

    /// Convert IEEE binary16 to `f32` exactly.
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = ((bits & 0x8000) as u32) << 16;
        let exponent = (bits >> 10) & 0x1f;
        let mantissa = bits & 0x03ff;

        let output = if exponent == 0 {
            if mantissa == 0 {
                sign
            } else {
                let mut normalized = mantissa as u32;
                let mut exponent32 = 127 - 14;
                while normalized & 0x400 == 0 {
                    normalized <<= 1;
                    exponent32 -= 1;
                }
                normalized &= 0x03ff;
                sign | ((exponent32 as u32) << 23) | (normalized << 13)
            }
        } else if exponent == 0x1f {
            sign | 0x7f80_0000 | ((mantissa as u32) << 13)
        } else {
            let exponent32 = exponent as u32 + (127 - 15);
            sign | (exponent32 << 23) | ((mantissa as u32) << 13)
        };

        f32::from_bits(output)
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.0 & 0x7c00 != 0x7c00
    }
}

impl From<f32> for F16 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<F16> for f32 {
    fn from(value: F16) -> Self {
        value.to_f32()
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F16")
            .field(&format_args!("0x{:04x}", self.0))
            .field(&self.to_f32())
            .finish()
    }
}

/// Mixed-precision forward result: binary16 context output and FP32 LSE.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatAttentionF16Output {
    pub output: Vec<F16>,
    pub lse: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_binary16_encodings_match_ieee754() {
        assert_eq!(F16::from_f32(0.0).to_bits(), 0x0000);
        assert_eq!(F16::from_f32(-0.0).to_bits(), 0x8000);
        assert_eq!(F16::from_f32(1.0).to_bits(), 0x3c00);
        assert_eq!(F16::from_f32(-2.0).to_bits(), 0xc000);
        assert_eq!(F16::from_f32(65_504.0).to_bits(), 0x7bff);
        assert_eq!(F16::from_f32(f32::INFINITY).to_bits(), 0x7c00);
    }

    #[test]
    fn binary16_decoding_handles_normals_and_subnormals() {
        assert_eq!(F16::from_bits(0x3c00).to_f32(), 1.0);
        assert_eq!(F16::from_bits(0xc000).to_f32(), -2.0);
        assert_eq!(F16::from_bits(0x0400).to_f32(), 2.0f32.powi(-14));
        assert_eq!(F16::from_bits(0x0001).to_f32(), 2.0f32.powi(-24));
    }

    #[test]
    fn finite_roundtrip_has_binary16_scale_error() {
        for value in [
            -17.75f32,
            -1.0,
            -0.33325,
            0.0,
            0.1,
            1.0,
            3.141_592_7,
            1024.5,
        ] {
            let decoded = F16::from_f32(value).to_f32();
            let tolerance = 5.0e-4 * value.abs().max(1.0);
            assert!((decoded - value).abs() <= tolerance, "{value} -> {decoded}");
        }
    }

    #[test]
    fn non_finite_classification_is_explicit() {
        assert!(F16::from_f32(1.0).is_finite());
        assert!(!F16::from_f32(f32::INFINITY).is_finite());
        assert!(!F16::from_f32(f32::NAN).is_finite());
    }
}
