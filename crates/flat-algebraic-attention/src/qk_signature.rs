use core::fmt;

use crate::f2::{F2Error, F2Vector};

const BIT_INDEX_MIX: u64 = 0x9e37_79b9_7f4a_7c15;
const COORD_INDEX_MIX: u64 = 0xbf58_476d_1ce4_e5b9;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum QkSignatureError {
    ZeroInputDimension,
    ZeroOutputBits,
    InputDimensionMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        index: usize,
    },
    IndexTooLarge {
        kind: &'static str,
        index: usize,
    },
    ProjectionOverflow {
        bit_index: usize,
    },
    F2(F2Error),
}

impl fmt::Display for QkSignatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroInputDimension => {
                write!(formatter, "Q/K signature input dimension must be non-zero")
            }
            Self::ZeroOutputBits => {
                write!(formatter, "Q/K signature output width must be non-zero")
            }
            Self::InputDimensionMismatch { expected, actual } => write!(
                formatter,
                "Q/K signature input dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteInput { index } => write!(
                formatter,
                "Q/K signature input contains a non-finite value at index {index}"
            ),
            Self::IndexTooLarge { kind, index } => write!(
                formatter,
                "Q/K signature {kind} index {index} cannot be represented in the deterministic u64 projector key"
            ),
            Self::ProjectionOverflow { bit_index } => write!(
                formatter,
                "Q/K signature projection for output bit {bit_index} became non-finite"
            ),
            Self::F2(error) => write!(formatter, "Q/K signature F2 packing failed: {error}"),
        }
    }
}

impl std::error::Error for QkSignatureError {}

impl From<F2Error> for QkSignatureError {
    fn from(error: F2Error) -> Self {
        Self::F2(error)
    }
}

/// Deterministic signed-hyperplane map from a numerical Q/K vector into F2.
///
/// The hyperplane coefficients are Rademacher values (-1 or +1) derived
/// solely from the projector seed and the output/input indices. This is a
/// research oracle, not a performance implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedHyperplaneProjector {
    input_dim: usize,
    output_bits: usize,
    seed: u64,
}

impl SignedHyperplaneProjector {
    pub fn new(input_dim: usize, output_bits: usize, seed: u64) -> Result<Self, QkSignatureError> {
        if input_dim == 0 {
            return Err(QkSignatureError::ZeroInputDimension);
        }
        if output_bits == 0 {
            return Err(QkSignatureError::ZeroOutputBits);
        }

        // Fail before projection if the configured dimensions cannot be
        // represented in the stable u64 key space used by the projector.
        checked_index("input", input_dim - 1)?;
        checked_index("output", output_bits - 1)?;

        Ok(Self {
            input_dim,
            output_bits,
            seed,
        })
    }

    #[must_use]
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    #[must_use]
    pub const fn output_bits(&self) -> usize {
        self.output_bits
    }

    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    pub fn project(&self, input: &[f32]) -> Result<F2Vector, QkSignatureError> {
        if input.len() != self.input_dim {
            return Err(QkSignatureError::InputDimensionMismatch {
                expected: self.input_dim,
                actual: input.len(),
            });
        }
        for (index, value) in input.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(QkSignatureError::NonFiniteInput { index });
            }
        }

        let mut bits = Vec::with_capacity(self.output_bits);
        for bit_index in 0..self.output_bits {
            let bit_key = checked_index("output", bit_index)?;
            let mut projection = 0.0f64;

            for (coord_index, value) in input.iter().copied().enumerate() {
                let coord_key = checked_index("input", coord_index)?;
                let key = self.seed
                    ^ bit_key.wrapping_mul(BIT_INDEX_MIX)
                    ^ coord_key.wrapping_mul(COORD_INDEX_MIX);
                let sign = if mix64(key) & 1 == 0 { -1.0 } else { 1.0 };
                projection += sign * f64::from(value);
            }

            if !projection.is_finite() {
                return Err(QkSignatureError::ProjectionOverflow { bit_index });
            }
            bits.push(projection >= 0.0);
        }

        Ok(F2Vector::from_bools(&bits)?)
    }
}

fn checked_index(kind: &'static str, index: usize) -> Result<u64, QkSignatureError> {
    u64::try_from(index).map_err(|_| QkSignatureError::IndexTooLarge { kind, index })
}

/// SplitMix64 finalizer used as a stable pure mixing function.
///
/// Wrapping arithmetic is part of this explicitly defined hash-style mixing
/// operation; dimension/index conversion is validated before values reach it.
#[must_use]
const fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_projection_is_bit_exact() {
        let projector = SignedHyperplaneProjector::new(4, 65, 0x1234).unwrap();
        let input = [0.25, -0.5, 1.25, 0.75];

        let first = projector.project(&input).unwrap();
        let second = projector.project(&input).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.bit_len(), 65);
        assert_eq!(first.words().len(), 2);
    }

    #[test]
    fn positive_scaling_preserves_signature() {
        let projector = SignedHyperplaneProjector::new(4, 128, 0x514b_5349_474e_3133).unwrap();
        let input = [0.25, -0.5, 1.25, 0.75];
        let scaled = input.map(|value| value * 3.5);

        assert_eq!(
            projector.project(&input).unwrap(),
            projector.project(&scaled).unwrap()
        );
    }

    #[test]
    fn zero_dimensions_fail_closed() {
        assert_eq!(
            SignedHyperplaneProjector::new(0, 8, 1),
            Err(QkSignatureError::ZeroInputDimension)
        );
        assert_eq!(
            SignedHyperplaneProjector::new(8, 0, 1),
            Err(QkSignatureError::ZeroOutputBits)
        );
    }

    #[test]
    fn wrong_input_width_fails_closed() {
        let projector = SignedHyperplaneProjector::new(4, 8, 1).unwrap();

        assert_eq!(
            projector.project(&[1.0, 2.0]),
            Err(QkSignatureError::InputDimensionMismatch {
                expected: 4,
                actual: 2,
            })
        );
    }

    #[test]
    fn non_finite_input_fails_closed() {
        let projector = SignedHyperplaneProjector::new(3, 8, 1).unwrap();

        assert_eq!(
            projector.project(&[1.0, f32::NAN, 2.0]),
            Err(QkSignatureError::NonFiniteInput { index: 1 })
        );
        assert_eq!(
            projector.project(&[1.0, f32::INFINITY, 2.0]),
            Err(QkSignatureError::NonFiniteInput { index: 1 })
        );
    }
}
