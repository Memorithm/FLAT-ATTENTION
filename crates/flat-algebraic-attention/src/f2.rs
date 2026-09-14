use core::fmt;

pub const F2_WORD_BITS: usize = u64::BITS as usize;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum F2Error {
    ZeroDimension,
    WordCountMismatch {
        bit_len: usize,
        expected_words: usize,
        actual_words: usize,
    },
    NonZeroTailBits,
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        index: usize,
        bit_len: usize,
    },
    ArithmeticOverflow,
}

impl fmt::Display for F2Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(formatter, "F2 dimensions must be non-zero"),
            Self::WordCountMismatch {
                bit_len,
                expected_words,
                actual_words,
            } => write!(
                formatter,
                "F2 vector with {bit_len} bits requires {expected_words} u64 words, got {actual_words}"
            ),
            Self::NonZeroTailBits => write!(
                formatter,
                "unused high bits in the final F2 vector word must be zero"
            ),
            Self::DimensionMismatch { expected, actual } => write!(
                formatter,
                "F2 dimension mismatch: expected {expected} bits, got {actual}"
            ),
            Self::IndexOutOfBounds { index, bit_len } => write!(
                formatter,
                "F2 bit index {index} is outside 0..{bit_len}"
            ),
            Self::ArithmeticOverflow => write!(formatter, "F2 arithmetic accounting overflowed"),
        }
    }
}

impl std::error::Error for F2Error {}

fn words_for_bits(bit_len: usize) -> Result<usize, F2Error> {
    if bit_len == 0 {
        return Err(F2Error::ZeroDimension);
    }
    Ok(bit_len.div_ceil(F2_WORD_BITS))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F2Vector {
    bit_len: usize,
    words: Vec<u64>,
}

impl F2Vector {
    pub fn new(bit_len: usize, words: Vec<u64>) -> Result<Self, F2Error> {
        let expected_words = words_for_bits(bit_len)?;
        if words.len() != expected_words {
            return Err(F2Error::WordCountMismatch {
                bit_len,
                expected_words,
                actual_words: words.len(),
            });
        }

        let tail_bits = bit_len % F2_WORD_BITS;
        if tail_bits != 0 {
            let valid_mask = (1u64 << tail_bits) - 1;
            if words.last().copied().unwrap_or_default() & !valid_mask != 0 {
                return Err(F2Error::NonZeroTailBits);
            }
        }

        Ok(Self { bit_len, words })
    }

    pub fn from_bools(bits: &[bool]) -> Result<Self, F2Error> {
        let word_count = words_for_bits(bits.len())?;
        let mut words = vec![0u64; word_count];
        for (index, bit) in bits.iter().copied().enumerate() {
            if bit {
                words[index / F2_WORD_BITS] |= 1u64 << (index % F2_WORD_BITS);
            }
        }
        Self::new(bits.len(), words)
    }

    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn bit(&self, index: usize) -> Result<bool, F2Error> {
        if index >= self.bit_len {
            return Err(F2Error::IndexOutOfBounds {
                index,
                bit_len: self.bit_len,
            });
        }
        Ok((self.words[index / F2_WORD_BITS] & (1u64 << (index % F2_WORD_BITS))) != 0)
    }

    pub fn xor(&self, other: &Self) -> Result<Self, F2Error> {
        self.require_same_width(other)?;
        let words = self
            .words
            .iter()
            .zip(&other.words)
            .map(|(left, right)| left ^ right)
            .collect();
        Self::new(self.bit_len, words)
    }

    pub fn dot(&self, other: &Self) -> Result<bool, F2Error> {
        self.require_same_width(other)?;
        let parity = self
            .words
            .iter()
            .zip(&other.words)
            .fold(0u32, |parity, (left, right)| {
                parity ^ ((left & right).count_ones() & 1)
            });
        Ok(parity != 0)
    }

    pub fn hamming_distance(&self, other: &Self) -> Result<usize, F2Error> {
        self.require_same_width(other)?;
        self.words
            .iter()
            .zip(&other.words)
            .try_fold(0usize, |total, (left, right)| {
                total
                    .checked_add((left ^ right).count_ones() as usize)
                    .ok_or(F2Error::ArithmeticOverflow)
            })
    }

    fn require_same_width(&self, other: &Self) -> Result<(), F2Error> {
        if self.bit_len != other.bit_len {
            return Err(F2Error::DimensionMismatch {
                expected: self.bit_len,
                actual: other.bit_len,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F2LinearMap {
    input_bits: usize,
    rows: Vec<F2Vector>,
}

impl F2LinearMap {
    pub fn new(input_bits: usize, rows: Vec<F2Vector>) -> Result<Self, F2Error> {
        words_for_bits(input_bits)?;
        if rows.is_empty() {
            return Err(F2Error::ZeroDimension);
        }
        for row in &rows {
            if row.bit_len() != input_bits {
                return Err(F2Error::DimensionMismatch {
                    expected: input_bits,
                    actual: row.bit_len(),
                });
            }
        }
        Ok(Self { input_bits, rows })
    }

    #[must_use]
    pub fn input_bits(&self) -> usize {
        self.input_bits
    }

    #[must_use]
    pub fn output_bits(&self) -> usize {
        self.rows.len()
    }

    #[must_use]
    pub fn rows(&self) -> &[F2Vector] {
        &self.rows
    }

    pub fn apply(&self, input: &F2Vector) -> Result<F2Vector, F2Error> {
        if input.bit_len() != self.input_bits {
            return Err(F2Error::DimensionMismatch {
                expected: self.input_bits,
                actual: input.bit_len(),
            });
        }
        let output = self
            .rows
            .iter()
            .map(|row| row.dot(input))
            .collect::<Result<Vec<_>, _>>()?;
        F2Vector::from_bools(&output)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F2AffinePredicate {
    coefficients: F2Vector,
    bias: bool,
}

impl F2AffinePredicate {
    #[must_use]
    pub fn new(coefficients: F2Vector, bias: bool) -> Self {
        Self { coefficients, bias }
    }

    #[must_use]
    pub fn input_bits(&self) -> usize {
        self.coefficients.bit_len()
    }

    #[must_use]
    pub fn bias(&self) -> bool {
        self.bias
    }

    #[must_use]
    pub fn coefficients(&self) -> &F2Vector {
        &self.coefficients
    }

    pub fn evaluate(&self, input: &F2Vector) -> Result<bool, F2Error> {
        Ok(self.coefficients.dot(input)? ^ self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_vector_round_trips_across_word_boundary() {
        let mut bits = vec![false; 65];
        bits[0] = true;
        bits[63] = true;
        bits[64] = true;
        let vector = F2Vector::from_bools(&bits).unwrap();

        assert_eq!(vector.bit_len(), 65);
        assert_eq!(vector.words(), &[1 | (1u64 << 63), 1]);
        assert!(vector.bit(0).unwrap());
        assert!(vector.bit(63).unwrap());
        assert!(vector.bit(64).unwrap());
    }

    #[test]
    fn rejects_nonzero_padding_bits() {
        assert_eq!(
            F2Vector::new(65, vec![0, 0b10]),
            Err(F2Error::NonZeroTailBits)
        );
    }

    #[test]
    fn xor_is_field_addition_and_dot_is_mod_two() {
        let left = F2Vector::from_bools(&[true, true, false, true]).unwrap();
        let right = F2Vector::from_bools(&[true, false, true, true]).unwrap();

        assert_eq!(
            left.xor(&right).unwrap(),
            F2Vector::from_bools(&[false, true, true, false]).unwrap()
        );
        assert!(!left.dot(&right).unwrap());
        assert_eq!(left.hamming_distance(&right).unwrap(), 2);
    }

    #[test]
    fn linear_map_applies_row_dot_products() {
        let map = F2LinearMap::new(
            3,
            vec![
                F2Vector::from_bools(&[true, false, true]).unwrap(),
                F2Vector::from_bools(&[false, true, true]).unwrap(),
            ],
        )
        .unwrap();
        let input = F2Vector::from_bools(&[true, true, false]).unwrap();

        assert_eq!(
            map.apply(&input).unwrap(),
            F2Vector::from_bools(&[true, true]).unwrap()
        );
    }

    #[test]
    fn affine_predicate_adds_constant_term() {
        let predicate =
            F2AffinePredicate::new(F2Vector::from_bools(&[true, true, false]).unwrap(), true);
        let input = F2Vector::from_bools(&[true, false, true]).unwrap();

        assert!(!predicate.evaluate(&input).unwrap());
    }

    #[test]
    fn dimension_mismatch_fails_closed() {
        let left = F2Vector::from_bools(&[true, false]).unwrap();
        let right = F2Vector::from_bools(&[true]).unwrap();

        assert_eq!(
            left.dot(&right),
            Err(F2Error::DimensionMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }
}
