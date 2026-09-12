use core::fmt;

pub const BOOLEAN_ATTENTION_SIGNATURE_SCHEMA_VERSION: u32 = 1;
pub const BOOLEAN_ATTENTION_SIGNATURE_WORD_BITS: usize = u64::BITS as usize;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BooleanAttentionSignatureError {
    ZeroBits,
    WordCountMismatch {
        bits: usize,
        expected_words: usize,
        actual_words: usize,
    },
    NonZeroTailBits,
    WidthMismatch {
        query_bits: usize,
        key_bits: usize,
    },
    ThresholdOutOfRange {
        threshold: usize,
        bits: usize,
    },
}

impl fmt::Display for BooleanAttentionSignatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBits => {
                write!(f, "Boolean attention signature must contain at least one bit")
            }
            Self::WordCountMismatch {
                bits,
                expected_words,
                actual_words,
            } => write!(
                f,
                "Boolean attention signature with {bits} bits requires {expected_words} u64 words, got {actual_words}"
            ),
            Self::NonZeroTailBits => write!(
                f,
                "unused high bits in the final Boolean attention signature word must be zero"
            ),
            Self::WidthMismatch {
                query_bits,
                key_bits,
            } => write!(
                f,
                "Boolean attention query/key widths differ: query={query_bits}, key={key_bits}"
            ),
            Self::ThresholdOutOfRange { threshold, bits } => write!(
                f,
                "Boolean attention Hamming threshold {threshold} exceeds signature width {bits}"
            ),
        }
    }
}

impl std::error::Error for BooleanAttentionSignatureError {}

fn words_for_bits(bits: usize) -> Result<usize, BooleanAttentionSignatureError> {
    if bits == 0 {
        return Err(BooleanAttentionSignatureError::ZeroBits);
    }
    Ok(bits.div_ceil(BOOLEAN_ATTENTION_SIGNATURE_WORD_BITS))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanAttentionSignature {
    bits: usize,
    words: Vec<u64>,
}

impl BooleanAttentionSignature {
    pub fn new(bits: usize, words: Vec<u64>) -> Result<Self, BooleanAttentionSignatureError> {
        let expected_words = words_for_bits(bits)?;
        if words.len() != expected_words {
            return Err(BooleanAttentionSignatureError::WordCountMismatch {
                bits,
                expected_words,
                actual_words: words.len(),
            });
        }

        let tail_bits = bits % BOOLEAN_ATTENTION_SIGNATURE_WORD_BITS;
        if tail_bits != 0 {
            let valid_mask = (1u64 << tail_bits) - 1;
            if words.last().copied().unwrap_or_default() & !valid_mask != 0 {
                return Err(BooleanAttentionSignatureError::NonZeroTailBits);
            }
        }

        Ok(Self { bits, words })
    }

    #[must_use]
    pub fn bits(&self) -> usize {
        self.bits
    }

    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn hamming_distance(&self, other: &Self) -> Result<usize, BooleanAttentionSignatureError> {
        if self.bits != other.bits {
            return Err(BooleanAttentionSignatureError::WidthMismatch {
                query_bits: self.bits,
                key_bits: other.bits,
            });
        }
        Ok(self
            .words
            .iter()
            .zip(&other.words)
            .map(|(query, key)| (query ^ key).count_ones() as usize)
            .sum())
    }

    pub fn xnor_match_count(&self, other: &Self) -> Result<usize, BooleanAttentionSignatureError> {
        let distance = self.hamming_distance(other)?;
        Ok(self.bits - distance)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HammingAdmissionRule {
    pub max_distance: usize,
}

impl HammingAdmissionRule {
    pub fn new(max_distance: usize, bits: usize) -> Result<Self, BooleanAttentionSignatureError> {
        if max_distance > bits {
            return Err(BooleanAttentionSignatureError::ThresholdOutOfRange {
                threshold: max_distance,
                bits,
            });
        }
        Ok(Self { max_distance })
    }

    pub fn admits(
        &self,
        query: &BooleanAttentionSignature,
        key: &BooleanAttentionSignature,
    ) -> Result<bool, BooleanAttentionSignatureError> {
        Ok(query.hamming_distance(key)? <= self.max_distance)
    }
}
