use core::fmt;

pub const BOOLEAN_ATTENTION_MASK_SCHEMA_VERSION: u32 = 1;
pub const BOOLEAN_ATTENTION_MASK_WORD_BITS: usize = u64::BITS as usize;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BooleanAttentionMaskError {
    ZeroBlocks,
    WordCountMismatch {
        blocks: usize,
        expected_words: usize,
        actual_words: usize,
    },
    NonZeroTailBits,
    BlockOutOfRange {
        block: usize,
        blocks: usize,
    },
    StorageOverflow,
}

impl fmt::Display for BooleanAttentionMaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBlocks => write!(f, "Boolean attention mask must contain at least one block"),
            Self::WordCountMismatch {
                blocks,
                expected_words,
                actual_words,
            } => write!(
                f,
                "Boolean attention mask with {blocks} blocks requires {expected_words} u64 words, got {actual_words}"
            ),
            Self::NonZeroTailBits => write!(
                f,
                "unused high bits in the final Boolean attention mask word must be zero"
            ),
            Self::BlockOutOfRange { block, blocks } => {
                write!(f, "Boolean attention block {block} is outside 0..{blocks}")
            }
            Self::StorageOverflow => write!(f, "Boolean attention mask storage accounting overflowed"),
        }
    }
}

impl std::error::Error for BooleanAttentionMaskError {}

fn words_for_blocks(blocks: usize) -> Result<usize, BooleanAttentionMaskError> {
    if blocks == 0 {
        return Err(BooleanAttentionMaskError::ZeroBlocks);
    }
    Ok(blocks.div_ceil(BOOLEAN_ATTENTION_MASK_WORD_BITS))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanAttentionMask {
    blocks: usize,
    words: Vec<u64>,
}

impl BooleanAttentionMask {
    pub fn new(blocks: usize, words: Vec<u64>) -> Result<Self, BooleanAttentionMaskError> {
        let expected_words = words_for_blocks(blocks)?;
        if words.len() != expected_words {
            return Err(BooleanAttentionMaskError::WordCountMismatch {
                blocks,
                expected_words,
                actual_words: words.len(),
            });
        }

        let tail_bits = blocks % BOOLEAN_ATTENTION_MASK_WORD_BITS;
        if tail_bits != 0 {
            let valid_mask = (1u64 << tail_bits) - 1;
            if words.last().copied().unwrap_or_default() & !valid_mask != 0 {
                return Err(BooleanAttentionMaskError::NonZeroTailBits);
            }
        }

        Ok(Self { blocks, words })
    }

    pub fn from_admissions(admitted: &[bool]) -> Result<Self, BooleanAttentionMaskError> {
        let word_count = words_for_blocks(admitted.len())?;
        let mut words = vec![0u64; word_count];
        for (block, keep) in admitted.iter().copied().enumerate() {
            if keep {
                words[block / BOOLEAN_ATTENTION_MASK_WORD_BITS] |=
                    1u64 << (block % BOOLEAN_ATTENTION_MASK_WORD_BITS);
            }
        }
        Self::new(admitted.len(), words)
    }

    #[must_use]
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn is_admitted(&self, block: usize) -> Result<bool, BooleanAttentionMaskError> {
        if block >= self.blocks {
            return Err(BooleanAttentionMaskError::BlockOutOfRange {
                block,
                blocks: self.blocks,
            });
        }
        Ok((self.words[block / BOOLEAN_ATTENTION_MASK_WORD_BITS]
            & (1u64 << (block % BOOLEAN_ATTENTION_MASK_WORD_BITS)))
            != 0)
    }

    #[must_use]
    pub fn admitted_blocks(&self) -> Vec<usize> {
        (0..self.blocks)
            .filter(|&block| {
                let word = self.words[block / BOOLEAN_ATTENTION_MASK_WORD_BITS];
                (word & (1u64 << (block % BOOLEAN_ATTENTION_MASK_WORD_BITS))) != 0
            })
            .collect()
    }

    #[must_use]
    pub fn admitted_count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    #[must_use]
    pub fn logical_bits(&self) -> usize {
        self.blocks
    }

    pub fn physical_bytes(&self) -> Result<usize, BooleanAttentionMaskError> {
        self.words
            .len()
            .checked_mul(core::mem::size_of::<u64>())
            .ok_or(BooleanAttentionMaskError::StorageOverflow)
    }
}
