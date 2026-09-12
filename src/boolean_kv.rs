use core::fmt;

pub const BOOLEAN_KV_SCHEMA_VERSION: u32 = 1;
pub const BOOLEAN_KV_WORD_BITS: usize = u64::BITS as usize;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BooleanKvError {
    ZeroSignatureBits,
    WordCountMismatch {
        bit_len: usize,
        expected_words: usize,
        actual_words: usize,
    },
    NonZeroTailBits,
    SignatureWidthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidMaxDistance {
        max_distance: usize,
        signature_bits: usize,
    },
    ZeroLimit,
    GenerationOverflow,
    StorageOverflow,
}

impl fmt::Display for BooleanKvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSignatureBits => write!(f, "Boolean KV signature width must be non-zero"),
            Self::WordCountMismatch {
                bit_len,
                expected_words,
                actual_words,
            } => write!(
                f,
                "Boolean KV signature with {bit_len} bits requires {expected_words} u64 words, got {actual_words}"
            ),
            Self::NonZeroTailBits => write!(
                f,
                "unused high bits in the final Boolean KV word must be zero"
            ),
            Self::SignatureWidthMismatch { expected, actual } => write!(
                f,
                "Boolean KV signature width mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidMaxDistance {
                max_distance,
                signature_bits,
            } => write!(
                f,
                "Boolean KV max Hamming distance {max_distance} exceeds signature width {signature_bits}"
            ),
            Self::ZeroLimit => write!(f, "Boolean KV result limit must be non-zero"),
            Self::GenerationOverflow => write!(f, "Boolean KV generation counter overflowed"),
            Self::StorageOverflow => write!(f, "Boolean KV storage accounting overflowed usize"),
        }
    }
}

impl std::error::Error for BooleanKvError {}

fn words_for_bits(bit_len: usize) -> Result<usize, BooleanKvError> {
    if bit_len == 0 {
        return Err(BooleanKvError::ZeroSignatureBits);
    }
    Ok(bit_len.div_ceil(BOOLEAN_KV_WORD_BITS))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedBooleanSignature {
    bit_len: usize,
    words: Vec<u64>,
}

impl PackedBooleanSignature {
    pub fn new(bit_len: usize, words: Vec<u64>) -> Result<Self, BooleanKvError> {
        let expected_words = words_for_bits(bit_len)?;
        if words.len() != expected_words {
            return Err(BooleanKvError::WordCountMismatch {
                bit_len,
                expected_words,
                actual_words: words.len(),
            });
        }
        let tail_bits = bit_len % BOOLEAN_KV_WORD_BITS;
        if tail_bits != 0 {
            let valid_mask = (1u64 << tail_bits) - 1;
            if words.last().copied().unwrap_or_default() & !valid_mask != 0 {
                return Err(BooleanKvError::NonZeroTailBits);
            }
        }
        Ok(Self { bit_len, words })
    }

    pub fn from_bools(bits: &[bool]) -> Result<Self, BooleanKvError> {
        let word_count = words_for_bits(bits.len())?;
        let mut words = vec![0u64; word_count];
        for (index, bit) in bits.iter().copied().enumerate() {
            if bit {
                words[index / BOOLEAN_KV_WORD_BITS] |= 1u64 << (index % BOOLEAN_KV_WORD_BITS);
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

    #[must_use]
    pub fn logical_bits(&self) -> usize {
        self.bit_len
    }

    pub fn physical_bits(&self) -> Result<usize, BooleanKvError> {
        self.words
            .len()
            .checked_mul(BOOLEAN_KV_WORD_BITS)
            .ok_or(BooleanKvError::StorageOverflow)
    }

    pub fn physical_bytes(&self) -> Result<usize, BooleanKvError> {
        self.words
            .len()
            .checked_mul(core::mem::size_of::<u64>())
            .ok_or(BooleanKvError::StorageOverflow)
    }

    pub fn hamming_distance(&self, other: &Self) -> Result<usize, BooleanKvError> {
        self.require_same_width(other)?;
        self.words
            .iter()
            .zip(&other.words)
            .try_fold(0usize, |total, (left, right)| {
                total
                    .checked_add((left ^ right).count_ones() as usize)
                    .ok_or(BooleanKvError::StorageOverflow)
            })
    }

    pub fn xnor_matches(&self, other: &Self) -> Result<usize, BooleanKvError> {
        Ok(self.bit_len - self.hamming_distance(other)?)
    }

    fn require_same_width(&self, other: &Self) -> Result<(), BooleanKvError> {
        if self.bit_len != other.bit_len {
            return Err(BooleanKvError::SignatureWidthMismatch {
                expected: self.bit_len,
                actual: other.bit_len,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanKvPage {
    pub logical_page: usize,
    pub generation: u64,
    pub key_signature: PackedBooleanSignature,
    pub value_signature: Option<PackedBooleanSignature>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanKvAccounting {
    pub pages: usize,
    pub signature_bits: usize,
    pub key_logical_bits: usize,
    pub value_logical_bits: usize,
    pub key_physical_bytes: usize,
    pub value_physical_bytes: usize,
    pub total_physical_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanKvMatch {
    pub logical_page: usize,
    pub hamming_distance: usize,
    pub xnor_matches: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanKvCache {
    signature_bits: usize,
    generation: u64,
    pages: Vec<BooleanKvPage>,
}

impl BooleanKvCache {
    pub fn new(signature_bits: usize) -> Result<Self, BooleanKvError> {
        words_for_bits(signature_bits)?;
        Ok(Self {
            signature_bits,
            generation: 0,
            pages: Vec::new(),
        })
    }

    #[must_use]
    pub fn signature_bits(&self) -> usize {
        self.signature_bits
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    pub fn append(
        &mut self,
        key_signature: PackedBooleanSignature,
        value_signature: Option<PackedBooleanSignature>,
    ) -> Result<&BooleanKvPage, BooleanKvError> {
        self.require_width(&key_signature)?;
        if let Some(value) = value_signature.as_ref() {
            self.require_width(value)?;
        }
        let page = BooleanKvPage {
            logical_page: self.pages.len(),
            generation: self.generation,
            key_signature,
            value_signature,
        };
        self.pages.push(page);
        Ok(self.pages.last().expect("page was just pushed"))
    }

    #[must_use]
    pub fn page(&self, logical_page: usize) -> Option<&BooleanKvPage> {
        self.pages
            .get(logical_page)
            .filter(|page| page.generation == self.generation)
    }

    pub fn reset(&mut self) -> Result<(), BooleanKvError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(BooleanKvError::GenerationOverflow)?;
        self.pages.clear();
        Ok(())
    }

    pub fn accounting(&self) -> Result<BooleanKvAccounting, BooleanKvError> {
        let mut key_logical_bits = 0usize;
        let mut value_logical_bits = 0usize;
        let mut key_physical_bytes = 0usize;
        let mut value_physical_bytes = 0usize;

        for page in &self.pages {
            key_logical_bits = key_logical_bits
                .checked_add(page.key_signature.logical_bits())
                .ok_or(BooleanKvError::StorageOverflow)?;
            key_physical_bytes = key_physical_bytes
                .checked_add(page.key_signature.physical_bytes()?)
                .ok_or(BooleanKvError::StorageOverflow)?;
            if let Some(value) = page.value_signature.as_ref() {
                value_logical_bits = value_logical_bits
                    .checked_add(value.logical_bits())
                    .ok_or(BooleanKvError::StorageOverflow)?;
                value_physical_bytes = value_physical_bytes
                    .checked_add(value.physical_bytes()?)
                    .ok_or(BooleanKvError::StorageOverflow)?;
            }
        }
        let total_physical_bytes = key_physical_bytes
            .checked_add(value_physical_bytes)
            .ok_or(BooleanKvError::StorageOverflow)?;
        Ok(BooleanKvAccounting {
            pages: self.pages.len(),
            signature_bits: self.signature_bits,
            key_logical_bits,
            value_logical_bits,
            key_physical_bytes,
            value_physical_bytes,
            total_physical_bytes,
        })
    }

    pub fn search_hamming(
        &self,
        query: &PackedBooleanSignature,
        max_distance: usize,
        limit: Option<usize>,
    ) -> Result<Vec<BooleanKvMatch>, BooleanKvError> {
        self.require_width(query)?;
        if max_distance > self.signature_bits {
            return Err(BooleanKvError::InvalidMaxDistance {
                max_distance,
                signature_bits: self.signature_bits,
            });
        }
        if matches!(limit, Some(0)) {
            return Err(BooleanKvError::ZeroLimit);
        }

        let mut matches = Vec::new();
        for page in &self.pages {
            let distance = query.hamming_distance(&page.key_signature)?;
            if distance <= max_distance {
                matches.push(BooleanKvMatch {
                    logical_page: page.logical_page,
                    hamming_distance: distance,
                    xnor_matches: self.signature_bits - distance,
                });
            }
        }
        matches.sort_unstable_by_key(|item| (item.hamming_distance, item.logical_page));
        if let Some(limit) = limit {
            matches.truncate(limit);
        }
        Ok(matches)
    }

    fn require_width(&self, signature: &PackedBooleanSignature) -> Result<(), BooleanKvError> {
        if signature.bit_len() != self.signature_bits {
            return Err(BooleanKvError::SignatureWidthMismatch {
                expected: self.signature_bits,
                actual: signature.bit_len(),
            });
        }
        Ok(())
    }
}
