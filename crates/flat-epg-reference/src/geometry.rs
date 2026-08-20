use core::fmt;
use flat_attention::FlatAttentionError;

/// Stable version of the reference representation contract.
pub const EPG_CONTRACT_VERSION: u32 = 1;

/// Four-dimensional rotation family used by the EPG portion of a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum So4Geometry {
    /// Canonical isoclinic control: both planes share one frequency.
    Isoclinic,
    /// Canonical double rotation using two consecutive RoPE frequencies.
    Biplanar,
}

/// Versioned head-local hybrid positional-geometry descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpgEmbeddingConfig {
    /// Contract version.
    pub version: u32,
    /// RoPE-compatible base frequency.
    pub theta: f32,
    /// Absolute position added to token indices in this attention call.
    pub position_offset: usize,
    /// Number of trailing head channels assigned to four-dimensional blocks.
    pub so4_dims: usize,
    /// Geometry used by the four-dimensional blocks.
    pub so4_geometry: So4Geometry,
}

impl EpgEmbeddingConfig {
    /// Construct a generation-1 descriptor.
    pub const fn v1(
        theta: f32,
        position_offset: usize,
        so4_dims: usize,
        so4_geometry: So4Geometry,
    ) -> Self {
        Self {
            version: EPG_CONTRACT_VERSION,
            theta,
            position_offset,
            so4_dims,
            so4_geometry,
        }
    }

    /// Number of leading channels retaining ordinary interleaved `SO(2)` RoPE.
    pub const fn so2_dims(self, head_dim: usize) -> usize {
        head_dim - self.so4_dims
    }

    /// Validate the representation against one attention head.
    pub fn validate(self, head_dim: usize, seq_len: usize) -> Result<(), EpgError> {
        if self.version != EPG_CONTRACT_VERSION {
            return Err(EpgError::UnsupportedContractVersion(self.version));
        }
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(EpgError::InvalidHeadDim(head_dim));
        }
        if self.so4_dims > head_dim || !self.so4_dims.is_multiple_of(4) {
            return Err(EpgError::InvalidSo4Tail {
                head_dim,
                so4_dims: self.so4_dims,
            });
        }
        if !(head_dim - self.so4_dims).is_multiple_of(2) {
            return Err(EpgError::InvalidSo4Tail {
                head_dim,
                so4_dims: self.so4_dims,
            });
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(EpgError::InvalidTheta(self.theta));
        }
        self.position_offset
            .checked_add(seq_len.saturating_sub(1))
            .ok_or(EpgError::PositionOverflow)?;
        Ok(())
    }
}

/// Failures in the EPG representation contract or wrapped FLAT contract.
#[derive(Debug, Clone, PartialEq)]
pub enum EpgError {
    /// Descriptor version is not implemented by this reference crate.
    UnsupportedContractVersion(u32),
    /// Head dimension is zero or odd.
    InvalidHeadDim(usize),
    /// Four-dimensional tail is not a valid multiple-of-four suffix.
    InvalidSo4Tail {
        /// Full head dimension.
        head_dim: usize,
        /// Requested SO(4) suffix dimension.
        so4_dims: usize,
    },
    /// Base frequency is non-finite or non-positive.
    InvalidTheta(f32),
    /// Position arithmetic overflowed `usize`.
    PositionOverflow,
    /// Input tensor has the wrong scalar length.
    LengthMismatch {
        /// Tensor name.
        tensor: &'static str,
        /// Actual number of scalars.
        actual: usize,
        /// Required number of scalars.
        expected: usize,
    },
    /// Input contains a non-finite scalar.
    NonFiniteInput {
        /// Tensor name.
        tensor: &'static str,
        /// Offending scalar index.
        index: usize,
    },
    /// Error reported by FLAT's public shape/configuration contract.
    Flat(FlatAttentionError),
}

impl From<FlatAttentionError> for EpgError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Flat(value)
    }
}

impl fmt::Display for EpgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedContractVersion(v) => write!(f, "unsupported EPG contract version {v}"),
            Self::InvalidHeadDim(d) => write!(f, "EPG head_dim must be non-zero and even, got {d}"),
            Self::InvalidSo4Tail { head_dim, so4_dims } => write!(
                f,
                "EPG so4_dims must be a multiple of four not exceeding head_dim; got {so4_dims}/{head_dim}"
            ),
            Self::InvalidTheta(t) => write!(f, "EPG theta must be finite and positive, got {t}"),
            Self::PositionOverflow => write!(f, "EPG position offset overflows the index space"),
            Self::LengthMismatch { tensor, actual, expected } => write!(
                f,
                "tensor {tensor} contains {actual} elements, expected {expected}"
            ),
            Self::NonFiniteInput { tensor, index } => write!(
                f,
                "tensor {tensor} contains a non-finite value at index {index}"
            ),
            Self::Flat(error) => write!(f, "FLAT contract error: {error}"),
        }
    }
}

impl std::error::Error for EpgError {}
