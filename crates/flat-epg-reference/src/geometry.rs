use core::fmt;

use epg_core::{
    EpgContractError, EpgGeometryDescriptor, EpgGeometryKind, EpgPositionDomain, So4Geometry,
};
use flat_attention::FlatAttentionError;

pub use epg_core::EPG_CONTRACT_VERSION;
pub use epg_core::So4Geometry as CoreSo4Geometry;

/// Execution-local EPG configuration used by the scalar FLAT oracle.
///
/// The mathematical representation (`geometry`) and the position origin
/// (`position`) are intentionally separate. Moving a query to another absolute
/// offset does not create a new model representation capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpgEmbeddingConfig {
    /// Stable geometry descriptor shared with runtimes and other backends.
    pub geometry: EpgGeometryDescriptor,
    /// Absolute position domain for this execution.
    pub position: EpgPositionDomain,
}

impl EpgEmbeddingConfig {
    /// Construct ordinary SO(2) RoPE in an EPG execution domain.
    pub fn so2(theta: f32, position_offset: usize) -> Result<Self, EpgError> {
        Ok(Self {
            geometry: EpgGeometryDescriptor::so2(theta)?,
            position: EpgPositionDomain::new(
                u64::try_from(position_offset).map_err(|_| EpgError::PositionOverflow)?,
            ),
        })
    }

    /// Construct a hybrid SO(2)/SO(4) execution descriptor.
    pub fn hybrid_so4(
        theta: f32,
        position_offset: usize,
        so4_dims: usize,
        so4_geometry: So4Geometry,
    ) -> Result<Self, EpgError> {
        Ok(Self {
            geometry: EpgGeometryDescriptor::hybrid_so4(
                theta,
                u32::try_from(so4_dims).map_err(|_| EpgError::DimensionOverflow)?,
                so4_geometry,
            )?,
            position: EpgPositionDomain::new(
                u64::try_from(position_offset).map_err(|_| EpgError::PositionOverflow)?,
            ),
        })
    }

    /// RoPE-compatible base frequency.
    pub const fn theta(self) -> f32 {
        self.geometry.theta()
    }

    /// Number of trailing dimensions assigned to SO(4) blocks.
    pub const fn so4_dims(self) -> usize {
        self.geometry.so4_dims() as usize
    }

    /// SO(4) control family when this is a hybrid descriptor.
    pub const fn so4_geometry(self) -> Option<So4Geometry> {
        match self.geometry.kind() {
            EpgGeometryKind::So2 => None,
            EpgGeometryKind::HybridSo4(geometry) => Some(geometry),
        }
    }

    /// Number of leading dimensions retaining ordinary interleaved SO(2) RoPE.
    pub fn so2_dims(self, head_dim: usize) -> Result<usize, EpgError> {
        let head_dim = u32::try_from(head_dim).map_err(|_| EpgError::DimensionOverflow)?;
        Ok(self.geometry.so2_dims(head_dim)? as usize)
    }

    /// Resolve one local token index to an absolute position representable by
    /// the scalar oracle.
    pub fn resolve_position(self, local_index: usize) -> Result<usize, EpgError> {
        let local_index = u64::try_from(local_index).map_err(|_| EpgError::PositionOverflow)?;
        let absolute = self.position.resolve(local_index)?;
        usize::try_from(absolute).map_err(|_| EpgError::PositionOverflow)
    }

    /// Validate the representation against one attention head and sequence.
    pub fn validate(self, head_dim: usize, seq_len: usize) -> Result<(), EpgError> {
        let head_dim = u32::try_from(head_dim).map_err(|_| EpgError::DimensionOverflow)?;
        self.geometry.validate_head_dim(head_dim)?;
        self.resolve_position(seq_len.saturating_sub(1))?;
        Ok(())
    }
}

/// Failures in the EPG representation contract, oracle adapter, or wrapped
/// FLAT attention contract.
#[derive(Debug, Clone, PartialEq)]
pub enum EpgError {
    /// Runtime-neutral EPG contract error.
    Contract(EpgContractError),
    /// A host dimension does not fit the portable EPG contract index space.
    DimensionOverflow,
    /// Position arithmetic cannot be represented by the host oracle.
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

impl From<EpgContractError> for EpgError {
    fn from(value: EpgContractError) -> Self {
        Self::Contract(value)
    }
}

impl From<FlatAttentionError> for EpgError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Flat(value)
    }
}

impl fmt::Display for EpgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(f, "EPG contract error: {error}"),
            Self::DimensionOverflow => write!(f, "EPG dimension exceeds the portable index space"),
            Self::PositionOverflow => write!(f, "EPG position exceeds the host index space"),
            Self::LengthMismatch {
                tensor,
                actual,
                expected,
            } => write!(
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
