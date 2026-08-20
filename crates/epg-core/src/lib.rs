//! Runtime-neutral core contracts for Elastic Positional Geometry (EPG).
//!
//! `epg-core` describes a positional geometry. It does **not** implement
//! attention, a KV cache, a runtime policy, or any GPU backend. Keeping this
//! layer dependency-free makes the contract movable to its own repository and
//! usable by FLAT-ATTENTION, ElasticXxx, SLHAv2 adapters, or other consumers.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::fmt;

/// Current stable EPG representation-contract version.
pub const EPG_CONTRACT_VERSION: u32 = 1;

/// Four-dimensional control family for a hybrid SO(2)/SO(4) head.
///
/// Generation 1 intentionally contains only controls whose mathematical
/// behaviour is easy to qualify before introducing a genuinely richer EPG
/// geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum So4Geometry {
    /// Canonical isoclinic control: both orthogonal 2D planes in each 4-channel
    /// block use the same angular frequency.
    Isoclinic,
    /// Canonical double rotation using two consecutive RoPE frequencies.
    /// This is an explicit RoPE-equivalence control, not a novelty claim.
    Biplanar,
}

/// Stable family identifier for a positional geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpgGeometryKind {
    /// Ordinary interleaved SO(2) RoPE over the complete head.
    So2,
    /// Leading SO(2) channels plus trailing four-dimensional blocks.
    HybridSo4(So4Geometry),
}

impl EpgGeometryKind {
    /// Stable textual identifier suitable for runtime capability registries.
    pub const fn representation_id(self) -> &'static str {
        match self {
            Self::So2 => "epg.so2",
            Self::HybridSo4(So4Geometry::Isoclinic) => "epg.so4.isoclinic",
            Self::HybridSo4(So4Geometry::Biplanar) => "epg.so4.biplanar",
        }
    }
}

/// Versioned positional-geometry descriptor for one attention head.
///
/// `theta_bits` stores the exact IEEE-754 representation of the base frequency
/// so the descriptor remains `Eq`/`Hash` and can be used safely as a runtime
/// capability/cache key. Use [`EpgGeometryDescriptor::theta`] to recover the
/// floating-point value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpgGeometryDescriptor {
    version: u32,
    theta_bits: u32,
    kind: EpgGeometryKind,
    so4_dims: u32,
}

impl EpgGeometryDescriptor {
    /// Construct ordinary SO(2) RoPE.
    pub fn so2(theta: f32) -> Result<Self, EpgContractError> {
        Self::build(theta, EpgGeometryKind::So2, 0)
    }

    /// Construct a hybrid geometry with a trailing SO(4) suffix.
    pub fn hybrid_so4(
        theta: f32,
        so4_dims: u32,
        geometry: So4Geometry,
    ) -> Result<Self, EpgContractError> {
        if so4_dims == 0 || !so4_dims.is_multiple_of(4) {
            return Err(EpgContractError::InvalidSo4Dims(so4_dims));
        }
        Self::build(theta, EpgGeometryKind::HybridSo4(geometry), so4_dims)
    }

    fn build(
        theta: f32,
        kind: EpgGeometryKind,
        so4_dims: u32,
    ) -> Result<Self, EpgContractError> {
        if !theta.is_finite() || theta <= 0.0 {
            return Err(EpgContractError::InvalidTheta(theta.to_bits()));
        }
        Ok(Self {
            version: EPG_CONTRACT_VERSION,
            theta_bits: theta.to_bits(),
            kind,
            so4_dims,
        })
    }

    /// Representation-contract version.
    pub const fn version(self) -> u32 {
        self.version
    }

    /// RoPE-compatible base frequency.
    pub const fn theta(self) -> f32 {
        f32::from_bits(self.theta_bits)
    }

    /// Exact IEEE-754 bits used in the descriptor.
    pub const fn theta_bits(self) -> u32 {
        self.theta_bits
    }

    /// Geometry family.
    pub const fn kind(self) -> EpgGeometryKind {
        self.kind
    }

    /// Number of trailing dimensions assigned to SO(4) blocks.
    pub const fn so4_dims(self) -> u32 {
        self.so4_dims
    }

    /// Stable representation identifier for capability/runtime registries.
    pub const fn representation_id(self) -> &'static str {
        self.kind.representation_id()
    }

    /// Validate the descriptor for a concrete attention-head dimension.
    pub fn validate_head_dim(self, head_dim: u32) -> Result<(), EpgContractError> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(EpgContractError::InvalidHeadDim(head_dim));
        }
        match self.kind {
            EpgGeometryKind::So2 => {
                if self.so4_dims != 0 {
                    return Err(EpgContractError::InconsistentDescriptor);
                }
            }
            EpgGeometryKind::HybridSo4(_) => {
                if self.so4_dims == 0
                    || !self.so4_dims.is_multiple_of(4)
                    || self.so4_dims > head_dim
                {
                    return Err(EpgContractError::InvalidSo4Tail {
                        head_dim,
                        so4_dims: self.so4_dims,
                    });
                }
                if !(head_dim - self.so4_dims).is_multiple_of(2) {
                    return Err(EpgContractError::InvalidSo4Tail {
                        head_dim,
                        so4_dims: self.so4_dims,
                    });
                }
            }
        }
        Ok(())
    }

    /// Number of leading SO(2) dimensions for a validated head.
    pub fn so2_dims(self, head_dim: u32) -> Result<u32, EpgContractError> {
        self.validate_head_dim(head_dim)?;
        Ok(head_dim - self.so4_dims)
    }
}

/// Position origin for one execution domain.
///
/// The origin is deliberately separate from [`EpgGeometryDescriptor`]: moving a
/// query within a sequence must not create a new mathematical representation
/// capability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct EpgPositionDomain {
    offset: u64,
}

impl EpgPositionDomain {
    /// Construct a position domain from its absolute origin.
    pub const fn new(offset: u64) -> Self {
        Self { offset }
    }

    /// Absolute position origin.
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Resolve a local token index to an absolute position.
    pub fn resolve(self, local_index: u64) -> Result<u64, EpgContractError> {
        self.offset
            .checked_add(local_index)
            .ok_or(EpgContractError::PositionOverflow)
    }
}

/// Invalid EPG representation contract or execution coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpgContractError {
    /// Base frequency is non-finite or non-positive. The payload contains its
    /// exact IEEE-754 bits so the error remains Eq.
    InvalidTheta(u32),
    /// Head dimension is zero or odd.
    InvalidHeadDim(u32),
    /// SO(4) suffix requested at construction is zero or not a multiple of four.
    InvalidSo4Dims(u32),
    /// SO(4) suffix does not fit the concrete head dimension.
    InvalidSo4Tail {
        /// Full head dimension.
        head_dim: u32,
        /// Requested SO(4) suffix dimension.
        so4_dims: u32,
    },
    /// Descriptor fields contradict their geometry family.
    InconsistentDescriptor,
    /// Absolute position arithmetic overflowed `u64`.
    PositionOverflow,
}

impl fmt::Display for EpgContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidTheta(bits) => write!(
                f,
                "EPG theta must be finite and positive, got {:?}",
                f32::from_bits(bits)
            ),
            Self::InvalidHeadDim(head_dim) => {
                write!(f, "EPG head_dim must be non-zero and even, got {head_dim}")
            }
            Self::InvalidSo4Dims(so4_dims) => write!(
                f,
                "EPG SO(4) suffix must be a non-zero multiple of four, got {so4_dims}"
            ),
            Self::InvalidSo4Tail { head_dim, so4_dims } => write!(
                f,
                "EPG SO(4) suffix {so4_dims} is invalid for head_dim {head_dim}"
            ),
            Self::InconsistentDescriptor => write!(f, "inconsistent EPG geometry descriptor"),
            Self::PositionOverflow => write!(f, "EPG absolute position overflow"),
        }
    }
}

impl std::error::Error for EpgContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representation_identity_is_stable_and_offset_independent() {
        let geometry = EpgGeometryDescriptor::hybrid_so4(
            10_000.0,
            32,
            So4Geometry::Isoclinic,
        )
        .unwrap();
        assert_eq!(geometry.representation_id(), "epg.so4.isoclinic");
        assert_eq!(geometry.theta_bits(), 10_000.0f32.to_bits());
        assert_eq!(EpgPositionDomain::new(1).offset(), 1);
        assert_eq!(EpgPositionDomain::new(1_000_000).offset(), 1_000_000);
    }

    #[test]
    fn hybrid_suffix_is_validated_against_head_dim() {
        let geometry =
            EpgGeometryDescriptor::hybrid_so4(10_000.0, 32, So4Geometry::Biplanar).unwrap();
        assert!(geometry.validate_head_dim(128).is_ok());
        assert_eq!(geometry.so2_dims(128).unwrap(), 96);
        assert!(geometry.validate_head_dim(16).is_err());
    }

    #[test]
    fn position_domain_checks_overflow() {
        let domain = EpgPositionDomain::new(u64::MAX);
        assert_eq!(domain.resolve(1), Err(EpgContractError::PositionOverflow));
    }
}
