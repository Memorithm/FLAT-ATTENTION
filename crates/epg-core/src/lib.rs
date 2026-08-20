//! Runtime-neutral contracts for Elastic Positional Geometry (EPG).
//!
//! This crate describes representation identity and position domains only. It
//! deliberately contains no attention, cache, runtime-policy, or GPU code.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::fmt;

/// Current EPG contract version.
pub const EPG_CONTRACT_VERSION: u32 = 1;

/// Qualified four-dimensional control family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum So4Geometry {
    /// Equal angular frequency in both orthogonal planes of each 4D block.
    Isoclinic,
    /// Consecutive RoPE frequencies in the two orthogonal planes.
    Biplanar,
}

/// Stable positional-geometry family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpgGeometryKind {
    /// Ordinary interleaved SO(2) RoPE over the complete head.
    So2,
    /// Leading SO(2) channels followed by SO(4) blocks.
    HybridSo4(So4Geometry),
}

impl EpgGeometryKind {
    /// Stable capability identifier.
    pub const fn representation_id(self) -> &'static str {
        match self {
            Self::So2 => "epg.so2",
            Self::HybridSo4(So4Geometry::Isoclinic) => "epg.so4.isoclinic",
            Self::HybridSo4(So4Geometry::Biplanar) => "epg.so4.biplanar",
        }
    }
}

/// Versioned mathematical representation descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpgGeometryDescriptor {
    theta_bits: u32,
    kind: EpgGeometryKind,
    so4_dims: u32,
}

impl EpgGeometryDescriptor {
    /// Construct ordinary SO(2) RoPE.
    pub fn so2(theta: f32) -> Result<Self, EpgContractError> {
        Self::build(theta, EpgGeometryKind::So2, 0)
    }

    /// Construct a hybrid descriptor with a non-empty 4D suffix.
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

    fn build(theta: f32, kind: EpgGeometryKind, so4_dims: u32) -> Result<Self, EpgContractError> {
        if !theta.is_finite() || theta <= 0.0 {
            return Err(EpgContractError::InvalidTheta(theta.to_bits()));
        }
        Ok(Self {
            theta_bits: theta.to_bits(),
            kind,
            so4_dims,
        })
    }

    /// Contract version.
    pub const fn version(self) -> u32 {
        EPG_CONTRACT_VERSION
    }

    /// Base frequency.
    pub const fn theta(self) -> f32 {
        f32::from_bits(self.theta_bits)
    }

    /// Exact base-frequency bits.
    pub const fn theta_bits(self) -> u32 {
        self.theta_bits
    }

    /// Geometry family.
    pub const fn kind(self) -> EpgGeometryKind {
        self.kind
    }

    /// Number of trailing SO(4) dimensions.
    pub const fn so4_dims(self) -> u32 {
        self.so4_dims
    }

    /// Stable representation identifier.
    pub const fn representation_id(self) -> &'static str {
        self.kind.representation_id()
    }

    /// Validate this descriptor for a concrete head dimension.
    pub fn validate_head_dim(self, head_dim: u32) -> Result<(), EpgContractError> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(EpgContractError::InvalidHeadDim(head_dim));
        }
        if self.so4_dims > head_dim {
            return Err(EpgContractError::InvalidSo4Tail {
                head_dim,
                so4_dims: self.so4_dims,
            });
        }
        Ok(())
    }

    /// Number of leading SO(2) dimensions.
    pub fn so2_dims(self, head_dim: u32) -> Result<u32, EpgContractError> {
        self.validate_head_dim(head_dim)?;
        Ok(head_dim - self.so4_dims)
    }
}

/// Absolute origin of one execution-local position domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct EpgPositionDomain(u64);

impl EpgPositionDomain {
    /// Construct a domain from its absolute origin.
    pub const fn new(offset: u64) -> Self {
        Self(offset)
    }

    /// Return the absolute origin.
    pub const fn offset(self) -> u64 {
        self.0
    }

    /// Resolve a local token index.
    pub fn resolve(self, local_index: u64) -> Result<u64, EpgContractError> {
        self.0
            .checked_add(local_index)
            .ok_or(EpgContractError::PositionOverflow)
    }
}

/// Invalid EPG descriptor or position coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpgContractError {
    /// Theta is not finite and positive; payload stores IEEE-754 bits.
    InvalidTheta(u32),
    /// Head dimension is zero or odd.
    InvalidHeadDim(u32),
    /// SO(4) suffix is zero or not divisible by four.
    InvalidSo4Dims(u32),
    /// SO(4) suffix exceeds the concrete head dimension.
    InvalidSo4Tail {
        /// Head dimension.
        head_dim: u32,
        /// SO(4) suffix dimension.
        so4_dims: u32,
    },
    /// Absolute position arithmetic overflowed.
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
            Self::InvalidHeadDim(value) => write!(f, "invalid EPG head dimension {value}"),
            Self::InvalidSo4Dims(value) => write!(f, "invalid EPG SO(4) suffix {value}"),
            Self::InvalidSo4Tail { head_dim, so4_dims } => {
                write!(
                    f,
                    "EPG SO(4) suffix {so4_dims} exceeds head dimension {head_dim}"
                )
            }
            Self::PositionOverflow => write!(f, "EPG absolute position overflow"),
        }
    }
}

impl std::error::Error for EpgContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representation_identity_does_not_include_position() {
        let geometry =
            EpgGeometryDescriptor::hybrid_so4(10_000.0, 32, So4Geometry::Isoclinic).unwrap();
        assert_eq!(geometry.representation_id(), "epg.so4.isoclinic");
        assert_eq!(EpgPositionDomain::new(3).offset(), 3);
        assert_eq!(EpgPositionDomain::new(30_000).offset(), 30_000);
    }

    #[test]
    fn descriptor_and_position_validation_are_checked() {
        let geometry =
            EpgGeometryDescriptor::hybrid_so4(10_000.0, 32, So4Geometry::Biplanar).unwrap();
        assert_eq!(geometry.so2_dims(128).unwrap(), 96);
        assert!(geometry.validate_head_dim(16).is_err());
        assert_eq!(
            EpgPositionDomain::new(u64::MAX).resolve(1),
            Err(EpgContractError::PositionOverflow)
        );
    }
}
