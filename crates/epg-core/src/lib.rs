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
    /// Stable family identifier.
    pub const fn representation_id(self) -> &'static str {
        match self {
            Self::So2 => "epg.so2",
            Self::HybridSo4(So4Geometry::Isoclinic) => "epg.so4.isoclinic",
            Self::HybridSo4(So4Geometry::Biplanar) => "epg.so4.biplanar",
        }
    }
}

/// Exact, versioned identity of one concrete EPG transform contract.
///
/// Unlike the family identifier returned by [`EpgGeometryKind::representation_id`],
/// this key includes the parameters required to distinguish the concrete
/// transform: contract version, geometry family, exact IEEE-754 theta bits,
/// concrete head dimension, and SO(4) suffix dimension. The head dimension is
/// part of the identity because rotary frequencies are defined relative to that
/// dimension. Execution-local position offsets are deliberately absent.
///
/// The canonical [`fmt::Display`] form is suitable for adapters that need a
/// stable string identifier for a generic representation registry or cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpgRepresentationKey {
    version: u32,
    kind: EpgGeometryKind,
    theta_bits: u32,
    head_dim: u32,
    so4_dims: u32,
}

impl EpgRepresentationKey {
    /// EPG contract version carried by this exact key.
    pub const fn version(self) -> u32 {
        self.version
    }

    /// Geometry family carried by this exact key.
    pub const fn kind(self) -> EpgGeometryKind {
        self.kind
    }

    /// Stable family identifier, without the parameter suffix.
    pub const fn family_id(self) -> &'static str {
        self.kind.representation_id()
    }

    /// Exact IEEE-754 bits of the base frequency.
    pub const fn theta_bits(self) -> u32 {
        self.theta_bits
    }

    /// Concrete attention head dimension used by the transform.
    pub const fn head_dim(self) -> u32 {
        self.head_dim
    }

    /// Number of trailing SO(4) dimensions.
    pub const fn so4_dims(self) -> u32 {
        self.so4_dims
    }
}

impl fmt::Display for EpgRepresentationKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}@v{};theta_bits={:08x};head_dim={};so4_dims={}",
            self.kind.representation_id(),
            self.version,
            self.theta_bits,
            self.head_dim,
            self.so4_dims
        )
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

    /// Stable geometry-family identifier.
    ///
    /// This intentionally omits theta, head dimension, and SO(4) dimensions.
    /// Consumers that need exact cache/model compatibility must use
    /// [`Self::representation_key`] with the concrete head dimension.
    pub const fn representation_id(self) -> &'static str {
        self.kind.representation_id()
    }

    /// Exact versioned representation key for a concrete head dimension.
    ///
    /// Construction validates the head dimension and SO(4) suffix before the
    /// key can enter a cache/model compatibility registry.
    pub fn representation_key(
        self,
        head_dim: u32,
    ) -> Result<EpgRepresentationKey, EpgContractError> {
        self.validate_head_dim(head_dim)?;
        Ok(EpgRepresentationKey {
            version: EPG_CONTRACT_VERSION,
            kind: self.kind,
            theta_bits: self.theta_bits,
            head_dim,
            so4_dims: self.so4_dims,
        })
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
        assert_eq!(
            geometry.representation_key(128).unwrap().to_string(),
            "epg.so4.isoclinic@v1;theta_bits=461c4000;head_dim=128;so4_dims=32"
        );
        assert_eq!(EpgPositionDomain::new(3).offset(), 3);
        assert_eq!(EpgPositionDomain::new(30_000).offset(), 30_000);
    }

    #[test]
    fn exact_representation_key_distinguishes_transform_parameters() {
        let base = EpgGeometryDescriptor::hybrid_so4(10_000.0, 32, So4Geometry::Isoclinic).unwrap();
        let different_theta =
            EpgGeometryDescriptor::hybrid_so4(500_000.0, 32, So4Geometry::Isoclinic).unwrap();
        let different_tail =
            EpgGeometryDescriptor::hybrid_so4(10_000.0, 64, So4Geometry::Isoclinic).unwrap();
        let different_family =
            EpgGeometryDescriptor::hybrid_so4(10_000.0, 32, So4Geometry::Biplanar).unwrap();

        let base_128 = base.representation_key(128).unwrap();
        assert_ne!(base_128, base.representation_key(64).unwrap());
        assert_ne!(base_128, different_theta.representation_key(128).unwrap());
        assert_ne!(base_128, different_tail.representation_key(128).unwrap());
        assert_ne!(base_128, different_family.representation_key(128).unwrap());
        assert_eq!(base_128.head_dim(), 128);
        assert_eq!(base_128.theta_bits(), 0x461c_4000);
        assert_eq!(
            different_theta
                .representation_key(128)
                .unwrap()
                .theta_bits(),
            0x48f4_2400
        );
    }

    #[test]
    fn exact_representation_key_rejects_invalid_concrete_shape() {
        let geometry =
            EpgGeometryDescriptor::hybrid_so4(10_000.0, 32, So4Geometry::Isoclinic).unwrap();
        assert_eq!(
            geometry.representation_key(16),
            Err(EpgContractError::InvalidSo4Tail {
                head_dim: 16,
                so4_dims: 32,
            })
        );
        assert_eq!(
            geometry.representation_key(127),
            Err(EpgContractError::InvalidHeadDim(127))
        );
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
