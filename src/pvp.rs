//! FLAT Pascal Vector Projection (PVP) u32 reference projection.
//!
//! This module derives a portable WGPU physical representation from the frozen
//! logical PVP contract owned by SML-GENIUS and qualified in SciRust:
//!
//! - logical order: address-major gate bitplanes;
//! - host/WGPU physical word: u32;
//! - physical shape: `[K, ceil(G / 32)]`;
//! - transform: staged subset-zeta/Pascal XOR butterfly.
//!
//! The module is research-only. It does not change attention routing or make a
//! performance claim.

use core::fmt;

/// Logical source contract shared with the SciRust PVP reference.
pub const PVP_LOGICAL_SCHEMA_V1: &str = "pvp-bitplanes/v1";
/// FLAT portable u32 projection schema.
pub const FLAT_PVP_U32_SCHEMA_V1: &str = "flat.pvp-u32/v1";
/// Portable scalar-WGSL workgroup size.
pub const FLAT_PVP_SCALAR_WORKGROUP_SIZE: u32 = 64;
/// Handwritten scalar-u32 reference shader.
pub const FLAT_PVP_SCALAR_WGSL: &str = include_str!("../shaders/flat_pvp_scalar.wgsl");

/// Fail-closed PVP representation/execution error.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlatPvpError {
    /// Address domain is empty.
    ZeroAddresses,
    /// Address count is not a power of two.
    AddressesNotPowerOfTwo { addresses: usize },
    /// Gate bank is empty.
    ZeroGates,
    /// Checked arithmetic overflowed.
    ArithmeticOverflow,
    /// Packed storage has an unexpected word count.
    StorageLengthMismatch {
        expected_words: usize,
        actual_words: usize,
    },
    /// Canonical tail padding is non-zero.
    NonZeroPadding { major_index: usize },
    /// WGSL u32 index space cannot represent a dimension.
    IndexSpaceExceeded { value: usize },
    /// Bound state buffer is too small.
    BufferTooSmall {
        required_bytes: u64,
        actual_bytes: u64,
    },
    /// Required dispatch exceeds the device limit.
    DispatchLimit { required: u32, maximum: u32 },
    /// WGPU pipeline validation failed.
    Pipeline(String),
}

impl fmt::Display for FlatPvpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroAddresses => write!(f, "FLAT PVP requires at least one address"),
            Self::AddressesNotPowerOfTwo { addresses } => {
                write!(
                    f,
                    "FLAT PVP address count {addresses} is not a power of two"
                )
            }
            Self::ZeroGates => write!(f, "FLAT PVP requires at least one gate"),
            Self::ArithmeticOverflow => write!(f, "FLAT PVP size computation overflowed"),
            Self::StorageLengthMismatch {
                expected_words,
                actual_words,
            } => write!(
                f,
                "FLAT PVP storage has {actual_words} u32 words, expected {expected_words}"
            ),
            Self::NonZeroPadding { major_index } => {
                write!(
                    f,
                    "FLAT PVP canonical padding is non-zero at major index {major_index}"
                )
            }
            Self::IndexSpaceExceeded { value } => {
                write!(f, "FLAT PVP value {value} exceeds WGSL u32 index space")
            }
            Self::BufferTooSmall {
                required_bytes,
                actual_bytes,
            } => write!(
                f,
                "FLAT PVP state buffer has {actual_bytes} bytes, requires {required_bytes}"
            ),
            Self::DispatchLimit { required, maximum } => write!(
                f,
                "FLAT PVP requires {required} workgroups, device maximum is {maximum}"
            ),
            Self::Pipeline(message) => write!(f, "FLAT PVP pipeline failed: {message}"),
        }
    }
}

impl std::error::Error for FlatPvpError {}

/// Checked portable GPU projection of the PVP logical bitplane layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlatPvpU32LayoutV1 {
    addresses: usize,
    gates: usize,
    words_per_address: usize,
    storage_words: usize,
    logical_bits: usize,
    storage_bits: usize,
}

impl FlatPvpU32LayoutV1 {
    /// Construct a checked address-major u32 layout.
    pub fn new(addresses: usize, gates: usize) -> Result<Self, FlatPvpError> {
        if addresses == 0 {
            return Err(FlatPvpError::ZeroAddresses);
        }
        if !addresses.is_power_of_two() {
            return Err(FlatPvpError::AddressesNotPowerOfTwo { addresses });
        }
        if gates == 0 {
            return Err(FlatPvpError::ZeroGates);
        }
        let words_per_address = gates.div_ceil(32);
        let storage_words = addresses
            .checked_mul(words_per_address)
            .ok_or(FlatPvpError::ArithmeticOverflow)?;
        let logical_bits = addresses
            .checked_mul(gates)
            .ok_or(FlatPvpError::ArithmeticOverflow)?;
        let storage_bits = storage_words
            .checked_mul(32)
            .ok_or(FlatPvpError::ArithmeticOverflow)?;
        Ok(Self {
            addresses,
            gates,
            words_per_address,
            storage_words,
            logical_bits,
            storage_bits,
        })
    }

    #[must_use]
    pub const fn addresses(self) -> usize {
        self.addresses
    }

    #[must_use]
    pub const fn gates(self) -> usize {
        self.gates
    }

    #[must_use]
    pub const fn words_per_address(self) -> usize {
        self.words_per_address
    }

    #[must_use]
    pub const fn storage_words(self) -> usize {
        self.storage_words
    }

    #[must_use]
    pub const fn logical_bits(self) -> usize {
        self.logical_bits
    }

    #[must_use]
    pub const fn storage_bits(self) -> usize {
        self.storage_bits
    }

    #[must_use]
    pub const fn padding_bits(self) -> usize {
        self.storage_bits - self.logical_bits
    }

    #[must_use]
    pub const fn stages(self) -> u32 {
        self.addresses.ilog2()
    }

    /// Stable contract identity. The logical source schema is explicit because
    /// the physical u32 packing differs from SciRust's CPU u64 packing.
    #[must_use]
    pub fn canonical_record(self) -> String {
        format!(
            "{};source_logical={};addresses={};gates={};word_bits=32;order=address-major-gate-word;words_per_address={};storage_bits={};padding_bits={}",
            FLAT_PVP_U32_SCHEMA_V1,
            PVP_LOGICAL_SCHEMA_V1,
            self.addresses,
            self.gates,
            self.words_per_address,
            self.storage_bits,
            self.padding_bits()
        )
    }

    fn gate_major_words(self) -> Result<usize, FlatPvpError> {
        self.gates
            .checked_mul(self.addresses.div_ceil(64))
            .ok_or(FlatPvpError::ArithmeticOverflow)
    }

    #[cfg(feature = "wgpu")]
    fn storage_bytes(self) -> Result<u64, FlatPvpError> {
        let bytes = self
            .storage_words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(FlatPvpError::ArithmeticOverflow)?;
        u64::try_from(bytes).map_err(|_| FlatPvpError::ArithmeticOverflow)
    }
}

/// Owned canonical FLAT address-major u32 bitplanes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatPvpU32BitplanesV1 {
    layout: FlatPvpU32LayoutV1,
    words: Vec<u32>,
}

impl FlatPvpU32BitplanesV1 {
    #[must_use]
    pub fn zeroed(layout: FlatPvpU32LayoutV1) -> Self {
        Self {
            layout,
            words: vec![0; layout.storage_words()],
        }
    }

    pub fn from_words(layout: FlatPvpU32LayoutV1, words: Vec<u32>) -> Result<Self, FlatPvpError> {
        if words.len() != layout.storage_words() {
            return Err(FlatPvpError::StorageLengthMismatch {
                expected_words: layout.storage_words(),
                actual_words: words.len(),
            });
        }
        let value = Self { layout, words };
        value.validate_padding_zero()?;
        Ok(value)
    }

    /// Project the same canonical gate-major u64 logical source accepted by
    /// SciRust PVP-0 into FLAT's portable u32 address-major storage.
    pub fn from_gate_major_u64(
        layout: FlatPvpU32LayoutV1,
        gate_major: &[u64],
    ) -> Result<Self, FlatPvpError> {
        validate_gate_major_u64(layout, gate_major)?;
        let address_words = layout.addresses().div_ceil(64);
        let mut output = Self::zeroed(layout);
        for gate in 0..layout.gates() {
            let source_base = gate
                .checked_mul(address_words)
                .ok_or(FlatPvpError::ArithmeticOverflow)?;
            for address in 0..layout.addresses() {
                if gate_major[source_base + address / 64] & (1_u64 << (address % 64)) != 0 {
                    let index = address
                        .checked_mul(layout.words_per_address())
                        .and_then(|base| base.checked_add(gate / 32))
                        .ok_or(FlatPvpError::ArithmeticOverflow)?;
                    output.words[index] |= 1_u32 << (gate % 32);
                }
            }
        }
        Ok(output)
    }

    #[must_use]
    pub const fn layout(&self) -> FlatPvpU32LayoutV1 {
        self.layout
    }

    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    fn validate_padding_zero(&self) -> Result<(), FlatPvpError> {
        let tail = self.layout.gates() % 32;
        if tail == 0 {
            return Ok(());
        }
        let mask = !((1_u32 << tail) - 1);
        let last = self.layout.words_per_address() - 1;
        for address in 0..self.layout.addresses() {
            let index = address
                .checked_mul(self.layout.words_per_address())
                .and_then(|base| base.checked_add(last))
                .ok_or(FlatPvpError::ArithmeticOverflow)?;
            if self.words[index] & mask != 0 {
                return Err(FlatPvpError::NonZeroPadding {
                    major_index: address,
                });
            }
        }
        Ok(())
    }
}

/// Exact host accounting for one full scalar-u32 transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatPvpTransformStatsV1 {
    pub stages: u32,
    pub logical_gate_xor_ops: u128,
    pub packed_u32_updates: u128,
    pub storage_words: usize,
    pub storage_bits: usize,
    pub scratch_words: usize,
}

/// Apply the scalar-u32 PVP butterfly in place.
pub fn pvp_subset_zeta_u32_scalar_in_place(
    bitplanes: &mut FlatPvpU32BitplanesV1,
) -> Result<FlatPvpTransformStatsV1, FlatPvpError> {
    bitplanes.validate_padding_zero()?;
    let layout = bitplanes.layout;
    let words_per_address = layout.words_per_address();
    let mut stride = 1_usize;
    while stride < layout.addresses() {
        let block = stride
            .checked_mul(2)
            .ok_or(FlatPvpError::ArithmeticOverflow)?;
        let mut block_start = 0_usize;
        while block_start < layout.addresses() {
            for offset in 0..stride {
                let source_address = block_start + offset;
                let target_address = source_address + stride;
                let source_base = source_address
                    .checked_mul(words_per_address)
                    .ok_or(FlatPvpError::ArithmeticOverflow)?;
                let target_base = target_address
                    .checked_mul(words_per_address)
                    .ok_or(FlatPvpError::ArithmeticOverflow)?;
                for word in 0..words_per_address {
                    bitplanes.words[target_base + word] ^= bitplanes.words[source_base + word];
                }
            }
            block_start = block_start
                .checked_add(block)
                .ok_or(FlatPvpError::ArithmeticOverflow)?;
        }
        stride = block;
    }
    bitplanes.validate_padding_zero()?;
    let pairs = (layout.addresses() / 2) as u128 * u128::from(layout.stages());
    Ok(FlatPvpTransformStatsV1 {
        stages: layout.stages(),
        logical_gate_xor_ops: pairs * layout.gates() as u128,
        packed_u32_updates: pairs * layout.words_per_address() as u128,
        storage_words: layout.storage_words(),
        storage_bits: layout.storage_bits(),
        scratch_words: 0,
    })
}

/// Independent direct-submask oracle over the canonical gate-major u64 source.
pub fn pvp_direct_subset_oracle_u32(
    layout: FlatPvpU32LayoutV1,
    gate_major: &[u64],
) -> Result<FlatPvpU32BitplanesV1, FlatPvpError> {
    validate_gate_major_u64(layout, gate_major)?;
    let address_words = layout.addresses().div_ceil(64);
    let mut output = FlatPvpU32BitplanesV1::zeroed(layout);
    for gate in 0..layout.gates() {
        let gate_base = gate
            .checked_mul(address_words)
            .ok_or(FlatPvpError::ArithmeticOverflow)?;
        for address in 0..layout.addresses() {
            let mut parity = false;
            let mut submask = address;
            loop {
                parity ^= gate_major[gate_base + submask / 64] & (1_u64 << (submask % 64)) != 0;
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & address;
            }
            if parity {
                let index = address
                    .checked_mul(layout.words_per_address())
                    .and_then(|base| base.checked_add(gate / 32))
                    .ok_or(FlatPvpError::ArithmeticOverflow)?;
                output.words[index] |= 1_u32 << (gate % 32);
            }
        }
    }
    Ok(output)
}

fn validate_gate_major_u64(
    layout: FlatPvpU32LayoutV1,
    gate_major: &[u64],
) -> Result<(), FlatPvpError> {
    let expected = layout.gate_major_words()?;
    if gate_major.len() != expected {
        return Err(FlatPvpError::StorageLengthMismatch {
            expected_words: expected,
            actual_words: gate_major.len(),
        });
    }
    let tail = layout.addresses() % 64;
    if tail == 0 {
        return Ok(());
    }
    let mask = !((1_u64 << tail) - 1);
    let words_per_gate = layout.addresses().div_ceil(64);
    let last = words_per_gate - 1;
    for gate in 0..layout.gates() {
        let index = gate
            .checked_mul(words_per_gate)
            .and_then(|base| base.checked_add(last))
            .ok_or(FlatPvpError::ArithmeticOverflow)?;
        if gate_major[index] & mask != 0 {
            return Err(FlatPvpError::NonZeroPadding { major_index: gate });
        }
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
/// Compiled scalar-u32 PVP stage pipeline.
pub struct WgpuPvpScalarPipeline {
    pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "wgpu")]
impl WgpuPvpScalarPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, FlatPvpError> {
        let pipeline = crate::wgpu_internal::create_pipeline(
            device,
            FLAT_PVP_SCALAR_WGSL,
            "flat-pvp-scalar-u32",
            "pvp_subset_zeta_stage",
        )
        .map_err(FlatPvpError::Pipeline)?;
        Ok(Self { pipeline })
    }

    /// Encode every butterfly stage over an already-resident read/write state.
    pub fn encode_all_stages(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        state: &wgpu::Buffer,
        layout: FlatPvpU32LayoutV1,
    ) -> Result<(), FlatPvpError> {
        let required = layout.storage_bytes()?;
        if state.size() < required {
            return Err(FlatPvpError::BufferTooSmall {
                required_bytes: required,
                actual_bytes: state.size(),
            });
        }
        let addresses =
            u32::try_from(layout.addresses()).map_err(|_| FlatPvpError::IndexSpaceExceeded {
                value: layout.addresses(),
            })?;
        let words_per_address = u32::try_from(layout.words_per_address()).map_err(|_| {
            FlatPvpError::IndexSpaceExceeded {
                value: layout.words_per_address(),
            }
        })?;
        let pair_count = addresses / 2;
        let invocations = pair_count
            .checked_mul(words_per_address)
            .ok_or(FlatPvpError::ArithmeticOverflow)?;
        let workgroups = invocations.div_ceil(FLAT_PVP_SCALAR_WORKGROUP_SIZE);
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        if workgroups > maximum {
            return Err(FlatPvpError::DispatchLimit {
                required: workgroups,
                maximum,
            });
        }

        let mut stride = 1_u32;
        while stride < addresses {
            let params = crate::wgpu_internal::encode_u32(&[
                addresses,
                words_per_address,
                stride,
                pair_count,
            ]);
            let uniform = crate::wgpu_internal::create_uniform_buffer_init(
                device,
                "flat-pvp-scalar-params",
                &params,
            );
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flat-pvp-scalar-bind-group"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: state.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("flat-pvp-scalar-stage"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            stride = stride
                .checked_mul(2)
                .ok_or(FlatPvpError::ArithmeticOverflow)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_gate_major(layout: FlatPvpU32LayoutV1) -> Vec<u64> {
        let words_per_gate = layout.addresses().div_ceil(64);
        let mut words = vec![0_u64; layout.gates() * words_per_gate];
        for gate in 0..layout.gates() {
            for address in 0..layout.addresses() {
                let bit = ((gate * 19 + address * 11 + (gate ^ address)) % 13) < 6;
                if bit {
                    words[gate * words_per_gate + address / 64] |= 1_u64 << (address % 64);
                }
            }
        }
        words
    }

    #[test]
    fn layout_is_versioned_and_exact() {
        let layout = FlatPvpU32LayoutV1::new(128, 65).unwrap();
        assert_eq!(layout.words_per_address(), 3);
        assert_eq!(layout.storage_words(), 384);
        assert_eq!(layout.logical_bits(), 8_320);
        assert_eq!(layout.storage_bits(), 12_288);
        assert_eq!(layout.padding_bits(), 3_968);
        assert_eq!(layout.stages(), 7);
        assert!(layout.canonical_record().contains(PVP_LOGICAL_SCHEMA_V1));
        assert!(layout.canonical_record().contains(FLAT_PVP_U32_SCHEMA_V1));
    }

    #[test]
    fn projected_scalar_matches_independent_direct_oracle() {
        let layout = FlatPvpU32LayoutV1::new(16, 97).unwrap();
        let source = fixture_gate_major(layout);
        let expected = pvp_direct_subset_oracle_u32(layout, &source).unwrap();
        let mut actual = FlatPvpU32BitplanesV1::from_gate_major_u64(layout, &source).unwrap();
        let stats = pvp_subset_zeta_u32_scalar_in_place(&mut actual).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(stats.stages, 4);
        assert_eq!(stats.logical_gate_xor_ops, 3_104);
        assert_eq!(stats.packed_u32_updates, 128);
        assert_eq!(stats.scratch_words, 0);
    }

    #[test]
    fn projected_scalar_is_self_inverse() {
        let layout = FlatPvpU32LayoutV1::new(64, 129).unwrap();
        let source = fixture_gate_major(layout);
        let original = FlatPvpU32BitplanesV1::from_gate_major_u64(layout, &source).unwrap();
        let mut value = original.clone();
        pvp_subset_zeta_u32_scalar_in_place(&mut value).unwrap();
        pvp_subset_zeta_u32_scalar_in_place(&mut value).unwrap();
        assert_eq!(value, original);
    }

    #[test]
    fn scalar_wgsl_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(FLAT_PVP_SCALAR_WGSL)
            .expect("FLAT PVP scalar WGSL must parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .expect("FLAT PVP scalar WGSL must validate");
    }
}
