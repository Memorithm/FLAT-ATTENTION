//! PVP-F2 vec4<u32> physical projection and WGPU reference.
//!
//! Logical ordering is unchanged from pvp-bitplanes/v1. The physical row is
//! padded to 128-bit groups so each WGPU storage element is one vec4<u32>.
//! This module changes representation width only; butterfly stage order and
//! Boolean semantics are identical to the scalar-u32 PVP reference.

use core::fmt;

pub const FLAT_PVP_VEC4_SCHEMA_V1: &str = "flat.pvp-vec4-u32/v1";
pub const PVP_LOGICAL_SCHEMA_V1: &str = "pvp-bitplanes/v1";
pub const FLAT_PVP_VEC4_WORKGROUP_SIZE: u32 = 64;
pub const FLAT_PVP_VEC4_WGSL: &str = include_str!("../shaders/flat_pvp_vec4.wgsl");

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlatPvpVec4Error {
    ZeroAddresses,
    AddressesNotPowerOfTwo { addresses: usize },
    ZeroGates,
    ArithmeticOverflow,
    StorageLengthMismatch {
        expected_words: usize,
        actual_words: usize,
    },
    NonZeroPadding { major_index: usize },
    IndexSpaceExceeded { value: usize },
    BufferTooSmall {
        required_bytes: u64,
        actual_bytes: u64,
    },
    DispatchLimit { required: u32, maximum: u32 },
    Pipeline(String),
}

impl fmt::Display for FlatPvpVec4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroAddresses => write!(f, "FLAT PVP vec4 requires at least one address"),
            Self::AddressesNotPowerOfTwo { addresses } => {
                write!(f, "FLAT PVP vec4 address count {addresses} is not a power of two")
            }
            Self::ZeroGates => write!(f, "FLAT PVP vec4 requires at least one gate"),
            Self::ArithmeticOverflow => write!(f, "FLAT PVP vec4 size computation overflowed"),
            Self::StorageLengthMismatch {
                expected_words,
                actual_words,
            } => write!(
                f,
                "FLAT PVP vec4 storage has {actual_words} words, expected {expected_words}"
            ),
            Self::NonZeroPadding { major_index } => {
                write!(f, "FLAT PVP vec4 major index {major_index} has non-zero padding")
            }
            Self::IndexSpaceExceeded { value } => {
                write!(f, "FLAT PVP vec4 value {value} exceeds WGSL u32 index space")
            }
            Self::BufferTooSmall {
                required_bytes,
                actual_bytes,
            } => write!(
                f,
                "FLAT PVP vec4 state buffer has {actual_bytes} bytes, requires {required_bytes}"
            ),
            Self::DispatchLimit { required, maximum } => write!(
                f,
                "FLAT PVP vec4 requires {required} workgroups, device maximum is {maximum}"
            ),
            Self::Pipeline(message) => write!(f, "FLAT PVP vec4 pipeline failed: {message}"),
        }
    }
}

impl std::error::Error for FlatPvpVec4Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlatPvpVec4LayoutV1 {
    addresses: usize,
    gates: usize,
    vectors_per_address: usize,
    storage_u32_words: usize,
    logical_bits: usize,
    storage_bits: usize,
}

impl FlatPvpVec4LayoutV1 {
    pub fn new(addresses: usize, gates: usize) -> Result<Self, FlatPvpVec4Error> {
        if addresses == 0 {
            return Err(FlatPvpVec4Error::ZeroAddresses);
        }
        if !addresses.is_power_of_two() {
            return Err(FlatPvpVec4Error::AddressesNotPowerOfTwo { addresses });
        }
        if gates == 0 {
            return Err(FlatPvpVec4Error::ZeroGates);
        }
        let vectors_per_address = gates
            .checked_add(127)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?
            / 128;
        let storage_vectors = addresses
            .checked_mul(vectors_per_address)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        let storage_u32_words = storage_vectors
            .checked_mul(4)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        let logical_bits = addresses
            .checked_mul(gates)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        let storage_bits = storage_u32_words
            .checked_mul(32)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        Ok(Self {
            addresses,
            gates,
            vectors_per_address,
            storage_u32_words,
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
    pub const fn vectors_per_address(self) -> usize {
        self.vectors_per_address
    }

    #[must_use]
    pub const fn storage_u32_words(self) -> usize {
        self.storage_u32_words
    }

    #[must_use]
    pub const fn storage_vectors(self) -> usize {
        self.storage_u32_words / 4
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

    #[must_use]
    pub fn canonical_record(self) -> String {
        format!(
            "{};source_logical={};addresses={};gates={};physical=vec4-u32;order=address-major-vector;vectors_per_address={};storage_bits={};padding_bits={}",
            FLAT_PVP_VEC4_SCHEMA_V1,
            PVP_LOGICAL_SCHEMA_V1,
            self.addresses,
            self.gates,
            self.vectors_per_address,
            self.storage_bits,
            self.padding_bits()
        )
    }

    fn storage_bytes(self) -> Result<u64, FlatPvpVec4Error> {
        let bytes = self
            .storage_u32_words
            .checked_mul(4)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        u64::try_from(bytes).map_err(|_| FlatPvpVec4Error::ArithmeticOverflow)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatPvpVec4BitplanesV1 {
    layout: FlatPvpVec4LayoutV1,
    words: Vec<u32>,
}

impl FlatPvpVec4BitplanesV1 {
    #[must_use]
    pub fn zeroed(layout: FlatPvpVec4LayoutV1) -> Self {
        Self {
            layout,
            words: vec![0; layout.storage_u32_words()],
        }
    }

    pub fn from_words(
        layout: FlatPvpVec4LayoutV1,
        words: Vec<u32>,
    ) -> Result<Self, FlatPvpVec4Error> {
        if words.len() != layout.storage_u32_words() {
            return Err(FlatPvpVec4Error::StorageLengthMismatch {
                expected_words: layout.storage_u32_words(),
                actual_words: words.len(),
            });
        }
        let value = Self { layout, words };
        value.validate_padding_zero()?;
        Ok(value)
    }

    pub fn from_gate_major_u64(
        layout: FlatPvpVec4LayoutV1,
        gate_major: &[u64],
    ) -> Result<Self, FlatPvpVec4Error> {
        let address_words = layout
            .addresses()
            .checked_add(63)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?
            / 64;
        let expected = layout
            .gates()
            .checked_mul(address_words)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        if gate_major.len() != expected {
            return Err(FlatPvpVec4Error::StorageLengthMismatch {
                expected_words: expected,
                actual_words: gate_major.len(),
            });
        }
        if layout.addresses() % 64 != 0 {
            let tail = layout.addresses() % 64;
            let mask = !((1_u64 << tail) - 1);
            for gate in 0..layout.gates() {
                if gate_major[gate * address_words + address_words - 1] & mask != 0 {
                    return Err(FlatPvpVec4Error::NonZeroPadding { major_index: gate });
                }
            }
        }

        let mut output = Self::zeroed(layout);
        let row_words = layout.vectors_per_address() * 4;
        for gate in 0..layout.gates() {
            let source_base = gate
                .checked_mul(address_words)
                .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
            for address in 0..layout.addresses() {
                if gate_major[source_base + address / 64] & (1_u64 << (address % 64)) != 0 {
                    output.words[address * row_words + gate / 32] |= 1_u32 << (gate % 32);
                }
            }
        }
        Ok(output)
    }

    #[must_use]
    pub const fn layout(&self) -> FlatPvpVec4LayoutV1 {
        self.layout
    }

    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn get(&self, address: usize, gate: usize) -> Option<bool> {
        if address >= self.layout.addresses() || gate >= self.layout.gates() {
            return None;
        }
        let row_words = self.layout.vectors_per_address() * 4;
        Some(self.words[address * row_words + gate / 32] & (1_u32 << (gate % 32)) != 0)
    }

    fn validate_padding_zero(&self) -> Result<(), FlatPvpVec4Error> {
        let row_words = self.layout.vectors_per_address() * 4;
        let live_words = self
            .layout
            .gates()
            .checked_add(31)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?
            / 32;
        let tail = self.layout.gates() % 32;

        for address in 0..self.layout.addresses() {
            let base = address * row_words;
            if tail != 0 {
                let mask = !((1_u32 << tail) - 1);
                if self.words[base + live_words - 1] & mask != 0 {
                    return Err(FlatPvpVec4Error::NonZeroPadding { major_index: address });
                }
            }
            for word in live_words..row_words {
                if self.words[base + word] != 0 {
                    return Err(FlatPvpVec4Error::NonZeroPadding { major_index: address });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatPvpVec4StatsV1 {
    pub stages: u32,
    pub logical_gate_xor_ops: u128,
    pub vec4_updates: u128,
    pub storage_vectors: usize,
    pub storage_bits: usize,
    pub scratch_vectors: usize,
}

pub fn pvp_subset_zeta_vec4_host_in_place(
    bitplanes: &mut FlatPvpVec4BitplanesV1,
) -> Result<FlatPvpVec4StatsV1, FlatPvpVec4Error> {
    bitplanes.validate_padding_zero()?;
    let layout = bitplanes.layout;
    let row_words = layout.vectors_per_address() * 4;
    let mut stride = 1_usize;
    while stride < layout.addresses() {
        let block = stride
            .checked_mul(2)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        let mut block_start = 0_usize;
        while block_start < layout.addresses() {
            for offset in 0..stride {
                let source = (block_start + offset) * row_words;
                let target = (block_start + offset + stride) * row_words;
                for vector in 0..layout.vectors_per_address() {
                    let lane = vector * 4;
                    bitplanes.words[target + lane] ^= bitplanes.words[source + lane];
                    bitplanes.words[target + lane + 1] ^= bitplanes.words[source + lane + 1];
                    bitplanes.words[target + lane + 2] ^= bitplanes.words[source + lane + 2];
                    bitplanes.words[target + lane + 3] ^= bitplanes.words[source + lane + 3];
                }
            }
            block_start = block_start
                .checked_add(block)
                .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        }
        stride = block;
    }
    bitplanes.validate_padding_zero()?;
    let pairs = (layout.addresses() / 2) as u128 * u128::from(layout.stages());
    Ok(FlatPvpVec4StatsV1 {
        stages: layout.stages(),
        logical_gate_xor_ops: pairs * layout.gates() as u128,
        vec4_updates: pairs * layout.vectors_per_address() as u128,
        storage_vectors: layout.storage_vectors(),
        storage_bits: layout.storage_bits(),
        scratch_vectors: 0,
    })
}

#[cfg(feature = "wgpu")]
pub struct WgpuPvpVec4Pipeline {
    pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "wgpu")]
impl WgpuPvpVec4Pipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, FlatPvpVec4Error> {
        let pipeline = crate::wgpu_internal::create_pipeline(
            device,
            FLAT_PVP_VEC4_WGSL,
            "flat-pvp-vec4-u32",
            "pvp_subset_zeta_stage_vec4",
        )
        .map_err(FlatPvpVec4Error::Pipeline)?;
        Ok(Self { pipeline })
    }

    pub fn encode_all_stages(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        state: &wgpu::Buffer,
        layout: FlatPvpVec4LayoutV1,
    ) -> Result<(), FlatPvpVec4Error> {
        let required = layout.storage_bytes()?;
        if state.size() < required {
            return Err(FlatPvpVec4Error::BufferTooSmall {
                required_bytes: required,
                actual_bytes: state.size(),
            });
        }
        let addresses =
            u32::try_from(layout.addresses()).map_err(|_| FlatPvpVec4Error::IndexSpaceExceeded {
                value: layout.addresses(),
            })?;
        let vectors_per_address =
            u32::try_from(layout.vectors_per_address()).map_err(|_| {
                FlatPvpVec4Error::IndexSpaceExceeded {
                    value: layout.vectors_per_address(),
                }
            })?;
        let pair_count = addresses / 2;
        let invocations = pair_count
            .checked_mul(vectors_per_address)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        let workgroups = invocations.div_ceil(FLAT_PVP_VEC4_WORKGROUP_SIZE);
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        if workgroups > maximum {
            return Err(FlatPvpVec4Error::DispatchLimit {
                required: workgroups,
                maximum,
            });
        }

        let mut stride = 1_u32;
        while stride < addresses {
            let params = crate::wgpu_internal::encode_u32(&[
                addresses,
                vectors_per_address,
                stride,
                pair_count,
            ]);
            let uniform = crate::wgpu_internal::create_uniform_buffer_init(
                device,
                "flat-pvp-vec4-params",
                &params,
            );
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flat-pvp-vec4-bind-group"),
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
                    label: Some("flat-pvp-vec4-stage"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            stride = stride
                .checked_mul(2)
                .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pvp::{
        pvp_subset_zeta_u32_scalar_in_place, FlatPvpU32BitplanesV1, FlatPvpU32LayoutV1,
    };

    fn fixture_gate_major(addresses: usize, gates: usize) -> Vec<u64> {
        let words_per_gate = addresses.div_ceil(64);
        let mut words = vec![0_u64; gates * words_per_gate];
        for gate in 0..gates {
            for address in 0..addresses {
                if ((gate * 29 + address * 5 + (gate ^ address)) % 19) < 9 {
                    words[gate * words_per_gate + address / 64] |= 1_u64 << (address % 64);
                }
            }
        }
        words
    }

    #[test]
    fn vec4_projection_matches_scalar_u32_logical_bits() {
        for (addresses, gates) in [(16, 1), (32, 65), (64, 129), (128, 257)] {
            let source = fixture_gate_major(addresses, gates);
            let scalar_layout = FlatPvpU32LayoutV1::new(addresses, gates).unwrap();
            let vec4_layout = FlatPvpVec4LayoutV1::new(addresses, gates).unwrap();
            let mut scalar =
                FlatPvpU32BitplanesV1::from_gate_major_u64(scalar_layout, &source).unwrap();
            let mut vec4 =
                FlatPvpVec4BitplanesV1::from_gate_major_u64(vec4_layout, &source).unwrap();

            pvp_subset_zeta_u32_scalar_in_place(&mut scalar).unwrap();
            pvp_subset_zeta_vec4_host_in_place(&mut vec4).unwrap();

            for address in 0..addresses {
                for gate in 0..gates {
                    let scalar_word =
                        scalar.words()[address * scalar_layout.words_per_address() + gate / 32];
                    let scalar_bit = scalar_word & (1_u32 << (gate % 32)) != 0;
                    assert_eq!(vec4.get(address, gate), Some(scalar_bit));
                }
            }
        }
    }

    #[test]
    fn vec4_host_transform_is_self_inverse() {
        let layout = FlatPvpVec4LayoutV1::new(64, 257).unwrap();
        let source = fixture_gate_major(layout.addresses(), layout.gates());
        let original = FlatPvpVec4BitplanesV1::from_gate_major_u64(layout, &source).unwrap();
        let mut value = original.clone();
        pvp_subset_zeta_vec4_host_in_place(&mut value).unwrap();
        pvp_subset_zeta_vec4_host_in_place(&mut value).unwrap();
        assert_eq!(value, original);
    }

    #[test]
    fn vec4_wgsl_parses_and_validates() {
        let module =
            naga::front::wgsl::parse_str(FLAT_PVP_VEC4_WGSL).expect("PVP vec4 WGSL parses");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator.validate(&module).expect("PVP vec4 WGSL validates");
    }
}
