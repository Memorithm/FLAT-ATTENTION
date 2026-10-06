//! Research-only gate-major address-packed PVP realization.
//!
//! Each `vec4<u32>` carries 128 consecutive Boolean addresses of one gate,
//! rather than 128 gates at one address as in PVP2. Up to seven low-address
//! butterfly stages run in logical register values; remaining stages exchange
//! whole vectors. This is a separate physical contract, not a default route or
//! evidence of a performance improvement. Host conversion and residency costs
//! must be included when comparing different physical layouts.

use core::fmt;

pub const FLAT_PVP_PACKED_SCHEMA_V1: &str = "flat.pvp-gate-major-address-packed-vec4/v1";
pub const FLAT_PVP_PACKED_WORKGROUP_SIZE: u32 = 64;
pub const FLAT_PVP_PACKED_WGSL: &str = include_str!("../shaders/flat_pvp_gate_major_packed.wgsl");

/// Checked representation/execution rejection for the separate packed candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlatPvpPackedError {
    ZeroAddresses,
    AddressesNotPowerOfTwo {
        addresses: usize,
    },
    ZeroGates,
    ArithmeticOverflow,
    StorageLengthMismatch {
        expected_words: usize,
        actual_words: usize,
    },
    NonZeroPadding {
        gate: usize,
    },
    IndexSpaceExceeded {
        value: usize,
    },
    BufferTooSmall {
        required_bytes: u64,
        actual_bytes: u64,
    },
    MissingStorageUsage,
    DeviceLimit {
        name: &'static str,
        required: u64,
        maximum: u64,
    },
    Pipeline(String),
}

impl fmt::Display for FlatPvpPackedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroAddresses => write!(f, "packed PVP requires at least one address"),
            Self::AddressesNotPowerOfTwo { addresses } => write!(
                f,
                "packed PVP address count {addresses} is not a power of two"
            ),
            Self::ZeroGates => write!(f, "packed PVP requires at least one gate"),
            Self::ArithmeticOverflow => write!(f, "packed PVP checked arithmetic overflow"),
            Self::StorageLengthMismatch {
                expected_words,
                actual_words,
            } => write!(
                f,
                "packed PVP has {actual_words} words, expected {expected_words}"
            ),
            Self::NonZeroPadding { gate } => {
                write!(f, "packed PVP gate {gate} has nonzero address padding")
            }
            Self::IndexSpaceExceeded { value } => {
                write!(f, "packed PVP value {value} exceeds WGSL u32 index space")
            }
            Self::BufferTooSmall {
                required_bytes,
                actual_bytes,
            } => write!(
                f,
                "packed PVP buffer has {actual_bytes} bytes, requires {required_bytes}"
            ),
            Self::MissingStorageUsage => write!(f, "packed PVP state lacks STORAGE usage"),
            Self::DeviceLimit {
                name,
                required,
                maximum,
            } => write!(
                f,
                "packed PVP {name} requires {required}, device maximum is {maximum}"
            ),
            Self::Pipeline(message) => {
                write!(f, "packed PVP pipeline validation failed: {message}")
            }
        }
    }
}

impl std::error::Error for FlatPvpPackedError {}

/// Physical shape `[G, ceil(K / 128)] vec4<u32>`; unused address bits are zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlatPvpPackedLayoutV1 {
    addresses: usize,
    gates: usize,
    vectors_per_gate: usize,
    storage_u32_words: usize,
    logical_bits: usize,
    storage_bits: usize,
}

impl FlatPvpPackedLayoutV1 {
    pub fn new(addresses: usize, gates: usize) -> Result<Self, FlatPvpPackedError> {
        if addresses == 0 {
            return Err(FlatPvpPackedError::ZeroAddresses);
        }
        if !addresses.is_power_of_two() {
            return Err(FlatPvpPackedError::AddressesNotPowerOfTwo { addresses });
        }
        if gates == 0 {
            return Err(FlatPvpPackedError::ZeroGates);
        }
        let vectors_per_gate = addresses.div_ceil(128);
        let storage_u32_words = gates
            .checked_mul(vectors_per_gate)
            .and_then(|v| v.checked_mul(4))
            .ok_or(FlatPvpPackedError::ArithmeticOverflow)?;
        let logical_bits = addresses
            .checked_mul(gates)
            .ok_or(FlatPvpPackedError::ArithmeticOverflow)?;
        let storage_bits = storage_u32_words
            .checked_mul(32)
            .ok_or(FlatPvpPackedError::ArithmeticOverflow)?;
        // Ensure even a host-only layout has a representable byte allocation.
        storage_u32_words
            .checked_mul(4)
            .ok_or(FlatPvpPackedError::ArithmeticOverflow)?;
        Ok(Self {
            addresses,
            gates,
            vectors_per_gate,
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
    pub const fn vectors_per_gate(self) -> usize {
        self.vectors_per_gate
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
    pub fn logical_dispatches(self) -> u32 {
        if self.stages() == 0 {
            0
        } else {
            1 + self.stages().saturating_sub(7)
        }
    }
    pub fn storage_bytes(self) -> Result<u64, FlatPvpPackedError> {
        let bytes = self
            .storage_u32_words
            .checked_mul(4)
            .ok_or(FlatPvpPackedError::ArithmeticOverflow)?;
        u64::try_from(bytes).map_err(|_| FlatPvpPackedError::ArithmeticOverflow)
    }
    #[must_use]
    pub fn canonical_record(self) -> String {
        format!("{FLAT_PVP_PACKED_SCHEMA_V1};source_logical=pvp-bitplanes/v1;addresses={};gates={};word_bits=32;vector_words=4;order=gate-major-address-vector;vectors_per_gate={};storage_bits={};padding_bits={}", self.addresses, self.gates, self.vectors_per_gate, self.storage_bits, self.padding_bits())
    }
}

/// Owned canonical gate-major address-packed words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatPvpPackedBitplanesV1 {
    layout: FlatPvpPackedLayoutV1,
    words: Vec<u32>,
}

impl FlatPvpPackedBitplanesV1 {
    #[must_use]
    pub fn zeroed(layout: FlatPvpPackedLayoutV1) -> Self {
        Self {
            layout,
            words: vec![0; layout.storage_u32_words()],
        }
    }
    pub fn from_words(
        layout: FlatPvpPackedLayoutV1,
        words: Vec<u32>,
    ) -> Result<Self, FlatPvpPackedError> {
        if words.len() != layout.storage_u32_words() {
            return Err(FlatPvpPackedError::StorageLengthMismatch {
                expected_words: layout.storage_u32_words(),
                actual_words: words.len(),
            });
        }
        let value = Self { layout, words };
        value.validate_padding_zero()?;
        Ok(value)
    }
    /// Convert canonical `[G, ceil(K/64)] u64` without a bit-matrix transpose.
    pub fn from_gate_major_u64(
        layout: FlatPvpPackedLayoutV1,
        input: &[u64],
    ) -> Result<Self, FlatPvpPackedError> {
        let words_per_gate = layout.addresses().div_ceil(64);
        let expected = layout
            .gates()
            .checked_mul(words_per_gate)
            .ok_or(FlatPvpPackedError::ArithmeticOverflow)?;
        if input.len() != expected {
            return Err(FlatPvpPackedError::StorageLengthMismatch {
                expected_words: expected,
                actual_words: input.len(),
            });
        }
        if layout.addresses() < 64 {
            let mask = (1_u64 << layout.addresses()) - 1;
            for (gate, &word) in input.iter().enumerate() {
                if word & !mask != 0 {
                    return Err(FlatPvpPackedError::NonZeroPadding { gate });
                }
            }
        }
        let mut value = Self::zeroed(layout);
        let row_words = layout.vectors_per_gate() * 4;
        for gate in 0..layout.gates() {
            for word in 0..words_per_gate {
                let source = input[gate * words_per_gate + word];
                value.words[gate * row_words + word * 2] = source as u32;
                value.words[gate * row_words + word * 2 + 1] = (source >> 32) as u32;
            }
        }
        value.validate_padding_zero()?;
        Ok(value)
    }
    pub fn to_gate_major_u64(&self) -> Result<Vec<u64>, FlatPvpPackedError> {
        self.validate_padding_zero()?;
        let words_per_gate = self.layout.addresses().div_ceil(64);
        let length = self
            .layout
            .gates()
            .checked_mul(words_per_gate)
            .ok_or(FlatPvpPackedError::ArithmeticOverflow)?;
        let mut output = vec![0_u64; length];
        let row_words = self.layout.vectors_per_gate() * 4;
        for gate in 0..self.layout.gates() {
            for word in 0..words_per_gate {
                let index = gate * row_words + word * 2;
                output[gate * words_per_gate + word] =
                    u64::from(self.words[index]) | (u64::from(self.words[index + 1]) << 32);
            }
        }
        Ok(output)
    }
    #[must_use]
    pub const fn layout(&self) -> FlatPvpPackedLayoutV1 {
        self.layout
    }
    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }
    #[must_use]
    pub fn get(&self, address: usize, gate: usize) -> Option<bool> {
        if address >= self.layout.addresses() || gate >= self.layout.gates() {
            return None;
        }
        let index = gate * self.layout.vectors_per_gate() * 4 + address / 32;
        Some(self.words[index] & (1_u32 << (address % 32)) != 0)
    }
    pub fn validate_padding_zero(&self) -> Result<(), FlatPvpPackedError> {
        // K is a power of two: only K < 128 needs physical padding.
        if self.layout.addresses() >= 128 {
            return Ok(());
        }
        let addresses = self.layout.addresses();
        for gate in 0..self.layout.gates() {
            for lane in 0..4 {
                let remaining = addresses.saturating_sub(lane * 32);
                let valid_mask = match remaining {
                    0 => 0,
                    1..=31 => (1_u32 << remaining) - 1,
                    _ => u32::MAX,
                };
                if self.words[gate * 4 + lane] & !valid_mask != 0 {
                    return Err(FlatPvpPackedError::NonZeroPadding { gate });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatPvpPackedAccountingV1 {
    pub stages: u32,
    pub logical_gate_xor_ops: u128,
    pub logical_dispatches: u32,
    pub prefix_stages: u32,
    pub storage_u32_words: usize,
    pub padding_bits: usize,
    pub scratch_u32_words: usize,
}

fn accounting(layout: FlatPvpPackedLayoutV1) -> FlatPvpPackedAccountingV1 {
    FlatPvpPackedAccountingV1 {
        stages: layout.stages(),
        logical_gate_xor_ops: (layout.addresses() / 2) as u128
            * u128::from(layout.stages())
            * layout.gates() as u128,
        logical_dispatches: layout.logical_dispatches(),
        prefix_stages: layout.stages().min(7),
        storage_u32_words: layout.storage_u32_words(),
        padding_bits: layout.padding_bits(),
        scratch_u32_words: 0,
    }
}

/// Exact host emulation of the separate packed realization, not the oracle.
pub fn pvp_subset_zeta_packed_host_in_place(
    value: &mut FlatPvpPackedBitplanesV1,
) -> Result<FlatPvpPackedAccountingV1, FlatPvpPackedError> {
    value.validate_padding_zero()?;
    let layout = value.layout;
    let row_words = layout.vectors_per_gate() * 4;
    for gate in 0..layout.gates() {
        let row = &mut value.words[gate * row_words..(gate + 1) * row_words];
        for vector in row.chunks_exact_mut(4) {
            for (shift, mask) in [
                (1_u32, 0x5555_5555_u32),
                (2, 0x3333_3333),
                (4, 0x0f0f_0f0f),
                (8, 0x00ff_00ff),
                (16, 0x0000_ffff),
            ] {
                if (shift as usize) < layout.addresses() {
                    for word in vector.iter_mut() {
                        *word ^= (*word & mask) << shift;
                    }
                }
            }
            if layout.addresses() > 32 {
                vector[1] ^= vector[0];
                vector[3] ^= vector[2];
            }
            if layout.addresses() > 64 {
                vector[2] ^= vector[0];
                vector[3] ^= vector[1];
            }
        }
        let mut stride = 4_usize;
        while stride < row_words {
            for block in (0..row_words).step_by(stride * 2) {
                for offset in 0..stride {
                    row[block + stride + offset] ^= row[block + offset];
                }
            }
            stride *= 2;
        }
    }
    value.validate_padding_zero()?;
    Ok(accounting(layout))
}

/// Independent direct-submask oracle reading the original logical coefficients.
///
/// This intentionally expensive routine is for bounded correctness fixtures,
/// never a replacement for a large-domain kernel or a timing control.
pub fn pvp_direct_subset_packed_oracle(
    layout: FlatPvpPackedLayoutV1,
    input: &[u64],
) -> Result<FlatPvpPackedBitplanesV1, FlatPvpPackedError> {
    // Validate source independently of the transform, including input padding.
    FlatPvpPackedBitplanesV1::from_gate_major_u64(layout, input)?;
    let words_per_gate = layout.addresses().div_ceil(64);
    let mut output = FlatPvpPackedBitplanesV1::zeroed(layout);
    let row_words = layout.vectors_per_gate() * 4;
    for gate in 0..layout.gates() {
        for address in 0..layout.addresses() {
            let mut parity = false;
            let mut submask = address;
            loop {
                parity ^=
                    input[gate * words_per_gate + submask / 64] & (1_u64 << (submask % 64)) != 0;
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & address;
            }
            if parity {
                output.words[gate * row_words + address / 32] |= 1_u32 << (address % 32);
            }
        }
    }
    Ok(output)
}

#[cfg(feature = "wgpu")]
pub struct WgpuPvpPackedPipeline {
    prefix: wgpu::ComputePipeline,
    suffix: wgpu::ComputePipeline,
}

#[cfg(feature = "wgpu")]
struct PreparedStage {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _uniform: wgpu::Buffer,
    workgroups: u32,
}

/// Immutable plan for one geometry and one state buffer.
///
/// Construct uniforms and bind groups once with `prepare`. Repeated `encode`
/// creates no buffers or bind groups. State and uniforms are retained for the
/// complete plan lifetime. Pipeline/plan preparation is a separate cost and is
/// not evidence that host encoding, submission or execution costs disappear.
#[cfg(feature = "wgpu")]
pub struct WgpuPvpPackedPlan {
    stages: Vec<PreparedStage>,
    _state: wgpu::Buffer,
    layout: FlatPvpPackedLayoutV1,
}

#[cfg(feature = "wgpu")]
impl WgpuPvpPackedPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, FlatPvpPackedError> {
        let prefix = crate::wgpu_internal::create_pipeline(
            device,
            FLAT_PVP_PACKED_WGSL,
            "flat-pvp-packed-prefix",
            "pvp_packed_prefix",
        )
        .map_err(FlatPvpPackedError::Pipeline)?;
        let suffix = crate::wgpu_internal::create_pipeline(
            device,
            FLAT_PVP_PACKED_WGSL,
            "flat-pvp-packed-suffix",
            "pvp_packed_suffix",
        )
        .map_err(FlatPvpPackedError::Pipeline)?;
        Ok(Self { prefix, suffix })
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        state: &wgpu::Buffer,
        layout: FlatPvpPackedLayoutV1,
    ) -> Result<WgpuPvpPackedPlan, FlatPvpPackedError> {
        let required_bytes = layout.storage_bytes()?;
        if state.size() < required_bytes {
            return Err(FlatPvpPackedError::BufferTooSmall {
                required_bytes,
                actual_bytes: state.size(),
            });
        }
        if !state.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(FlatPvpPackedError::MissingStorageUsage);
        }
        let limits = device.limits();
        for (name, required, maximum) in [
            ("buffer bytes", required_bytes, limits.max_buffer_size),
            (
                "storage binding bytes",
                required_bytes,
                limits.max_storage_buffer_binding_size,
            ),
            (
                "workgroup invocations",
                u64::from(FLAT_PVP_PACKED_WORKGROUP_SIZE),
                u64::from(limits.max_compute_invocations_per_workgroup),
            ),
            (
                "workgroup x",
                u64::from(FLAT_PVP_PACKED_WORKGROUP_SIZE),
                u64::from(limits.max_compute_workgroup_size_x),
            ),
        ] {
            if required > maximum {
                return Err(FlatPvpPackedError::DeviceLimit {
                    name,
                    required,
                    maximum,
                });
            }
        }
        let checked_u32 = |value| {
            u32::try_from(value).map_err(|_| FlatPvpPackedError::IndexSpaceExceeded { value })
        };
        let addresses = checked_u32(layout.addresses())?;
        let gates = checked_u32(layout.gates())?;
        let vectors_per_gate = checked_u32(layout.vectors_per_gate())?;
        let total = checked_u32(layout.storage_vectors())?;
        let mut stages = Vec::with_capacity(layout.logical_dispatches() as usize);
        let mut add_stage = |pipeline: &wgpu::ComputePipeline,
                             stride: u32,
                             invocations: u32|
         -> Result<(), FlatPvpPackedError> {
            let workgroups = invocations.div_ceil(FLAT_PVP_PACKED_WORKGROUP_SIZE);
            if workgroups > limits.max_compute_workgroups_per_dimension {
                return Err(FlatPvpPackedError::DeviceLimit {
                    name: "workgroups x",
                    required: u64::from(workgroups),
                    maximum: u64::from(limits.max_compute_workgroups_per_dimension),
                });
            }
            let params =
                crate::wgpu_internal::encode_u32(&[addresses, gates, vectors_per_gate, stride]);
            let uniform = crate::wgpu_internal::create_uniform_buffer_init(
                device,
                "flat-pvp-packed-params",
                &params,
            );
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flat-pvp-packed-bind-group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: state,
                            offset: 0,
                            size: core::num::NonZeroU64::new(required_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            stages.push(PreparedStage {
                pipeline: pipeline.clone(),
                bind_group,
                _uniform: uniform,
                workgroups,
            });
            Ok(())
        };
        if layout.stages() != 0 {
            add_stage(&self.prefix, 0, total)?;
        }
        let mut stride = 1_u32;
        while stride < vectors_per_gate {
            add_stage(&self.suffix, stride, total / 2)?;
            stride = stride
                .checked_mul(2)
                .ok_or(FlatPvpPackedError::ArithmeticOverflow)?;
        }
        Ok(WgpuPvpPackedPlan {
            stages,
            _state: state.clone(),
            layout,
        })
    }
}

#[cfg(feature = "wgpu")]
impl WgpuPvpPackedPlan {
    #[must_use]
    pub const fn layout(&self) -> FlatPvpPackedLayoutV1 {
        self.layout
    }
    #[must_use]
    pub fn accounting(&self) -> FlatPvpPackedAccountingV1 {
        accounting(self.layout)
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for stage in &self.stages {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-pvp-packed-stage"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&stage.pipeline);
            pass.set_bind_group(0, &stage.bind_group, &[]);
            pass.dispatch_workgroups(stage.workgroups, 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_layout_and_padding_are_explicit() {
        assert_eq!(
            FlatPvpPackedLayoutV1::new(0, 1),
            Err(FlatPvpPackedError::ZeroAddresses)
        );
        assert!(matches!(
            FlatPvpPackedLayoutV1::new(3, 1),
            Err(FlatPvpPackedError::AddressesNotPowerOfTwo { .. })
        ));
        assert_eq!(
            FlatPvpPackedLayoutV1::new(2, 0),
            Err(FlatPvpPackedError::ZeroGates)
        );
        assert_eq!(
            FlatPvpPackedLayoutV1::new(128, usize::MAX),
            Err(FlatPvpPackedError::ArithmeticOverflow)
        );
        let layout = FlatPvpPackedLayoutV1::new(16, 3).unwrap();
        assert_eq!(layout.storage_u32_words(), 12);
        assert_eq!(layout.padding_bits(), 336);
        assert!(layout
            .canonical_record()
            .contains(FLAT_PVP_PACKED_SCHEMA_V1));
        assert!(matches!(
            FlatPvpPackedBitplanesV1::from_words(layout, vec![0; 11]),
            Err(FlatPvpPackedError::StorageLengthMismatch { .. })
        ));
        let mut words = vec![0; 12];
        words[1] = 1;
        assert_eq!(
            FlatPvpPackedBitplanesV1::from_words(layout, words),
            Err(FlatPvpPackedError::NonZeroPadding { gate: 0 })
        );
        assert!(matches!(
            FlatPvpPackedBitplanesV1::from_gate_major_u64(layout, &[1 << 16, 0, 0]),
            Err(FlatPvpPackedError::NonZeroPadding { .. })
        ));
    }

    #[test]
    fn direct_submask_fixture_and_inverse_cover_every_prefix_boundary() {
        for k in [1_usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let layout = FlatPvpPackedLayoutV1::new(k, 3).unwrap();
            let mut source = vec![0_u64; 3 * k.div_ceil(64)];
            for gate in 0..3 {
                for address in 0..k {
                    if ((gate * 13 + address * 31 + (gate ^ address)) % 17) < 8 {
                        source[gate * k.div_ceil(64) + address / 64] |= 1_u64 << (address % 64);
                    }
                }
            }
            let original = FlatPvpPackedBitplanesV1::from_gate_major_u64(layout, &source).unwrap();
            assert_eq!(original.to_gate_major_u64().unwrap(), source);
            let expected = pvp_direct_subset_packed_oracle(layout, &source).unwrap();
            let mut actual = original.clone();
            let stats = pvp_subset_zeta_packed_host_in_place(&mut actual).unwrap();
            assert_eq!(actual, expected, "K={k}");
            assert_eq!(
                stats.logical_dispatches,
                if k == 1 {
                    0
                } else {
                    1 + k.ilog2().saturating_sub(7)
                }
            );
            pvp_subset_zeta_packed_host_in_place(&mut actual).unwrap();
            assert_eq!(actual, original, "inverse K={k}");
        }
    }

    #[test]
    fn packed_shader_validates_without_optional_capabilities() {
        let module =
            naga::front::wgsl::parse_str(FLAT_PVP_PACKED_WGSL).expect("packed PVP WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("packed PVP WGSL validates");
    }
}
