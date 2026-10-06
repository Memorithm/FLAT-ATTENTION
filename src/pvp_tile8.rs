//! PVP3c correctness candidate: fuse three local butterfly stages in workgroup memory.
//!
//! One workgroup owns one tile of eight adjacent addresses and one 128-gate
//! vec4-u32 value per address. The workgroup executes strides 1, 2 and 4 with
//! explicit barriers, then delegates all remaining stages from stride 8 to the
//! qualified PVP2 vec4 kernel.
//!
//! This slice establishes correctness and explicit resource accounting only.
//! It makes no latency, throughput, bandwidth, or routing claim.

#[cfg(feature = "wgpu")]
use crate::pvp_vec4::WgpuPvpVec4Pipeline;
use crate::pvp_vec4::{
    pvp_subset_zeta_vec4_host_in_place, FlatPvpVec4BitplanesV1, FlatPvpVec4Error,
    FlatPvpVec4LayoutV1,
};

pub const FLAT_PVP_TILE8_SCHEMA_V1: &str = "flat.pvp-vec4-tile8/v1";
pub const FLAT_PVP_TILE8_WORKGROUP_SIZE: u32 = 8;
pub const FLAT_PVP_TILE8_WORKGROUP_STORAGE_BYTES: usize = 8 * 16;
pub const FLAT_PVP_TILE8_WGSL: &str = include_str!("../shaders/flat_pvp_vec4_tile8.wgsl");

/// Structural accounting for the tile8 candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatPvpTile8AccountingV1 {
    pub stages: u32,
    pub logical_gate_xor_ops: u128,
    pub vec4_updates: u128,
    pub logical_dispatches: u32,
    pub baseline_logical_dispatches: u32,
    pub state_vectors: usize,
    pub workgroup_storage_bytes: usize,
    pub scratch_state_vectors: usize,
}

/// Apply the exact PVP transform with the first three stages fused per 8-address tile.
pub fn pvp_subset_zeta_tile8_host(
    source: &FlatPvpVec4BitplanesV1,
) -> Result<(FlatPvpVec4BitplanesV1, FlatPvpTile8AccountingV1), FlatPvpVec4Error> {
    let layout = source.layout();
    if layout.addresses() < 8 {
        let mut value = source.clone();
        pvp_subset_zeta_vec4_host_in_place(&mut value)?;
        return Ok((value, accounting(layout)));
    }

    let row_words = layout
        .vectors_per_address()
        .checked_mul(4)
        .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
    let mut words = source.words().to_vec();

    for tile_start in (0..layout.addresses()).step_by(8) {
        for vector in 0..layout.vectors_per_address() {
            let lane = vector * 4;
            for component in 0..4 {
                let mut values = [0_u32; 8];
                for local in 0..8 {
                    values[local] = words[(tile_start + local) * row_words + lane + component];
                }
                for stride in [1_usize, 2, 4] {
                    for local in 0..8 {
                        if local & stride != 0 {
                            values[local] ^= values[local - stride];
                        }
                    }
                }
                for local in 0..8 {
                    words[(tile_start + local) * row_words + lane + component] = values[local];
                }
            }
        }
    }

    let mut stride = 8_usize;
    while stride < layout.addresses() {
        let block = stride
            .checked_mul(2)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        let mut block_start = 0_usize;
        while block_start < layout.addresses() {
            for offset in 0..stride {
                let source_base = (block_start + offset) * row_words;
                let target_base = (block_start + offset + stride) * row_words;
                for word in 0..row_words {
                    words[target_base + word] ^= words[source_base + word];
                }
            }
            block_start = block_start
                .checked_add(block)
                .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        }
        stride = block;
    }

    let value = FlatPvpVec4BitplanesV1::from_words(layout, words)?;
    Ok((value, accounting(layout)))
}

fn accounting(layout: FlatPvpVec4LayoutV1) -> FlatPvpTile8AccountingV1 {
    let pairs = (layout.addresses() / 2) as u128 * u128::from(layout.stages());
    let baseline_logical_dispatches = layout.stages();
    let logical_dispatches = if layout.stages() >= 3 {
        layout.stages() - 2
    } else {
        layout.stages()
    };
    FlatPvpTile8AccountingV1 {
        stages: layout.stages(),
        logical_gate_xor_ops: pairs * layout.gates() as u128,
        vec4_updates: pairs * layout.vectors_per_address() as u128,
        logical_dispatches,
        baseline_logical_dispatches,
        state_vectors: layout.storage_vectors(),
        workgroup_storage_bytes: FLAT_PVP_TILE8_WORKGROUP_STORAGE_BYTES,
        scratch_state_vectors: 0,
    }
}

#[cfg(feature = "wgpu")]
/// WGPU PVP3c pipeline with an 8-address workgroup-fused prefix.
pub struct WgpuPvpTile8Pipeline {
    tile8: wgpu::ComputePipeline,
    suffix: WgpuPvpVec4Pipeline,
}

#[cfg(feature = "wgpu")]
impl WgpuPvpTile8Pipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, FlatPvpVec4Error> {
        let tile8 = crate::wgpu_internal::create_pipeline(
            device,
            FLAT_PVP_TILE8_WGSL,
            "flat-pvp-vec4-tile8",
            "pvp_subset_zeta_tile8",
        )
        .map_err(FlatPvpVec4Error::Pipeline)?;
        Ok(Self {
            tile8,
            suffix: WgpuPvpVec4Pipeline::new(device)?,
        })
    }

    /// Encode the tile8 prefix followed by the qualified PVP2 suffix.
    pub fn encode_all_stages(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        state: &wgpu::Buffer,
        layout: FlatPvpVec4LayoutV1,
    ) -> Result<FlatPvpTile8AccountingV1, FlatPvpVec4Error> {
        if layout.addresses() < 8 {
            self.suffix
                .encode_all_stages(device, encoder, state, layout)?;
            return Ok(accounting(layout));
        }

        let required = layout
            .storage_u32_words()
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        let required = u64::try_from(required).map_err(|_| FlatPvpVec4Error::ArithmeticOverflow)?;
        if state.size() < required {
            return Err(FlatPvpVec4Error::BufferTooSmall {
                required_bytes: required,
                actual_bytes: state.size(),
            });
        }

        let addresses = u32::try_from(layout.addresses()).map_err(|_| {
            FlatPvpVec4Error::IndexSpaceExceeded {
                value: layout.addresses(),
            }
        })?;
        let vectors_per_address = u32::try_from(layout.vectors_per_address()).map_err(|_| {
            FlatPvpVec4Error::IndexSpaceExceeded {
                value: layout.vectors_per_address(),
            }
        })?;
        let tile_count = addresses / 8;
        let workgroups = tile_count
            .checked_mul(vectors_per_address)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        if workgroups > maximum {
            return Err(FlatPvpVec4Error::DispatchLimit {
                required: workgroups,
                maximum,
            });
        }
        if FLAT_PVP_TILE8_WORKGROUP_STORAGE_BYTES
            > device.limits().max_compute_workgroup_storage_size as usize
        {
            return Err(FlatPvpVec4Error::IndexSpaceExceeded {
                value: FLAT_PVP_TILE8_WORKGROUP_STORAGE_BYTES,
            });
        }

        let params =
            crate::wgpu_internal::encode_u32(&[addresses, vectors_per_address, tile_count, 0]);
        let uniform = crate::wgpu_internal::create_uniform_buffer_init(
            device,
            "flat-pvp-tile8-params",
            &params,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-pvp-tile8-bind-group"),
            layout: &self.tile8.get_bind_group_layout(0),
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
                label: Some("flat-pvp-tile8-prefix"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile8);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.suffix
            .encode_stages_from_stride(device, encoder, state, layout, 8)?;
        Ok(accounting(layout))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(layout: FlatPvpVec4LayoutV1) -> FlatPvpVec4BitplanesV1 {
        let words_per_gate = layout.addresses().div_ceil(64);
        let mut gate_major = vec![0_u64; layout.gates() * words_per_gate];
        for gate in 0..layout.gates() {
            for address in 0..layout.addresses() {
                if ((gate * 47 + address * 17 + (gate ^ address)) % 31) < 15 {
                    gate_major[gate * words_per_gate + address / 64] |= 1_u64 << (address % 64);
                }
            }
        }
        FlatPvpVec4BitplanesV1::from_gate_major_u64(layout, &gate_major).unwrap()
    }

    #[test]
    fn tile8_host_matches_full_vec4_reference() {
        for (addresses, gates) in [
            (1, 1),
            (2, 65),
            (4, 129),
            (8, 257),
            (16, 513),
            (64, 129),
            (128, 257),
        ] {
            let layout = FlatPvpVec4LayoutV1::new(addresses, gates).unwrap();
            let source = fixture(layout);
            let mut reference = source.clone();
            pvp_subset_zeta_vec4_host_in_place(&mut reference).unwrap();
            let (candidate, stats) = pvp_subset_zeta_tile8_host(&source).unwrap();
            assert_eq!(candidate, reference, "addresses={addresses} gates={gates}");
            assert_eq!(stats.workgroup_storage_bytes, 128);
            assert_eq!(stats.scratch_state_vectors, 0);
            if addresses >= 8 {
                assert_eq!(
                    stats.logical_dispatches + 2,
                    stats.baseline_logical_dispatches
                );
            }
        }
    }

    #[test]
    fn tile8_wgsl_parses_and_validates() {
        let module =
            naga::front::wgsl::parse_str(FLAT_PVP_TILE8_WGSL).expect("PVP tile8 WGSL parses");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .expect("PVP tile8 WGSL validates");
    }
}
