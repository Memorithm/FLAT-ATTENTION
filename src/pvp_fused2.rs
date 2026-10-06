//! PVP3a correctness candidate: fuse the first two vec4 butterfly stages.
//!
//! One invocation owns four adjacent addresses and one 128-gate vector. It
//! performs stride 1 then stride 2 entirely in registers. All remaining stages
//! are delegated to the qualified PVP2 vec4 stage kernel starting at stride 4.
//!
//! This is a structural dispatch-reduction candidate only. It carries no
//! latency, throughput, bandwidth, or default-routing claim.

#[cfg(feature = "wgpu")]
use crate::pvp_vec4::WgpuPvpVec4Pipeline;
use crate::pvp_vec4::{FlatPvpVec4BitplanesV1, FlatPvpVec4Error, FlatPvpVec4LayoutV1};

pub const FLAT_PVP_FUSED2_SCHEMA_V1: &str = "flat.pvp-vec4-fused2/v1";
pub const FLAT_PVP_FUSED2_WORKGROUP_SIZE: u32 = 64;
pub const FLAT_PVP_FUSED2_WGSL: &str = include_str!("../shaders/flat_pvp_vec4_fused2.wgsl");

/// Structural accounting for the fused-prefix candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatPvpFused2AccountingV1 {
    pub stages: u32,
    pub logical_gate_xor_ops: u128,
    pub vec4_updates: u128,
    pub logical_dispatches: u32,
    pub baseline_logical_dispatches: u32,
    pub state_vectors: usize,
    pub scratch_vectors: usize,
}

/// Apply the exact PVP transform with the first two stages fused on host.
///
/// This is an independent structural oracle for the WGPU candidate. It starts
/// from the same vec4 physical layout and preserves all canonical padding.
pub fn pvp_subset_zeta_fused2_host(
    source: &FlatPvpVec4BitplanesV1,
) -> Result<(FlatPvpVec4BitplanesV1, FlatPvpFused2AccountingV1), FlatPvpVec4Error> {
    let layout = source.layout();
    let row_words = layout
        .vectors_per_address()
        .checked_mul(4)
        .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
    let mut words = source.words().to_vec();

    if layout.addresses() >= 4 {
        for block_start in (0..layout.addresses()).step_by(4) {
            for vector in 0..layout.vectors_per_address() {
                let lane = vector * 4;
                let i0 = block_start * row_words + lane;
                let i1 = (block_start + 1) * row_words + lane;
                let i2 = (block_start + 2) * row_words + lane;
                let i3 = (block_start + 3) * row_words + lane;

                for component in 0..4 {
                    let a0 = words[i0 + component];
                    let mut a1 = words[i1 + component];
                    let mut a2 = words[i2 + component];
                    let mut a3 = words[i3 + component];
                    a1 ^= a0;
                    a3 ^= a2;
                    a2 ^= a0;
                    a3 ^= a1;
                    words[i0 + component] = a0;
                    words[i1 + component] = a1;
                    words[i2 + component] = a2;
                    words[i3 + component] = a3;
                }
            }
        }

        let mut stride = 4_usize;
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
    } else if layout.addresses() == 2 {
        for word in 0..row_words {
            words[row_words + word] ^= words[word];
        }
    }

    let value = FlatPvpVec4BitplanesV1::from_words(layout, words)?;
    Ok((value, accounting(layout)?))
}

fn accounting(
    layout: FlatPvpVec4LayoutV1,
) -> Result<FlatPvpFused2AccountingV1, FlatPvpVec4Error> {
    let pairs = (layout.addresses() / 2) as u128 * u128::from(layout.stages());
    let baseline_logical_dispatches = layout.stages();
    let logical_dispatches = if layout.stages() >= 2 {
        layout.stages() - 1
    } else {
        layout.stages()
    };
    Ok(FlatPvpFused2AccountingV1 {
        stages: layout.stages(),
        logical_gate_xor_ops: pairs * layout.gates() as u128,
        vec4_updates: pairs * layout.vectors_per_address() as u128,
        logical_dispatches,
        baseline_logical_dispatches,
        state_vectors: layout.storage_vectors(),
        scratch_vectors: 0,
    })
}

#[cfg(feature = "wgpu")]
/// WGPU PVP3a pipeline with a register-fused two-stage prefix.
pub struct WgpuPvpFused2Pipeline {
    fused: wgpu::ComputePipeline,
    suffix: WgpuPvpVec4Pipeline,
}

#[cfg(feature = "wgpu")]
impl WgpuPvpFused2Pipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, FlatPvpVec4Error> {
        let fused = crate::wgpu_internal::create_pipeline(
            device,
            FLAT_PVP_FUSED2_WGSL,
            "flat-pvp-vec4-fused2",
            "pvp_subset_zeta_fused2",
        )
        .map_err(FlatPvpVec4Error::Pipeline)?;
        Ok(Self {
            fused,
            suffix: WgpuPvpVec4Pipeline::new(device)?,
        })
    }

    /// Encode the fused prefix followed by the qualified PVP2 suffix.
    pub fn encode_all_stages(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        state: &wgpu::Buffer,
        layout: FlatPvpVec4LayoutV1,
    ) -> Result<FlatPvpFused2AccountingV1, FlatPvpVec4Error> {
        if layout.addresses() < 4 {
            self.suffix
                .encode_all_stages(device, encoder, state, layout)?;
            return accounting(layout);
        }

        let required = layout
            .storage_u32_words()
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        let required =
            u64::try_from(required).map_err(|_| FlatPvpVec4Error::ArithmeticOverflow)?;
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
        let block_count = addresses / 4;
        let invocations = block_count
            .checked_mul(vectors_per_address)
            .ok_or(FlatPvpVec4Error::ArithmeticOverflow)?;
        let workgroups = invocations.div_ceil(FLAT_PVP_FUSED2_WORKGROUP_SIZE);
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        if workgroups > maximum {
            return Err(FlatPvpVec4Error::DispatchLimit {
                required: workgroups,
                maximum,
            });
        }

        let params = crate::wgpu_internal::encode_u32(&[
            addresses,
            vectors_per_address,
            block_count,
            0,
        ]);
        let uniform = crate::wgpu_internal::create_uniform_buffer_init(
            device,
            "flat-pvp-fused2-params",
            &params,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-pvp-fused2-bind-group"),
            layout: &self.fused.get_bind_group_layout(0),
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
                label: Some("flat-pvp-fused2-prefix"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fused);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.suffix
            .encode_stages_from_stride(device, encoder, state, layout, 4)?;
        accounting(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pvp_vec4::pvp_subset_zeta_vec4_host_in_place;

    fn fixture(layout: FlatPvpVec4LayoutV1) -> FlatPvpVec4BitplanesV1 {
        let words_per_gate = layout.addresses().div_ceil(64);
        let mut gate_major = vec![0_u64; layout.gates() * words_per_gate];
        for gate in 0..layout.gates() {
            for address in 0..layout.addresses() {
                if ((gate * 37 + address * 13 + (gate ^ address)) % 29) < 14 {
                    gate_major[gate * words_per_gate + address / 64] |=
                        1_u64 << (address % 64);
                }
            }
        }
        FlatPvpVec4BitplanesV1::from_gate_major_u64(layout, &gate_major).unwrap()
    }

    #[test]
    fn fused2_host_matches_full_vec4_reference() {
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
            let (candidate, stats) = pvp_subset_zeta_fused2_host(&source).unwrap();
            assert_eq!(candidate, reference, "addresses={addresses} gates={gates}");
            assert_eq!(stats.stages, layout.stages());
            assert_eq!(stats.scratch_vectors, 0);
            assert!(stats.logical_dispatches <= stats.baseline_logical_dispatches);
            if addresses >= 4 {
                assert_eq!(
                    stats.logical_dispatches + 1,
                    stats.baseline_logical_dispatches
                );
            }
        }
    }

    #[test]
    fn fused2_wgsl_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(FLAT_PVP_FUSED2_WGSL)
            .expect("PVP fused2 WGSL parses");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .expect("PVP fused2 WGSL validates");
    }
}
