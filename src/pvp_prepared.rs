//! Experimental equal-preparation PVP controls. No routing or speed claim.
//!
//! Old APIs and kernels are unchanged. Every arm owns immutable uniforms and
//! bindings before encoding. PackedSevenStage delegates to the qualified packed
//! plan; PackedOneStage separates representation from low-stage fusion.
use core::fmt;
use crate::pvp_packed::FlatPvpPackedLayoutV1;
use crate::pvp_vec4::FlatPvpVec4LayoutV1;

pub const FLAT_PVP_PACKED_ONE_STAGE_WGSL: &str =
    include_str!("../shaders/flat_pvp_gate_major_one_stage.wgsl");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PvpPreparedArm { Vec4, Fused2, Tile8, PackedOneStage, PackedSevenStage }
impl PvpPreparedArm {
    pub const ALL: [Self; 5] = [Self::Vec4, Self::Fused2, Self::Tile8,
        Self::PackedOneStage, Self::PackedSevenStage];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vec4 => "vec4", Self::Fused2 => "fused2", Self::Tile8 => "tile8",
            Self::PackedOneStage => "packed1", Self::PackedSevenStage => "packed7",
        }
    }
    pub const fn is_packed(self) -> bool {
        matches!(self, Self::PackedOneStage | Self::PackedSevenStage)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PvpPreparedError {
    Layout(String), ArithmeticOverflow, IndexSpaceExceeded,
    ArmMismatch, MissingStorageUsage,
    BufferTooSmall { required: u64, actual: u64 },
    DeviceLimit { name: &'static str, required: u64, maximum: u64 },
    Pipeline(String),
}
impl fmt::Display for PvpPreparedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PVP prepared: {self:?}")
    }
}
impl std::error::Error for PvpPreparedError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kernel { Vec4, Fused2, Tile8, PackedOne, PackedPrefix, PackedSuffix }
#[derive(Debug, Clone, PartialEq, Eq)]
struct Stage { kernel: Kernel, params: [u32; 4], groups: u32 }
impl Stage {
    #[cfg(feature = "wgpu")]
    fn workgroup_size(&self) -> u32 { if self.kernel == Kernel::Tile8 { 8 } else { 64 } }
    #[cfg(feature = "wgpu")]
    fn shared_bytes(&self) -> u32 { if self.kernel == Kernel::Tile8 { 128 } else { 0 } }
}

/// Checked host recipe. Physical byte counts differ between the two layouts.
/// No buffer allocation or device execution occurs here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvpPreparedSpec {
    arm: PvpPreparedArm, addresses: usize, gates: usize, words: usize,
    bytes: u64, stages: Vec<Stage>,
}
impl PvpPreparedSpec {
    pub fn new(arm: PvpPreparedArm, addresses: usize, gates: usize) -> Result<Self, PvpPreparedError> {
        let old = FlatPvpVec4LayoutV1::new(addresses, gates)
            .map_err(|e| PvpPreparedError::Layout(e.to_string()))?;
        let packed = FlatPvpPackedLayoutV1::new(addresses, gates)
            .map_err(|e| PvpPreparedError::Layout(e.to_string()))?;
        let words = if arm.is_packed() { packed.storage_u32_words() } else { old.storage_u32_words() };
        let bytes = u64::try_from(words.checked_mul(4).ok_or(PvpPreparedError::ArithmeticOverflow)?)
            .map_err(|_| PvpPreparedError::ArithmeticOverflow)?;
        let narrow = |n: usize| u32::try_from(n).map_err(|_| PvpPreparedError::IndexSpaceExceeded);
        let k = narrow(addresses)?;
        let g = narrow(gates)?;
        let width = narrow(if arm.is_packed() { packed.vectors_per_gate() } else { old.vectors_per_address() })?;
        // Bound all shader storage-index multiplications, not just dispatch counts.
        let total = narrow(words / 4)?;
        let mut stages = Vec::new();
        if arm.is_packed() {
            if arm == PvpPreparedArm::PackedOneStage {
                for stage in 0..k.ilog2().min(7) {
                    stages.push(Stage { kernel: Kernel::PackedOne,
                        params: [k, g, width, stage], groups: total.div_ceil(64) });
                }
            } else if k > 1 {
                stages.push(Stage { kernel: Kernel::PackedPrefix,
                    params: [k, g, width, 0], groups: total.div_ceil(64) });
            }
            let mut stride = 1;
            while stride < width {
                stages.push(Stage { kernel: Kernel::PackedSuffix,
                    params: [k, g, width, stride], groups: (total / 2).div_ceil(64) });
                stride *= 2;
            }
        } else {
            let mut stride = 1;
            if arm == PvpPreparedArm::Fused2 && k >= 4 {
                stages.push(Stage { kernel: Kernel::Fused2,
                    params: [k, width, k / 4, 0], groups: (total / 4).div_ceil(64) });
                stride = 4;
            } else if arm == PvpPreparedArm::Tile8 && k >= 8 {
                stages.push(Stage { kernel: Kernel::Tile8,
                    params: [k, width, k / 8, 0], groups: total / 8 });
                stride = 8;
            }
            while stride < k {
                stages.push(Stage { kernel: Kernel::Vec4,
                    params: [k, width, stride, k / 2], groups: (total / 2).div_ceil(64) });
                stride *= 2;
            }
        }
        Ok(Self { arm, addresses, gates, words, bytes, stages })
    }
    pub const fn arm(&self) -> PvpPreparedArm { self.arm }
    pub const fn addresses(&self) -> usize { self.addresses }
    pub const fn gates(&self) -> usize { self.gates }
    pub const fn storage_u32_words(&self) -> usize { self.words }
    pub const fn storage_bytes(&self) -> u64 { self.bytes }
    pub fn dispatches(&self) -> usize { self.stages.len() }

    #[cfg(feature = "wgpu")]
    pub fn validate_limits(&self, limits: &wgpu::Limits) -> Result<(), PvpPreparedError> {
        let check = |name, required, maximum| {
            if required > maximum { Err(PvpPreparedError::DeviceLimit { name, required, maximum }) }
            else { Ok(()) }
        };
        check("buffer bytes", self.bytes, limits.max_buffer_size)?;
        check("storage binding bytes", self.bytes, limits.max_storage_buffer_binding_size)?;
        for stage in &self.stages {
            check("workgroups x", u64::from(stage.groups), u64::from(limits.max_compute_workgroups_per_dimension))?;
            check("workgroup x", u64::from(stage.workgroup_size()), u64::from(limits.max_compute_workgroup_size_x))?;
            check("workgroup invocations", u64::from(stage.workgroup_size()), u64::from(limits.max_compute_invocations_per_workgroup))?;
            check("workgroup storage bytes", u64::from(stage.shared_bytes()), u64::from(limits.max_compute_workgroup_storage_size))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wgpu")]
struct BoundStage { pipeline: wgpu::ComputePipeline, binding: wgpu::BindGroup,
    _uniform: wgpu::Buffer, groups: u32 }
#[cfg(feature = "wgpu")]
enum PreparedImplementation {
    Control(Vec<BoundStage>), Packed(crate::pvp_packed::WgpuPvpPackedPlan),
}
#[cfg(feature = "wgpu")]
pub struct WgpuPvpPreparedPlan {
    spec: PvpPreparedSpec, _state: wgpu::Buffer, implementation: PreparedImplementation,
}
#[cfg(feature = "wgpu")]
pub struct WgpuPvpPreparedPipeline {
    arm: PvpPreparedArm, kernels: Vec<(Kernel, wgpu::ComputePipeline)>,
    packed: Option<crate::pvp_packed::WgpuPvpPackedPipeline>,
}
#[cfg(feature = "wgpu")]
impl WgpuPvpPreparedPipeline {
    pub fn new(device: &wgpu::Device, arm: PvpPreparedArm) -> Result<Self, PvpPreparedError> {
        use crate::{pvp_vec4, pvp_fused2, pvp_tile8, pvp_packed};
        if arm == PvpPreparedArm::PackedSevenStage {
            return Ok(Self { arm, kernels: Vec::new(), packed: Some(pvp_packed::WgpuPvpPackedPipeline::new(device)
                .map_err(|e| PvpPreparedError::Pipeline(e.to_string()))?) });
        }
        let sources = if arm == PvpPreparedArm::PackedOneStage {
            vec![(Kernel::PackedOne, FLAT_PVP_PACKED_ONE_STAGE_WGSL, "pvp_packed_one_stage"),
                (Kernel::PackedSuffix, pvp_packed::FLAT_PVP_PACKED_WGSL, "pvp_packed_suffix")]
        } else {
            let mut sources = vec![(Kernel::Vec4, pvp_vec4::FLAT_PVP_VEC4_WGSL, "pvp_subset_zeta_stage_vec4")];
            if arm == PvpPreparedArm::Fused2 {
                sources.push((Kernel::Fused2, pvp_fused2::FLAT_PVP_FUSED2_WGSL, "pvp_subset_zeta_fused2"));
            } else if arm == PvpPreparedArm::Tile8 {
                sources.push((Kernel::Tile8, pvp_tile8::FLAT_PVP_TILE8_WGSL, "pvp_subset_zeta_tile8"));
            }
            sources
        };
        let mut kernels = Vec::new();
        for (kernel, source, entry) in sources {
            kernels.push((kernel, crate::wgpu_internal::create_pipeline(device, source, "pvp-prepared", entry)
                .map_err(PvpPreparedError::Pipeline)?));
        }
        Ok(Self { arm, kernels, packed: None })
    }
    pub fn prepare(&self, device: &wgpu::Device, state: &wgpu::Buffer, spec: &PvpPreparedSpec)
        -> Result<WgpuPvpPreparedPlan, PvpPreparedError> {
        if spec.arm != self.arm { return Err(PvpPreparedError::ArmMismatch); }
        if !state.usage().contains(wgpu::BufferUsages::STORAGE) { return Err(PvpPreparedError::MissingStorageUsage); }
        if state.size() < spec.bytes {
            return Err(PvpPreparedError::BufferTooSmall { required: spec.bytes, actual: state.size() });
        }
        spec.validate_limits(&device.limits())?;
        let implementation = if let Some(packed) = &self.packed {
            let layout = FlatPvpPackedLayoutV1::new(spec.addresses, spec.gates)
                .map_err(|e| PvpPreparedError::Layout(e.to_string()))?;
            PreparedImplementation::Packed(packed.prepare(device, state, layout)
                .map_err(|e| PvpPreparedError::Pipeline(e.to_string()))?)
        } else {
            let mut bound = Vec::with_capacity(spec.stages.len());
            for stage in &spec.stages {
                let pipeline = &self.kernels.iter().find(|(k, _)| *k == stage.kernel)
                    .expect("checked recipe uses only this arm's kernels").1;
                let params = crate::wgpu_internal::encode_u32(&stage.params);
                let uniform = crate::wgpu_internal::create_uniform_buffer_init(device, "pvp-prepared-uniform", &params);
                let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pvp-prepared-binding"), layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: state, offset: 0, size: core::num::NonZeroU64::new(spec.bytes),
                        }) },
                        wgpu::BindGroupEntry { binding: 1, resource: uniform.as_entire_binding() },
                    ],
                });
                bound.push(BoundStage { pipeline: pipeline.clone(), binding, _uniform: uniform, groups: stage.groups });
            }
            PreparedImplementation::Control(bound)
        };
        Ok(WgpuPvpPreparedPlan { spec: spec.clone(), _state: state.clone(), implementation })
    }
}
#[cfg(feature = "wgpu")]
impl WgpuPvpPreparedPlan {
    pub fn spec(&self) -> &PvpPreparedSpec { &self.spec }
    /// No buffer, uniform or bind-group creation. Caller owns submission/completion.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        match &self.implementation {
            PreparedImplementation::Packed(plan) => plan.encode(encoder),
            PreparedImplementation::Control(stages) => for stage in stages {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pvp-prepared-stage"), timestamp_writes: None,
                });
                pass.set_pipeline(&stage.pipeline);
                pass.set_bind_group(0, &stage.binding, &[]);
                pass.dispatch_workgroups(stage.groups, 1, 1);
            },
        }
    }
}
