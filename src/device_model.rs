//! Host-side device identity and capability model.
//!
//! These types describe the executing adapter (identity fingerprint, explicit
//! resource/capability limits) and the passive dispatch telemetry record. They
//! are pure host data with no WGPU dependency, so capability prefiltering,
//! candidate planning, and evidence records can run without a GPU runtime.
//! Adapter marketing names remain provenance only; selection must use the
//! explicit capability facts.

/// Stable logical identifier for an executable FLAT kernel family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeKernelId {
    Q4Portable,
    Q4Vec4Portable,
    Q4Vec4DoubleBuffered,
    Q4Subgroup,
    GroupedForwardPortable,
    ResidentDecodePortable,
    PagedDecodePortable,
    GroupedBackwardRecomputePortable,
}

/// Device/driver identity captured once when an owned WGPU context is created.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeDeviceFingerprint {
    /// Human-readable adapter name as reported by wgpu.
    pub name: String,
    /// Graphics backend family (Vulkan, Metal, D3D12).
    pub backend: String,
    /// Driver version string.
    pub driver: String,
    /// Extended driver description when the adapter provides one.
    pub driver_info: String,
    /// PCI vendor identifier.
    pub vendor: u32,
    /// PCI device identifier.
    pub device: u32,
}

impl RuntimeDeviceFingerprint {
    /// Deterministic canonical representation used by Phase I autotuning keys.
    ///
    /// Adapter names remain provenance only: candidate selection must use
    /// explicit capabilities/limits rather than marketing-name heuristics.
    pub fn canonical_record(&self) -> String {
        format!(
            "name={};backend={};driver={};driver_info={};vendor={:08x};device={:08x}",
            escape_component(&self.name),
            escape_component(&self.backend),
            escape_component(&self.driver),
            escape_component(&self.driver_info),
            self.vendor,
            self.device,
        )
    }

    /// Stable FNV-1a-64 fingerprint of [`Self::canonical_record`].
    ///
    /// This is a deterministic cache-key component, not a cryptographic
    /// authenticity primitive.
    pub fn stable_fingerprint(&self) -> u64 {
        fnv1a64(self.canonical_record().as_bytes())
    }
}

fn escape_component(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '%' => escaped.push_str("%25"),
            ';' => escaped.push_str("%3b"),
            '=' => escaped.push_str("%3d"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Explicit resource/capability limits of the selected WGPU device.
///
/// This is the M24 device capability model: deterministic host-side limits that
/// candidate generation must respect before pipeline creation. Adapter marketing
/// names are deliberately absent — selection uses these explicit limits, not
/// name heuristics. The record serializes deterministically so it can join the
/// autotuning cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeDeviceCapabilities {
    /// Device dispatch limit along every compute axis.
    pub max_workgroups_per_dimension: u32,
    /// Maximum invocations along the workgroup X axis.
    pub max_workgroup_size_x: u32,
    /// Maximum invocations along the workgroup Y axis.
    pub max_workgroup_size_y: u32,
    /// Maximum invocations along the workgroup Z axis.
    pub max_workgroup_size_z: u32,
    /// Workgroup-addressable memory in bytes.
    pub max_workgroup_storage_bytes: u32,
    /// Maximum number of bind groups per dispatch (wgpu 0.20 `max_bind_groups`).
    pub max_binding_entries: u32,
    /// Largest storage buffer binding in bytes.
    pub max_storage_buffer_binding_size: u32,
    /// Whether the adapter exposes WGPU subgroup operations.
    pub subgroup_supported: bool,
    /// Minimum subgroup width reported by the adapter.
    pub subgroup_min_size: u32,
    /// Maximum subgroup width reported by the adapter.
    pub subgroup_max_size: u32,
    /// Whether native shader-f16 is available (unused by packed-f16 I/O).
    pub f16_supported: bool,
}

impl RuntimeDeviceCapabilities {
    /// Deterministic canonical representation used by Phase I autotuning keys.
    pub fn canonical_record(&self) -> String {
        format!(
            "wgpd={};wgsx={};wgsy={};wgsz={};wgss={};bind={};sbuf={};sub={};submin={};submax={};f16={}",
            self.max_workgroups_per_dimension,
            self.max_workgroup_size_x,
            self.max_workgroup_size_y,
            self.max_workgroup_size_z,
            self.max_workgroup_storage_bytes,
            self.max_binding_entries,
            self.max_storage_buffer_binding_size,
            u8::from(self.subgroup_supported),
            self.subgroup_min_size,
            self.subgroup_max_size,
            u8::from(self.f16_supported),
        )
    }

    /// Stable FNV-1a-64 fingerprint of [`Self::canonical_record`].
    pub fn stable_fingerprint(&self) -> u64 {
        fnv1a64(self.canonical_record().as_bytes())
    }
}

/// Runtime-autotuner cache disposition attached to a dispatch record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AutotunerCacheStatus {
    #[default]
    NotApplicable,
    Hit,
    Miss,
}

/// Tile geometry selected for one dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeTileGeometry {
    /// Query rows staged per workgroup.
    pub query_rows: u32,
    /// K/V rows streamed per tile.
    pub kv_rows: u32,
    /// Resolved [x, y, z] dispatch geometry.
    pub workgroups: [u32; 3],
}

/// Passive metadata for one logical FLAT dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDispatchTelemetry {
    /// Kernel generation selected for the dispatch.
    pub kernel_id: RuntimeKernelId,
    /// Fingerprint of the executing adapter.
    pub device: RuntimeDeviceFingerprint,
    /// Tiling geometry used by the kernel.
    pub tile: RuntimeTileGeometry,
    /// Number of device dispatches issued for one logical pass.
    pub dispatch_count: u32,
    /// Buffers allocated by FLAT for this dispatch.
    pub temporary_allocation_count: u32,
    /// Total bytes of FLAT-owned temporary buffers.
    pub temporary_allocation_bytes: u64,
    /// Why a preferred generation was not selected, if any.
    pub fallback_reason: Option<String>,
    /// Externally-attached autotuner cache annotation.
    pub autotuner_cache: AutotunerCacheStatus,
}

impl RuntimeDispatchTelemetry {
    /// Attach an externally-owned autotuner cache decision without changing the
    /// measured dispatch or introducing runtime synchronization.
    pub fn with_autotuner_cache(mut self, status: AutotunerCacheStatus) -> Self {
        self.autotuner_cache = status;
        self
    }
}
