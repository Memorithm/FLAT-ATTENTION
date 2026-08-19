//! Passive runtime telemetry for FLAT execution decisions.
//!
//! Telemetry is deliberately host-side metadata. Building or reading a snapshot
//! never submits GPU work, polls a device, maps a buffer, or introduces a
//! synchronization boundary. Callers choose whether to request/record it.

use crate::WgpuKernelVariant;

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

impl From<WgpuKernelVariant> for RuntimeKernelId {
    fn from(value: WgpuKernelVariant) -> Self {
        match value {
            WgpuKernelVariant::Q4Portable => Self::Q4Portable,
            WgpuKernelVariant::Q4Vec4Portable => Self::Q4Vec4Portable,
            WgpuKernelVariant::Q4Vec4DoubleBuffered => Self::Q4Vec4DoubleBuffered,
            WgpuKernelVariant::Q4Subgroup => Self::Q4Subgroup,
        }
    }
}

/// Device/driver identity captured once when an owned WGPU context is created.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeDeviceFingerprint {
    pub name: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub vendor: u32,
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
    pub query_rows: u32,
    pub kv_rows: u32,
    pub workgroups: [u32; 3],
}

/// Passive metadata for one logical FLAT dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDispatchTelemetry {
    pub kernel_id: RuntimeKernelId,
    pub device: RuntimeDeviceFingerprint,
    pub tile: RuntimeTileGeometry,
    pub dispatch_count: u32,
    pub temporary_allocation_count: u32,
    pub temporary_allocation_bytes: u64,
    pub fallback_reason: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint_fixture() -> RuntimeDeviceFingerprint {
        RuntimeDeviceFingerprint {
            name: "NVIDIA Tegra NVIDIA Thor".into(),
            backend: "Vulkan".into(),
            driver: "580.00".into(),
            driver_info: "test-driver-info".into(),
            vendor: 0x10de,
            device: 0x0001,
        }
    }

    #[test]
    fn device_fingerprint_is_deterministic_and_driver_sensitive() {
        let baseline = fingerprint_fixture();
        assert_eq!(baseline.stable_fingerprint(), baseline.clone().stable_fingerprint());

        let mut changed = baseline.clone();
        changed.driver.push_str("-changed");
        assert_ne!(baseline.stable_fingerprint(), changed.stable_fingerprint());
    }

    #[test]
    fn canonical_fingerprint_record_preserves_utf8_and_escapes_delimiters() {
        let mut fingerprint = fingerprint_fixture();
        fingerprint.name = "GPU été;a=b%".into();
        assert!(fingerprint
            .canonical_record()
            .contains("name=GPU été%3ba%3db%25;"));
    }

    #[test]
    fn autotuner_annotation_is_pure_metadata() {
        let telemetry = RuntimeDispatchTelemetry {
            kernel_id: RuntimeKernelId::Q4Portable,
            device: RuntimeDeviceFingerprint {
                name: "test".into(),
                backend: "Vulkan".into(),
                driver: "driver".into(),
                driver_info: "info".into(),
                vendor: 1,
                device: 2,
            },
            tile: RuntimeTileGeometry {
                query_rows: 4,
                kv_rows: 8,
                workgroups: [1, 1, 1],
            },
            dispatch_count: 1,
            temporary_allocation_count: 1,
            temporary_allocation_bytes: 32,
            fallback_reason: None,
            autotuner_cache: AutotunerCacheStatus::NotApplicable,
        };
        assert_eq!(
            telemetry
                .clone()
                .with_autotuner_cache(AutotunerCacheStatus::Hit)
                .autotuner_cache,
            AutotunerCacheStatus::Hit
        );
        assert_eq!(
            telemetry.autotuner_cache,
            AutotunerCacheStatus::NotApplicable
        );
    }
}
