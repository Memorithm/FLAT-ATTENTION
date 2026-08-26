//! Passive runtime telemetry for FLAT execution decisions.
//!
//! Telemetry is deliberately host-side metadata. Building or reading a snapshot
//! never submits GPU work, polls a device, maps a buffer, or introduces a
//! synchronization boundary. Callers choose whether to request/record it.

use crate::device_model::RuntimeKernelId;
use crate::WgpuKernelVariant;

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

#[cfg(test)]
mod tests {
    use crate::device_model::{
        AutotunerCacheStatus, RuntimeDeviceCapabilities, RuntimeDeviceFingerprint,
        RuntimeDispatchTelemetry, RuntimeTileGeometry,
    };

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
        assert_eq!(
            baseline.stable_fingerprint(),
            baseline.clone().stable_fingerprint()
        );

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

    fn capabilities_fixture() -> RuntimeDeviceCapabilities {
        RuntimeDeviceCapabilities {
            max_workgroups_per_dimension: 65535,
            max_workgroup_size_x: 1024,
            max_workgroup_size_y: 1024,
            max_workgroup_size_z: 64,
            max_workgroup_storage_bytes: 32768,
            max_binding_entries: 12,
            max_storage_buffer_binding_size: 1 << 30,
            subgroup_supported: true,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f16_supported: true,
        }
    }

    #[test]
    fn capabilities_record_is_deterministic_and_limit_sensitive() {
        let baseline = capabilities_fixture();
        assert_eq!(baseline.stable_fingerprint(), baseline.stable_fingerprint());

        let mut changed = baseline;
        changed.max_workgroup_storage_bytes = 16384;
        assert_ne!(baseline.stable_fingerprint(), changed.stable_fingerprint());

        let mut f16_flip = baseline;
        f16_flip.f16_supported = false;
        assert_ne!(baseline.stable_fingerprint(), f16_flip.stable_fingerprint());
    }

    #[test]
    fn capabilities_record_is_dispatch_boundary_explicit() {
        let capabilities = capabilities_fixture();
        assert!(capabilities.max_workgroups_per_dimension >= 1);
        assert!(capabilities.max_workgroup_size_x >= 1);
        assert!(capabilities.max_workgroup_storage_bytes >= 1);
        assert!(capabilities.max_binding_entries >= 1);
        assert!(capabilities.max_storage_buffer_binding_size >= 1);
        assert!(capabilities.subgroup_max_size >= capabilities.subgroup_min_size);
    }
}
