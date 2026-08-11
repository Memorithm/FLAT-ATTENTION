//! Passive runtime telemetry for FLAT execution decisions.
//!
//! Telemetry is deliberately host-side metadata. Building or reading a snapshot
//! never submits GPU work, polls a device, maps a buffer, or introduces a
//! synchronization boundary. Callers choose whether to request/record it.

use crate::{GroupedAttentionShape, WgpuKernelVariant, WGSL_KV_TILE, WGSL_QUERY_ROWS};

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

/// Derive the qualified grouped-forward Q4 workgroup geometry without creating
/// GPU resources or synchronizing a device.
pub fn grouped_q4_geometry(
    shape: GroupedAttentionShape,
) -> Result<RuntimeTileGeometry, crate::FlatAttentionError> {
    shape.validate()?;
    let query_tiles = shape.seq_len.div_ceil(WGSL_QUERY_ROWS);
    let batch_heads = shape
        .batch
        .checked_mul(shape.q_heads)
        .ok_or(crate::FlatAttentionError::ShapeOverflow)?;
    Ok(RuntimeTileGeometry {
        query_rows: u32::try_from(WGSL_QUERY_ROWS)
            .map_err(|_| crate::FlatAttentionError::ShapeOverflow)?,
        kv_rows: u32::try_from(WGSL_KV_TILE)
            .map_err(|_| crate::FlatAttentionError::ShapeOverflow)?,
        workgroups: [
            u32::try_from(query_tiles).map_err(|_| crate::FlatAttentionError::ShapeOverflow)?,
            u32::try_from(batch_heads).map_err(|_| crate::FlatAttentionError::ShapeOverflow)?,
            1,
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_geometry_is_deterministic_and_native_to_query_heads() {
        let shape = GroupedAttentionShape {
            batch: 2,
            q_heads: 8,
            kv_heads: 2,
            seq_len: 17,
            head_dim: 64,
        };
        let a = grouped_q4_geometry(shape).unwrap();
        let b = grouped_q4_geometry(shape).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.query_rows, WGSL_QUERY_ROWS as u32);
        assert_eq!(a.kv_rows, WGSL_KV_TILE as u32);
        assert_eq!(a.workgroups, [5, 16, 1]);
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
