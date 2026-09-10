#![cfg(feature = "wgpu")]

use flat_attention::api::research_da_luc::{
    DalucBitOrder, DalucCodebookScope, DalucFloatDType, DalucKeyRepresentation,
    DalucKvViewContract, DalucLogicalKvShape, DalucPaddingRule, DalucPhysicalLayout,
    DalucResidualSemantics, DalucRowOrder, DalucStorageTopology, DalucValueRepresentation,
    DA_LUC_KV_VIEW_SCHEMA_VERSION,
};
use flat_attention::api::research_da_luc_oracle::tiering::{
    route_by_recency, DalucPrecisionTier, DalucTierId, DalucTierQuota,
};
use flat_attention::api::research_da_luc_paged_binding::DalucPagedTierBindingError;
use flat_attention::api::research_da_luc_paged_cache_binding::{
    bind_paged_tier_plan_to_cache, DalucCacheScopedPagedTierBinding,
    DalucCacheScopedPagedTierBindingError,
};
use flat_attention::paged_kv::{PagedKvConfig, WgpuPagedKvCache, WgpuPagedKvCacheError};

struct DeviceHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn harness() -> Option<DeviceHarness> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }));
    let Ok(adapter) = adapter else {
        if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
            panic!(
                "FDAL6 cache-scoped binding requires a WGPU adapter in the mandatory device gate"
            );
        }
        eprintln!("WGPU adapter unavailable; optional FDAL6 cache-scoped binding test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-fdal6-cache-scoped-binding-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("FDAL6 cache-scoped request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn source_buffer(device: &wgpu::Device, rows: usize) -> wgpu::Buffer {
    let bytes = rows
        .checked_mul(2)
        .and_then(|value| value.checked_mul(32))
        .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
        .unwrap() as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-fdal6-cache-scoped-binding-source"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn append_rows(harness: &DeviceHarness, cache: &mut WgpuPagedKvCache, rows: usize) {
    let k = source_buffer(&harness.device, rows);
    let v = source_buffer(&harness.device, rows);
    cache
        .append_and_submit(&harness.device, &harness.queue, &k, &v, rows)
        .unwrap();
}

fn seeded_cache(harness: &DeviceHarness, kv_len: usize) -> WgpuPagedKvCache {
    let mut cache = WgpuPagedKvCache::new(
        &harness.device,
        PagedKvConfig {
            page_size: 4,
            physical_pages: 4,
        },
        2,
        32,
    )
    .unwrap();
    append_rows(harness, &mut cache, kv_len);
    cache
}

fn contract(kv_len: usize) -> DalucKvViewContract {
    DalucKvViewContract {
        schema_version: DA_LUC_KV_VIEW_SCHEMA_VERSION,
        shape: DalucLogicalKvShape {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            kv_len,
            key_head_dim: 32,
            value_head_dim: 32,
        },
        keys: DalucKeyRepresentation {
            subspace_dim: 4,
            codebook_entries: 16,
            codebook_dtype: DalucFloatDType::F32,
            codebook_scope: DalucCodebookScope::PerKvHead,
            index_bits: 4,
            index_bit_order: DalucBitOrder::Lsb0,
            residual: DalucResidualSemantics::None,
        },
        values: DalucValueRepresentation::Dense {
            dtype: DalucFloatDType::F32,
        },
        layout: DalucPhysicalLayout {
            row_order: DalucRowOrder::BatchTokenHead,
            topology: DalucStorageTopology::Paged {
                page_size: 4,
                physical_pages_per_batch: 4,
            },
            plane_alignment_bytes: 4,
            padding: DalucPaddingRule::None,
        },
    }
}

fn scoped_binding(cache: &WgpuPagedKvCache, kv_len: usize) -> DalucCacheScopedPagedTierBinding {
    let contract = contract(kv_len);
    let tiers = [
        DalucPrecisionTier {
            id: DalucTierId(10),
            keys: contract.keys,
            values: contract.values,
        },
        DalucPrecisionTier {
            id: DalucTierId(20),
            keys: contract.keys,
            values: contract.values,
        },
    ];
    let segments = kv_len.div_ceil(4);
    let quotas = [
        DalucTierQuota {
            tier_id: DalucTierId(10),
            segments: 1,
        },
        DalucTierQuota {
            tier_id: DalucTierId(20),
            segments: segments.saturating_sub(1),
        },
    ];
    let plan = route_by_recency(contract, 4, &tiers, &quotas).unwrap();
    bind_paged_tier_plan_to_cache(cache, contract, &tiers, &plan).unwrap()
}

#[test]
fn cache_scoped_binding_accepts_the_origin_cache() {
    let Some(harness) = harness() else {
        return;
    };
    let cache = seeded_cache(&harness, 8);
    let scoped = scoped_binding(&cache, 8);

    assert_eq!(scoped.binding().kv_len, 8);
    assert_eq!(scoped.binding().generation, cache.generation());
    assert_eq!(scoped.binding().branch_epoch, cache.branch_epoch());
    scoped.validate_cache(&cache).unwrap();
}

#[test]
fn cache_scoped_binding_retains_an_independent_validated_tier_catalog() {
    let Some(harness) = harness() else {
        return;
    };
    let cache = seeded_cache(&harness, 8);
    let contract = contract(8);
    let mut tiers = [
        DalucPrecisionTier {
            id: DalucTierId(10),
            keys: contract.keys,
            values: contract.values,
        },
        DalucPrecisionTier {
            id: DalucTierId(20),
            keys: contract.keys,
            values: DalucValueRepresentation::Dense {
                dtype: DalucFloatDType::F16,
            },
        },
    ];
    let quotas = [
        DalucTierQuota {
            tier_id: DalucTierId(10),
            segments: 1,
        },
        DalucTierQuota {
            tier_id: DalucTierId(20),
            segments: 1,
        },
    ];
    let plan = route_by_recency(contract, 4, &tiers, &quotas).unwrap();
    let scoped = bind_paged_tier_plan_to_cache(&cache, contract, &tiers, &plan).unwrap();

    assert_eq!(scoped.tiers(), &tiers);
    tiers[0].id = DalucTierId(99);
    tiers[0].values = DalucValueRepresentation::Dense {
        dtype: DalucFloatDType::Bf16,
    };

    assert_eq!(scoped.tiers()[0].id, DalucTierId(10));
    assert_eq!(
        scoped.tiers()[0].values,
        DalucValueRepresentation::Dense {
            dtype: DalucFloatDType::F32,
        }
    );
    assert_eq!(scoped.tiers()[1].id, DalucTierId(20));
    assert_eq!(
        scoped.tiers()[1].values,
        DalucValueRepresentation::Dense {
            dtype: DalucFloatDType::F16,
        }
    );
}

#[test]
fn cache_scoped_binding_rejects_identical_foreign_cache_metadata() {
    let Some(harness) = harness() else {
        return;
    };
    let origin = seeded_cache(&harness, 8);
    let foreign = seeded_cache(&harness, 8);
    let scoped = scoped_binding(&origin, 8);

    // The portable FDAL6 binding intentionally accepts equal exposed metadata.
    scoped
        .binding()
        .validate_observation(&foreign.observation().unwrap())
        .unwrap();

    assert_eq!(
        scoped.validate_cache(&foreign),
        Err(DalucCacheScopedPagedTierBindingError::Cache(
            WgpuPagedKvCacheError::ForeignCheckpoint,
        ))
    );
}

#[test]
fn append_makes_metadata_stale_then_truncate_invalidates_lineage() {
    let Some(harness) = harness() else {
        return;
    };
    let mut cache = seeded_cache(&harness, 8);
    let scoped = scoped_binding(&cache, 8);
    let checkpoint_branch = cache.branch_epoch();

    append_rows(&harness, &mut cache, 2);
    assert_eq!(
        scoped.validate_cache(&cache),
        Err(DalucCacheScopedPagedTierBindingError::Binding(
            DalucPagedTierBindingError::BindingKvLenMismatch {
                binding: 8,
                observation: 10,
            },
        ))
    );

    cache.truncate(8).unwrap();
    assert_eq!(
        scoped.validate_cache(&cache),
        Err(DalucCacheScopedPagedTierBindingError::Cache(
            WgpuPagedKvCacheError::CheckpointBranchMismatch {
                checkpoint_branch_epoch: checkpoint_branch,
                current_branch_epoch: cache.branch_epoch(),
            },
        ))
    );
}

#[test]
fn reset_invalidates_cache_scoped_binding_even_after_same_length_reappend() {
    let Some(harness) = harness() else {
        return;
    };
    let mut cache = seeded_cache(&harness, 8);
    let scoped = scoped_binding(&cache, 8);
    let checkpoint_generation = cache.generation();

    cache.reset().unwrap();
    append_rows(&harness, &mut cache, 8);
    assert_eq!(cache.len(), 8);
    assert_eq!(
        scoped.validate_cache(&cache),
        Err(DalucCacheScopedPagedTierBindingError::Cache(
            WgpuPagedKvCacheError::CheckpointGenerationMismatch {
                checkpoint_generation,
                current_generation: cache.generation(),
            },
        ))
    );
}

#[test]
fn externally_recorded_writes_remain_fail_closed_for_fdal_freshness() {
    let Some(harness) = harness() else {
        return;
    };
    let mut cache = seeded_cache(&harness, 8);
    let scoped = scoped_binding(&cache, 8);
    let k = source_buffer(&harness.device, 1);
    let v = source_buffer(&harness.device, 1);
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-fdal6-cache-scoped-external-append"),
        });
    cache.record_append(&mut encoder, &k, &v, 1).unwrap();

    assert_eq!(
        scoped.validate_cache(&cache),
        Err(DalucCacheScopedPagedTierBindingError::Binding(
            DalucPagedTierBindingError::UnsubmittedRecordedWrites,
        ))
    );
    drop(encoder);
}
