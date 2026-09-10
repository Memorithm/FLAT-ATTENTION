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
use flat_attention::api::research_da_luc_paged_binding::{
    bind_paged_tier_plan, DalucPagedTierBinding, DalucPagedTierBindingError,
    DA_LUC_PAGED_TIER_BINDING_VERSION,
};
use flat_attention::paged_kv::{PagedKvConfig, WgpuPagedKvCache};

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
            panic!("FDAL6 paged tier binding requires a WGPU adapter in the mandatory device gate");
        }
        eprintln!("WGPU adapter unavailable; optional FDAL6 paged binding test skipped");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-fdal6-paged-tier-binding-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("FDAL6 request_device failed: {error}"));
    Some(DeviceHarness { device, queue })
}

fn source_buffer(
    device: &wgpu::Device,
    rows: usize,
    kv_heads: usize,
    head_dim: usize,
) -> wgpu::Buffer {
    let bytes = rows
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
        .unwrap() as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flat-fdal6-paged-tier-binding-source"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn append_rows(harness: &DeviceHarness, cache: &mut WgpuPagedKvCache, rows: usize) {
    let k = source_buffer(&harness.device, rows, 2, 32);
    let v = source_buffer(&harness.device, rows, 2, 32);
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

fn contract(
    kv_len: usize,
    kv_heads: usize,
    page_size: usize,
    physical_pages: usize,
) -> DalucKvViewContract {
    DalucKvViewContract {
        schema_version: DA_LUC_KV_VIEW_SCHEMA_VERSION,
        shape: DalucLogicalKvShape {
            batch: 1,
            q_heads: 4,
            kv_heads,
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
                page_size,
                physical_pages_per_batch: physical_pages,
            },
            plane_alignment_bytes: 4,
            padding: DalucPaddingRule::None,
        },
    }
}

fn tiers(contract: DalucKvViewContract) -> [DalucPrecisionTier; 2] {
    [
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
    ]
}

fn quotas(first: usize, second: usize) -> [DalucTierQuota; 2] {
    [
        DalucTierQuota {
            tier_id: DalucTierId(10),
            segments: first,
        },
        DalucTierQuota {
            tier_id: DalucTierId(20),
            segments: second,
        },
    ]
}

fn binding_for(cache: &WgpuPagedKvCache, kv_len: usize) -> DalucPagedTierBinding {
    let observation = cache.observation().unwrap();
    let contract = contract(kv_len, 2, 4, 4);
    let tiers = tiers(contract);
    let segment_count = kv_len.div_ceil(4);
    let plan = route_by_recency(
        contract,
        4,
        &tiers,
        &quotas(1, segment_count.saturating_sub(1)),
    )
    .unwrap();
    bind_paged_tier_plan(&observation, contract, &tiers, &plan).unwrap()
}

#[test]
fn binds_logical_tiers_to_observed_physical_pages_without_payload_io() {
    let Some(harness) = harness() else {
        return;
    };
    let cache = seeded_cache(&harness, 10);
    let observation = cache.observation().unwrap();
    let contract = contract(10, 2, 4, 4);
    let tiers = tiers(contract);
    let plan = route_by_recency(contract, 4, &tiers, &quotas(1, 2)).unwrap();

    let binding = bind_paged_tier_plan(&observation, contract, &tiers, &plan).unwrap();
    assert_eq!(binding.binding_version, DA_LUC_PAGED_TIER_BINDING_VERSION);
    assert_eq!(
        binding.observation_schema_version,
        observation.schema_version()
    );
    assert_eq!(binding.kv_len, 10);
    assert_eq!(binding.page_size, 4);
    assert_eq!(binding.physical_pages, 4);
    assert_eq!(binding.generation, cache.generation());
    assert_eq!(binding.branch_epoch, cache.branch_epoch());
    assert_eq!(binding.assignments.len(), 3);

    assert_eq!(binding.assignments[0].logical_page, 0);
    assert_eq!(binding.assignments[0].physical_page, 0);
    assert_eq!(binding.assignments[0].start_token, 0);
    assert_eq!(binding.assignments[0].end_token_exclusive, 4);
    assert_eq!(binding.assignments[0].tier_id, DalucTierId(20));
    assert_eq!(binding.assignments[1].physical_page, 1);
    assert_eq!(binding.assignments[1].tier_id, DalucTierId(20));
    assert_eq!(binding.assignments[2].physical_page, 2);
    assert_eq!(binding.assignments[2].start_token, 8);
    assert_eq!(binding.assignments[2].end_token_exclusive, 10);
    assert_eq!(binding.assignments[2].tier_id, DalucTierId(10));
    assert!(binding
        .assignments
        .iter()
        .all(|assignment| assignment.generation == cache.generation()));
    binding.validate_observation(&observation).unwrap();
}

#[test]
fn rejects_observation_with_externally_recorded_unsubmitted_writes() {
    let Some(harness) = harness() else {
        return;
    };
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
    let k = source_buffer(&harness.device, 10, 2, 32);
    let v = source_buffer(&harness.device, 10, 2, 32);
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-fdal6-unsubmitted-append"),
        });
    cache.record_append(&mut encoder, &k, &v, 10).unwrap();
    let observation = cache.observation().unwrap();
    assert!(observation.has_unsubmitted_recorded_writes());

    let contract = contract(10, 2, 4, 4);
    let tiers = tiers(contract);
    let plan = route_by_recency(contract, 4, &tiers, &quotas(1, 2)).unwrap();
    assert_eq!(
        bind_paged_tier_plan(&observation, contract, &tiers, &plan),
        Err(DalucPagedTierBindingError::UnsubmittedRecordedWrites)
    );

    drop(encoder);
}

#[test]
fn rejects_routing_segments_that_do_not_match_page_geometry() {
    let Some(harness) = harness() else {
        return;
    };
    let cache = seeded_cache(&harness, 10);
    let observation = cache.observation().unwrap();
    let contract = contract(10, 2, 4, 4);
    let tiers = tiers(contract);
    let plan = route_by_recency(contract, 5, &tiers, &quotas(1, 1)).unwrap();

    assert!(matches!(
        bind_paged_tier_plan(&observation, contract, &tiers, &plan),
        Err(DalucPagedTierBindingError::SegmentSizeMismatch {
            segment_size: 5,
            page_size: 4,
        })
    ));
}

#[test]
fn rejects_contract_shape_and_page_geometry_mismatches() {
    let Some(harness) = harness() else {
        return;
    };
    let cache = seeded_cache(&harness, 10);
    let observation = cache.observation().unwrap();

    let wrong_heads = contract(10, 1, 4, 4);
    let wrong_head_tiers = tiers(wrong_heads);
    let wrong_head_plan =
        route_by_recency(wrong_heads, 4, &wrong_head_tiers, &quotas(1, 2)).unwrap();
    assert_eq!(
        bind_paged_tier_plan(
            &observation,
            wrong_heads,
            &wrong_head_tiers,
            &wrong_head_plan,
        ),
        Err(DalucPagedTierBindingError::KvHeadsMismatch {
            contract: 1,
            observation: 2,
        })
    );

    let wrong_pages = contract(10, 2, 5, 3);
    let wrong_page_tiers = tiers(wrong_pages);
    let wrong_page_plan =
        route_by_recency(wrong_pages, 5, &wrong_page_tiers, &quotas(1, 1)).unwrap();
    assert!(matches!(
        bind_paged_tier_plan(
            &observation,
            wrong_pages,
            &wrong_page_tiers,
            &wrong_page_plan,
        ),
        Err(DalucPagedTierBindingError::PageGeometryMismatch { .. })
    ));
}

#[test]
fn freshness_rejects_append_after_binding() {
    let Some(harness) = harness() else {
        return;
    };
    let mut cache = seeded_cache(&harness, 8);
    let binding = binding_for(&cache, 8);

    append_rows(&harness, &mut cache, 2);
    let current = cache.observation().unwrap();
    assert_eq!(
        binding.validate_observation(&current),
        Err(DalucPagedTierBindingError::BindingKvLenMismatch {
            binding: 8,
            observation: 10,
        })
    );
}

#[test]
fn freshness_rejects_truncate_reappend_even_when_length_returns_to_same_value() {
    let Some(harness) = harness() else {
        return;
    };
    let mut cache = seeded_cache(&harness, 10);
    let binding = binding_for(&cache, 10);
    assert_eq!(binding.branch_epoch, 0);

    cache.truncate(8).unwrap();
    append_rows(&harness, &mut cache, 2);
    let current = cache.observation().unwrap();
    assert_eq!(current.topology().telemetry().live_tokens, 10);
    assert_eq!(
        binding.validate_observation(&current),
        Err(DalucPagedTierBindingError::BindingBranchEpochMismatch {
            binding: 0,
            observation: 1,
        })
    );
}

#[test]
fn freshness_rejects_reset_reappend_even_when_length_returns_to_same_value() {
    let Some(harness) = harness() else {
        return;
    };
    let mut cache = seeded_cache(&harness, 10);
    let binding = binding_for(&cache, 10);
    assert_eq!(binding.generation, 0);

    cache.reset().unwrap();
    append_rows(&harness, &mut cache, 10);
    let current = cache.observation().unwrap();
    assert_eq!(current.topology().telemetry().live_tokens, 10);
    assert_eq!(
        binding.validate_observation(&current),
        Err(DalucPagedTierBindingError::BindingGenerationMismatch {
            binding: 0,
            observation: 1,
        })
    );
}

#[test]
fn freshness_rejects_current_observation_tainted_by_external_recording() {
    let Some(harness) = harness() else {
        return;
    };
    let mut cache = seeded_cache(&harness, 8);
    let binding = binding_for(&cache, 8);
    let k = source_buffer(&harness.device, 2, 2, 32);
    let v = source_buffer(&harness.device, 2, 2, 32);
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-fdal6-stale-binding-unsubmitted-append"),
        });
    cache.record_append(&mut encoder, &k, &v, 2).unwrap();

    let current = cache.observation().unwrap();
    assert_eq!(
        binding.validate_observation(&current),
        Err(DalucPagedTierBindingError::UnsubmittedRecordedWrites)
    );
    drop(encoder);
}

#[test]
fn freshness_rejects_mutated_binding_metadata() {
    let Some(harness) = harness() else {
        return;
    };
    let cache = seeded_cache(&harness, 10);
    let current = cache.observation().unwrap();

    let mut wrong_version = binding_for(&cache, 10);
    wrong_version.binding_version += 1;
    assert_eq!(
        wrong_version.validate_observation(&current),
        Err(DalucPagedTierBindingError::UnsupportedBindingVersion {
            actual: DA_LUC_PAGED_TIER_BINDING_VERSION + 1,
            supported: DA_LUC_PAGED_TIER_BINDING_VERSION,
        })
    );

    let mut wrong_page = binding_for(&cache, 10);
    wrong_page.assignments[0].physical_page = 3;
    assert_eq!(
        wrong_page.validate_observation(&current),
        Err(DalucPagedTierBindingError::BindingAssignmentPageMismatch {
            logical_page: 0,
        })
    );
}

#[test]
fn freshness_check_does_not_claim_cache_instance_identity() {
    let Some(harness) = harness() else {
        return;
    };
    let first = seeded_cache(&harness, 10);
    let second = seeded_cache(&harness, 10);
    let binding = binding_for(&first, 10);
    let second_observation = second.observation().unwrap();

    binding.validate_observation(&second_observation).unwrap();
}
