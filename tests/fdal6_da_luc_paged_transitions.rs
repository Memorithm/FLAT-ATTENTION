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
    DalucCacheScopedPagedTierBindingError as ScopedError, DalucPagedTierTransition,
    DalucPagedTierTransitionError as TransitionError, DA_LUC_PAGED_TIER_TRANSITION_VERSION,
};
use flat_attention::paged_kv::{PagedKvConfig, WgpuPagedKvCache, WgpuPagedKvCacheError};

fn harness() -> Option<(wgpu::Device, wgpu::Queue)> {
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
            panic!("FDAL6 page transition tests require a WGPU adapter");
        }
        eprintln!("WGPU unavailable; optional FDAL6 page transition tests skipped");
        return None;
    };
    Some(
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fdal6-page-transition-tests"),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .expect("request FDAL6 test device"),
    )
}

fn source(device: &wgpu::Device, rows: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fdal6-page-transition-source"),
        size: u64::try_from(rows.checked_mul(2 * 32 * 4).unwrap()).unwrap(),
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn append(device: &wgpu::Device, queue: &wgpu::Queue, cache: &mut WgpuPagedKvCache, rows: usize) {
    let k = source(device, rows);
    let v = source(device, rows);
    cache.append_and_submit(device, queue, &k, &v, rows).unwrap();
}

fn seeded(device: &wgpu::Device, queue: &wgpu::Queue) -> WgpuPagedKvCache {
    let mut cache = WgpuPagedKvCache::new(
        device,
        PagedKvConfig {
            page_size: 4,
            physical_pages: 4,
        },
        2,
        32,
    )
    .unwrap();
    append(device, queue, &mut cache, 10);
    cache
}

fn view(kv_len: usize) -> DalucKvViewContract {
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

fn catalog() -> [DalucPrecisionTier; 2] {
    let contract = view(10);
    [
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
    ]
}

fn bind(
    cache: &WgpuPagedKvCache,
    contract: DalucKvViewContract,
    tiers: &[DalucPrecisionTier],
) -> DalucCacheScopedPagedTierBinding {
    let segments = contract.shape.kv_len.div_ceil(4);
    let quotas = tiers
        .iter()
        .map(|tier| DalucTierQuota {
            tier_id: tier.id,
            segments: if tiers.len() == 1 {
                segments
            } else if tier.id == DalucTierId(10) {
                1
            } else if tier.id == DalucTierId(20) {
                segments - 1
            } else {
                0
            },
        })
        .collect::<Vec<_>>();
    let plan = route_by_recency(contract, 4, tiers, &quotas).unwrap();
    bind_paged_tier_plan_to_cache(cache, contract, tiers, &plan).unwrap()
}

#[test]
fn unchanged_assignment_is_empty_and_does_not_mutate_the_cache() {
    let Some((device, queue)) = harness() else {
        return;
    };
    let cache = seeded(&device, &queue);
    let before = cache.observation().unwrap();
    let previous = bind(&cache, view(10), &catalog());
    let next = previous.clone();
    let plan = next.transitions_from(&previous, &cache).unwrap();
    assert_eq!(plan.schema_version(), DA_LUC_PAGED_TIER_TRANSITION_VERSION);
    assert!(plan.transitions().is_empty());
    assert_eq!(plan.previous().contract(), view(10));
    assert_eq!(plan.next().tiers(), &catalog());
    plan.validate_cache(&cache).unwrap();
    assert_eq!(cache.observation().unwrap(), before);
    let mut copied = previous.contract();
    copied.layout.row_order = DalucRowOrder::BatchHeadToken;
    assert_ne!(previous.contract(), copied);
}

#[test]
fn reordered_catalog_produces_canonical_page_changes_with_partial_tail() {
    let Some((device, queue)) = harness() else {
        return;
    };
    let cache = seeded(&device, &queue);
    let before = cache.observation().unwrap();
    let tiers = catalog();
    let previous = bind(&cache, view(10), &tiers);
    let next = bind(&cache, view(10), &[tiers[1], tiers[0]]);
    let plan = next.transitions_from(&previous, &cache).unwrap();
    assert_eq!(
        plan.transitions(),
        &[
            DalucPagedTierTransition {
                logical_page: 0,
                physical_page: 0,
                start_token: 0,
                end_token_exclusive: 4,
                generation: cache.generation(),
                branch_epoch: cache.branch_epoch(),
                from_tier: DalucTierId(20),
                to_tier: DalucTierId(10),
            },
            DalucPagedTierTransition {
                logical_page: 2,
                physical_page: 2,
                start_token: 8,
                end_token_exclusive: 10,
                generation: cache.generation(),
                branch_epoch: cache.branch_epoch(),
                from_tier: DalucTierId(10),
                to_tier: DalucTierId(20),
            },
        ]
    );
    assert_eq!(
        next.transitions_from(&previous, &cache).unwrap().transitions(),
        plan.transitions()
    );
    plan.validate_cache(&cache).unwrap();
    assert_eq!(cache.observation().unwrap(), before);
}

#[test]
fn redefined_keys_or_values_are_rejected_even_with_equal_binding_metadata() {
    let Some((device, queue)) = harness() else {
        return;
    };
    let cache = seeded(&device, &queue);
    let previous = bind(&cache, view(10), &catalog());
    for change_keys in [false, true] {
        let mut tiers = catalog();
        if change_keys {
            tiers[1].keys.index_bit_order = DalucBitOrder::Msb0;
        } else {
            tiers[1].values = DalucValueRepresentation::Dense {
                dtype: DalucFloatDType::Bf16,
            };
        }
        let next = bind(&cache, view(10), &tiers);
        assert_eq!(previous.binding(), next.binding());
        assert_eq!(
            next.transitions_from(&previous, &cache).unwrap_err(),
            TransitionError::IncompatibleTierCatalogs
        );
    }
}

#[test]
fn added_or_missing_tier_ids_are_rejected() {
    let Some((device, queue)) = harness() else {
        return;
    };
    let cache = seeded(&device, &queue);
    let tiers = catalog();
    let previous = bind(&cache, view(10), &tiers);
    let extra = DalucPrecisionTier {
        id: DalucTierId(30),
        ..tiers[0]
    };
    for changed in [vec![tiers[0]], vec![tiers[0], tiers[1], extra]] {
        let next = bind(&cache, view(10), &changed);
        assert_eq!(
            next.transitions_from(&previous, &cache).unwrap_err(),
            TransitionError::IncompatibleTierCatalogs
        );
    }
}

#[test]
fn changed_row_order_is_rejected_even_when_page_metadata_matches() {
    let Some((device, queue)) = harness() else {
        return;
    };
    let cache = seeded(&device, &queue);
    let previous = bind(&cache, view(10), &catalog());
    let mut different = view(10);
    different.layout.row_order = DalucRowOrder::BatchHeadToken;
    let next = bind(&cache, different, &catalog());
    assert_eq!(previous.binding(), next.binding());
    assert_eq!(
        next.transitions_from(&previous, &cache).unwrap_err(),
        TransitionError::IncompatibleViewContracts
    );
}

#[test]
fn foreign_previous_next_and_replay_fail_closed_including_empty_plans() {
    let Some((device, queue)) = harness() else {
        return;
    };
    let cache = seeded(&device, &queue);
    let foreign = seeded(&device, &queue);
    let origin_binding = bind(&cache, view(10), &catalog());
    let foreign_binding = bind(&foreign, view(10), &catalog());
    let error = ScopedError::Cache(WgpuPagedKvCacheError::ForeignCheckpoint);
    assert_eq!(
        origin_binding
            .transitions_from(&foreign_binding, &cache)
            .unwrap_err(),
        TransitionError::PreviousBinding(error.clone())
    );
    assert_eq!(
        foreign_binding
            .transitions_from(&origin_binding, &cache)
            .unwrap_err(),
        TransitionError::NextBinding(error.clone())
    );
    let plan = origin_binding
        .transitions_from(&origin_binding, &cache)
        .unwrap();
    assert_eq!(
        plan.validate_cache(&foreign),
        Err(TransitionError::PreviousBinding(error))
    );
}

#[test]
fn replay_rejects_append_truncate_reappend_and_reset_reappend() {
    let Some((device, queue)) = harness() else {
        return;
    };
    for mutation in 0..3 {
        let mut cache = seeded(&device, &queue);
        let tiers = catalog();
        let previous = bind(&cache, view(10), &tiers);
        let next = bind(&cache, view(10), &[tiers[1], tiers[0]]);
        let plan = next.transitions_from(&previous, &cache).unwrap();
        let error = match mutation {
            0 => {
                append(&device, &queue, &mut cache, 1);
                ScopedError::Binding(DalucPagedTierBindingError::BindingKvLenMismatch {
                    binding: 10,
                    observation: 11,
                })
            }
            1 => {
                cache.truncate(8).unwrap();
                append(&device, &queue, &mut cache, 2);
                ScopedError::Cache(WgpuPagedKvCacheError::CheckpointBranchMismatch {
                    checkpoint_branch_epoch: 0,
                    current_branch_epoch: cache.branch_epoch(),
                })
            }
            _ => {
                cache.reset().unwrap();
                append(&device, &queue, &mut cache, 10);
                ScopedError::Cache(WgpuPagedKvCacheError::CheckpointGenerationMismatch {
                    checkpoint_generation: 0,
                    current_generation: cache.generation(),
                })
            }
        };
        assert_eq!(
            plan.validate_cache(&cache),
            Err(TransitionError::PreviousBinding(error))
        );
    }
}

#[test]
fn stale_previous_and_stale_next_are_both_rejected_at_plan_creation() {
    let Some((device, queue)) = harness() else {
        return;
    };
    let mut cache = seeded(&device, &queue);
    let old = bind(&cache, view(10), &catalog());
    append(&device, &queue, &mut cache, 1);
    let current = bind(&cache, view(11), &catalog());
    let error = ScopedError::Binding(DalucPagedTierBindingError::BindingKvLenMismatch {
        binding: 10,
        observation: 11,
    });
    assert_eq!(
        current.transitions_from(&old, &cache).unwrap_err(),
        TransitionError::PreviousBinding(error.clone())
    );
    assert_eq!(
        old.transitions_from(&current, &cache).unwrap_err(),
        TransitionError::NextBinding(error)
    );
}

#[test]
fn external_recording_rejects_plan_creation_and_replay() {
    let Some((device, queue)) = harness() else {
        return;
    };
    let mut cache = seeded(&device, &queue);
    let binding = bind(&cache, view(10), &catalog());
    let plan = binding.transitions_from(&binding, &cache).unwrap();
    let k = source(&device, 1);
    let v = source(&device, 1);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    cache.record_append(&mut encoder, &k, &v, 1).unwrap();
    let error = TransitionError::PreviousBinding(ScopedError::Binding(
        DalucPagedTierBindingError::UnsubmittedRecordedWrites,
    ));
    assert_eq!(plan.validate_cache(&cache), Err(error.clone()));
    assert_eq!(
        binding.transitions_from(&binding, &cache).unwrap_err(),
        error
    );
    drop(encoder);
}
